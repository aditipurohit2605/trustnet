"""
Central Model Registry and Hardware Lifecycle Manager for TrustNet Stage 10.

Manages:
1. One-time synchronous loading of core pipeline models:
   - Text classifier / NLI model
   - Retriever with BM25 and FAISS dense indexes
   - Cross-encoder reranker
   - CLIP multimodal semantic scorer
   - EasyOCR multilingual reader
   - Fusion heads (PyTorch MLP / LightGBM)
   - Temperature calibrator
   - Confidence abstainer
   - Source reliability weighting catalog
2. On-demand lazy loading of heavy secondary models (MarianMT translator, BLIP captioner)
   with automated idle-unload timer to release VRAM/RAM when inactive.
3. Diagnostic readiness checks and per-model memory measurement (GPU VRAM stats via
   torch.cuda memory tracking with safe CPU RSS fallback via psutil).
"""

import asyncio
import gc
import logging
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Union

import psutil
import torch

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Singleton-style registry managing model instances, memory profiling, and lazy idle lifecycles.
    """

    def __init__(
        self,
        idle_timeout_seconds: int = 60,
        lazy_model_names: Optional[List[str]] = None,
    ):
        self.idle_timeout_seconds = idle_timeout_seconds
        self.lazy_model_names = lazy_model_names or ["translator", "captioner"]

        self._core_models: Dict[str, Any] = {}
        self._lazy_models: Dict[str, Any] = {}
        self._lazy_last_used: Dict[str, float] = {}
        self._memory_stats: Dict[str, Dict[str, Any]] = {}
        self._idle_task: Optional[asyncio.Task] = None
        self._is_ready: bool = False

    def is_cuda_available(self) -> bool:
        """Checks if active PyTorch build has accessible CUDA hardware."""
        return torch.cuda.is_available()

    def _measure_load(self, model_name: str, loader_fn: Callable[[], Any]) -> Any:
        """
        Executes model loader function while profiling memory and latency:
        - If CUDA is available: measures `torch.cuda.memory_allocated()` delta.
        - Always: measures resident set size (RSS) memory delta via `psutil.Process()`.
        - Records elapsed wall-clock load duration via `time.perf_counter()`.
        """
        cuda = self.is_cuda_available()
        process = psutil.Process()

        # Pre-load baselines
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            gpu_before = torch.cuda.memory_allocated()
        else:
            gpu_before = 0

        cpu_before = process.memory_info().rss
        start_time = time.perf_counter()

        # Instantiate / load model
        try:
            model_instance = loader_fn()
        except Exception as e:
            logger.warning(f"Model loader for '{model_name}' encountered an error: {e}")
            model_instance = None

        load_duration = time.perf_counter() - start_time

        # Post-load measurements
        cpu_after = process.memory_info().rss
        if cuda:
            gpu_after = torch.cuda.memory_allocated()
            gpu_peak = torch.cuda.max_memory_allocated()
        else:
            gpu_after = 0
            gpu_peak = 0

        gpu_delta_mb = round((gpu_after - gpu_before) / (1024 * 1024), 2)
        cpu_delta_mb = round((cpu_after - cpu_before) / (1024 * 1024), 2)

        self._memory_stats[model_name] = {
            "loaded": model_instance is not None,
            "load_time_seconds": round(load_duration, 4),
            "gpu_allocated_delta_mb": max(0.0, gpu_delta_mb),
            "gpu_peak_mb": round(gpu_peak / (1024 * 1024), 2) if cuda else 0.0,
            "cpu_rss_delta_mb": max(0.0, cpu_delta_mb),
            "device": "cuda" if cuda else "cpu",
        }

        logger.info(
            f"Loaded model '{model_name}' in {load_duration:.3f}s. "
            f"[GPU Delta: {gpu_delta_mb} MB, CPU Delta: {cpu_delta_mb} MB]"
        )
        return model_instance

    def load_core_models(self) -> None:
        """
        Loads all required core pipeline models once at startup.
        Ensures models are never re-instantiated on a per-request basis.
        """
        logger.info("Initializing TrustNet Core Model Registry...")

        # 1. Source Reliability Assigner
        def _load_source_weights():
            from trustnet.reasoning.source_reliability import SourceReliabilityAssigner
            return SourceReliabilityAssigner(config_path="configs/source_reliability.yaml")
        self._core_models["source_weights"] = self._measure_load("source_weights", _load_source_weights)

        # 2. Temperature Calibrator
        def _load_calibrator():
            from trustnet.calibration.temperature import TemperatureScaler
            scaler = TemperatureScaler()
            cal_file = Path("outputs/stage7/temperature.json")
            if cal_file.exists():
                scaler.load(cal_file)
            return scaler
        self._core_models["calibrators"] = self._measure_load("calibrators", _load_calibrator)

        # 3. Abstainer
        def _load_abstainer():
            from trustnet.calibration.abstain import Abstainer
            abstainer = Abstainer()
            abst_file = Path("outputs/stage7/abstainer.json")
            if abst_file.exists():
                abstainer.load(abst_file)
            return abstainer
        self._core_models["abstainer"] = self._measure_load("abstainer", _load_abstainer)

        # 4. Multimodal Fusion Predictor
        def _load_fusion():
            from trustnet.fusion.predict import MultimodalPredictor
            return MultimodalPredictor.from_artifacts(
                temperature_path="outputs/stage7/temperature.json",
                abstainer_path="outputs/stage7/abstainer.json",
            )
        self._core_models["fusion_heads"] = self._measure_load("fusion_heads", _load_fusion)

        # 5. Text Extraction / ClaimExtractor
        def _load_extractor():
            from trustnet.text.extraction.extractor import ClaimExtractor
            return ClaimExtractor()
        self._core_models["extractor"] = self._measure_load("extractor", _load_extractor)

        # 6. Multi-Mode Retriever (BM25, FAISS Dense, Reranker)
        def _load_retriever():
            from trustnet.retrieval.retriever import Retriever
            retriever = Retriever(config_or_path="configs/stage3_retrieval.yaml")
            try:
                retriever.build_or_load_indices()
            except Exception as e:
                logger.warning(f"Retriever indices could not be loaded: {e}; using unindexed fallback.")
            return retriever
        self._core_models["retriever"] = self._measure_load("retriever", _load_retriever)
        if self._core_models["retriever"] is not None:
            self._core_models["reranker"] = getattr(self._core_models["retriever"], "reranker", None)
        else:
            self._core_models["reranker"] = None

        # 7. Vision Semantic Scorer (CLIP)
        def _load_clip():
            from trustnet.vision.clip_scorer import ClipScorer
            return ClipScorer(device="cuda" if self.is_cuda_available() else "cpu")
        self._core_models["clip"] = self._measure_load("clip", _load_clip)

        # 8. Vision OCR Reader
        def _load_ocr():
            from trustnet.vision.ocr import OCRReader
            return OCRReader(gpu=self.is_cuda_available())
        self._core_models["ocr"] = self._measure_load("ocr", _load_ocr)

        # 9. Text NLI / Conflict Model (if checkpoint exists)
        def _load_text_model():
            text_ckpt = Path("outputs/stage6/text_model_train_text")
            if text_ckpt.exists():
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                tok = AutoTokenizer.from_pretrained(str(text_ckpt))
                mdl = AutoModelForSequenceClassification.from_pretrained(str(text_ckpt))
                device = "cuda" if self.is_cuda_available() else "cpu"
                mdl.to(device)
                mdl.eval()
                return {"model": mdl, "tokenizer": tok}
            return None
        self._core_models["text_model"] = self._measure_load("text_model", _load_text_model)

        self._is_ready = True
        logger.info("Core Model Registry initialization complete.")

    def _load_lazy_model(self, name: str) -> Any:
        """Loads a designated lazy model on first request and records its memory profile."""
        if name == "translator":
            def _load_trans():
                from trustnet.lang.translate import Translator
                return Translator.get_instance(
                    device="cuda" if self.is_cuda_available() else "cpu",
                    lazy_load=False,
                )
            instance = self._measure_load("translator", _load_trans)
        elif name == "captioner":
            def _load_cap():
                from trustnet.vision.caption import ImageCaptioner
                return ImageCaptioner(device="cuda" if self.is_cuda_available() else "cpu")
            instance = self._measure_load("captioner", _load_cap)
        else:
            raise KeyError(f"Unknown lazy model '{name}'")

        self._lazy_models[name] = instance
        self._lazy_last_used[name] = time.time()
        return instance

    def get(self, name: str) -> Any:
        """
        Retrieves a loaded model instance by identifier.
        For lazy models, loads them transparently if not in memory and refreshes their idle timer.
        """
        if name in self._core_models:
            return self._core_models[name]

        if name in self.lazy_model_names:
            if name not in self._lazy_models or self._lazy_models[name] is None:
                self._load_lazy_model(name)
            self._lazy_last_used[name] = time.time()
            return self._lazy_models[name]

        raise KeyError(f"Model '{name}' is not registered in the ModelRegistry.")

    async def _idle_check_loop(self) -> None:
        """Background coroutine that regularly unloads idle lazy models to conserve VRAM/RAM."""
        while True:
            await asyncio.sleep(10)
            now = time.time()
            for name in list(self._lazy_models.keys()):
                instance = self._lazy_models.get(name)
                if instance is None:
                    continue
                last_used = self._lazy_last_used.get(name, 0.0)
                if now - last_used > self.idle_timeout_seconds:
                    logger.info(f"Lazy model '{name}' has been idle for >{self.idle_timeout_seconds}s; unloading from VRAM...")
                    try:
                        if hasattr(instance, "free_memory"):
                            instance.free_memory()
                        del self._lazy_models[name]
                        gc.collect()
                        if self.is_cuda_available():
                            torch.cuda.empty_cache()
                        logger.info(f"Unloaded lazy model '{name}'.")
                    except Exception as e:
                        logger.warning(f"Error unloading lazy model '{name}': {e}")

    def start_idle_timer(self) -> None:
        """Spawns the idle cleanup background task within the running event loop."""
        if self._idle_task is None or self._idle_task.done():
            loop = asyncio.get_event_loop()
            self._idle_task = loop.create_task(self._idle_check_loop())
            logger.info(f"Started lazy model idle watcher (timeout: {self.idle_timeout_seconds}s).")

    def stop_idle_timer(self) -> None:
        """Cancels the idle cleanup background task."""
        if self._idle_task and not self._idle_task.done():
            self._idle_task.cancel()
            self._idle_task = None
            logger.info("Stopped lazy model idle watcher.")

    def readiness(self) -> Dict[str, Any]:
        """Diagnostic report of registry operational readiness and loaded models."""
        lazy_status = {
            m: ("loaded" if self._lazy_models.get(m) is not None else "unloaded")
            for m in self.lazy_model_names
        }
        return {
            "ready": self._is_ready,
            "device": "cuda" if self.is_cuda_available() else "cpu",
            "cuda_available": self.is_cuda_available(),
            "core_models": {k: (v is not None) for k, v in self._core_models.items()},
            "lazy_models": lazy_status,
        }

    def memory_report(self) -> Dict[str, Any]:
        """
        Generates a comprehensive snapshot of system and per-model memory consumption:
        - Host RAM RSS via psutil
        - Dedicated GPU VRAM stats (allocated, reserved, peak) if CUDA is present
        - Per-model initial delta measurements
        """
        process = psutil.Process()
        cpu_rss_mb = round(process.memory_info().rss / (1024 * 1024), 2)

        cuda = self.is_cuda_available()
        gpu_allocated_mb = round(torch.cuda.memory_allocated() / (1024 * 1024), 2) if cuda else 0.0
        gpu_reserved_mb = round(torch.cuda.memory_reserved() / (1024 * 1024), 2) if cuda else 0.0
        gpu_peak_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2) if cuda else 0.0

        return {
            "cuda_available": cuda,
            "device": "cuda" if cuda else "cpu",
            "total_cpu_rss_mb": cpu_rss_mb,
            "gpu_allocated_mb": gpu_allocated_mb,
            "gpu_reserved_mb": gpu_reserved_mb,
            "gpu_peak_mb": gpu_peak_mb,
            "per_model_breakdown": self._memory_stats,
        }
