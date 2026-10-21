"""
Pipeline Runner and Orchestration Engine for TrustNet Stage 10.

Executes verification variants ('text_only', 'text_retrieval', 'text_image', 'full')
with:
1. Dynamic per-stage progress reporting and latency profiling.
2. Component-level fault tolerance (graceful degradation without job failure).
3. ReportEnvelope construction with git version provenance and configuration fingerprinting.
"""

import hashlib
import json
import logging
from pathlib import Path
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional, Union
import uuid

import yaml

from trustnet.api.schemas import PipelineVariant, ReportEnvelope, VerifyOptions
from trustnet.pipeline import (
    apply_reasoning_to_report,
    run_extraction_for_image_text,
    run_retrieval_for_message,
    verify_claim,
)
from trustnet.schemas import Claim, ClaimReport, Evidence, ExtractedClaim, ImageSignals

logger = logging.getLogger(__name__)


def get_git_commit_hash() -> str:
    """Attempts to retrieve the current repository git commit hash."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=2,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()[:16]
    except Exception:
        pass
    return "0.1.0-release"


def compute_config_hash(config_path: Union[str, Path] = "configs/stage10_api.yaml") -> str:
    """Computes a deterministic SHA-256 fingerprint of the active pipeline configuration."""
    try:
        p = Path(config_path)
        if p.exists():
            content = p.read_bytes()
            return hashlib.sha256(content).hexdigest()[:16]
    except Exception:
        pass
    return "default_cfg_hash"


class PipelineRunner:
    """
    Executes claim verification for asynchronous jobs using pre-loaded models from ModelRegistry.
    """

    def __init__(self, registry: Any, config: Dict[str, Any]):
        self.registry = registry
        self.config = config
        self.pipeline_version = get_git_commit_hash()
        self.config_hash = compute_config_hash()

    def run_job(
        self,
        text: str,
        image_path: Optional[str] = None,
        options: Optional[VerifyOptions] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> ReportEnvelope:
        """
        Executes verification across distinct stages with fine-grained error capture and timing.
        """
        options = options or VerifyOptions()
        variant = options.variant.value if hasattr(options.variant, "value") else str(options.variant)
        report_id = f"rep_{uuid.uuid4().hex[:16]}"

        timings: Dict[str, float] = {}
        degraded: List[str] = []
        warnings: List[str] = []
        reports: List[ClaimReport] = []

        total_start = time.perf_counter()

        def _update_progress(stage: str, progress: float) -> None:
            if progress_callback:
                try:
                    progress_callback(stage, progress)
                except Exception as e:
                    logger.debug(f"Progress callback failed: {e}")

        # ----------------------------------------------------------------------
        # STAGE 1: Claim Extraction & Vision Signal Ingestion (0.0 -> 0.25)
        # ----------------------------------------------------------------------
        _update_progress("claim_extraction", 0.10)
        t_start = time.perf_counter()

        extracted_claims: List[ExtractedClaim] = []
        source_mentions: List[Any] = []
        image_signals: Optional[ImageSignals] = None

        # 1a. Extract text claims
        try:
            extractor = self.registry.get("extractor")
            if text and text.strip():
                ext_res = extractor.extract(text)
                extracted_claims.extend([c for c in ext_res.claims if c.checkable])
                source_mentions.extend(ext_res.source_mentions)
        except Exception as e:
            logger.warning(f"Claim extraction degraded on text: {e}")
            degraded.append("claim_extractor")
            warnings.append(f"Text claim extraction failed: {str(e)}")

        # 1b. Ingest image signals if variant utilizes images and image is present
        if variant in ("text_image", "full") and image_path:
            try:
                ocr_reader = self.registry.get("ocr")
                clip_scorer = self.registry.get("clip")

                ocr_text, ocr_conf, ocr_lang = ocr_reader.read(image_path)
                clip_sim = clip_scorer.score(image_path, text) if text else None

                image_signals = ImageSignals(
                    ocr_text=ocr_text,
                    ocr_confidence=ocr_conf,
                    ocr_language=ocr_lang,
                    clip_similarity=clip_sim,
                )

                # Extract claims from OCR text if available
                if ocr_text and ocr_text.strip():
                    img_ext = run_extraction_for_image_text(
                        image=image_path,
                        extractor=extractor,
                        ocr_reader=ocr_reader,
                    )
                    extracted_claims.extend(img_ext.get("checkable_claims", []))
            except Exception as e:
                logger.warning(f"Image analysis degraded: {e}")
                degraded.append("vision_signals")
                warnings.append(f"Vision feature extraction failed: {str(e)}")

        # Deduplicate checkable claims by normalized text
        seen_texts = set()
        unique_claims: List[Claim] = []
        for c in extracted_claims:
            norm_key = c.text.strip().lower()
            if norm_key not in seen_texts:
                seen_texts.add(norm_key)
                unique_claims.append(c.to_claim())

        # If no atomic claims could be extracted but text was provided, verify text as single claim
        if not unique_claims and text and text.strip():
            claim_id = f"claim_{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}"
            unique_claims.append(Claim(id=claim_id, text=text.strip()))

        # Enforce max claims per message limit
        max_claims = self.config.get("max_claims_per_message", 10)
        if len(unique_claims) > max_claims:
            warnings.append(f"Input contained {len(unique_claims)} claims; truncated to {max_claims}.")
            unique_claims = unique_claims[:max_claims]

        timings["claim_extraction"] = round(time.perf_counter() - t_start, 4)

        # ----------------------------------------------------------------------
        # STAGE 2: Evidence Retrieval (0.25 -> 0.50)
        # ----------------------------------------------------------------------
        _update_progress("evidence_retrieval", 0.35)
        t_start = time.perf_counter()

        claim_evidence_map: Dict[str, List[Evidence]] = {}

        if variant in ("text_retrieval", "full"):
            try:
                retriever = self.registry.get("retriever")
                for claim in unique_claims:
                    ev_items = retriever.search(
                        claim_text=claim.text,
                        mode="hybrid_rerank" if self.registry.get("reranker") else "hybrid",
                        top_k=5,
                    )
                    claim_evidence_map[claim.id] = ev_items
            except Exception as e:
                logger.warning(f"Retriever degraded: {e}")
                degraded.append("retriever")
                warnings.append(f"Evidence retrieval failed: {str(e)}; continuing with empty evidence.")
                for claim in unique_claims:
                    claim_evidence_map[claim.id] = []
        else:
            # text_only or text_image variant deliberately skips retrieval
            for claim in unique_claims:
                claim_evidence_map[claim.id] = []

        timings["evidence_retrieval"] = round(time.perf_counter() - t_start, 4)

        # ----------------------------------------------------------------------
        # STAGE 3: Calibrated Multimodal Prediction & Abstention (0.50 -> 0.75)
        # ----------------------------------------------------------------------
        _update_progress("multimodal_prediction", 0.60)
        t_start = time.perf_counter()

        predictor = self.registry.get("fusion_heads")

        for claim in unique_claims:
            ev = claim_evidence_map.get(claim.id, [])
            try:
                # Use image_path only if variant includes images
                active_image_path = image_path if variant in ("text_image", "full") else None
                report = verify_claim(
                    claim=claim,
                    evidence=ev,
                    image_path=active_image_path,
                    predictor=predictor,
                    enable_reasoning=False,  # Reasoning executed separately in Stage 4 for granular timing
                )
                reports.append(report)
            except Exception as e:
                logger.warning(f"Prediction failed for claim {claim.id}: {e}")
                degraded.append("prediction_fusion")
                warnings.append(f"Fusion model prediction degraded for claim: {str(e)}")
                # Construct graceful fallback report
                from trustnet.schemas import Verdict
                fallback_report = ClaimReport(
                    claim=claim,
                    evidence=ev,
                    verdict=Verdict.INSUFFICIENT_EVIDENCE,
                    confidence=0.3333,
                    abstained=True,
                    abstention_reason=f"Prediction degraded due to component error: {e}",
                )
                reports.append(fallback_report)

        timings["multimodal_prediction"] = round(time.perf_counter() - t_start, 4)

        # ----------------------------------------------------------------------
        # STAGE 4: Stage 8 Reasoning Layer (0.75 -> 0.95)
        # ----------------------------------------------------------------------
        if options.enable_reasoning:
            _update_progress("reasoning_layer", 0.85)
            t_start = time.perf_counter()

            enriched_reports: List[ClaimReport] = []
            for report in reports:
                try:
                    enriched = apply_reasoning_to_report(
                        report=report,
                        reasoning_config="configs/stage8_reasoning.yaml",
                        image_signals=image_signals,
                        source_mentions=source_mentions,
                        original_message=text,
                    )
                    enriched_reports.append(enriched)
                except Exception as e:
                    logger.warning(f"Reasoning layer degraded: {e}")
                    degraded.append("reasoning_layer")
                    warnings.append(f"Reasoning analysis failed: {str(e)}")
                    enriched_reports.append(report)
            reports = enriched_reports
            timings["reasoning_layer"] = round(time.perf_counter() - t_start, 4)

        # ----------------------------------------------------------------------
        # STAGE 5: Finalizing & Envelope Packaging (0.95 -> 1.0)
        # ----------------------------------------------------------------------
        _update_progress("finalizing", 0.98)
        timings["total"] = round(time.perf_counter() - total_start, 4)

        # Model versions catalog
        model_versions = {
            "retriever": "BAAI/bge-m3",
            "reranker": "BAAI/bge-reranker-v2-m3",
            "text_classifier": "outputs/stage6/text_model_train_text",
            "fusion": "mlp_stage6",
            "calibrator": "temperature_scaling_stage7",
            "abstainer": "abstainer_stage7",
            "clip": "openclip_ViT-B-32",
            "ocr": "easyocr_multilingual",
        }

        envelope = ReportEnvelope(
            report_id=report_id,
            pipeline_version=self.pipeline_version,
            model_versions=model_versions,
            config_hash=self.config_hash,
            variant=variant,
            timings=timings,
            degraded=list(dict.fromkeys(degraded)),  # Deduplicate while preserving order
            claims=reports,
            original_text=text,
            warnings=warnings,
        )

        _update_progress("done", 1.0)
        return envelope
