"""
Image Analyzer orchestrator for TrustNet Stage 5.

Coordinates visual signal extraction:
- Optical Character Recognition (OCR) with spatial reading order
- Safe EXIF metadata extraction (strictly omitting GPS)
- Visual quality and screenshot heuristics
- Reverse image reuse search against the indexed image bank
- CLIP vision-language semantic alignment and mismatch scoring
- Optional lazy image captioning

Robustness guarantee:
Each submodule is isolated with try/except wrappers so an individual component failure
leaves that specific field empty/None, logs a warning, and allows the remaining signals
to be extracted safely without terminating execution.

Privacy guarantee:
Zero disk persistence of input images.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image
import yaml

from trustnet.schemas import ImageSignals, ReuseMatch
from trustnet.vision.caption import ImageCaptioner
from trustnet.vision.clip_scorer import ClipScorer, MismatchCalibrator
from trustnet.vision.exif import ExifReader
from trustnet.vision.image_bank import ImageBank
from trustnet.vision.ocr import OCRReader
from trustnet.vision.quality import QualityChecker

logger = logging.getLogger(__name__)


class ImageAnalyzer:
    """
    Unified analyzer producing ImageSignals from in-memory images.
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        ocr_reader: Optional[OCRReader] = None,
        clip_scorer: Optional[ClipScorer] = None,
        calibrator: Optional[MismatchCalibrator] = None,
        image_bank: Optional[ImageBank] = None,
        captioner: Optional[ImageCaptioner] = None,
        quality_checker: Optional[QualityChecker] = None,
    ):
        """
        Initializes the analyzer either from a YAML configuration file or provided components.
        """
        # Load config if available
        cfg: Dict[str, Any] = {}
        if config_path is not None:
            cpath = Path(config_path)
            if cpath.exists():
                with open(cpath, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}

        # 1. OCR Reader
        if ocr_reader is not None:
            self.ocr_reader = ocr_reader
        else:
            langs = cfg.get("ocr_languages", ["en", "hi"])
            min_conf = float(cfg.get("ocr_min_confidence", 0.3))
            self.ocr_reader = OCRReader(languages=langs, min_confidence=min_conf, gpu=False)

        # 2. Quality Checker
        self.quality_checker = quality_checker or QualityChecker()

        # 3. Clip Scorer
        if clip_scorer is not None:
            self.clip_scorer = clip_scorer
        else:
            clip_model = cfg.get("clip_model", "ViT-B-32")
            pretrained = cfg.get("pretrained", "laion2b_s34b_b79k")
            device = cfg.get("device", "auto")
            fp16 = bool(cfg.get("fp16", True))
            text_encoder_mode = cfg.get("text_encoder_mode", "clip")
            self.clip_scorer = ClipScorer(
                model_name=clip_model,
                pretrained=pretrained,
                device=device,
                fp16=fp16,
                text_encoder_mode=text_encoder_mode,
            )

        # 4. Mismatch Calibrator
        self.calibrator = calibrator or MismatchCalibrator()

        # 5. Image Bank
        if image_bank is not None:
            self.image_bank = image_bank
        else:
            phash_size = int(cfg.get("phash_size", 8))
            phash_thresh = int(cfg.get("phash_match_threshold", 10))
            embed_thresh = float(cfg.get("embedding_match_threshold", 0.85))
            bank_dir = cfg.get("image_bank_dir", "data/image_bank")

            self.image_bank = ImageBank(
                clip_scorer=self.clip_scorer,
                phash_size=phash_size,
                phash_match_threshold=phash_thresh,
                embedding_match_threshold=embed_thresh,
            )
            # If bank directory exists on disk, index it
            if bank_dir and Path(bank_dir).exists():
                self.image_bank.build_from_directory(bank_dir)

        # 6. Optional Captioner
        if captioner is not None:
            self.captioner = captioner
        else:
            caption_model = cfg.get("caption_model")
            device = cfg.get("device", "auto")
            self.captioner = ImageCaptioner(model_id=caption_model, device=device)

    def analyze(
        self,
        image: Union[Image.Image, np.ndarray, bytes, str, Path],
        claim_text: Optional[str] = None,
    ) -> ImageSignals:
        """
        Runs full multimodal feature extraction for an image and an optional claim.
        Zero disk persistence is performed.

        Args:
            image: In-memory image representation (PIL, numpy array, bytes) or file path.
            claim_text: Optional text of the claim being verified against the visual evidence.

        Returns:
            ImageSignals instance containing extracted signals.
        """
        signals = ImageSignals()

        # 1. EXIF Metadata Extraction (Privacy safe: strictly excludes GPS)
        try:
            signals.exif_summary = ExifReader.extract(image)
        except Exception as e:
            logger.warning(f"ImageAnalyzer: EXIF extraction failed: {e}")
            signals.exif_summary = {}

        # 2. Quality and Forensics Heuristics
        try:
            signals.quality_flags = self.quality_checker.analyze(image)
        except Exception as e:
            logger.warning(f"ImageAnalyzer: Quality analysis failed: {e}")
            signals.quality_flags = []

        # 3. Optical Character Recognition (OCR)
        try:
            text, conf, lang = self.ocr_reader.read(image)
            if text:
                signals.ocr_text = text
                signals.ocr_confidence = round(conf, 4)
                signals.ocr_language = lang
        except Exception as e:
            logger.warning(f"ImageAnalyzer: OCR step failed: {e}")
            signals.ocr_text = None
            signals.ocr_confidence = None
            signals.ocr_language = None

        # 4. Image Bank Reuse Search
        try:
            signals.reuse_matches = self.image_bank.search(image)
        except Exception as e:
            logger.warning(f"ImageAnalyzer: Reuse search failed: {e}")
            signals.reuse_matches = []

        # 5. Vision-Language Alignment & Mismatch Scoring (if claim_text provided)
        if claim_text and claim_text.strip():
            try:
                sim = self.clip_scorer.similarity(image, claim_text.strip())
                mismatch = self.calibrator.predict_mismatch(sim)
                signals.clip_similarity = round(float(sim), 4)
                signals.mismatch_score = round(float(mismatch), 4)
            except Exception as e:
                logger.warning(f"ImageAnalyzer: CLIP scoring failed: {e}")
                signals.clip_similarity = None
                signals.mismatch_score = None

        # 6. Optional Image Captioning (if enabled)
        if self.captioner.enabled:
            try:
                signals.caption = self.captioner.generate_caption(image)
            except Exception as e:
                logger.warning(f"ImageAnalyzer: Captioning step failed: {e}")
                signals.caption = None

        return signals
