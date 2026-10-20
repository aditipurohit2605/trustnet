"""
Vision-language scoring and calibration using OpenCLIP.

Features:
- Batched image and text encoding with L2 normalization.
- Half-precision (fp16) acceleration when running on CUDA; safe fp32 fallback on CPU.
- Cosine similarity computation between images and claims.
- MismatchCalibrator: converts raw CLIP cosine similarity into a calibrated mismatch score
  in [0, 1] via empirical percentile ranking against a validation distribution of matched pairs.
- Serialization of calibrator models to and from JSON.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import open_clip
from PIL import Image
import torch

from trustnet.vision.ocr import _to_numpy_rgb

logger = logging.getLogger(__name__)


def _to_pil_image(image: Union[Image.Image, np.ndarray, bytes, str, Path]) -> Image.Image:
    """
    Converts various input formats into a PIL RGB Image in RAM without persisting to disk.
    """
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    rgb_np = _to_numpy_rgb(image)
    return Image.fromarray(rgb_np)


class ClipScorer:
    """
    Computes vision-language semantic alignment scores using OpenCLIP models.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "auto",
        fp16: bool = True,
        text_encoder_mode: str = "translate_first",
    ):
        """
        Args:
            model_name: Architecture tag (e.g. 'ViT-B-32' or multilingual 'xlm-roberta-base-ViT-B-32').
            pretrained: Checkpoint tag (e.g. 'laion2b_s34b_b79k' or multilingual 'laion5b_s13b_b90k').
            device: 'auto', 'cuda', or 'cpu'.
            fp16: Whether to use half-precision on GPU (auto-disabled on CPU).
            text_encoder_mode: 'translate_first' (translates Hindi/Hinglish to English before encoding)
                              or 'multilingual' (uses native multilingual tokenizer like xlm-roberta-base-ViT-B-32).
        """
        self.model_name = model_name
        self.pretrained = pretrained
        self.text_encoder_mode = text_encoder_mode
        self._is_native_multilingual = "xlm" in model_name.lower() or "muril" in model_name.lower()

        # Resolve compute device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # PyTorch CPU half-precision does not support all operations natively;
        # enable fp16 strictly on CUDA devices.
        self.use_fp16 = fp16 and (self.device.type == "cuda")

        logger.info(
            f"Loading OpenCLIP ({model_name}, {pretrained}) on device={self.device}, "
            f"fp16={self.use_fp16}, text_encoder_mode={text_encoder_mode}..."
        )

        # Load OpenCLIP model, transform pipeline, and tokenizer
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
            device=self.device,
        )
        self.model.eval()

        if self.use_fp16:
            self.model = self.model.half()

        self.tokenizer = open_clip.get_tokenizer(self.model_name)

    def encode_image(
        self,
        images: Union[Image.Image, np.ndarray, bytes, str, Path, List[Any]],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Encodes one or more images into L2-normalized CLIP visual embeddings.

        Args:
            images: Single image or list of images.
            batch_size: Mini-batch size for inference.

        Returns:
            numpy array of shape [N, embedding_dim] normalized to unit L2 norm.
        """
        if not isinstance(images, list):
            images = [images]

        all_embeddings: List[np.ndarray] = []

        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch_imgs = images[i : i + batch_size]
                # Preprocess PIL images to torch tensors [C, H, W]
                tensors = [self.preprocess(_to_pil_image(img)) for img in batch_imgs]
                batch_tensor = torch.stack(tensors).to(self.device)

                if self.use_fp16:
                    batch_tensor = batch_tensor.half()

                # Extract visual features from the vision transformer trunk
                features = self.model.encode_image(batch_tensor)
                # L2 normalize so dot product equals cosine similarity
                normalized = torch.nn.functional.normalize(features, p=2, dim=-1)
                all_embeddings.append(normalized.float().cpu().numpy())

        if not all_embeddings:
            return np.empty((0, 512), dtype=np.float32)

        return np.concatenate(all_embeddings, axis=0)

    def encode_text(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Encodes one or more text queries into L2-normalized CLIP text embeddings.

        Args:
            texts: Single string or list of strings.
            batch_size: Mini-batch size for inference.

        Returns:
            numpy array of shape [N, embedding_dim] normalized to unit L2 norm.
        """
        if isinstance(texts, str):
            texts = [texts]

        # Handle text_encoder_mode for non-English queries
        processed_texts: List[str] = []
        for t in texts:
            needs_translation = False
            # Check for Devanagari unicode range \u0900-\u097F or non-ASCII
            has_devanagari = any("\u0900" <= ch <= "\u097f" for ch in t)

            if self.text_encoder_mode == "translate_first":
                needs_translation = has_devanagari
            elif self.text_encoder_mode == "multilingual":
                # If model is not natively multilingual (e.g., standard English ViT-B-32),
                # translate to avoid OOV token collapse while documenting requirement
                if not self._is_native_multilingual and has_devanagari:
                    logger.debug(
                        f"Current OpenCLIP checkpoint '{self.model_name}' is English-only. "
                        f"To tokenize natively, configure 'xlm-roberta-base-ViT-B-32' (tag: 'laion5b_s13b_b90k'). "
                        f"Translating query to English for compatibility."
                    )
                    needs_translation = True

            if needs_translation:
                try:
                    from trustnet.lang.translate import Translator
                    translator = Translator.get_instance()
                    translated = translator.translate(t, src_lang="hi", tgt_lang="en")
                    processed_texts.append(translated)
                except Exception as ex:
                    logger.warning(f"ClipScorer translation failed for '{t}': {ex}")
                    processed_texts.append(t)
            else:
                processed_texts.append(t)

        all_embeddings: List[np.ndarray] = []

        with torch.no_grad():
            for i in range(0, len(processed_texts), batch_size):
                batch_texts = processed_texts[i : i + batch_size]
                tokens = self.tokenizer(batch_texts).to(self.device)

                features = self.model.encode_text(tokens)
                normalized = torch.nn.functional.normalize(features, p=2, dim=-1)
                all_embeddings.append(normalized.float().cpu().numpy())

        if not all_embeddings:
            return np.empty((0, 512), dtype=np.float32)

        return np.concatenate(all_embeddings, axis=0)

    def similarity(
        self,
        image: Union[Image.Image, np.ndarray, bytes, str, Path],
        text: str,
    ) -> float:
        """
        Computes cosine similarity between an image and a text string.

        Returns:
            Cosine similarity float in [-1.0, 1.0].
        """
        img_emb = self.encode_image(image)  # [1, D]
        txt_emb = self.encode_text(text)   # [1, D]

        # Inner product of normalized vectors = cosine similarity
        sim = float(np.dot(img_emb[0], txt_emb[0]))
        return float(np.clip(sim, -1.0, 1.0))

    def batch_similarity(
        self,
        images: List[Any],
        texts: List[str],
        batch_size: int = 32,
    ) -> List[float]:
        """
        Computes pairwise cosine similarities for aligned (image_i, text_i) pairs.
        """
        if len(images) != len(texts):
            raise ValueError(f"Images count ({len(images)}) != texts count ({len(texts)})")

        img_embs = self.encode_image(images, batch_size=batch_size)
        txt_embs = self.encode_text(texts, batch_size=batch_size)

        # Vectorized row-wise dot product
        similarities = np.sum(img_embs * txt_embs, axis=1)
        return [float(np.clip(s, -1.0, 1.0)) for s in similarities]


class MismatchCalibrator:
    """
    Calibrates raw CLIP image-text similarity into a probabilistic mismatch score [0, 1].

    Methodology:
    Fit on empirical matched image-text pairs (e.g. validation split).
    A high cosine similarity (e.g., matching or exceeding the 95th percentile of matched pairs)
    corresponds to a low mismatch score near 0.0.
    A low cosine similarity (falling below the matched distribution) corresponds to a high
    mismatch score approaching 1.0.

    Percentile formula:
      mismatch_score = 1.0 - percentile_rank(raw_similarity, matched_distribution)
    where percentile_rank = fraction of matched validation pairs with similarity <= raw_similarity.
    """

    def __init__(self, sorted_similarities: Optional[List[float]] = None):
        if sorted_similarities:
            self.sorted_similarities = np.array(sorted(sorted_similarities), dtype=np.float32)
        else:
            self.sorted_similarities = np.array([], dtype=np.float32)

    @property
    def is_fitted(self) -> bool:
        return len(self.sorted_similarities) > 0

    def fit(self, similarities: List[float]) -> "MismatchCalibrator":
        """
        Fits the calibrator using an array of raw cosine similarities from matched pairs.
        """
        clean = [float(s) for s in similarities if not np.isnan(s)]
        if not clean:
            raise ValueError("Cannot fit MismatchCalibrator with empty or NaN similarities.")
        self.sorted_similarities = np.array(sorted(clean), dtype=np.float32)
        logger.info(
            f"MismatchCalibrator fit on {len(self.sorted_similarities)} matched pairs. "
            f"Min: {self.sorted_similarities[0]:.4f}, "
            f"Median: {np.median(self.sorted_similarities):.4f}, "
            f"Max: {self.sorted_similarities[-1]:.4f}"
        )
        return self

    def predict_mismatch(self, raw_similarity: float) -> float:
        """
        Transforms a raw cosine similarity into a mismatch score in [0.0, 1.0].
        Higher value = more mismatched.
        """
        if not self.is_fitted:
            # Fallback heuristic if calibrator is not yet fitted:
            # Maps typical CLIP range [0.10, 0.35] inversely to [1.0, 0.0]
            clipped = np.clip(raw_similarity, 0.10, 0.35)
            return float(1.0 - (clipped - 0.10) / (0.35 - 0.10))

        # Compute empirical percentile rank via binary search
        n = len(self.sorted_similarities)
        idx = np.searchsorted(self.sorted_similarities, raw_similarity, side="right")
        percentile_rank = float(idx) / float(n)

        # Mismatch score is 1.0 minus the matched percentile rank
        # If raw_similarity is higher than all matched pairs, rank = 1.0 -> mismatch = 0.0
        # If raw_similarity is lower than all matched pairs, rank = 0.0 -> mismatch = 1.0
        mismatch_score = float(np.clip(1.0 - percentile_rank, 0.0, 1.0))
        return mismatch_score

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Saves calibrator parameters and quantiles to JSON.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        if not self.is_fitted:
            raise ValueError("Cannot save unfitted MismatchCalibrator.")

        quantiles = {
            f"p{q}": float(np.percentile(self.sorted_similarities, q))
            for q in [0, 5, 10, 25, 50, 75, 90, 95, 100]
        }

        data = {
            "type": "MismatchCalibrator_percentile",
            "num_samples": int(len(self.sorted_similarities)),
            "quantiles": quantiles,
            "sorted_similarities": self.sorted_similarities.tolist(),
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved MismatchCalibrator to {path}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "MismatchCalibrator":
        """
        Loads a saved MismatchCalibrator from JSON.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Calibrator file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        sims = data.get("sorted_similarities", [])
        calibrator = cls(sorted_similarities=sims)
        logger.info(f"Loaded MismatchCalibrator from {path} with {len(sims)} calibration points.")
        return calibrator
