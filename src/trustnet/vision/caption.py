"""
Optional Image Captioning module for TrustNet.

Features:
- Disabled by default when caption_model is null.
- Fully lazy loading: models and weights are loaded into VRAM only when actively generating.
- Memory cleanup: explicitly unloads weights, executes garbage collection, and clears
  CUDA memory cache immediately after generation.
- Never writes uploaded images to disk.
"""

import gc
import logging
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from PIL import Image
import torch

from trustnet.vision.clip_scorer import _to_pil_image

logger = logging.getLogger(__name__)


class ImageCaptioner:
    """
    Lazy-loaded, optional vision-to-text caption generator.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        device: str = "auto",
    ):
        """
        Args:
            model_id: Hugging Face model identifier (e.g. 'Salesforce/blip-image-captioning-base')
                      or None to disable captioning.
            device: 'auto', 'cuda', or 'cpu'.
        """
        self.model_id = model_id
        self.enabled = model_id is not None and str(model_id).strip().lower() not in ("none", "null", "")

        if device == "auto":
            self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device_str = device

        if not self.enabled:
            logger.debug("ImageCaptioner is disabled (model_id is null).")
        else:
            logger.info(f"ImageCaptioner configured with model_id={self.model_id} on {self.device_str} (lazy load).")

    def generate_caption(
        self,
        image: Union[Image.Image, np.ndarray, bytes, str, Path],
        max_new_tokens: int = 50,
    ) -> Optional[str]:
        """
        Generates a natural language description of the image.
        If disabled, immediately returns None.
        If enabled, loads the model, generates the caption, and immediately frees memory.

        Args:
            image: In-memory image representation or file path.
            max_new_tokens: Maximum tokens in generated description.

        Returns:
            Generated caption string, or None if disabled/failed.
        """
        if not self.enabled:
            return None

        pil_img = _to_pil_image(image)

        caption: Optional[str] = None
        model = None
        processor = None

        try:
            logger.info(f"Lazily loading caption model '{self.model_id}' onto {self.device_str}...")
            # Lazy import to avoid importing transformers caption modules when unused
            from transformers import BlipForConditionalGeneration, BlipProcessor

            processor = BlipProcessor.from_pretrained(self.model_id)
            model = BlipForConditionalGeneration.from_pretrained(self.model_id)
            device = torch.device(self.device_str)
            model.to(device)
            model.eval()

            # Preprocess image into PyTorch tensor
            inputs = processor(images=pil_img, return_tensors="pt").to(device)

            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

            caption = processor.decode(generated_ids[0], skip_special_tokens=True).strip()
            logger.debug(f"Generated caption: '{caption}'")

        except Exception as e:
            logger.warning(f"Failed to generate caption with '{self.model_id}': {e}")
            caption = None

        finally:
            # Memory safety: explicitly delete model and clear GPU / CPU RAM
            if model is not None:
                del model
            if processor is not None:
                del processor

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return caption
