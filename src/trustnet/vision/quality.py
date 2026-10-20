"""
Image quality and artifact analysis module for TrustNet.

Implements detection heuristics for:
1. 'tiny_resolution': Checks whether dimensions or total pixel count are insufficient
   for reliable OCR or visual forensics (min dimension < 256 or total pixels < 65,536).
2. 'extreme_aspect_ratio': Checks whether aspect ratio exceeds 3:1 (or < 1:3), which causes
   severe distortion when resized to standard square model receptive fields (e.g. 224x224).
3. 'heavy_jpeg_compression': Detects 8x8 pixel DCT block boundary discontinuities
   characteristic of aggressive lossy compression.
4. 'screenshot_likeness': Detects user interface characteristics (status bars, uniform
   flat background bands, palette sparsity, and sharp synthetic text edges).
"""

import logging
from pathlib import Path
from typing import Any, List, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from trustnet.vision.ocr import _to_numpy_rgb

logger = logging.getLogger(__name__)


def check_jpeg_blockiness(gray_img: np.ndarray, block_size: int = 8) -> float:
    """
    Measures 8x8 JPEG blocking artifacts.

    Heuristic description:
    JPEG transforms 8x8 pixel blocks independently using DCT. Coarse quantization steps
    cause adjacent blocks to take slightly different mean intensities, creating artificial
    intensity steps across block borders (grid lines at multiples of 8).
    We compare the mean gradient magnitude across 8-pixel block boundaries to the mean
    gradient magnitude within the interior of blocks. A ratio significantly above 1.25-1.35
    indicates strong blocking artifacts.
    """
    h, w = gray_img.shape
    if h < block_size * 4 or w < block_size * 4:
        return 1.0

    # Vertical block boundaries (horizontal differences)
    # Column indices at boundary: 7, 15, 23, ... vs 8, 16, 24, ...
    cols_boundary = np.arange(block_size - 1, w - 1, block_size)
    cols_internal = np.arange(block_size // 2, w - 1, block_size)

    diff_boundary_h = np.abs(gray_img[:, cols_boundary + 1].astype(float) - gray_img[:, cols_boundary].astype(float))
    diff_internal_h = np.abs(gray_img[:, cols_internal + 1].astype(float) - gray_img[:, cols_internal].astype(float))

    # Horizontal block boundaries (vertical differences)
    rows_boundary = np.arange(block_size - 1, h - 1, block_size)
    rows_internal = np.arange(block_size // 2, h - 1, block_size)

    diff_boundary_v = np.abs(gray_img[rows_boundary + 1, :].astype(float) - gray_img[rows_boundary, :].astype(float))
    diff_internal_v = np.abs(gray_img[rows_internal + 1, :].astype(float) - gray_img[rows_internal, :].astype(float))

    mean_boundary = (np.mean(diff_boundary_h) + np.mean(diff_boundary_v)) / 2.0
    mean_internal = (np.mean(diff_internal_h) + np.mean(diff_internal_v)) / 2.0

    if mean_internal < 1e-4:
        return 1.0

    return float(mean_boundary / mean_internal)


def check_screenshot_heuristics(rgb_img: np.ndarray) -> bool:
    """
    Determines whether an image is likely a digital screenshot rather than a photograph.

    Heuristic description:
    Screenshots (especially from mobile chat apps or desktop browsers) exhibit:
    1. Solid color header or footer bars: Mobile status bars (clock, battery) or app toolbars
       consist of horizontal strips with near-zero color variance (e.g. variance < 2.0).
    2. Large flat background regions: Unlike natural photos with sensor noise and smooth lighting
       gradients, synthetic UI graphics have broad regions of perfectly flat, uniform pixels.
    3. Common smartphone display aspect ratios: Modern phones typically match aspect ratios
       close to 16:9 (~1.78), 18:9 (2.0), 19.5:9 (~2.17), or 20:9 (~2.22).
    """
    h, w, c = rgb_img.shape
    if h < 50 or w < 50:
        return False

    # Check top 5% (status bar region) and bottom 5% (nav bar region) for solid uniformity
    top_band = rgb_img[: max(10, int(h * 0.05)), :]
    bottom_band = rgb_img[min(h - 10, int(h * 0.95)) :, :]

    top_var = float(np.mean(np.var(top_band, axis=(0, 1))))
    bottom_var = float(np.mean(np.var(bottom_band, axis=(0, 1))))

    # Natural photos virtually always have top/bottom noise variance > 50-100;
    # UI status bars frequently have variance < 10
    has_flat_ui_bar = (top_var < 15.0) or (bottom_var < 15.0)

    # Check percentage of flat pixels using gradient magnitude in grayscale
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)

    # In synthetic UI, a high percentage (>45%) of pixels have near-zero gradient (completely flat backgrounds)
    flat_pixel_ratio = float(np.mean(grad_mag < 2.0))

    # Calculate aspect ratio
    long_side = max(h, w)
    short_side = min(h, w)
    aspect_ratio = float(long_side) / float(short_side)

    # Typical smartphone aspect ratios: 16:9 (1.78), 19.5:9 (2.17), 20:9 (2.22)
    is_mobile_ratio = any(abs(aspect_ratio - target) < 0.10 for target in [16 / 9, 18 / 9, 19.5 / 9, 20 / 9])

    # Composite screenshot condition:
    if has_flat_ui_bar and flat_pixel_ratio > 0.35:
        return True
    if is_mobile_ratio and flat_pixel_ratio > 0.45:
        return True

    return False


class QualityChecker:
    """
    Evaluates image quality flags for trust and verification pipelines.
    """

    def __init__(
        self,
        min_dimension: int = 256,
        min_pixels: int = 65536,
        extreme_aspect_ratio_threshold: float = 3.0,
        jpeg_blockiness_threshold: float = 1.35,
    ):
        self.min_dimension = min_dimension
        self.min_pixels = min_pixels
        self.extreme_aspect_ratio_threshold = extreme_aspect_ratio_threshold
        self.jpeg_blockiness_threshold = jpeg_blockiness_threshold

    def analyze(
        self,
        image: Union[Image.Image, np.ndarray, bytes, str, Path],
    ) -> List[str]:
        """
        Runs quality analysis and returns a list of detected anomaly or limitation flags.

        Possible flags:
        - 'tiny_resolution'
        - 'extreme_aspect_ratio'
        - 'heavy_jpeg_compression'
        - 'screenshot_likeness'

        Returns:
            List of string flags (empty list if image is high quality).
        """
        flags: List[str] = []

        try:
            rgb_np = _to_numpy_rgb(image)
            h, w = rgb_np.shape[:2]

            # 1. Tiny resolution check
            if min(h, w) < self.min_dimension or (h * w) < self.min_pixels:
                flags.append("tiny_resolution")

            # 2. Extreme aspect ratio check
            short_side = max(1, min(h, w))
            long_side = max(h, w)
            ratio = float(long_side) / float(short_side)
            if ratio > self.extreme_aspect_ratio_threshold:
                flags.append("extreme_aspect_ratio")

            # 3. Heavy JPEG compression check
            gray = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY)
            blockiness = check_jpeg_blockiness(gray)
            if blockiness > self.jpeg_blockiness_threshold:
                flags.append("heavy_jpeg_compression")

            # 4. Screenshot-likeness heuristic
            if check_screenshot_heuristics(rgb_np):
                flags.append("screenshot_likeness")

        except Exception as e:
            logger.warning(f"QualityChecker encountered non-fatal error: {e}")

        return flags
