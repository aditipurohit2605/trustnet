"""
Optical Character Recognition (OCR) module for TrustNet.

Extracts text from images using EasyOCR with multilingual support (English and Hindi).
Features:
- Filters low-confidence detected bounding boxes.
- Reconstructs 2D bounding boxes into natural reading order (top-to-bottom lines, left-to-right words).
- Infers primary language (English vs. Hindi) based on Unicode script frequency.
- Never writes or persists uploaded images to disk.
"""

import io
import logging
import re
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import cv2
import easyocr
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def _to_numpy_rgb(image: Union[Image.Image, np.ndarray, bytes, str, Path]) -> np.ndarray:
    """
    Converts various image input representations into a standard uint8 RGB numpy array.
    Zero disk writes are performed: all operations occur purely in-memory.
    """
    if isinstance(image, Image.Image):
        # Convert PIL Image to RGB mode and cast to numpy array
        return np.array(image.convert("RGB"), dtype=np.uint8)

    if isinstance(image, (str, Path)):
        # Read from disk if a path is provided, convert BGR to RGB
        img_np = cv2.imread(str(image))
        if img_np is None:
            raise ValueError(f"Could not read image from path: {image}")
        return cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    if isinstance(image, bytes):
        # Decode in-memory byte buffer
        nparr = np.frombuffer(image, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_np is None:
            raise ValueError("Could not decode image from provided byte buffer.")
        return cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            # Grayscale -> RGB
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3:
            # Assume already RGB if 3 channels, ensure uint8
            return image.astype(np.uint8)

    raise TypeError(f"Unsupported image type: {type(image)}")


class OCRReader:
    """
    Multilingual OCR reader leveraging EasyOCR with spatial reading-order reconstruction.
    """

    def __init__(
        self,
        languages: Optional[List[str]] = None,
        min_confidence: float = 0.3,
        gpu: bool = False,
    ):
        """
        Args:
            languages: ISO language codes for EasyOCR (e.g., ['en', 'hi']).
            min_confidence: Minimum recognition confidence threshold (0.0 to 1.0).
            gpu: Whether to utilize GPU acceleration if available.
        """
        self.languages = languages or ["en", "hi"]
        self.min_confidence = min_confidence
        self.gpu = gpu

        logger.info(f"Initializing EasyOCR reader with languages={self.languages}, gpu={self.gpu}...")
        # Initialize the EasyOCR reader instance once and reuse across calls
        self._reader = easyocr.Reader(self.languages, gpu=self.gpu)

    def _sort_boxes_reading_order(
        self,
        detections: List[Tuple[List[List[float]], str, float]],
    ) -> List[Tuple[List[List[float]], str, float]]:
        """
        Sorts 2D bounding boxes into natural reading order:
        - Primary direction: Top-to-bottom lines.
        - Secondary direction: Left-to-right within each line.

        Algorithm:
        1. For each detection polygon, compute its bounding rectangle (ymin, ymax, xmin, xmax),
           vertical center (y_center), and box height.
        2. Calculate the median box height across all detections to determine a robust
           line-grouping vertical threshold (e.g., 50% of median height).
        3. Sort all boxes by their vertical center (y_center).
        4. Group boxes whose y_centers fall within the vertical threshold into the same line.
        5. Within each line, sort boxes horizontally by xmin (left-to-right).
        6. Flatten the grouped lines back into a single ordered list of detections.
        """
        if not detections:
            return []

        # Extract geometric features for each detected text box
        box_meta = []
        heights = []
        for det in detections:
            pts = det[0]  # [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            y_center = (ymin + ymax) / 2.0
            height = max(1.0, ymax - ymin)
            heights.append(height)
            box_meta.append({
                "det": det,
                "xmin": xmin,
                "xmax": xmax,
                "ymin": ymin,
                "ymax": ymax,
                "y_center": y_center,
                "height": height,
            })

        # Vertical line proximity tolerance based on half the median text height
        median_height = float(np.median(heights))
        y_tolerance = max(8.0, median_height * 0.5)

        # Sort all boxes vertically first
        box_meta.sort(key=lambda b: b["y_center"])

        # Cluster boxes into distinct horizontal lines
        lines: List[List[dict]] = []
        for b in box_meta:
            if not lines:
                lines.append([b])
            else:
                # Compare current box y_center with the average y_center of the current line
                line_y_avg = sum(item["y_center"] for item in lines[-1]) / len(lines[-1])
                if abs(b["y_center"] - line_y_avg) <= y_tolerance:
                    lines[-1].append(b)
                else:
                    lines.append([b])

        # Sort boxes within each line from left to right (by xmin)
        ordered_detections = []
        for line in lines:
            line.sort(key=lambda b: b["xmin"])
            for b in line:
                ordered_detections.append(b["det"])

        return ordered_detections

    def _detect_language(self, text: str) -> str:
        """
        Infers the primary language of the text based on script code points:
        - Devanagari script (\u0900 - \u097F) -> "hi"
        - Latin script (a-zA-Z) -> "en"
        - Fallback: "unknown"
        """
        if not text.strip():
            return "unknown"

        # Count occurrences of Devanagari characters
        hindi_chars = len(re.findall(r"[\u0900-\u097F]", text))
        # Count occurrences of Latin characters
        latin_chars = len(re.findall(r"[a-zA-Z]", text))

        if hindi_chars > latin_chars and hindi_chars > 0:
            return "hi"
        elif latin_chars > 0:
            return "en"
        return "unknown"

    def read(
        self,
        image: Union[Image.Image, np.ndarray, bytes, str, Path],
    ) -> Tuple[str, float, str]:
        """
        Performs OCR on an image and returns formatted text, mean confidence, and language.

        Args:
            image: Image in memory (PIL Image, numpy array, bytes) or file path.

        Returns:
            Tuple of:
            - text (str): Extracted text formatted with line breaks between reading lines.
            - mean_confidence (float): Average confidence score across all accepted text boxes (0 to 1).
            - detected_language (str): "en", "hi", or "unknown".
        """
        # Convert input into standardized RGB uint8 numpy array in RAM
        img_np = _to_numpy_rgb(image)

        # Run EasyOCR detection and recognition
        # Output format: [ (bbox, text, confidence), ... ]
        raw_detections = self._reader.readtext(img_np)

        # 1. Filter low-confidence boxes below min_confidence threshold
        filtered_detections = [
            (pts, text.strip(), float(conf))
            for (pts, text, conf) in raw_detections
            if float(conf) >= self.min_confidence and text.strip()
        ]

        if not filtered_detections:
            logger.debug("No OCR text detected above confidence threshold.")
            return ("", 0.0, "unknown")

        # 2. Sort detected boxes into natural human reading order
        ordered_detections = self._sort_boxes_reading_order(filtered_detections)

        # 3. Reconstruct text lines: group boxes that share a reading line
        # We re-run line grouping to insert proper newline characters between visual lines
        ordered_texts = []
        confidences = []

        # Heights and vertical coordinates for line breaks
        heights = [
            max(1.0, max(p[1] for p in det[0]) - min(p[1] for p in det[0]))
            for det in ordered_detections
        ]
        median_height = float(np.median(heights))
        y_tolerance = max(8.0, median_height * 0.5)

        current_line: List[str] = []
        current_line_y = None

        for pts, text, conf in ordered_detections:
            confidences.append(conf)
            y_center = sum(p[1] for p in pts) / 4.0

            if current_line_y is None:
                current_line.append(text)
                current_line_y = y_center
            elif abs(y_center - current_line_y) <= y_tolerance:
                current_line.append(text)
            else:
                # Flush previous line with spaces, start new line
                ordered_texts.append(" ".join(current_line))
                current_line = [text]
                current_line_y = y_center

        if current_line:
            ordered_texts.append(" ".join(current_line))

        full_text = "\n".join(ordered_texts).strip()
        mean_conf = float(np.mean(confidences)) if confidences else 0.0
        detected_lang = self._detect_language(full_text)

        return (full_text, mean_conf, detected_lang)
