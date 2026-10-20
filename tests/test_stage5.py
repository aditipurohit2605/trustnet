"""
Comprehensive test suite for TrustNet Stage 5: Vision Module.

Verifies:
1. ExifReader never returns GPS keys (tested with synthetic EXIF containing GPS).
2. Perceptual hash (pHash) is stable under image resizing.
3. ImageBank search finds an exact copy with Hamming distance 0.
4. A failing OCR step does not crash ImageAnalyzer (graceful error handling).
5. CPU smoke test across 5 diverse test images.
"""

import io
from pathlib import Path
from unittest.mock import MagicMock

import imagehash
import numpy as np
import pytest
from PIL import Image, ImageDraw

from trustnet.schemas import ImageSignals, ReuseMatch
from trustnet.vision.analyzer import ImageAnalyzer
from trustnet.vision.clip_scorer import ClipScorer, MismatchCalibrator
from trustnet.vision.exif import ExifReader
from trustnet.vision.image_bank import ImageBank
from trustnet.vision.ocr import OCRReader
from trustnet.vision.quality import QualityChecker


def test_exif_reader_never_returns_gps():
    """
    Ensures that ExifReader extracts safe metadata (Make, Model) while strictly
    filtering out all GPS location tags without exception.
    """
    img = Image.new("RGB", (120, 120), color=(255, 255, 255))
    exif = img.getexif()

    # Add safe metadata tags
    exif[271] = "TrustNetTestMake"  # Make
    exif[272] = "TrustNetTestModel"  # Model
    exif[305] = "TrustNetSoftware"   # Software

    # Inject GPS IFD tags (Tag 0x8825 is GPSInfo)
    gps_ifd = exif.get_ifd(0x8825)
    gps_ifd[1] = "N"  # GPSLatitudeRef
    gps_ifd[2] = (37, 46, 0)  # GPSLatitude (San Francisco)
    gps_ifd[3] = "W"  # GPSLongitudeRef
    gps_ifd[4] = (122, 25, 0)  # GPSLongitude
    gps_ifd[7] = (12, 0, 0)  # GPSTimeStamp

    # Save to in-memory JPEG byte buffer
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    img_bytes = buf.getvalue()

    # Extract metadata using ExifReader
    extracted = ExifReader.extract(img_bytes)

    # Verify safe fields were successfully recovered
    assert "make" in extracted
    assert extracted["make"] == "TrustNetTestMake"
    assert extracted.get("model") == "TrustNetTestModel"

    # CRITICAL PRIVACY GUARANTEE:
    # Absolutely NO key containing 'gps' should ever be returned
    for key in extracted.keys():
        assert "gps" not in key.lower(), f"Privacy leak: found GPS key '{key}' in extracted EXIF!"

    # Also test with PIL Image directly
    img_with_exif = Image.open(io.BytesIO(img_bytes))
    extracted_pil = ExifReader.extract(img_with_exif)
    assert not any("gps" in k.lower() for k in extracted_pil.keys())


def test_phash_stability_under_resize():
    """
    Verifies that perceptual hashing (pHash) remains invariant or near-invariant
    under image resizing operations (Hamming distance <= 4).
    """
    bank_img_path = Path("data/image_bank/bank_img_001.png")
    if bank_img_path.exists():
        img = Image.open(bank_img_path)
    else:
        img = Image.new("RGB", (320, 240), color=(40, 120, 200))
        d = ImageDraw.Draw(img)
        d.rectangle([20, 20, 300, 220], outline=(255, 255, 255), width=3)
        d.text((30, 40), "Reference Scene", fill=(255, 255, 255))

    # Compute original pHash
    h_orig = imagehash.phash(img, hash_size=8)

    # Downscale (50%) and upscale (150%) transformations with Lanczos resampling
    downscaled = img.resize((max(16, int(img.width * 0.5)), max(16, int(img.height * 0.5))), Image.Resampling.LANCZOS)
    upscaled = img.resize((int(img.width * 1.5), int(img.height * 1.5)), Image.Resampling.LANCZOS)

    h_down = imagehash.phash(downscaled, hash_size=8)
    h_up = imagehash.phash(upscaled, hash_size=8)

    # Hamming distance should be minimal (typically 0 or <= 2, well within threshold 10)
    dist_down = int(h_orig - h_down)
    dist_up = int(h_orig - h_up)

    assert dist_down <= 4, f"pHash unstable under downscaling: distance={dist_down}"
    assert dist_up <= 4, f"pHash unstable under upscaling: distance={dist_up}"



def test_bank_search_exact_copy():
    """
    Verifies that querying an image bank with an exact identical image
    retrieves the image with Hamming distance 0 and rank 1.
    """
    bank = ImageBank(phash_size=8, phash_match_threshold=10)

    # Create reference images
    ref1 = Image.new("RGB", (200, 200), color=(30, 180, 60))
    ref2 = Image.new("RGB", (200, 200), color=(180, 30, 60))

    bank.add_image("id_nature", ref1, known_context="Nature photo", source="SourceA")
    bank.add_image("id_urban", ref2, known_context="Urban scene", source="SourceB")

    # Search using exact copy of ref1
    matches = bank.search(ref1, top_k=2)

    assert len(matches) >= 1
    top_match = matches[0]

    assert top_match.bank_id == "id_nature"
    assert top_match.phash_distance == 0
    assert top_match.known_context == "Nature photo"
    assert top_match.source == "SourceA"


def test_failing_ocr_step_does_not_crash_analyzer():
    """
    Ensures that if the OCR step fails with an unexpected exception,
    ImageAnalyzer logs a warning and completes signal extraction safely.
    """
    # Create mock OCR reader that intentionally raises an unhandled exception
    failing_ocr = MagicMock(spec=OCRReader)
    failing_ocr.read.side_effect = RuntimeError("Fatal crash in OCR native C++ backend!")

    # Instantiate analyzer with failing mock OCR
    analyzer = ImageAnalyzer(
        ocr_reader=failing_ocr,
        quality_checker=QualityChecker(),
        clip_scorer=None,  # Skips clip loading for fast unit testing
        image_bank=ImageBank(),
    )

    test_img = Image.new("RGB", (200, 200), color=(255, 255, 255))

    # analyze() must NOT raise an exception
    signals = analyzer.analyze(test_img)

    # OCR fields must be gracefully None
    assert signals.ocr_text is None
    assert signals.ocr_confidence is None
    assert signals.ocr_language is None

    # Other non-failing components must still produce valid outputs
    assert isinstance(signals.quality_flags, list)
    assert isinstance(signals.exif_summary, dict)
    assert isinstance(signals.reuse_matches, list)


def test_cpu_smoke_test_5_images():
    """
    End-to-end CPU smoke test running ImageAnalyzer on 5 diverse images:
    1. Standard natural color image
    2. Screenshot with flat UI bars
    3. Tiny resolution image
    4. Text-containing document image
    5. Extreme aspect ratio image
    """
    analyzer = ImageAnalyzer(config_path="configs/stage5_vision.yaml")

    # Image 1: Clean natural-like image
    img1 = Image.new("RGB", (320, 240), color=(50, 120, 180))
    d1 = ImageDraw.Draw(img1)
    d1.ellipse([40, 40, 280, 200], fill=(220, 180, 50))

    # Image 2: Screenshot simulation (flat status bar + flat background)
    img2 = Image.new("RGB", (270, 585), color=(245, 245, 245))  # 19.5:9 phone ratio
    d2 = ImageDraw.Draw(img2)
    d2.rectangle([0, 0, 270, 30], fill=(10, 10, 10))  # Black status bar
    d2.rectangle([0, 555, 270, 585], fill=(10, 10, 10))  # Black nav bar

    # Image 3: Tiny resolution image
    img3 = Image.new("RGB", (80, 80), color=(200, 50, 50))

    # Image 4: Image containing rendered text
    img4 = Image.new("RGB", (400, 150), color=(255, 255, 255))
    d4 = ImageDraw.Draw(img4)
    d4.text((20, 40), "Water levels are normal and safe.", fill=(0, 0, 0))

    # Image 5: Extreme aspect ratio image (e.g. 8:1 banner)
    img5 = Image.new("RGB", (800, 90), color=(100, 200, 100))

    test_images = [img1, img2, img3, img4, img5]
    claim_text = "The reservoir water levels are normal."

    for idx, img in enumerate(test_images, start=1):
        signals = analyzer.analyze(img, claim_text=claim_text)

        assert isinstance(signals, ImageSignals), f"Image {idx} did not return ImageSignals"
        assert signals.clip_similarity is not None, f"Image {idx} should have clip_similarity"
        assert signals.mismatch_score is not None, f"Image {idx} should have mismatch_score"
        assert 0.0 <= signals.mismatch_score <= 1.0, f"Image {idx} mismatch_score out of range [0, 1]"

    # Verify quality flags on expected anomalies
    signals_tiny = analyzer.analyze(img3)
    assert "tiny_resolution" in signals_tiny.quality_flags

    signals_extreme = analyzer.analyze(img5)
    assert "extreme_aspect_ratio" in signals_extreme.quality_flags
