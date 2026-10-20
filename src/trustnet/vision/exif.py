"""
Safe EXIF metadata extractor for TrustNet.

Extracts non-sensitive provenance metadata:
- Capture timestamp (DateTimeOriginal)
- Device manufacturer (Make)
- Device model (Model)
- Software/editor identifier (Software)

Privacy Guarantees:
- Strictly filters out, ignores, and NEVER returns or logs GPS coordinates,
  geotags, or location metadata (any tag with 'GPS' or related identifiers).
- Never writes uploaded images to disk.
- Returns an empty dictionary if no EXIF is present or upon parsing errors.
"""

import io
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import exifread
from PIL import Image

logger = logging.getLogger(__name__)

# Strict allowlist of safe metadata keys to return
SAFE_TAG_MAPPING = {
    "EXIF DateTimeOriginal": "datetime_original",
    "EXIF DateTimeDigitized": "datetime_digitized",
    "Image DateTime": "datetime",
    "Image Make": "make",
    "Image Model": "model",
    "Image Software": "software",
}


class ExifReader:
    """
    Extracts privacy-safe EXIF metadata from in-memory images or paths.
    """

    @staticmethod
    def extract(
        image: Union[Image.Image, bytes, str, Path],
    ) -> Dict[str, Any]:
        """
        Extracts safe EXIF metadata.
        Never returns or logs GPS or sensitive location attributes.

        Args:
            image: In-memory PIL Image, bytes buffer, or file path.

        Returns:
            Dictionary containing only safe keys:
            {'datetime_original', 'make', 'model', 'software'}.
            Returns empty dict {} if no EXIF data exists.
        """
        summary: Dict[str, Any] = {}

        try:
            # 1. Obtain readable byte buffer for exifread
            buf: Optional[io.BytesIO] = None

            if isinstance(image, bytes):
                buf = io.BytesIO(image)
            elif isinstance(image, (str, Path)):
                path = Path(image)
                if path.exists() and path.is_file():
                    with open(path, "rb") as f:
                        buf = io.BytesIO(f.read())
            elif isinstance(image, Image.Image):
                # Inspect PIL's raw EXIF if present
                raw_exif = image.info.get("exif")
                if raw_exif and isinstance(raw_exif, bytes):
                    buf = io.BytesIO(raw_exif)
                else:
                    # Fallback: check PIL getexif()
                    try:
                        exif_obj = image.getexif()
                        if exif_obj:
                            # Standard Exif tags by ID
                            # 271: Make, 272: Model, 305: Software, 306: DateTime, 36867: DateTimeOriginal
                            pil_tag_map = {
                                271: "make",
                                272: "model",
                                305: "software",
                                306: "datetime",
                                36867: "datetime_original",
                                36868: "datetime_digitized",
                            }
                            for tag_id, val in exif_obj.items():
                                if tag_id in pil_tag_map and val is not None:
                                    cleaned_val = str(val).strip().replace("\x00", "")
                                    if cleaned_val:
                                        summary[pil_tag_map[tag_id]] = cleaned_val

                            # Check EXIF IFD sub-dict (tag 34665)
                            try:
                                ifd = exif_obj.get_ifd(0x8769)
                                if ifd:
                                    if 36867 in ifd and ifd[36867]:
                                        summary["datetime_original"] = str(ifd[36867]).strip().replace("\x00", "")
                                    if 36868 in ifd and ifd[36868]:
                                        summary["datetime_digitized"] = str(ifd[36868]).strip().replace("\x00", "")
                            except Exception:
                                pass
                    except Exception as e:
                        logger.debug(f"PIL getexif inspection encountered: {e}")

            # 2. If we have a byte buffer, process with exifread for full tag parsing
            if buf is not None:
                buf.seek(0)
                # details=False avoids extracting thumbnail or binary bloat
                tags = exifread.process_file(buf, details=False)

                for tag_key, tag_value in tags.items():
                    # PRIVACY SAFEGUARD:
                    # Discard ANY tag with 'GPS' in the tag name without logging its content
                    if "gps" in tag_key.lower():
                        continue

                    # Only accept explicitly allowlisted safe tags
                    if tag_key in SAFE_TAG_MAPPING:
                        field_name = SAFE_TAG_MAPPING[tag_key]
                        val_str = str(tag_value).strip().replace("\x00", "")
                        if val_str and val_str.lower() != "none":
                            summary[field_name] = val_str

        except Exception as e:
            # Failure in EXIF extraction must never abort the pipeline
            logger.warning(f"EXIF extraction encountered a non-fatal error: {e}")
            return {}

        # Ensure datetime_original is populated if general datetime exists
        if "datetime_original" not in summary and "datetime" in summary:
            summary["datetime_original"] = summary["datetime"]

        # Ensure NO GPS keys ever leaked
        clean_summary = {
            k: v for k, v in summary.items()
            if "gps" not in k.lower()
        }

        return clean_summary
