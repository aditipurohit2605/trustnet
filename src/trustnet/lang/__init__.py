"""
Multilingual language support modules for TrustNet (Stage 9).

Provides:
- Script and language detection for Hindi and Hinglish (code-mixed Latin script).
- Text normalization, Devanagari digit conversion, and transliteration.
- Neural translation wrapper with disk caching and aggressive memory freeing.
"""

from trustnet.lang.detect import detect_language, detect_script
from trustnet.lang.normalize import normalize_text
from trustnet.lang.translate import Translator

__all__ = [
    "detect_language",
    "detect_script",
    "normalize_text",
    "Translator",
]
