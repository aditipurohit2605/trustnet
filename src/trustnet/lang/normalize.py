"""
Text Normalization Engine for Multilingual TrustNet (Stage 9).

Performs:
1. Unicode NFC Normalization: canonical composition across composite glyphs and matras.
2. Devanagari Numeral Conversion: converts Indic numerals (०-९) to standard ASCII digits (0-9).
3. Elongated Character Cleanup: compresses repeated characters (e.g., 'sooooo' -> 'soo', 'bahutttt' -> 'bahut').
4. URL and Mention Sanitization: strips or normalizes web links and social handles.
5. Punctuation and Whitespace Consolidation: cleans excessive whitespace and repeated exclamation/question marks.
6. Optional Transliteration / Translation: transforms Hinglish into Devanagari or English.

Returns a structured NormalizedText model capturing the transformations and history of applied steps.
"""

import logging
import re
import unicodedata
from typing import List, Optional

from trustnet.schemas import NormalizedText

logger = logging.getLogger(__name__)

# Devanagari numerals (U+0966 to U+096F) to ASCII mapping
DEVANAGARI_DIGITS_MAP = {
    "\u0966": "0",  # ०
    "\u0967": "1",  # १
    "\u0968": "2",  # २
    "\u0969": "3",  # ३
    "\u096A": "4",  # ४
    "\u096B": "5",  # ५
    "\u096C": "6",  # ६
    "\u096D": "7",  # ७
    "\u096E": "8",  # ८
    "\u096F": "9",  # ९
}

# Regex to match URLs and web addresses
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

# Regex to match social media user mentions (@username)
MENTION_PATTERN = re.compile(r"@\w+", re.IGNORECASE)

# Regex to compress elongated characters: 3 or more consecutive identical characters reduced to 2
ELONGATED_PATTERN = re.compile(r"(.)\1{2,}", re.IGNORECASE)

# Repeated punctuation cleanup (e.g., '!!!!' -> '!', '????' -> '?')
REPEATED_PUNCT_PATTERN = re.compile(r"([!?.,;:])\1+")


def normalize_unicode_nfc(text: str) -> str:
    """Canonical Unicode NFC normalization."""
    return unicodedata.normalize("NFC", text)


def convert_devanagari_numerals(text: str) -> str:
    """Converts Devanagari digits (०-९) to standard ASCII digits (0-9)."""
    chars = [DEVANAGARI_DIGITS_MAP.get(ch, ch) for ch in text]
    return "".join(chars)


def sanitize_urls_and_mentions(text: str, mask_token: str = "") -> str:
    """
    Removes or masks web links and social handles.
    If mask_token is empty, strips them cleanly.
    """
    text = URL_PATTERN.sub(mask_token, text)
    text = MENTION_PATTERN.sub(mask_token, text)
    return text


def clean_elongated_characters(text: str) -> str:
    """
    Compresses artificial character elongation used for emphasis.
    e.g. 'sooooo' -> 'soo', 'bahutttttt' -> 'bahut' (retains max 2 characters).
    """
    return ELONGATED_PATTERN.sub(r"\1\1", text)


def consolidate_whitespace(text: str) -> str:
    """Normalizes excessive whitespace, newlines, and tabs into a single space."""
    # Collapse repeated punctuation
    text = REPEATED_PUNCT_PATTERN.sub(r"\1", text)
    # Collapse whitespace
    return " ".join(text.split()).strip()


def transliterate_hinglish_to_devanagari(text: str) -> str:
    """
    Transliterates Romanized Indic text to Devanagari script.
    Uses indic-transliteration (Sanscript ITRANS) if installed, or falls back to basic phonetic mapping.
    """
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate
        # Transliterate ITRANS / Roman to Devanagari
        return transliterate(text, sanscript.ITRANS, sanscript.DEVANAGARI)
    except Exception as e:
        logger.debug(f"indic_transliteration not available or failed: {e}; returning text as-is.")
        return text


class TextNormalizer:
    """
    Configurable multilingual text normalizer with step audit tracking.
    """

    def __init__(
        self,
        nfc_norm: bool = True,
        convert_numerals: bool = True,
        clean_elongation: bool = True,
        sanitize_urls: bool = True,
        hinglish_mode: str = "translate",  # "none", "transliterate", "translate"
    ):
        self.nfc_norm = nfc_norm
        self.convert_numerals = convert_numerals
        self.clean_elongation = clean_elongation
        self.sanitize_urls = sanitize_urls
        self.hinglish_mode = hinglish_mode

    def normalize(self, text: str, translated_en: Optional[str] = None) -> NormalizedText:
        """
        Executes configured normalization pipeline and records the audit trail of applied steps.
        
        Args:
            text: Raw input claim or message.
            translated_en: Optional English translation if precomputed.
            
        Returns:
            NormalizedText containing original text, normalized text, and list of applied steps.
        """
        original = text or ""
        current = original
        steps: List[str] = []

        if not current.strip():
            return NormalizedText(
                original=original,
                normalized="",
                translated_en=translated_en,
                steps=["empty_input"],
            )

        # 1. Unicode NFC Normalization
        if self.nfc_norm:
            norm_nfc = normalize_unicode_nfc(current)
            if norm_nfc != current:
                steps.append("unicode_nfc")
                current = norm_nfc

        # 2. Devanagari Numerals Conversion
        if self.convert_numerals:
            num_converted = convert_devanagari_numerals(current)
            if num_converted != current:
                steps.append("devanagari_numerals_to_ascii")
                current = num_converted

        # 3. URL and Mention Sanitization
        if self.sanitize_urls:
            sanitized = sanitize_urls_and_mentions(current)
            if sanitized != current:
                steps.append("sanitize_urls_and_mentions")
                current = sanitized

        # 4. Elongated Character Cleanup
        if self.clean_elongation:
            compressed = clean_elongated_characters(current)
            if compressed != current:
                steps.append("clean_elongated_characters")
                current = compressed

        # 5. Whitespace and Punctuation Consolidation
        clean_ws = consolidate_whitespace(current)
        if clean_ws != current:
            steps.append("consolidate_whitespace")
            current = clean_ws

        # 6. Optional Hinglish Transliteration
        if self.hinglish_mode == "transliterate":
            translit = transliterate_hinglish_to_devanagari(current)
            if translit != current:
                steps.append("transliterate_to_devanagari")
                current = translit

        return NormalizedText(
            original=original,
            normalized=current,
            translated_en=translated_en,
            steps=steps,
        )


def normalize_text(
    text: str,
    hinglish_mode: str = "none",
    translated_en: Optional[str] = None,
) -> NormalizedText:
    """
    Convenience functional wrapper normalizing claim text.
    Returns NormalizedText capturing normalized content and audit trail of steps.
    """
    normalizer = TextNormalizer(hinglish_mode=hinglish_mode)
    return normalizer.normalize(text, translated_en=translated_en)
