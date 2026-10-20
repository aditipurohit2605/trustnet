"""
Script and Language Identification for Multilingual TrustNet (Stage 9).

Detects:
1. Script (Devanagari, Latin, Mixed, or Unknown) via Unicode block character ranges.
2. Language code ('hi', 'en', 'hi-Latn' / Hinglish, or 'und' for undetermined).
3. Code-mixing detection (is_code_mixed: True if substantial Hindi and English or mixed scripts coexist).

Robustness:
- Never crashes on empty strings, whitespace-only, numbers-only, or emoji-only inputs.
"""

import logging
import re
from typing import Dict, Optional, Set, Tuple

from trustnet.schemas import LanguageInfo

logger = logging.getLogger(__name__)

# Unicode Character Ranges
# Devanagari Unicode Block: U+0900 to U+097F
DEVANAGARI_RANGE = (0x0900, 0x097F)
# Basic Latin Alphabetic Characters: A-Z, a-z
LATIN_RANGE_UPPER = (0x0041, 0x005A)
LATIN_RANGE_LOWER = (0x0061, 0x007A)

# Curated Romanized Hindi (Hinglish) Lexical Indicators
# High-frequency grammatical markers, verbs, pronouns, and sensational/news vocabulary in Roman script
# Note: Common English function words (such as 'the' and 'to') are excluded to prevent English false positives
HINGLISH_VOCABULARY: Set[str] = {
    # Pronouns & determiners
    "yeh", "ye", "woh", "wo", "hum", "aap", "tum", "mera", "meri", "mere", "tera", "teri", "tere",
    "unka", "unki", "unke", "iska", "iski", "iske", "kiska", "kiski", "kiske", "sab", "sabko", "kisi",
    # Auxiliary verbs & copulas
    "hai", "hain", "tha", "thi", "hoga", "hogi", "hoge", "raha", "rahi", "rahe", "hua", "hui", "hue",
    # Common verbs & imperatives
    "karo", "karein", "kare", "kiya", "bhejo", "dekho", "dekhein", "bolo", "bataya", "bataye", "aaya", "aayi",
    "gaya", "gayi", "milega", "dekh", "sun", "samjho", "padho",
    # Negations & question particles
    "nahi", "nahin", "na", "mat", "kya", "kyun", "kyu", "kaise", "kab", "kahan", "kitna", "kitni",
    # Adverbs, prepositions & conjunctions
    "abhi", "aaj", "kal", "parso", "parson", "pehle", "baad", "saath", "par", "se", "ko", "mein", "me", "ne",
    "aur", "lekin", "magar", "kyunki", "isliye", "bhi", "hi", "toh",
    # Common nouns & forward rhetoric
    "sach", "jhooth", "khabar", "samachar", "sarkar", "desh", "log", "bhai", "yaar", "dost", "dhyan",
    "khatra", "jaldi", "turant", "viral", "asli", "nakli", "afwah",
}

# English common stopwords for detecting bilingual code-mixing
ENGLISH_STOPWORDS: Set[str] = {
    "the", "is", "at", "which", "on", "and", "a", "an", "in", "to", "of", "for", "with",
    "that", "by", "from", "as", "are", "was", "were", "this", "be", "or", "have", "has",
    "had", "it", "not", "but", "all", "they", "we", "you", "he", "she", "his", "her",
    "official", "report", "confirmed", "government", "news", "hospital", "police", "warning",
}


class ScriptResult(str):
    """
    Subclass of str representing the detected script name.
    Supports both direct string comparisons (e.g. res == 'Devanagari')
    and tuple unpacking (script, conf = detect_script(text)).
    """
    def __new__(cls, script: str, confidence: float = 1.0):
        obj = str.__new__(cls, script)
        obj.confidence = confidence
        return obj

    def __iter__(self):
        return iter((str(self), self.confidence))


def count_script_characters(text: str) -> Dict[str, int]:
    """
    Counts characters belonging to specific Unicode script ranges.
    Ignores whitespace, standard punctuation, ASCII digits, and emojis.
    """
    devanagari_count = 0
    latin_count = 0
    other_count = 0

    for ch in text:
        cp = ord(ch)
        if DEVANAGARI_RANGE[0] <= cp <= DEVANAGARI_RANGE[1]:
            devanagari_count += 1
        elif (LATIN_RANGE_UPPER[0] <= cp <= LATIN_RANGE_UPPER[1]) or (LATIN_RANGE_LOWER[0] <= cp <= LATIN_RANGE_LOWER[1]):
            latin_count += 1
        elif not ch.isspace() and not ch.isnumeric():
            other_count += 1

    return {
        "devanagari": devanagari_count,
        "latin": latin_count,
        "other": other_count,
        "total_alpha": devanagari_count + latin_count,
    }


def detect_script(text: str) -> ScriptResult:
    """
    Determines the dominant script of a string.
    
    Returns:
        ScriptResult instance supporting both string equality and tuple unpacking.
    """
    counts = count_script_characters(text)
    total_alpha = counts["total_alpha"]

    if total_alpha == 0:
        return ScriptResult("unknown", 0.0)

    dev_ratio = counts["devanagari"] / total_alpha
    lat_ratio = counts["latin"] / total_alpha

    # Mixed script condition: both scripts represent >= 15% of alphabetical characters
    if dev_ratio >= 0.15 and lat_ratio >= 0.15:
        return ScriptResult("mixed", round(float(min(dev_ratio, lat_ratio) * 2.0), 3))

    if dev_ratio >= 0.50:
        return ScriptResult("Devanagari", round(float(dev_ratio), 3))

    if lat_ratio >= 0.50:
        return ScriptResult("Latin", round(float(lat_ratio), 3))

    return ScriptResult("unknown", 0.50)


def detect_language(text: str) -> LanguageInfo:
    """
    Analyzes input text for script, language code, confidence, and code-mixing.
    
    Guarantees:
    - Never throws an exception on empty, emoji-only, or whitespace-only inputs.
    - Accurately distinguishes Devanagari Hindi ('hi'), Latin English ('en'),
      and Romanized Hindi / Hinglish ('hi-Latn').
    - Reports is_code_mixed=True when Hindi and English vocabulary or Devanagari
      and Latin scripts are combined.
      
    Returns:
        LanguageInfo object containing code, script, confidence, and is_code_mixed.
    """
    if not text or not text.strip():
        return LanguageInfo(code="und", script="unknown", confidence=0.0, is_code_mixed=False)

    script, script_conf = detect_script(text)

    # Handle Emoji-only or symbol-only inputs
    if script == "unknown":
        return LanguageInfo(code="und", script="unknown", confidence=0.0, is_code_mixed=False)

    # 1. Pure Devanagari Script
    if script == "Devanagari":
        counts = count_script_characters(text)
        is_mixed = counts["latin"] > 0
        return LanguageInfo(
            code="hi",
            script="Devanagari",
            confidence=max(0.70, script_conf),
            is_code_mixed=is_mixed,
        )

    # 2. Mixed Script (Devanagari + Latin characters co-occurring in the text)
    if script == "mixed":
        return LanguageInfo(
            code="hi-Latn",
            script="mixed",
            confidence=0.85,
            is_code_mixed=True,
        )

    # 3. Latin Script: Differentiate English from Romanized Hindi (Hinglish)
    words = re.findall(r"\b[A-Za-z]+\b", text.lower())
    if not words:
        return LanguageInfo(code="en", script="Latin", confidence=0.5, is_code_mixed=False)

    hinglish_hits = sum(1 for w in words if w in HINGLISH_VOCABULARY)
    english_hits = sum(1 for w in words if w in ENGLISH_STOPWORDS)
    total_words = len(words)

    hinglish_density = hinglish_hits / total_words
    english_density = english_hits / total_words

    # If English hits outnumber Hinglish hits and Hinglish density is low, classify as English
    if english_hits > hinglish_hits and hinglish_density < 0.25:
        conf = min(0.99, round(0.70 + english_density * 0.30, 3))
        return LanguageInfo(
            code="en",
            script="Latin",
            confidence=conf,
            is_code_mixed=False,
        )

    # If Romanized Hindi markers are detected
    if hinglish_hits >= 2 or (hinglish_density >= 0.20 and hinglish_hits >= 1):
        # Code-mixed if both Hindi markers and English tokens exist
        is_code_mixed = english_hits >= 1 or (total_words - hinglish_hits >= 2)
        conf = min(0.99, round(0.60 + hinglish_density * 0.40, 3))
        return LanguageInfo(
            code="hi-Latn",
            script="Latin",
            confidence=conf,
            is_code_mixed=is_code_mixed,
        )

    # Default Latin text without significant Indic vocabulary is classified as English
    conf = min(0.99, round(0.70 + english_density * 0.30, 3))
    return LanguageInfo(
        code="en",
        script="Latin",
        confidence=conf,
        is_code_mixed=False,
    )
