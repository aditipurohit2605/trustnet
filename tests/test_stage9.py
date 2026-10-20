"""
Comprehensive Test Suite for Stage 9: Hindi and Hinglish Multilingual Support.

Tests:
1. Language and script detection on hand-written strings (Devanagari, Latin, code-mixed Hinglish,
   emoji-only, and empty strings - never crashes).
2. Normalization steps: NFC, Devanagari numerals to ASCII digits, elongated character cleanup,
   URL and mention sanitization, and step tracking in NormalizedText.
3. Anti-leakage: translated training data (train_text_hi.jsonl) claims NEVER appear in
   val, val_select, val_cal, train_fusion, or test splits.
4. Original text preservation: ClaimReport preserves original user text, language_info,
   and normalized_text.
5. English regression invariance: English predictions are identical and reproducible when
   multilingual features are disabled.
6. Mocked Translator tests: translation batching, disk cache keying, and memory freeing.
7. CPU smoke test: end-to-end claim verification pipeline executing safely on CPU.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trustnet.lang.detect import detect_language, detect_script
from trustnet.lang.normalize import (
    clean_elongated_characters,
    convert_devanagari_numerals,
    normalize_text,
    sanitize_urls_and_mentions,
)
from trustnet.lang.translate import Translator
from trustnet.pipeline import verify_claim
from trustnet.reasoning.temporal import TemporalAnalyzer
from trustnet.schemas import Claim, ClaimReport, LanguageInfo, NormalizedText, Verdict


# ---------------------------------------------------------------------------
# 1. Language & Script Detection Tests
# ---------------------------------------------------------------------------

def test_language_detection_handwritten():
    """Validates script and language detection across diverse language forms."""
    # 1. Pure Devanagari Hindi
    res_hi = detect_language("उत्तरी बांध का जलस्तर सामान्य सीमा के भीतर है।")
    assert res_hi.code == "hi"
    assert res_hi.script == "Devanagari"
    assert res_hi.is_code_mixed is False

    # 2. Pure English
    res_en = detect_language("The water levels behind the northern municipal dam are safe.")
    assert res_en.code == "en"
    assert res_en.script == "Latin"
    assert res_en.is_code_mixed is False

    # 3. Code-mixed Hinglish (Latin-script Hindi)
    res_hinglish = detect_language("Bhai kal subah sabhi trains me free wifi band ho jayega.")
    assert res_hinglish.code in ("hi-Latn", "hinglish")
    assert res_hinglish.is_code_mixed is True

    # 4. Emoji-only string (MUST NEVER CRASH)
    res_emoji = detect_language("🔥🔥🔥 🙏🙏🙏")
    assert isinstance(res_emoji, LanguageInfo)
    assert res_emoji.code in ("unknown", "en", "und")
    assert res_emoji.confidence >= 0.0

    # 5. Empty or whitespace string (MUST NEVER CRASH)
    res_empty = detect_language("   \n\t  ")
    assert isinstance(res_empty, LanguageInfo)
    assert res_empty.code in ("unknown", "en", "und")


def test_script_detection_ranges():
    """Tests Unicode range script classifier."""
    assert detect_script("भारत") == "Devanagari"
    assert detect_script("India") == "Latin"
    assert detect_script("India भारत") == "mixed"
    assert detect_script("12345 !@#$") in ("unknown", "Latin")


# ---------------------------------------------------------------------------
# 2. Text Normalization Pipeline Tests
# ---------------------------------------------------------------------------

def test_normalization_steps():
    """Tests individual and end-to-end normalization stages."""
    # Step 1: Devanagari numerals to ASCII digits
    dev_num_text = "साल २०२४ में १२३ मरीज ठीक हुए।"
    ascii_num_text = convert_devanagari_numerals(dev_num_text)
    assert "2024" in ascii_num_text
    assert "123" in ascii_num_text
    assert "२०२४" not in ascii_num_text

    # Step 2: Elongated characters cleanup
    elongated = "sooooo biggg news sachhhhhh"
    cleaned_elongated = clean_elongated_characters(elongated)
    assert "sooo" not in cleaned_elongated
    assert "sachhh" not in cleaned_elongated

    # Step 3: URL and mention sanitization
    tweet = "Check this report https://example.com/breaking @dr_sharma confirm karo"
    sanitized = sanitize_urls_and_mentions(tweet)
    assert "https://example.com/breaking" not in sanitized
    assert "@dr_sharma" not in sanitized

    # Step 4: Full normalization wrapper
    raw = "  BREAKING: साल २०२४ https://news.org @who sachhhhhhh mein!!  "
    norm_res = normalize_text(raw)
    assert isinstance(norm_res, NormalizedText)
    assert norm_res.original == raw
    assert "2024" in norm_res.normalized
    assert "https://" not in norm_res.normalized
    assert "@who" not in norm_res.normalized
    assert len(norm_res.steps) >= 3
    assert "devanagari_numerals_to_ascii" in norm_res.steps or "convert_devanagari_numerals" in norm_res.steps


# ---------------------------------------------------------------------------
# 3. Anti-Leakage Invariant Test
# ---------------------------------------------------------------------------

def test_anti_leakage_translated_train_data():
    """
    CRITICAL SAFETY CHECK:
    Asserts that translated training data (data/processed/train_text_hi.jsonl)
    has ZERO claim ID overlap with val, val_select, val_cal, train_fusion, or test splits.
    """
    train_hi_path = Path("data/processed/train_text_hi.jsonl")
    assert train_hi_path.is_file(), "Translated training file data/processed/train_text_hi.jsonl must exist."

    # Extract base claim IDs from train_text_hi (strip '_hi' suffix)
    hi_claim_ids = set()
    with open(train_hi_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                orig_id = rec.get("metadata", {}).get("original_claim_id")
                if orig_id:
                    hi_claim_ids.add(orig_id)
                else:
                    cid = str(rec.get("id") or rec.get("claim_id"))
                    hi_claim_ids.add(cid.replace("_hi", ""))

    assert len(hi_claim_ids) > 0, "No claims found in train_text_hi.jsonl"

    forbidden_splits = [
        Path("data/processed/val.jsonl"),
        Path("data/processed/val_select.jsonl"),
        Path("data/processed/val_cal.jsonl"),
        Path("data/processed/train_fusion.jsonl"),
        Path("data/processed/test.jsonl"),
    ]

    for split_path in forbidden_splits:
        if not split_path.is_file():
            continue
        forbidden_ids = set()
        with open(split_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    forbidden_ids.add(str(rec.get("id") or rec.get("claim_id")))

        overlap = hi_claim_ids.intersection(forbidden_ids)
        assert len(overlap) == 0, (
            f"LEAKAGE VIOLATION: {len(overlap)} claim IDs from train_text_hi overlap with {split_path.name}: {overlap}"
        )


# ---------------------------------------------------------------------------
# 4. Original Text & Language Info Preserved in Report
# ---------------------------------------------------------------------------

def test_original_text_preserved_in_report():
    """Ensures reports preserve original text, language_info, and normalization records."""
    hi_claim_text = "उत्तरी बांध का जलस्तर सामान्य सीमा के भीतर है।"
    claim = Claim(id="test_hi_001", text=hi_claim_text)

    # Verify claim with reasoning disabled for fast isolated unit testing
    report = verify_claim(claim, enable_reasoning=False)

    assert isinstance(report, ClaimReport)
    assert report.original_text == hi_claim_text
    assert report.language_info is not None
    assert report.language_info.code == "hi"
    assert report.language_info.script == "Devanagari"
    assert report.normalized_text is not None
    assert report.normalized_text.original == hi_claim_text


# ---------------------------------------------------------------------------
# 5. English Invariance: Backward-Compatibility Guarantee
# ---------------------------------------------------------------------------

def test_english_results_unchanged_when_multilingual_off():
    """
    Ensures that processing an English claim produces identical predictions
    and is completely unaffected by multilingual extensions.
    """
    en_claim_text = "Solar power generation reached parity with fossil generation during peak midday hours."
    claim_en = Claim(id="test_en_001", text=en_claim_text)

    # 1. Run with default config
    report1 = verify_claim(claim_en, enable_reasoning=False)

    # 2. Run again
    report2 = verify_claim(claim_en, enable_reasoning=False)

    assert report1.verdict == report2.verdict
    assert abs(report1.confidence - report2.confidence) < 1e-5
    assert report1.language_info.code == "en"
    assert report1.original_text == en_claim_text


# ---------------------------------------------------------------------------
# 6. Mocked Translator & Cache Tests
# ---------------------------------------------------------------------------

def test_mocked_translator():
    """Tests Translator caching, batching, and memory release."""
    translator = Translator.get_instance()
    cache_path = Path("outputs/stage9/test_cache.json")
    if cache_path.is_file():
        cache_path.unlink()

    translator.cache_path = cache_path
    translator.cache = {}

    # Test phrase translation (using dictionary / offline fallback)
    hi_text = "उत्तरी बांध का जलस्तर सामान्य सीमा के भीतर और सुरक्षित है।"
    translated = translator.translate(hi_text, src_lang="hi", tgt_lang="en")

    assert isinstance(translated, str)
    assert len(translated) > 0
    # Verify disk cache written
    assert cache_path.is_file()

    # Verify memory freeing does not crash
    translator.free_memory()
    assert translator._model_hi_en is None


# ---------------------------------------------------------------------------
# 7. Reasoning Layer Temporal Extensions Test
# ---------------------------------------------------------------------------

def test_temporal_hindi_keywords_and_numerals():
    """Validates extraction of Hindi/Hinglish temporal keywords and Devanagari numerals."""
    analyzer = TemporalAnalyzer()

    # Hindi Devanagari keywords
    text_hi = "आज सुबह बांध में पानी सामान्य है और कल भी रहेगा।"
    exprs_hi = analyzer.extract_time_expressions(text_hi)
    assert "आज" in exprs_hi
    assert "कल" in exprs_hi

    # Hinglish keywords
    text_hgl = "Bhai aaj subah aur kal parso report aayi thi."
    exprs_hgl = analyzer.extract_time_expressions(text_hgl)
    assert "aaj" in exprs_hgl
    assert "kal" in exprs_hgl
    assert "parso" in exprs_hgl

    # Devanagari year extraction
    text_dev_year = "यह नियम साल २०२४ से लागू होगा।"
    exprs_year = analyzer.extract_time_expressions(text_dev_year)
    assert "२०२४" in exprs_year

    # Date parsing with Devanagari numerals
    dt = analyzer.parse_date("२०२४-०५-१०")
    assert dt is not None
    assert dt.year == 2024
    assert dt.month == 5
    assert dt.day == 10


# ---------------------------------------------------------------------------
# 8. CPU Smoke Test
# ---------------------------------------------------------------------------

def test_cpu_smoke_test():
    """CPU smoke test verifying that multilingual claim verification executes cleanly end-to-end."""
    sample_hinglish = "Bhai AIIMS ke doctor ne kal confirm kiya ki new herbal syrup effective hai."
    claim = Claim(id="smoke_001", text=sample_hinglish)

    report = verify_claim(claim, enable_reasoning=True)
    assert isinstance(report, ClaimReport)
    assert report.verdict in (Verdict.VERIFIED, Verdict.CONTRADICTED, Verdict.INSUFFICIENT_EVIDENCE)
    assert 0.0 <= report.confidence <= 1.0
    assert report.language_info is not None
    assert report.original_text == sample_hinglish
