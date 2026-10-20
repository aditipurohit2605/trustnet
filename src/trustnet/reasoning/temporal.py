"""
Temporal Freshness and Staleness Engine for TrustNet Stage 8.

Features:
- Rule-based time expression extraction for English and Hindi/Hinglish (today, yesterday, aaj, kal, abhi, etc.).
- Compares claim timestamps against Evidence dates using dateparser.
- Marks is_stale=True when newest evidence is older than stale_days or when claim asserts
  a present-tense event without recent corroboration.
- Anti-hallucination guarantee: missing dates are reported as missing, never guessed.
"""

from datetime import datetime, timezone
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import dateparser

from trustnet.schemas import Evidence, TemporalInfo

logger = logging.getLogger(__name__)

DEVANAGARI_DIGITS_MAP = str.maketrans("०१२३४५६७८९", "0123456789")

# Keywords indicating present-tense or breaking assertions (English, Hinglish, Devanagari)
NOW_KEYWORDS = {
    "today", "now", "currently", "breaking", "urgent", "this morning", "tonight",
    "aaj", "abhi", "filhal", "is samay", "is waqt", "haali me", "aaj subah", "turant",
    "आज", "अभी", "फिलहाल", "इस समय", "इस वक्त", "हाल ही में", "आज सुबह", "आज शाम", "आज रात", "तुरंत",
}

# General English, Hinglish, and Hindi temporal keywords for extraction
TEMPORAL_KEYWORDS = {
    "today", "yesterday", "tomorrow", "recently", "now", "currently",
    "last night", "this morning", "last week", "this week", "last month", "this year",
    "aaj", "kal", "parso", "parson", "abhi", "filhal", "is samay", "is waqt",
    "subah", "shaam", "raat", "haali me", "turant",
    "आज", "कल", "परसों", "परसो", "अभी", "फिलहाल", "हाल ही में", "इस समय", "इस वक्त",
    "सुबह", "शाम", "रात", "तुरंत", "बीते कल", "आने वाले कल",
}

# Regex for explicit year, month, or formatted date strings (ASCII digits)
DATE_REGEX = re.compile(
    r"\b(?:"
    r"\d{4}[-/]\d{1,2}[-/]\d{1,2}"  # YYYY-MM-DD or YYYY/MM/DD
    r"|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}"  # DD-MM-YYYY or MM/DD/YYYY
    r"|(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    r"\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}"  # Month DD, YYYY
    r"|\b(?:20\d\d|19\d\d)\b"  # 4-digit years
    r")\b",
    re.IGNORECASE,
)

# Regex matching Devanagari 4-digit years (e.g. २०२४, २०१९)
DEVANAGARI_YEAR_REGEX = re.compile(r"[०-९]{4}")


class TemporalAnalyzer:
    """
    Extracts temporal expressions and determines evidentiary freshness and staleness.
    """

    def __init__(
        self,
        stale_days: int = 90,
        recent_cutoff_days: int = 14,
        reference_date: Optional[datetime] = None,
    ):
        self.stale_days = stale_days
        self.recent_cutoff_days = recent_cutoff_days
        self.reference_date = reference_date or datetime.now()

    def extract_time_expressions(self, text: str) -> List[str]:
        """
        Extracts temporal expressions (keywords, relative terms, and explicit dates) from text.
        Preserves exact substring occurrences.
        """
        if not text:
            return []

        expressions: List[str] = []
        tokens = re.findall(r"\b[\w\-']+\b", text.lower())

        # 1. Match multi-word keywords
        t_lower = text.lower()
        for kw in sorted(TEMPORAL_KEYWORDS, key=len, reverse=True):
            if re.search(rf"\b{re.escape(kw)}\b", t_lower):
                if kw not in expressions:
                    expressions.append(kw)

        # 2. Match Devanagari years (e.g., २०२४)
        dev_years = DEVANAGARI_YEAR_REGEX.findall(text)
        for dy in dev_years:
            if dy not in expressions:
                expressions.append(dy)

        # 3. Match explicit formatted dates and years via regex
        matches = DATE_REGEX.findall(text)
        for m in matches:
            m_clean = m.strip()
            if m_clean and m_clean.lower() not in expressions:
                expressions.append(m_clean)

        return expressions

    def parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """
        Parses an ISO or freeform date string safely using dateparser.
        Supports Devanagari numerals by normalizing them to ASCII digits.
        Returns None if missing or unparseable (never guesses).
        """
        if not date_str or not isinstance(date_str, str) or not date_str.strip():
            return None

        # Clean string and normalize Devanagari numerals
        clean_str = date_str.strip().translate(DEVANAGARI_DIGITS_MAP)
        try:
            # First try ISO format
            if len(clean_str) >= 10 and clean_str[4] == "-" and clean_str[7] == "-":
                return datetime.strptime(clean_str[:10], "%Y-%m-%d")

            # Fall back to dateparser
            parsed = dateparser.parse(
                clean_str,
                settings={"PREFER_DAY_OF_MONTH": "first", "REQUIRE_PARTS": ["year"]},
            )
            return parsed
        except Exception as e:
            logger.debug(f"Failed to parse date '{date_str}': {e}")
            return None

    def analyze(
        self,
        claim_text: str,
        evidence_list: List[Evidence],
    ) -> TemporalInfo:
        """
        Analyzes temporal alignment between claim and evidence.
        """
        time_expressions = self.extract_time_expressions(claim_text)
        claim_implies_now = any(
            any(kw in expr.lower() for kw in NOW_KEYWORDS)
            for expr in time_expressions
        )

        parsed_evidence_dates: List[Tuple[datetime, str, Evidence]] = []
        missing_date_count = 0

        for ev in evidence_list:
            ev_date_raw = ev.date
            if not ev_date_raw and ev.metadata:
                ev_date_raw = ev.metadata.get("date")

            parsed_dt = self.parse_date(ev_date_raw)
            if parsed_dt is not None:
                parsed_evidence_dates.append((parsed_dt, ev_date_raw or "", ev))
            else:
                missing_date_count += 1

        # Case 1: No dates found in any evidence
        if not parsed_evidence_dates:
            if claim_implies_now:
                return TemporalInfo(
                    claim_time_expressions=time_expressions,
                    newest_evidence_date=None,
                    oldest_evidence_date=None,
                    is_stale=True,
                    explanation=(
                        "Claim asserts a present-tense or breaking event ('today'/'now'/'abhi'), "
                        "but no retrieved evidence contains verifiable publication dates."
                    ),
                )
            return TemporalInfo(
                claim_time_expressions=time_expressions,
                newest_evidence_date=None,
                oldest_evidence_date=None,
                is_stale=False,
                explanation="No dated evidence available; evidence dates were missing or unparseable.",
            )

        # Sort dates chronologically
        parsed_evidence_dates.sort(key=lambda item: item[0])
        oldest_dt, oldest_str, _ = parsed_evidence_dates[0]
        newest_dt, newest_str, _ = parsed_evidence_dates[-1]

        # Calculate age of newest evidence relative to reference date
        age_days = (self.reference_date.date() - newest_dt.date()).days

        is_stale = False
        explanation_parts = []

        if age_days > self.stale_days:
            is_stale = True
            explanation_parts.append(
                f"Newest evidence ({newest_str}) is {age_days} days old, exceeding the {self.stale_days}-day staleness limit."
            )
        elif claim_implies_now and age_days > self.recent_cutoff_days:
            is_stale = True
            explanation_parts.append(
                f"Claim implies an immediate or present occurrence, but the most recent corroborating evidence is {age_days} days old."
            )
        else:
            explanation_parts.append(
                f"Evidence is timely: newest source is from {newest_str} ({max(0, age_days)} days old)."
            )

        if missing_date_count > 0:
            explanation_parts.append(f"({missing_date_count} evidence item(s) lacked publication dates).")

        return TemporalInfo(
            claim_time_expressions=time_expressions,
            newest_evidence_date=newest_str,
            oldest_evidence_date=oldest_str,
            is_stale=is_stale,
            explanation=" ".join(explanation_parts),
        )
