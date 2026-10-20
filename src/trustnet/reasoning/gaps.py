"""
Rule-Based Evidence Gap Analysis Engine for TrustNet Stage 8.

Identifies missing verification pillars, unconfirmed geographic assertions,
and evidentiary voids using explicit rules over structured pipeline representations:
1. no_primary_source: No government or wire news agency evidence.
2. no_dated_evidence: All retrieved evidence passages lack timestamps.
3. stale_evidence: Newest evidence exceeds staleness threshold.
4. no_location_confirmation: Claim mentions a location uncorroborated in evidence (gazetteer-based regex).
5. conflicting_sources: Authoritative evidence contradicts core proposition.
6. image_origin_unknown: Attached image has no reuse matches and no EXIF timestamps.
7. image_text_mismatch_high: High visual-textual semantic mismatch (score >= 0.70).
8. source_unnamed: Unnamed or generic source attribution in original message.
9. low_retrieval_scores: Maximum evidence retrieval score falls below threshold (< 0.50).

Descriptions and verification instructions are loaded from data/eval/gap_templates.yaml.
"""

import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Union

import yaml

from trustnet.schemas import ConflictInfo, Evidence, Gap, ImageSignals, SourceMention, TemporalInfo

logger = logging.getLogger(__name__)

DEFAULT_GAP_TEMPLATES_PATH = Path("data/eval/gap_templates.yaml")

# Primary source types considered authoritative ground-truth pillars
PRIMARY_SOURCE_TYPES = {"government", "wire", "academic"}

# Location Gazetteer: Curated list of countries, states, and major urban centers
# Method used: Case-insensitive whole-word boundary regex over geographical dictionary
COMMON_LOCATIONS = {
    # Indian Metros & Cities
    "delhi", "new delhi", "mumbai", "bangalore", "bengaluru", "kolkata", "chennai",
    "hyderabad", "kochi", "pune", "ahmedabad", "jaipur", "lucknow", "varanasi", "patna",
    "bhopal", "chandigarh", "guwahati", "srinagar", "shimla", "dehradun", "amritsar",
    # Indian States
    "kerala", "karnataka", "maharashtra", "tamil nadu", "uttar pradesh", "bihar",
    "west bengal", "punjab", "haryana", "rajasthan", "gujarat", "assam", "odisha",
    # Global / Regional
    "india", "antarctica", "europe", "uk", "usa", "china",
}


class GapAnalyzer:
    """
    Evaluates evidence sufficiency across 9 deterministic verification dimensions.
    """

    def __init__(
        self,
        templates_path: Union[str, Path] = DEFAULT_GAP_TEMPLATES_PATH,
        retrieval_score_threshold: float = 0.50,
        mismatch_threshold: float = 0.70,
    ):
        self.templates_path = Path(templates_path)
        self.retrieval_score_threshold = retrieval_score_threshold
        self.mismatch_threshold = mismatch_threshold
        self.templates: Dict[str, Dict[str, str]] = {}
        self._load_templates()

    def _load_templates(self) -> None:
        """Loads fixed gap descriptions and remediation instructions from YAML."""
        if not self.templates_path.exists():
            logger.warning(f"Gap templates file {self.templates_path} not found; using built-in defaults.")
            self.templates = {
                "no_primary_source": {
                    "description": "No primary official, government, or wire agency evidence passage was retrieved.",
                    "how_to_check": "Cross-reference with official ministry portals or news wire feeds.",
                },
                "no_dated_evidence": {
                    "description": "None of the retrieved evidence passages contain verifiable publication timestamps.",
                    "how_to_check": "Inspect official press release archives or news bylines.",
                },
                "stale_evidence": {
                    "description": "The most recent retrieved evidence is older than the staleness threshold.",
                    "how_to_check": "Search for recent developments published within the last 30 to 90 days.",
                },
                "no_location_confirmation": {
                    "description": "The claim specifies a location that is not corroborated by any retrieved evidence.",
                    "how_to_check": "Check local district administration notices or regional municipal bulletins.",
                },
                "conflicting_sources": {
                    "description": "Reputable sources present direct contradictions regarding the core factual proposition.",
                    "how_to_check": "Compare specific statements cited by both supporting and refuting sources.",
                },
                "image_origin_unknown": {
                    "description": "The visual asset has no known provenance, reverse image matches, or EXIF metadata.",
                    "how_to_check": "Perform a reverse image search across Google Images or TinEye.",
                },
                "image_text_mismatch_high": {
                    "description": "Significant semantic mismatch detected between the image and textual claim.",
                    "how_to_check": "Verify if an archival image has been repurposed with misleading captioning.",
                },
                "source_unnamed": {
                    "description": "The claim attributes information to an unnamed, anonymous, or vague insider source.",
                    "how_to_check": "Look for on-the-record statements attributed to named authorized spokespersons.",
                },
                "low_retrieval_scores": {
                    "description": "Retrieved evidence has weak semantic relevance scores, indicating insufficient corpus coverage.",
                    "how_to_check": "Broaden keyword search queries across external independent knowledge bases.",
                },
            }
            return

        with open(self.templates_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        self.templates = data.get("gaps", {})
        logger.info(f"Loaded {len(self.templates)} gap templates from {self.templates_path}")

    def _extract_mentioned_locations(self, text: str) -> List[str]:
        """
        Detects mentioned geographic locations using whole-word boundary matching
        against the curated gazetteer.
        Method: Deterministic regex whole-word search over COMMON_LOCATIONS.
        """
        mentioned = []
        t_lower = text.lower()
        for loc in sorted(COMMON_LOCATIONS, key=len, reverse=True):
            if re.search(rf"\b{re.escape(loc)}\b", t_lower):
                mentioned.append(loc)
        return mentioned

    def analyze(
        self,
        claim_text: str,
        evidence_list: List[Evidence],
        conflict_info: Optional[ConflictInfo] = None,
        temporal_info: Optional[TemporalInfo] = None,
        image_signals: Optional[ImageSignals] = None,
        source_mentions: Optional[List[SourceMention]] = None,
    ) -> List[Gap]:
        """
        Runs comprehensive rule-based gap analysis.
        """
        gaps: List[Gap] = []
        triggered_codes: Set[str] = set()

        # 1. Rule: no_primary_source
        has_primary = any(
            (ev.source_type or "").lower() in PRIMARY_SOURCE_TYPES
            for ev in evidence_list
        )
        if not has_primary:
            triggered_codes.add("no_primary_source")

        # 2. Rule: no_dated_evidence
        has_dates = any(
            bool(ev.date or (ev.metadata and ev.metadata.get("date")))
            for ev in evidence_list
        )
        if not has_dates:
            triggered_codes.add("no_dated_evidence")

        # 3. Rule: stale_evidence
        if temporal_info is not None and temporal_info.is_stale:
            triggered_codes.add("stale_evidence")

        # 4. Rule: no_location_confirmation
        claim_locs = self._extract_mentioned_locations(claim_text)
        if claim_locs:
            evidence_text_combined = " ".join([ev.text for ev in evidence_list]).lower()
            unconfirmed_locs = [
                loc for loc in claim_locs
                if not re.search(rf"\b{re.escape(loc)}\b", evidence_text_combined)
            ]
            if unconfirmed_locs:
                triggered_codes.add("no_location_confirmation")

        # 5. Rule: conflicting_sources
        if conflict_info is not None and conflict_info.has_conflict:
            triggered_codes.add("conflicting_sources")

        # 6. Rule: image_origin_unknown
        if image_signals is not None:
            # Check if image has no reuse matches and no EXIF timestamps
            has_reuse = len(image_signals.reuse_matches) > 0
            has_exif_time = bool(image_signals.exif_summary and any(
                k.lower() in ["datetime", "datetimeoriginal", "createdate"]
                for k in image_signals.exif_summary.keys()
            ))
            if not has_reuse and not has_exif_time:
                triggered_codes.add("image_origin_unknown")

            # 7. Rule: image_text_mismatch_high
            if image_signals.mismatch_score is not None and image_signals.mismatch_score >= self.mismatch_threshold:
                triggered_codes.add("image_text_mismatch_high")

        # 8. Rule: source_unnamed
        if source_mentions:
            for sm in source_mentions:
                sm_text = sm.text.lower()
                # Check for generic/anonymous attribution patterns
                if any(w in sm_text for w in ["cousin", "friend", "insider", "sources", "anonymous", "whatsapp", "someone", "group", "family"]):
                    triggered_codes.add("source_unnamed")
                    break
        else:
            # Fallback lexical check on claim text for unnamed source attributions
            c_lower = claim_text.lower()
            if any(w in c_lower for w in ["insider", "sources say", "sources claim", "anonymous", "family groups", "whatsapp group", "someone told me"]):
                triggered_codes.add("source_unnamed")


        # 9. Rule: low_retrieval_scores
        if not evidence_list:
            triggered_codes.add("low_retrieval_scores")
        else:
            max_score = max([float(ev.score or 0.0) for ev in evidence_list])
            if max_score < self.retrieval_score_threshold:
                triggered_codes.add("low_retrieval_scores")

        # Assemble Gap objects from templates
        for code in sorted(list(triggered_codes)):
            tmpl = self.templates.get(code, {})
            desc = tmpl.get("description", f"Identified verification gap: {code}")
            check = tmpl.get("how_to_check", "Conduct independent verification across primary public records.")
            gaps.append(Gap(code=code, description=desc, how_to_check=check))

        return gaps
