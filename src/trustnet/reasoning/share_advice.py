"""
Actionable Sharing Advice Engine for TrustNet Stage 8.

Implements a transparent, readable decision table mapping:
(verdict, calibrated_confidence, abstained, conflict, stale) -> ShareAdvice

ShareAdvice Options:
- SAFE_TO_SHARE: Fully verified by high-confidence, timely, non-conflicting evidence.
- SHARE_WITH_CAUTION: Verified historically but stale, or verified with moderate confidence,
  or insufficient evidence.
- DO_NOT_SHARE_AS_CONFIRMED: Contradicted claim, authoritative conflict, or abstained due to uncertainty.

Strict Requirements:
1. Each row emits exactly one reason code and one plain-language reason from SHARE_ADVICE_TEMPLATES.
2. No plain-language reason string contains the substring " or ".
3. Verdict and recommended action are always separate fields.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from trustnet.schemas import ConflictInfo, ShareAdvice, TemporalInfo, Verdict


# Plain-Language Reason Template Table (Strictly no " or " substrings)
SHARE_ADVICE_TEMPLATES: Dict[str, str] = {
    "contradicted_by_evidence": "Claim is refuted by reliable evidence.",
    "authoritative_conflict": "Authoritative sources conflict on this claim.",
    "high_uncertainty_abstain": "The verification system abstained due to high uncertainty.",
    "insufficient_evidence": "Available evidence is insufficient to verify the claim.",
    "stale_evidence": "Claim is supported by evidence, but the evidence is outdated.",
    "verified_high_confidence": "Corroborated by high-confidence, fresh, consistent evidence.",
    "verified_moderate_confidence": "Supported by evidence with moderate calibrated confidence (<0.80).",
    "inconclusive_fallback": "Verification status is inconclusive; proceed with caution.",
}


# Transparent Decision Table Specification
# Rules are evaluated sequentially in order of precedence.
DECISION_TABLE: List[Dict[str, Any]] = [
    {
        "row_id": "rule_1_contradicted",
        "condition": "verdict == contradicted",
        "advice": ShareAdvice.DO_NOT_SHARE_AS_CONFIRMED,
        "reason_code": "contradicted_by_evidence",
        "rationale": SHARE_ADVICE_TEMPLATES["contradicted_by_evidence"],
    },
    {
        "row_id": "rule_2_active_conflict",
        "condition": "has_conflict == True",
        "advice": ShareAdvice.DO_NOT_SHARE_AS_CONFIRMED,
        "reason_code": "authoritative_conflict",
        "rationale": SHARE_ADVICE_TEMPLATES["authoritative_conflict"],
    },
    {
        "row_id": "rule_3_abstained",
        "condition": "abstained == True",
        "advice": ShareAdvice.DO_NOT_SHARE_AS_CONFIRMED,
        "reason_code": "high_uncertainty_abstain",
        "rationale": SHARE_ADVICE_TEMPLATES["high_uncertainty_abstain"],
    },
    {
        "row_id": "rule_4_insufficient_evidence",
        "condition": "verdict == insufficient_evidence",
        "advice": ShareAdvice.SHARE_WITH_CAUTION,
        "reason_code": "insufficient_evidence",
        "rationale": SHARE_ADVICE_TEMPLATES["insufficient_evidence"],
    },
    {
        "row_id": "rule_5_verified_stale",
        "condition": "verdict == verified and is_stale == True",
        "advice": ShareAdvice.SHARE_WITH_CAUTION,
        "reason_code": "stale_evidence",
        "rationale": SHARE_ADVICE_TEMPLATES["stale_evidence"],
    },
    {
        "row_id": "rule_6_verified_high_confidence",
        "condition": "verdict == verified and confidence >= 0.80 and not is_stale and not has_conflict and not abstained",
        "advice": ShareAdvice.SAFE_TO_SHARE,
        "reason_code": "verified_high_confidence",
        "rationale": SHARE_ADVICE_TEMPLATES["verified_high_confidence"],
    },
    {
        "row_id": "rule_7_verified_moderate_confidence",
        "condition": "verdict == verified and confidence < 0.80",
        "advice": ShareAdvice.SHARE_WITH_CAUTION,
        "reason_code": "verified_moderate_confidence",
        "rationale": SHARE_ADVICE_TEMPLATES["verified_moderate_confidence"],
    },
]


def determine_share_advice(
    verdict: Union[Verdict, str],
    confidence: float,
    abstained: bool = False,
    conflict: Optional[Union[ConflictInfo, bool]] = None,
    temporal: Optional[Union[TemporalInfo, bool]] = None,
    return_code: bool = False,
) -> Union[Tuple[ShareAdvice, str], Tuple[ShareAdvice, str, str]]:
    """
    Evaluates sharing advice via the transparent decision table.

    Args:
        verdict: Predicted verdict (Verdict enum or string).
        confidence: Calibrated confidence score in [0, 1].
        abstained: True if the model abstained.
        conflict: ConflictInfo object or boolean flag indicating conflict.
        temporal: TemporalInfo object or boolean flag indicating staleness.
        return_code: If True, returns a 3-tuple (advice, reason_code, rationale).
                     Defaults to False returning (advice, rationale) for backward compatibility.

    Returns:
        Tuple of (advice, rationale) or (advice, reason_code, rationale).
    """
    # Normalize inputs
    v_str = str(verdict.value if isinstance(verdict, Verdict) else verdict).lower()
    conf = float(confidence)

    has_conflict = False
    if isinstance(conflict, ConflictInfo):
        has_conflict = conflict.has_conflict
    elif isinstance(conflict, bool):
        has_conflict = conflict

    is_stale = False
    if isinstance(temporal, TemporalInfo):
        is_stale = temporal.is_stale
    elif isinstance(temporal, bool):
        is_stale = temporal

    # 1. Row 1: Contradicted
    if v_str in ("contradicted", "refuted", "false"):
        code = "contradicted_by_evidence"
        advice = ShareAdvice.DO_NOT_SHARE_AS_CONFIRMED

    # 2. Row 2: Active Conflict
    elif has_conflict:
        code = "authoritative_conflict"
        advice = ShareAdvice.DO_NOT_SHARE_AS_CONFIRMED

    # 3. Row 3: Abstained
    elif abstained:
        code = "high_uncertainty_abstain"
        advice = ShareAdvice.DO_NOT_SHARE_AS_CONFIRMED

    # 4. Row 4: Insufficient Evidence
    elif v_str in ("insufficient_evidence", "insufficient", "unverified"):
        code = "insufficient_evidence"
        advice = ShareAdvice.SHARE_WITH_CAUTION

    # 5. Row 5: Verified but Stale
    elif v_str == "verified" and is_stale:
        code = "stale_evidence"
        advice = ShareAdvice.SHARE_WITH_CAUTION

    # 6. Row 6: Verified with High Confidence
    elif v_str == "verified" and conf >= 0.80:
        code = "verified_high_confidence"
        advice = ShareAdvice.SAFE_TO_SHARE

    # 7. Row 7: Verified with Moderate Confidence (< 0.80)
    elif v_str == "verified":
        code = "verified_moderate_confidence"
        advice = ShareAdvice.SHARE_WITH_CAUTION

    # Fallback default
    else:
        code = "inconclusive_fallback"
        advice = ShareAdvice.SHARE_WITH_CAUTION

    rationale = SHARE_ADVICE_TEMPLATES.get(code, "Verification status is inconclusive; proceed with caution.")

    if return_code:
        return advice, code, rationale
    return advice, rationale


def get_share_advice_decision(
    verdict: Union[Verdict, str],
    confidence: float,
    abstained: bool = False,
    conflict: Optional[Union[ConflictInfo, bool]] = None,
    temporal: Optional[Union[TemporalInfo, bool]] = None,
) -> Tuple[ShareAdvice, str, str]:
    """
    Returns the complete (advice, reason_code, rationale) decision tuple.
    """
    res = determine_share_advice(
        verdict=verdict,
        confidence=confidence,
        abstained=abstained,
        conflict=conflict,
        temporal=temporal,
        return_code=True,
    )
    return res  # type: ignore
