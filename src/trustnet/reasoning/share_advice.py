"""
Actionable Sharing Advice Engine for TrustNet Stage 8.

Implements a transparent, readable decision table mapping:
(verdict, calibrated_confidence, abstained, conflict, stale) -> ShareAdvice

ShareAdvice Options:
- SAFE_TO_SHARE: Fully verified by high-confidence, timely, non-conflicting evidence.
- SHARE_WITH_CAUTION: Verified historically but stale, or verified with moderate confidence,
  or insufficient evidence.
- DO_NOT_SHARE_AS_CONFIRMED: Contradicted/false claim, authoritative conflict, or abstained due to uncertainty.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from trustnet.schemas import ConflictInfo, ShareAdvice, TemporalInfo, Verdict


# Transparent Decision Table Specification
# Rules are evaluated sequentially in order of precedence.
DECISION_TABLE: List[Dict[str, Any]] = [
    {
        "row_id": "rule_1_contradicted",
        "condition": "verdict == contradicted",
        "advice": ShareAdvice.DO_NOT_SHARE_AS_CONFIRMED,
        "rationale": "Claim has been contradicted or refuted by evidence.",
    },
    {
        "row_id": "rule_2_active_conflict",
        "condition": "has_conflict == True",
        "advice": ShareAdvice.DO_NOT_SHARE_AS_CONFIRMED,
        "rationale": "Authoritative sources directly conflict on this claim.",
    },
    {
        "row_id": "rule_3_abstained",
        "condition": "abstained == True",
        "advice": ShareAdvice.DO_NOT_SHARE_AS_CONFIRMED,
        "rationale": "The verification system abstained due to high uncertainty.",
    },
    {
        "row_id": "rule_4_insufficient_evidence",
        "condition": "verdict == insufficient_evidence",
        "advice": ShareAdvice.SHARE_WITH_CAUTION,
        "rationale": "Available evidence is insufficient to verify or refute the claim.",
    },
    {
        "row_id": "rule_5_verified_stale",
        "condition": "verdict == verified and is_stale == True",
        "advice": ShareAdvice.SHARE_WITH_CAUTION,
        "rationale": "Claim is supported by evidence, but the evidence is stale or outdated.",
    },
    {
        "row_id": "rule_6_verified_high_confidence",
        "condition": "verdict == verified and confidence >= 0.80 and not is_stale and not has_conflict and not abstained",
        "advice": ShareAdvice.SAFE_TO_SHARE,
        "rationale": "Corroborated by high-confidence, fresh, and consistent evidence.",
    },
    {
        "row_id": "rule_7_verified_moderate_confidence",
        "condition": "verdict == verified and confidence < 0.80",
        "advice": ShareAdvice.SHARE_WITH_CAUTION,
        "rationale": "Supported by evidence, but calibrated confidence is moderate (<0.80).",
    },
]


def determine_share_advice(
    verdict: Union[Verdict, str],
    confidence: float,
    abstained: bool = False,
    conflict: Optional[Union[ConflictInfo, bool]] = None,
    temporal: Optional[Union[TemporalInfo, bool]] = None,
) -> Tuple[ShareAdvice, str]:
    """
    Evaluates sharing advice via the transparent decision table.
    
    Args:
        verdict: Predicted verdict (Verdict enum or string).
        confidence: Calibrated confidence score in [0, 1].
        abstained: True if the model abstained.
        conflict: ConflictInfo object or boolean flag indicating conflict.
        temporal: TemporalInfo object or boolean flag indicating staleness.
        
    Returns:
        Tuple of (ShareAdvice enum value, explanation string).
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
        return ShareAdvice.DO_NOT_SHARE_AS_CONFIRMED, "Claim has been contradicted or refuted by evidence."

    # 2. Row 2: Active Conflict
    if has_conflict:
        return ShareAdvice.DO_NOT_SHARE_AS_CONFIRMED, "Authoritative sources directly conflict on this claim."

    # 3. Row 3: Abstained
    if abstained:
        return ShareAdvice.DO_NOT_SHARE_AS_CONFIRMED, "The verification system abstained due to high uncertainty."

    # 4. Row 4: Insufficient Evidence
    if v_str in ("insufficient_evidence", "insufficient", "unverified"):
        return ShareAdvice.SHARE_WITH_CAUTION, "Available evidence is insufficient to verify or refute the claim."

    # 5. Row 5: Verified but Stale
    if v_str == "verified" and is_stale:
        return ShareAdvice.SHARE_WITH_CAUTION, "Claim is supported by evidence, but the evidence is stale or outdated."

    # 6. Row 6: Verified with High Confidence
    if v_str == "verified" and conf >= 0.80:
        return ShareAdvice.SAFE_TO_SHARE, "Corroborated by high-confidence, fresh, and consistent evidence."

    # 7. Row 7: Verified with Moderate Confidence (< 0.80)
    if v_str == "verified":
        return ShareAdvice.SHARE_WITH_CAUTION, "Supported by evidence, but calibrated confidence is moderate (<0.80)."

    # Fallback default
    return ShareAdvice.SHARE_WITH_CAUTION, "Verification status is inconclusive; share with caution."
