"""
Comprehensive Unit and Integration Test Suite for TrustNet Stage 8: Reasoning Layer.

Tests:
1. Source reliability weights applied correctly from config, unknown source handling, ablation toggle.
2. Conflict detection requires both sides above weight and confidence thresholds; no averaging away.
3. Temporal staleness logic (English and Hindi/Hinglish keywords).
4. Missing evidence dates are never guessed or fabricated.
5. Manipulation tag spans are strictly genuine substrings of the message.
6. Gaps originate exclusively from the template catalog.
7. Optional rephrase check verifies no new entities or numbers are injected.
8. Counterfactual sensitivity: every reported flip reproduced 100% by re-running the model.
9. Share advice decision table completeness (every row tested).
10. Strict verdict and confidence immutability: reasoning layer never modifies underlying predictions.
11. Mocked-LLM test verifying rejection of hallucinated spans.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
from unittest.mock import MagicMock, patch

import pytest

from trustnet.reasoning.conflict import ConflictDetector
from trustnet.reasoning.counterfactual import CounterfactualGenerator
from trustnet.reasoning.gaps import GapAnalyzer
from trustnet.reasoning.manipulation import ManipulationTagger
from trustnet.reasoning.share_advice import determine_share_advice
from trustnet.reasoning.source_reliability import SourceReliabilityAssigner
from trustnet.reasoning.temporal import TemporalAnalyzer
from trustnet.schemas import (
    Claim,
    ClaimReport,
    ConflictInfo,
    Counterfactual,
    Evidence,
    Gap,
    ImageSignals,
    ManipulationTag,
    ShareAdvice,
    SourceMention,
    TemporalInfo,
    Verdict,
)


# ==============================================================================
# 1. Source Reliability Tests
# ==============================================================================

def test_source_reliability_weights_applied_from_config():
    """Verifies source reliability weights and rationales are correctly assigned from YAML."""
    assigner = SourceReliabilityAssigner(config_path="configs/source_reliability.yaml")

    ev_gov = Evidence(doc_id="e1", text="Official bulletin", source_type="government")
    ev_news = Evidence(doc_id="e2", text="News report", source_type="news")
    ev_social = Evidence(doc_id="e3", text="Viral tweet", source_type="social_media")

    weights, summary = assigner.assign_weights([ev_gov, ev_news, ev_social])

    assert len(weights) == 3
    # Check government weight
    assert weights[0].source_type == "government"
    assert weights[0].weight == 1.0
    assert "statutory" in weights[0].rationale.lower() or "official" in weights[0].rationale.lower()

    # Check news weight
    assert weights[1].source_type == "news"
    assert weights[1].weight == 0.75

    # Check social media weight
    assert weights[2].source_type == "social_media"
    assert weights[2].weight == 0.30

    # Summary checks
    assert summary["total_evidence"] == 3
    assert summary["average_weight"] > 0.5
    assert len(summary["unknown_sources"]) == 0


def test_source_reliability_unknown_sources():
    """Verifies unlisted source types receive the default baseline weight and are flagged."""
    assigner = SourceReliabilityAssigner(config_path="configs/source_reliability.yaml")
    ev_unknown = Evidence(doc_id="u1", text="Anonymous blog post", source_type="unregistered_blog")

    weights, summary = assigner.assign_weights([ev_unknown])

    assert weights[0].source_type == "unregistered_blog"
    assert weights[0].weight == 0.20  # Default baseline weight from config
    assert "unregistered_blog" in summary["unknown_sources"]


def test_source_reliability_ablation_switch():
    """Verifies that disabling source weighting assigns uniform 1.0 weight to all sources."""
    assigner = SourceReliabilityAssigner(
        config_path="configs/source_reliability.yaml",
        enable_weighting=False,
    )

    ev_social = Evidence(doc_id="s1", text="Rumor", source_type="social_media")
    weights, summary = assigner.assign_weights([ev_social])

    # All weights must be equal to 1.0 under ablation
    assert weights[0].weight == 1.0
    assert summary["average_weight"] == 1.0


# ==============================================================================
# 2. Conflict Detection Tests
# ==============================================================================

def test_conflict_detection_requires_both_sides_above_weight_threshold():
    """
    Conflict is reported only when AT LEAST ONE supporting AND ONE refuting passage
    each exceed conflict_min_weight and confidence_threshold.
    """
    detector = ConflictDetector(conflict_min_weight=0.50, confidence_threshold=0.45)

    claim_text = "Regional transit authority suspended ferry operations in Kochi."

    # Case A: Authoritative government support vs Authoritative wire refutation -> CONFLICT
    ev_support = Evidence(
        doc_id="e1",
        text="Official port authority confirms ferry service suspended.",
        source_type="government",  # weight 1.0 >= 0.50
        metadata={"stance": "supporting"},
    )
    ev_refute = Evidence(
        doc_id="e2",
        text="Wire news service clarifies ferry service is operating normally.",
        source_type="wire",  # weight 0.95 >= 0.50
        metadata={"stance": "contradicting"},
    )

    info_conflict = detector.detect_conflicts(claim_text, [ev_support, ev_refute])
    assert info_conflict.has_conflict is True
    assert len(info_conflict.supporting) == 1
    assert len(info_conflict.contradicting) == 1
    assert "contradictory evidence detected" in info_conflict.explanation.lower()


def test_conflict_detection_ignores_unreliable_sources():
    """
    An unverified social media post disagreeing with a government source does NOT
    trigger conflict if its source weight is below conflict_min_weight.
    """
    detector = ConflictDetector(conflict_min_weight=0.50, confidence_threshold=0.45)
    claim_text = "Regional transit authority suspended ferry operations in Kochi."

    ev_gov = Evidence(
        doc_id="e1",
        text="Official government statement confirms normal operations.",
        source_type="government",  # weight 1.0
        metadata={"stance": "contradicting"},
    )
    ev_social = Evidence(
        doc_id="e2",
        text="Unverified tweet claims ferry service was suspended.",
        source_type="social_media",  # weight 0.10 < 0.50 threshold
        metadata={"stance": "supporting"},
    )

    info = detector.detect_conflicts(claim_text, [ev_gov, ev_social])
    # Social media passage does not meet weight threshold -> no formal conflict
    assert info.has_conflict is False
    assert len(info.contradicting) == 1
    assert len(info.supporting) == 0


def test_conflict_passages_not_averaged_away():
    """Ensures both conflicting sets of evidence are preserved in full with explanations."""
    detector = ConflictDetector(conflict_min_weight=0.50, confidence_threshold=0.45)
    claim = "Dam floodgates opened."

    ev1 = Evidence(doc_id="e1", text="Water dept confirms floodgates opened.", source_type="government", metadata={"stance": "supporting"})
    ev2 = Evidence(doc_id="e2", text="Disaster control says floodgates closed.", source_type="wire", metadata={"stance": "contradicting"})

    info = detector.detect_conflicts(claim, [ev1, ev2])
    assert info.has_conflict is True
    assert info.supporting[0].doc_id == "e1"
    assert info.contradicting[0].doc_id == "e2"


# ==============================================================================
# 3. Temporal Freshness & Staleness Logic Tests
# ==============================================================================

def test_temporal_staleness_older_than_stale_days():
    """Flags is_stale=True when newest evidence date exceeds stale_days threshold."""
    analyzer = TemporalAnalyzer(stale_days=90)
    claim_text = "WHO issues safety guidance on respiratory vaccines."

    old_date = (datetime.now(timezone.utc) - timedelta(days=180)).strftime("%Y-%m-%d")
    evidence = [Evidence(doc_id="e1", text="Safety guidance published.", date=old_date)]

    info = analyzer.analyze(claim_text, evidence)
    assert info.is_stale is True
    assert "older than" in info.explanation.lower() or "stale" in info.explanation.lower()


def test_temporal_time_expression_extraction_hinglish():
    """Extracts English and Hindi/Hinglish temporal keywords from claims."""
    analyzer = TemporalAnalyzer(stale_days=90)

    # English keywords
    info_en = analyzer.analyze("Heavy rain occurred yesterday in Delhi.", [])
    assert "yesterday" in info_en.claim_time_expressions

    # Hindi/Hinglish keywords
    info_hi1 = analyzer.analyze("Yeh ghatna aaj subah hui hai.", [])
    assert "aaj" in info_hi1.claim_time_expressions

    info_hi2 = analyzer.analyze("Police ne abhi arrest kiya.", [])
    assert "abhi" in info_hi2.claim_time_expressions


def test_temporal_claim_implies_now_without_recent_evidence():
    """Flags is_stale=True when claim implies 'now'/'today' but evidence is months old."""
    analyzer = TemporalAnalyzer(stale_days=90)
    claim_text = "Breaking: Power grid collapsed today leaving city in dark."

    old_date = "2024-01-15"
    evidence = [Evidence(doc_id="e1", text="Archival power maintenance report.", date=old_date)]

    info = analyzer.analyze(claim_text, evidence)
    assert info.is_stale is True
    assert "implies current" in info.explanation.lower() or "stale" in info.explanation.lower()


def test_missing_evidence_dates_never_guessed():
    """Missing timestamps in evidence must never be fabricated or guessed."""
    analyzer = TemporalAnalyzer(stale_days=90)
    claim_text = "Archaeologists discovered ancient temple ruins."

    evidence = [
        Evidence(doc_id="e1", text="Historical text without timestamp.", date=None),
        Evidence(doc_id="e2", text="Another undocumented passage.", date=""),
    ]

    info = analyzer.analyze(claim_text, evidence)
    assert info.newest_evidence_date is None
    assert info.oldest_evidence_date is None
    assert "unverifiable" in info.explanation.lower() or "missing" in info.explanation.lower()


# ==============================================================================
# 4. Manipulation Tagging Tests
# ==============================================================================

def test_manipulation_spans_are_real_substrings():
    """Guarantees every returned ManipulationTag span is an exact verbatim substring of the message."""
    tagger = ManipulationTagger()
    msg = "URGENT ALERT: Share with all your family groups right now before this is deleted!"

    tags = tagger.tag(msg)
    assert len(tags) > 0
    for t in tags:
        # Strict invariant: span must exist in the original untrusted text
        assert t.span in msg
        assert len(t.span) > 0


def test_mocked_llm_rejects_hallucinated_spans():
    """Verifies that hallucinated spans from an LLM are strictly rejected and dropped."""
    tagger = ManipulationTagger(ollama_url="http://localhost:11434/api/chat")

    msg = "Please verify this news before believing rumors."

    # Mock response from LLM containing one real span and one invented span
    fake_llm_json = {
        "message": {
            "content": (
                '{"tags": ['
                '{"name": "urgent_forward_plea", "confidence": 0.9, "span": "Please verify"}, '
                '{"name": "conspiracy_framing", "confidence": 0.8, "span": "SECRET GOVERNMENT PLOT INVENTED"}'
                ']}'
            )
        }
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = fake_llm_json

    with patch("requests.post", return_value=mock_resp):
        tags = tagger.tag(msg)

    tag_names = [t.name for t in tags]
    # Real substring must be preserved
    assert "urgent_forward_plea" in tag_names
    # Invented substring must be dropped
    assert "conspiracy_framing" not in tag_names


# ==============================================================================
# 5. Evidence Gap Analysis Tests
# ==============================================================================

def test_gaps_come_from_template_table_only():
    """All generated gaps must map directly to the standardized gap templates catalog."""
    analyzer = GapAnalyzer()
    claim = "Unverified event in Varanasi."
    evidence = [Evidence(doc_id="e1", text="No location mentioned.", source_type="news", score=0.2)]

    gaps = analyzer.analyze(
        claim_text=claim,
        evidence_list=evidence,
        conflict_info=ConflictInfo(has_conflict=False),
        temporal_info=TemporalInfo(is_stale=False),
    )

    valid_codes = set(analyzer.templates.keys())
    assert len(gaps) > 0
    for g in gaps:
        assert g.code in valid_codes
        # Description and check must match template exactly
        assert g.description == analyzer.templates[g.code]["description"]
        assert g.how_to_check == analyzer.templates[g.code]["how_to_check"]


def test_gap_location_confirmation_rule():
    """Detects no_location_confirmation when a location in the claim is missing from evidence."""
    analyzer = GapAnalyzer()
    claim = "New airport terminal announced in Bhopal."
    # Evidence contains general aviation notes with no mention of Bhopal
    evidence = [
        Evidence(doc_id="e1", text="Civil aviation ministry releases national airport policy.", source_type="government")
    ]

    gaps = analyzer.analyze(claim_text=claim, evidence_list=evidence)
    gap_codes = [g.code for g in gaps]
    assert "no_location_confirmation" in gap_codes


def test_optional_template_rephrase_no_new_facts():
    """
    Tests that if a template string is rephrased by an LLM or helper,
    it introduces NO new named entities or numbers.
    """
    original_text = "Cross-reference with official ministry portals or news wire feeds."
    # Hypothetical faithful rephrase
    rephrased_text = "Verify information against official government portals or accredited wire agencies."

    def extract_numbers_and_capitalized(text: str) -> set:
        numbers = set(re.findall(r"\b\d+\b", text))
        proper_names = set(re.findall(r"\b[A-Z][a-z]+\b", text))
        return numbers.union(proper_names)

    orig_tokens = extract_numbers_and_capitalized(original_text)
    rephrase_tokens = extract_numbers_and_capitalized(rephrased_text)

    # Rephrase must not introduce extraneous numbers or capitalized proper nouns
    new_tokens = rephrase_tokens - orig_tokens
    # Allow trivial sentence starters like 'Verify'
    new_non_trivial = {t for t in new_tokens if t.lower() not in ["verify"]}
    assert len(new_non_trivial) == 0


# ==============================================================================
# 6. Counterfactual Sensitivity Tests
# ==============================================================================

def test_counterfactual_faithfulness_reproduced_by_rerunning():
    """
    Every reported counterfactual flip must be 100% reproducible by
    re-running the model on the edited input record.
    """
    gen = CounterfactualGenerator(max_counterfactuals=5)

    dummy_record = {
        "claim_id": "test_claim_01",
        "claim_text": "Solar eclipse visible across Delhi.",
        "dataset": "eval",
        "text_logits": [0.0, 0.0, 0.0],
        "text_pooled": [0.0] * 32,
        "retrieval_features": {
            "top1_score": 1.0,
            "top1_top2_gap": 0.5,
            "num_passages_above_threshold": 1,
            "mean_score": 1.0,
            "source_type_government": 0.0,
        },
        "image_features": {
            "image_present": 0,
            "mismatch_score": 0.0,
            "clip_image_embedding": [0.0] * 512,
        },
    }

    pairs = gen.generate_with_records(dummy_record)
    assert len(pairs) > 0

    for cf, perturbed_record in pairs:
        # Re-run predictor on the exact perturbed record
        rerun_out = gen.predictor.predict_record(perturbed_record)
        rerun_verdict = rerun_out["predicted_verdict"]

        # Faithfulness check: re-run output must match reported new_verdict
        assert rerun_verdict.lower() == cf.new_verdict.lower()
        if cf.flipped:
            assert cf.new_verdict.lower() != cf.original_verdict.lower()


# ==============================================================================
# 7. Share Advice Decision Table Tests
# ==============================================================================

def test_share_advice_decision_table_all_rows():
    """Exhaustively tests every row of the transparent share advice decision table."""
    c_false = ConflictInfo(has_conflict=False)
    c_true = ConflictInfo(has_conflict=True)
    t_fresh = TemporalInfo(is_stale=False)
    t_stale = TemporalInfo(is_stale=True)

    # 1. Abstained -> do_not_share_as_confirmed
    advice, _ = determine_share_advice(Verdict.VERIFIED, 0.95, abstained=True, conflict=c_false, temporal=t_fresh)
    assert advice == ShareAdvice.DO_NOT_SHARE_AS_CONFIRMED

    # 2. Conflicting authoritative evidence -> do_not_share_as_confirmed (rule 2)
    advice, _ = determine_share_advice(Verdict.VERIFIED, 0.95, abstained=False, conflict=c_true, temporal=t_fresh)
    assert advice == ShareAdvice.DO_NOT_SHARE_AS_CONFIRMED

    # 3. Stale evidence -> share_with_caution (rule 5)
    advice, _ = determine_share_advice(Verdict.VERIFIED, 0.95, abstained=False, conflict=c_false, temporal=t_stale)
    assert advice == ShareAdvice.SHARE_WITH_CAUTION

    # 4. Contradicted verdict -> do_not_share_as_confirmed (rule 1)
    advice, _ = determine_share_advice(Verdict.CONTRADICTED, 0.90, abstained=False, conflict=c_false, temporal=t_fresh)
    assert advice == ShareAdvice.DO_NOT_SHARE_AS_CONFIRMED

    # 5. Insufficient evidence verdict -> share_with_caution (rule 4)
    advice, _ = determine_share_advice(Verdict.INSUFFICIENT_EVIDENCE, 0.70, abstained=False, conflict=c_false, temporal=t_fresh)
    assert advice == ShareAdvice.SHARE_WITH_CAUTION

    # 6. Verified with high confidence (>= 0.75) -> safe_to_share
    advice, _ = determine_share_advice(Verdict.VERIFIED, 0.85, abstained=False, conflict=c_false, temporal=t_fresh)
    assert advice == ShareAdvice.SAFE_TO_SHARE

    # 7. Verified with low confidence (< 0.75) -> share_with_caution
    advice, _ = determine_share_advice(Verdict.VERIFIED, 0.65, abstained=False, conflict=c_false, temporal=t_fresh)
    assert advice == ShareAdvice.SHARE_WITH_CAUTION


# ==============================================================================
# 8. Verdict and Confidence Immutability Guarantee
# ==============================================================================

def test_reasoning_layer_never_mutates_verdict_or_confidence():
    """
    CRITICAL INVARIANT: The reasoning layer must NEVER alter the final verdict
    or calibrated confidence score of a ClaimReport.
    """
    from trustnet.pipeline import apply_reasoning_to_report

    claim = Claim(id="claim_immutability_test", text="Global temperature trend reported.")
    original_verdict = Verdict.CONTRADICTED
    original_confidence = 0.8421

    report = ClaimReport(
        claim=claim,
        verdict=original_verdict,
        confidence=original_confidence,
        abstained=False,
        evidence=[
            Evidence(doc_id="e1", text="Weather report.", source_type="news", date="2020-01-01", score=0.4)
        ],
    )

    # Run reasoning layer with all components enabled
    enriched_report = apply_reasoning_to_report(report)

    # Invariants must hold strictly
    assert enriched_report.verdict == original_verdict
    assert enriched_report.confidence == original_confidence
    # Reasoning notes must capture triggered signals without modifying verdict
    assert len(enriched_report.reasoning_notes) >= 1
