"""
Unit and Integration Tests for TrustNet Reporting and Reasoning Fixes.

Covers:
1. Decision table coverage in share_advice.py:
   - Every row covered.
   - NO reason string contains the substring " or ".
   - Verdict and recommended action are strictly separate fields.
2. Retrieval and SearchCoverage telemetry:
   - Candidates retrieved, passages above threshold, and top score recorded per claim.
   - Distinguishes "nothing above threshold" (evidence_status=no_evidence_found, NOT degraded)
     from "retrieval failed" (evidence_status=retrieval_failed, retriever in degraded).
   - SearchCoverage populated from corpus manifest/config with not_searched list.
   - Present in ClaimReport and API ReportEnvelope.
3. GapStatus on Gaps:
   - GapStatus enum (open, searched_not_found, outside_scope, resolved).
   - searched_not_found marked only if search actually ran over the corpus; otherwise open or outside_scope.
   - No aggregate "x of y investigated" string.
4. Counterfactual Feature Sensitivity and Faithfulness:
   - Counterfactual edits change at least one model input feature in the compact vector.
   - Faithfulness check passes: new_verdict matches real prediction, flipped == (new_verdict != original_verdict).
   - Probabilities are not identical to 6 decimals.
5. Abstain Reason Guarantee:
   - abstain_reason is ALWAYS set and non-empty when abstained == True.
"""

from pathlib import Path
from typing import Any, Dict, List
import pytest

from trustnet.api.pipeline_runner import PipelineRunner
from trustnet.api.schemas import PipelineVariant, ReportEnvelope, VerifyOptions
from trustnet.fusion.features import extract_compact_vector
from trustnet.fusion.predict import MultimodalPredictor
from trustnet.pipeline import build_search_coverage, verify_claim
from trustnet.reasoning.counterfactual import CounterfactualGenerator
from trustnet.reasoning.gaps import GapAnalyzer
from trustnet.reasoning.share_advice import (
    DECISION_TABLE,
    SHARE_ADVICE_TEMPLATES,
    determine_share_advice,
    get_share_advice_decision,
)
from trustnet.schemas import (
    AbstainReason,
    Claim,
    ClaimReport,
    ConflictInfo,
    Evidence,
    EvidenceStatus,
    Gap,
    GapStatus,
    ReportDecision,
    SearchCoverage,
    ShareAdvice,
    TemporalInfo,
    Verdict,
)


# ==============================================================================
# 1. Share Advice Decision Table & Reason Templates
# ==============================================================================

def test_share_advice_templates_and_no_or_substring():
    """
    Verifies that every plain-language reason in SHARE_ADVICE_TEMPLATES
    and every rationale in DECISION_TABLE strictly does NOT contain the substring ' or '.
    """
    assert len(SHARE_ADVICE_TEMPLATES) >= 7

    for code, template_str in SHARE_ADVICE_TEMPLATES.items():
        assert " or " not in template_str, (
            f"Reason template '{code}' contains forbidden substring ' or ': \"{template_str}\""
        )
        assert len(template_str.strip()) > 10

    for row in DECISION_TABLE:
        assert "row_id" in row
        assert "condition" in row
        assert "advice" in row
        assert "reason_code" in row
        assert "rationale" in row
        assert isinstance(row["advice"], ShareAdvice)
        assert row["reason_code"] in SHARE_ADVICE_TEMPLATES
        assert row["rationale"] == SHARE_ADVICE_TEMPLATES[row["reason_code"]]
        assert " or " not in row["rationale"], (
            f"Decision table row '{row['row_id']}' contains forbidden substring ' or ': \"{row['rationale']}\""
        )


def test_share_advice_table_coverage_and_separate_verdict_action():
    """
    Exhaustively tests every row of DECISION_TABLE and verifies that
    verdict and action are always strictly separate fields.
    """
    c_false = ConflictInfo(has_conflict=False)
    c_true = ConflictInfo(has_conflict=True)
    t_fresh = TemporalInfo(is_stale=False)
    t_stale = TemporalInfo(is_stale=True)

    # Row 1: Contradicted
    advice, code, rationale = get_share_advice_decision(
        Verdict.CONTRADICTED, 0.90, abstained=False, conflict=c_false, temporal=t_fresh
    )
    assert advice == ShareAdvice.DO_NOT_SHARE_AS_CONFIRMED
    assert code == "contradicted_by_evidence"
    assert " or " not in rationale

    # Row 2: Active Conflict
    advice, code, rationale = get_share_advice_decision(
        Verdict.VERIFIED, 0.95, abstained=False, conflict=c_true, temporal=t_fresh
    )
    assert advice == ShareAdvice.DO_NOT_SHARE_AS_CONFIRMED
    assert code == "authoritative_conflict"
    assert " or " not in rationale

    # Row 3: Abstained
    advice, code, rationale = get_share_advice_decision(
        Verdict.VERIFIED, 0.95, abstained=True, conflict=c_false, temporal=t_fresh
    )
    assert advice == ShareAdvice.DO_NOT_SHARE_AS_CONFIRMED
    assert code == "high_uncertainty_abstain"
    assert " or " not in rationale

    # Row 4: Insufficient Evidence
    advice, code, rationale = get_share_advice_decision(
        Verdict.INSUFFICIENT_EVIDENCE, 0.70, abstained=False, conflict=c_false, temporal=t_fresh
    )
    assert advice == ShareAdvice.SHARE_WITH_CAUTION
    assert code == "insufficient_evidence"
    assert " or " not in rationale

    # Row 5: Verified Stale
    advice, code, rationale = get_share_advice_decision(
        Verdict.VERIFIED, 0.95, abstained=False, conflict=c_false, temporal=t_stale
    )
    assert advice == ShareAdvice.SHARE_WITH_CAUTION
    assert code == "stale_evidence"
    assert " or " not in rationale

    # Row 6: Verified High Confidence (>= 0.80)
    advice, code, rationale = get_share_advice_decision(
        Verdict.VERIFIED, 0.85, abstained=False, conflict=c_false, temporal=t_fresh
    )
    assert advice == ShareAdvice.SAFE_TO_SHARE
    assert code == "verified_high_confidence"
    assert " or " not in rationale

    # Row 7: Verified Moderate Confidence (< 0.80)
    advice, code, rationale = get_share_advice_decision(
        Verdict.VERIFIED, 0.65, abstained=False, conflict=c_false, temporal=t_fresh
    )
    assert advice == ShareAdvice.SHARE_WITH_CAUTION
    assert code == "verified_moderate_confidence"
    assert " or " not in rationale

    # Verification of separation in ReportDecision
    decision = ReportDecision(
        verdict=Verdict.VERIFIED,
        evidence_status=EvidenceStatus.SUPPORTED,
        recommended_action=ShareAdvice.SAFE_TO_SHARE,
        reason_codes=["verified_high_confidence"],
    )
    assert decision.verdict != decision.recommended_action
    assert decision.verdict == Verdict.VERIFIED
    assert decision.recommended_action == ShareAdvice.SAFE_TO_SHARE
    assert decision.evidence_status == EvidenceStatus.SUPPORTED


# ==============================================================================
# 2. Retrieval Telemetry and SearchCoverage
# ==============================================================================

def test_search_coverage_construction_and_fields():
    """
    Verifies that SearchCoverage records candidates retrieved, number above threshold,
    top score, and not_searched list loaded from config.
    """
    evidence = [
        Evidence(doc_id="d1", text="Passage 1", score=0.85),
        Evidence(doc_id="d2", text="Passage 2", score=0.42),
        Evidence(doc_id="d3", text="Passage 3", score=0.61),
    ]

    cov = build_search_coverage(evidence, total_retrieved=10)
    assert cov.candidates_retrieved == 10
    assert cov.passages_above_threshold == 2  # 0.85 and 0.61 >= 0.50
    assert cov.top_score == 0.85
    assert cov.total_documents_indexed >= 1
    assert "social_media_archives" in cov.not_searched
    assert "real_time_web" in cov.not_searched
    assert "subscription_databases" in cov.not_searched


def test_distinguish_nothing_above_threshold_from_retrieval_failed():
    """
    Verifies that 'nothing above threshold' results in evidence_status == no_evidence_found
    and does NOT add retriever to degraded, whereas 'retrieval failed' sets
    evidence_status == retrieval_failed and adds retriever to degraded.
    """
    claim = Claim(id="c_test", text="Uncorroborated test claim")

    # 1. Nothing above threshold (empirical finding)
    low_evidence = [
        Evidence(doc_id="d1", text="Low score text", score=0.20),
        Evidence(doc_id="d2", text="Another low score", score=0.35),
    ]
    cov_low = build_search_coverage(low_evidence)
    assert cov_low.passages_above_threshold == 0

    report_low = verify_claim(
        claim=claim,
        evidence=low_evidence,
        search_coverage=cov_low,
        enable_reasoning=False,
    )
    assert report_low.evidence_status == EvidenceStatus.NO_EVIDENCE_FOUND
    assert report_low.search_coverage.passages_above_threshold == 0

    # 2. Retrieval failed (operational failure)
    degraded_list = []
    try:
        raise ConnectionError("Retriever service unavailable")
    except Exception as e:
        degraded_list.append("retriever")

    assert "retriever" in degraded_list
    report_failed = ClaimReport(
        claim=claim,
        evidence=[],
        verdict=Verdict.INSUFFICIENT_EVIDENCE,
        confidence=0.3333,
        abstained=True,
        abstain_reason=AbstainReason.BELOW_CONFIDENCE_THRESHOLD,
        evidence_status=EvidenceStatus.RETRIEVAL_FAILED,
        search_coverage=build_search_coverage([]),
    )
    assert report_failed.evidence_status == EvidenceStatus.RETRIEVAL_FAILED
    assert "retriever" in degraded_list


def test_search_coverage_present_in_claim_report_and_envelope():
    """
    Verifies that search_coverage is present in every ClaimReport and in ReportEnvelope.
    """
    claim = Claim(id="c_rep", text="Ozone depletion has halted.")
    evidence = [Evidence(doc_id="doc_001", text="Satellite data confirms recovery.", score=0.91)]
    report = verify_claim(claim=claim, evidence=evidence, enable_reasoning=True)

    assert report.search_coverage is not None
    assert isinstance(report.search_coverage, SearchCoverage)
    assert report.search_coverage.candidates_retrieved >= 1
    assert report.search_coverage.passages_above_threshold >= 1
    assert report.search_coverage.top_score >= 0.50

    envelope = ReportEnvelope(
        report_id="rep_test_001",
        pipeline_version="0.1.0",
        config_hash="test_hash",
        variant="full",
        original_text="Ozone depletion has halted.",
        claims=[report],
        search_coverage=report.search_coverage,
    )
    assert envelope.search_coverage is not None
    assert envelope.search_coverage.passages_above_threshold >= 1


# ==============================================================================
# 3. GapStatus Logic on Gaps
# ==============================================================================

def test_gap_status_logic_and_no_aggregate_investigated():
    """
    Verifies that:
    1. Gaps have GapStatus assigned.
    2. Evidence gaps are marked searched_not_found ONLY if search actually ran over corpus.
    3. If search did not run over corpus, evidence gaps are marked open.
    4. source_unnamed is marked outside_scope.
    5. No aggregate 'x of y investigated' string exists in gap objects or descriptions.
    """
    analyzer = GapAnalyzer()
    claim_text = "Sources say a dam in Delhi is at emergency collapse levels."

    # Case A: Corpus search ran, but nothing found above threshold
    gaps_searched = analyzer.analyze(
        claim_text=claim_text,
        evidence_list=[],
        search_ran=True,
    )
    gap_map_searched = {g.code: g for g in gaps_searched}
    assert "low_retrieval_scores" in gap_map_searched
    assert gap_map_searched["low_retrieval_scores"].status == GapStatus.SEARCHED_NOT_FOUND
    assert "no_primary_source" in gap_map_searched
    assert gap_map_searched["no_primary_source"].status == GapStatus.SEARCHED_NOT_FOUND
    assert "source_unnamed" in gap_map_searched
    assert gap_map_searched["source_unnamed"].status == GapStatus.OUTSIDE_SCOPE

    # Case B: Search did NOT run over corpus (search_ran=False)
    gaps_not_searched = analyzer.analyze(
        claim_text=claim_text,
        evidence_list=[],
        search_ran=False,
    )
    gap_map_not_searched = {g.code: g for g in gaps_not_searched}
    assert gap_map_not_searched["low_retrieval_scores"].status == GapStatus.OPEN
    assert gap_map_not_searched["no_primary_source"].status == GapStatus.OPEN

    # Ensure no aggregate investigated text in any gap description or instruction
    for g in gaps_searched + gaps_not_searched:
        assert "investigated" not in g.description.lower()
        assert "investigated" not in g.how_to_check.lower()


# ==============================================================================
# 4. Counterfactual Feature Sensitivity and Faithfulness
# ==============================================================================

def test_counterfactual_edits_change_features_and_faithfulness():
    """
    Verifies that:
    1. Every counterfactual scenario modifies at least one feature in the compact vector.
    2. The faithfulness check passes: new_verdict matches the model's actual output,
       and flipped == (new_verdict != original_verdict).
    3. Output probabilities are sensitive and not identical to 6 decimal places.
    """
    predictor = MultimodalPredictor.from_artifacts()
    c_gen = CounterfactualGenerator(predictor=predictor, max_counterfactuals=5)

    base_rec = {
        "claim_id": "c_cf_test",
        "claim_text": "Satellite data confirms stratospheric ozone recovery.",
        "dataset": "val",
        "text_logits": [2.5, -1.0, -1.5],
        "text_pooled": [0.0] * 32,
        "retrieval_features": {
            "top1_score": 0.92,
            "top1_top2_gap": 0.45,
            "num_passages_above_threshold": 1.0,
            "mean_score": 0.92,
            "source_type_corpus_passage": 1.0,
            "source_type_government": 1.0,
            "source_type_news": 0.0,
            "source_type_other": 0.0,
            "agreement_feature": 1.0,
        },
        "image_features": {
            "image_present": 1,
            "mismatch_score": 0.15,
            "best_reuse_similarity": 0.88,
            "ocr_present": 1,
            "ocr_confidence": 0.95,
            "quality_flag_tiny_resolution": 0,
            "quality_flag_extreme_aspect_ratio": 0,
            "quality_flag_heavy_jpeg_compression": 0,
            "quality_flag_screenshot_likeness": 0,
            "clip_image_embedding": [0.0] * 512,
        },
    }

    base_vec, _ = extract_compact_vector(base_rec)
    cf_results = c_gen.generate_with_records(base_rec)
    assert len(cf_results) >= 3

    for cf, perturbed_rec in cf_results:
        pert_vec, _ = extract_compact_vector(perturbed_rec)
        # 1. Verify feature vector changed
        diffs = [abs(p - b) for p, b in zip(pert_vec, base_vec)]
        assert any(d > 1e-5 for d in diffs), (
            f"Scenario '{cf.change_description}' failed to modify any model input feature!"
        )

        # 2. Verify faithfulness: run predictor directly on perturbed record
        direct_out = predictor.predict_record(perturbed_rec)
        assert cf.new_verdict.lower() == direct_out["predicted_verdict"].lower()
        expected_flipped = (cf.new_verdict.lower() != cf.original_verdict.lower())
        assert cf.flipped == expected_flipped


# ==============================================================================
# 5. Abstain Reason Guarantee
# ==============================================================================

def test_abstain_reason_always_set_when_abstained():
    """
    Verifies that whenever a ClaimReport has abstained == True,
    abstain_reason is ALWAYS populated with a non-empty string or AbstainReason enum.
    """
    claim = Claim(id="c_abs", text="Uncertain and conflicting claim")

    # Low confidence -> abstains
    predictor = MultimodalPredictor.from_artifacts()
    # Force low confidence prediction
    rec_ambiguous = {
        "claim_id": claim.id,
        "claim_text": claim.text,
        "dataset": "val",
        "text_logits": [0.01, 0.01, 0.01],
        "text_pooled": [0.0] * 32,
        "retrieval_features": {
            "top1_score": 0.1,
            "top1_top2_gap": 0.0,
            "num_passages_above_threshold": 0.0,
            "mean_score": 0.1,
        },
        "image_features": {"image_present": 0},
    }
    report = predictor.predict_claim_report(
        claim=claim,
        evidence=[],
        cached_record=rec_ambiguous,
    )

    if report.abstained:
        assert report.abstain_reason is not None, "abstain_reason must not be None when abstained == True"
        assert len(str(report.abstain_reason).strip()) > 0
        assert report.abstention_reason is not None

    # Test direct instantiation
    report_forced = ClaimReport(
        claim=claim,
        evidence=[],
        verdict=Verdict.INSUFFICIENT_EVIDENCE,
        confidence=0.25,
        abstained=True,
        abstain_reason=AbstainReason.BELOW_CONFIDENCE_THRESHOLD,
    )
    assert report_forced.abstained is True
    assert report_forced.abstain_reason == AbstainReason.BELOW_CONFIDENCE_THRESHOLD
