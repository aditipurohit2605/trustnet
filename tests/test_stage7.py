"""
Unit and integration test suite for TrustNet Stage 7: Calibration, Uncertainty, and Abstention.

Tests:
1. Metric correctness on known synthetic examples (perfectly calibrated, always overconfident).
2. Metric comparison against netcal and scikit-learn within strict numerical tolerance.
3. Temperature scaling: learned T > 0 and improves NLL on overconfident logits.
4. Abstention: threshold is fit strictly from input data, correctly decides 'abstain',
   and compares max_prob vs normalized_entropy.
5. Anti-leakage guard: test split evaluation strictly blocked without '--final'.
6. Pipeline: ClaimReport preserves original claim text, sets abstained: bool, and provides probabilities.
7. CPU smoke test: runs full evaluation and OOD scripts on CPU.
"""

from pathlib import Path
import tempfile
import numpy as np
import pytest
from sklearn.metrics import log_loss
from netcal.metrics import ECE as NetcalECE

from trustnet.calibration.abstain import Abstainer, compare_confidence_metrics
from trustnet.calibration.conformal import SplitConformalPredictor
from trustnet.calibration.evaluate import run_evaluation
from trustnet.calibration.metrics import (
    adaptive_expected_calibration_error,
    classwise_expected_calibration_error,
    compute_aurc,
    compute_brier_score,
    compute_confidently_wrong_rate,
    compute_coverage_at_fixed_error_rate,
    compute_nll,
    compute_reliability_diagram_data,
    compute_risk_coverage_curve,
    compute_selective_accuracy,
    expected_calibration_error,
)
from trustnet.calibration.ood_eval import run_ood_evaluation
from trustnet.calibration.temperature import TemperatureScaler
from trustnet.fusion.predict import MultimodalPredictor
from trustnet.schemas import Claim, ClaimReport, Evidence, Verdict


# =========================================================================
# 1. Metric Correctness Tests on Synthetic Known Examples
# =========================================================================

def test_metrics_perfectly_calibrated():
    """
    Constructs a synthetic dataset where predictions are perfectly calibrated.
    Predicted confidence exactly matches empirical accuracy in each bin -> ECE should be 0.0.
    """
    # 10 samples with conf=0.8, 8 are correct
    probs_bin1 = np.array([[0.8, 0.2]] * 10)
    labels_bin1 = np.array([0] * 8 + [1] * 2)

    # 10 samples with conf=0.6, 6 are correct
    probs_bin2 = np.array([[0.6, 0.4]] * 10)
    labels_bin2 = np.array([0] * 6 + [1] * 4)

    probs = np.vstack([probs_bin1, probs_bin2])
    labels = np.concatenate([labels_bin1, labels_bin2])

    ece = expected_calibration_error(probs, labels, n_bins=10)
    assert ece == pytest.approx(0.0, abs=1e-5), f"Expected 0.0 ECE for perfectly calibrated data, got {ece}"


def test_metrics_always_overconfident():
    """
    Model predicts class 0 with 100% confidence for all samples, but accuracy is only 50%.
    ECE should be exactly |0.5 - 1.0| = 0.5.
    """
    probs = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
    ])
    labels = np.array([0, 0, 1, 1])  # 2 correct, 2 wrong -> acc = 0.5

    ece = expected_calibration_error(probs, labels, n_bins=5)
    assert ece == pytest.approx(0.5, abs=1e-5), f"Expected ECE 0.5 for overconfident model, got {ece}"


def test_metrics_comparison_with_netcal_and_sklearn():
    """
    Compares our scratch NumPy ECE and NLL against netcal and scikit-learn.
    """
    np.random.seed(42)
    # 3-class probability distribution for 20 samples
    logits = np.random.randn(20, 3)
    exp_l = np.exp(logits)
    probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)
    labels = np.random.choice([0, 1, 2], size=20)

    # NLL vs sklearn log_loss
    scratch_nll = compute_nll(probs, labels)
    sk_nll = log_loss(labels, probs, labels=[0, 1, 2])
    assert scratch_nll == pytest.approx(sk_nll, rel=1e-4), f"NLL mismatch: {scratch_nll} vs {sk_nll}"

    # ECE vs netcal ECE
    conf = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    correct = (preds == labels).astype(int)

    scratch_ece = expected_calibration_error(probs, labels, n_bins=5)
    net_ece = float(NetcalECE(bins=5).measure(conf, correct))
    assert scratch_ece == pytest.approx(net_ece, abs=1e-4), f"ECE mismatch: {scratch_ece} vs {net_ece}"


def test_metrics_risk_coverage_and_brier():
    """Tests Brier score, AURC, and selective accuracy computations."""
    probs = np.array([
        [0.9, 0.1],
        [0.8, 0.2],
        [0.7, 0.3],
        [0.4, 0.6],
    ])
    labels = np.array([0, 0, 0, 0])  # Last sample predicted 1, true is 0

    brier = compute_brier_score(probs, labels)
    assert 0.0 <= brier <= 2.0

    covs, risks = compute_risk_coverage_curve(probs, labels)
    aurc = compute_aurc(covs, risks)
    assert 0.0 <= aurc <= 1.0

    # At 75% coverage (top 3 samples), all are correct -> selective acc = 1.0
    sel_acc = compute_selective_accuracy(probs, labels, target_coverage=0.75)
    assert sel_acc == 1.0

    # Confidently wrong rate
    conf_wrong = compute_confidently_wrong_rate(probs, labels, cutoff=0.5)
    assert conf_wrong == 0.25  # 1 out of 4 samples (conf=0.6, wrong)


# =========================================================================
# 2. Temperature Scaling Tests
# =========================================================================

def test_temperature_scaling_improves_nll_on_overconfident():
    """
    Tests that temperature scaling finds T > 1 and strictly improves NLL on overconfident logits.
    """
    # Severely overconfident logits
    logits = np.array([
        [15.0, 1.0, 1.0],
        [1.0, 15.0, 1.0],
        [15.0, 1.0, 1.0],
        [1.0, 15.0, 1.0],
    ])
    labels = np.array([0, 0, 1, 1])  # 50% error rate despite large logit margins

    exp_l = np.exp(logits)
    raw_probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)
    raw_nll = compute_nll(raw_probs, labels)

    scaler = TemperatureScaler()
    scaler.fit(logits, labels)

    assert scaler.global_temperature > 1.0, f"Expected T > 1.0, got {scaler.global_temperature}"

    cal_probs = scaler.predict_proba(logits)
    cal_nll = compute_nll(cal_probs, labels)
    assert cal_nll < raw_nll, f"Expected calibrated NLL ({cal_nll}) < raw NLL ({raw_nll})"

    # Test serialization
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        scaler.save(tmp_path)
        loaded = TemperatureScaler.load(tmp_path)
        assert loaded.global_temperature == pytest.approx(scaler.global_temperature, rel=1e-5)
    finally:
        tmp_path.unlink(missing_ok=True)


# =========================================================================
# 3. Abstention Tests
# =========================================================================

def test_abstention_threshold_and_decisions():
    """
    Tests that Abstainer fits threshold strictly from input data and makes correct decisions.
    """
    probs = np.array([
        [0.95, 0.05],  # Correct
        [0.90, 0.10],  # Correct
        [0.65, 0.35],  # Wrong
        [0.60, 0.40],  # Wrong
    ])
    labels = np.array([0, 0, 1, 1])

    # For target error rate 0.0, it should only answer top 2 samples (conf >= 0.90)
    abstainer = Abstainer(target_error_rate=0.0, confidence_metric="max_prob")
    abstainer.fit(probs, labels)

    assert abstainer.threshold >= 0.90, f"Expected threshold >= 0.90, got {abstainer.threshold}"
    assert abstainer.val_cal_coverage == pytest.approx(0.5, abs=1e-4)

    # Test decisions
    test_probs = np.array([
        [0.92, 0.08],  # Should answer
        [0.55, 0.45],  # Should abstain
    ])
    decisions = abstainer.decide(test_probs)
    assert decisions[0] == "verified"
    assert decisions[1] == "abstain"

    # Compare confidence metrics
    comp = compare_confidence_metrics(probs, labels, target_error_rate=0.10)
    assert "max_prob" in comp
    assert "normalized_entropy" in comp


# =========================================================================
# 4. Anti-Leakage Guard Tests
# =========================================================================

def test_anti_leakage_guard_blocks_test_without_final():
    """
    Verifies that running evaluate on the test split without --final raises PermissionError.
    """
    with pytest.raises(PermissionError) as exc_info:
        run_evaluation(split="test", final=False)
    assert "--final" in str(exc_info.value)


# =========================================================================
# 5. Pipeline and ClaimReport Preservation Tests
# =========================================================================

def test_pipeline_preserves_claim_and_sets_abstention():
    """
    Tests that ClaimReport returned by MultimodalPredictor preserves original claim text,
    correctly sets abstained: bool, and provides probability distributions.
    """
    orig_claim_text = "Specific historical assertion that requires verification."
    claim = Claim(id="test_claim_001", text=orig_claim_text)
    evidence = [Evidence(doc_id="doc_1", text="Evidence snippet", score=0.8)]

    # Configure an abstainer with an impossible threshold (forces abstention)
    abstainer = Abstainer(target_error_rate=0.0)
    abstainer.threshold = 0.99
    abstainer.fitted = True

    predictor = MultimodalPredictor(abstainer=abstainer)
    report = predictor.predict_claim_report(claim=claim, evidence=evidence)

    assert isinstance(report, ClaimReport)
    assert report.claim.text == orig_claim_text, "Original claim text was modified!"
    assert report.claim.id == "test_claim_001"
    assert report.abstained is True, "Expected abstained == True"
    assert report.raw_probabilities is not None
    assert report.calibrated_probabilities is not None
    assert "verified" in report.calibrated_probabilities
    assert report.abstention_reason is not None


# =========================================================================
# 6. Conformal Prediction Set Test
# =========================================================================

def test_split_conformal_predictor():
    """Tests SplitConformalPredictor coverage and set generation."""
    probs = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.7, 0.2],
        [0.2, 0.2, 0.6],
    ])
    labels = np.array([0, 1, 2])

    conformal = SplitConformalPredictor(alpha=0.2)
    conformal.fit(probs, labels)
    eval_res = conformal.evaluate(probs, labels)

    assert eval_res["empirical_coverage"] >= 0.8
    assert eval_res["avg_set_size"] >= 1.0


# =========================================================================
# 7. CPU Smoke Test: Full Evaluation and OOD Evaluation
# =========================================================================

def test_cpu_smoke_test_val_cal_evaluation():
    """
    Executes Stage 7 evaluation on val_cal split and OOD evaluation on CPU,
    verifying artifact creation and valid JSON output.
    """
    out_dir = Path("outputs/stage7")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Run val_cal evaluation
    results = run_evaluation(
        config_path="configs/stage7_uncertainty.yaml",
        stage6_config_path="configs/stage6_fusion.yaml",
        cache_dir="outputs/stage6/cache",
        output_dir="outputs/stage7",
        split="val_cal",
        final=False,
    )

    assert (out_dir / "results.json").exists()
    assert (out_dir / "results.md").exists()
    assert (out_dir / "temperature.json").exists()
    assert (out_dir / "abstainer.json").exists()

    # Verify results content
    assert "uncalibrated" in results["results"]
    assert "temperature" in results["results"]
    assert "ensemble" in results["results"]

    # 2. Run OOD evaluation
    ood_results = run_ood_evaluation(
        ood_file="data/eval/indian_demo_set.jsonl",
        config_path="configs/stage7_uncertainty.yaml",
        temperature_path="outputs/stage7/temperature.json",
        abstainer_path="outputs/stage7/abstainer.json",
        output_dir="outputs/stage7",
    )

    assert (out_dir / "ood_results.json").exists()
    assert (out_dir / "ood_results.md").exists()
    op_plots = list((out_dir / "plots").glob("*.png"))
    assert len(op_plots) >= 4, f"Expected at least 4 plots, got {len(op_plots)}"
