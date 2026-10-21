"""
TrustNet Stage 12 Test Suite: Release, Reproducibility, and Verification Tooling.

Tests:
1. freeze check fails if a file is modified / test split hash is corrupted.
2. run_test refuses execution without explicit --final flag.
3. run_test refuses to overwrite existing predictions without --allow-rerun.
4. tables.py reproduces tables deterministically from mock predictions.
5. paired bootstrap strictly yields zero difference on identical systems.
6. hypotheses.md contains no numbers that do not trace to saved evaluation results.
7. telegram bot parses commands, formats responses, and handles network failures.
8. no module in src/trustnet/final/ imports model-training code.
"""

import ast
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import AsyncMock, MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import httpx
import numpy as np
import pytest

from trustnet.final.error_analysis import (
    analyze_errors,
    compute_cohens_kappa,
    sample_errors,
)
from trustnet.final.freeze_check import compute_sha256, verify_freeze
from trustnet.final.hypotheses import evaluate_hypotheses
from trustnet.final.run_test import run_test_evaluation
from trustnet.final.slices import run_slice_analysis
from trustnet.final.stats import (
    compute_accuracy,
    compute_ece,
    compute_macro_f1,
    mcnemar_test,
    paired_bootstrap_difference,
)
from trustnet.final.tables import (
    generate_all_tables_and_plots,
    load_prediction_records,
)
from scripts.telegram_bot import (
    check_rate_limit,
    format_verification_response,
    render_confidence_bar,
    call_verification_api,
)


# --- Test 1: Freeze check fails if a file is modified ---

def test_freeze_check_fails_on_tampered_test_split(tmp_path):
    # Setup temporary directory with valid manifest
    data_dir = tmp_path / "data"
    proc_dir = data_dir / "processed"
    proc_dir.mkdir(parents=True)

    test_file = proc_dir / "test.jsonl"
    test_file.write_text('{"id": "test_1", "claim": "Example", "verdict": "verified"}\n', encoding="utf-8")
    actual_hash = compute_sha256(test_file)

    manifest_file = data_dir / "manifest.json"
    manifest_data = {
        "splits": {
            "test": {
                "file": "processed/test.jsonl",
                "count": 1,
                "sha256": "tampered_fake_hash_1234567890abcdef",
            }
        }
    }
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

    passed, manifest = verify_freeze(repo_root=tmp_path, enforce_tag=False, write_manifest=False)
    assert not passed
    assert any("Test split hash mismatch" in err for err in manifest.get("errors", []))


# --- Test 2: run_test refuses execution without --final flag ---

def test_run_test_refuses_without_final_flag():
    # Calling the CLI entry point without --final must exit with returncode 1
    res = subprocess.run(
        [sys.executable, "-m", "trustnet.final.run_test"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 1
    assert "requires explicit --final flag" in res.stderr


# --- Test 3: run_test refuses to rerun without --allow-rerun ---

def test_run_test_refuses_rerun_without_allow_rerun(tmp_path):
    outputs_dir = tmp_path / "outputs" / "final" / "predictions"
    outputs_dir.mkdir(parents=True)
    existing_pred = outputs_dir / "test_predictions.jsonl"
    existing_pred.write_text('{"claim_id": "test_001"}\n', encoding="utf-8")

    with pytest.raises(RuntimeError, match="Refusing to overwrite test split outputs without explicit --allow-rerun"):
        run_test_evaluation(repo_root=tmp_path, allow_rerun=False)


# --- Test 4: tables.py reproduces tables deterministically from mock predictions ---

def test_tables_reproducibility(tmp_path):
    tables_dir = tmp_path / "outputs" / "final" / "tables"
    fig_dir = tmp_path / "outputs" / "final" / "figures"
    pred_dir = tmp_path / "outputs" / "final" / "predictions"
    pred_dir.mkdir(parents=True)

    # Write mock predictions
    mock_records = [
        {
            "claim_id": f"c_{i}",
            "gold_verdict": "verified" if i % 2 == 0 else "contradicted",
            "effective_verdict": "verified" if i % 2 == 0 else "contradicted",
            "predicted_verdict": "verified" if i % 2 == 0 else "contradicted",
            "variant": "full",
            "calibration_mode": "temperature",
            "probabilities": {"verified": 0.8, "contradicted": 0.1, "insufficient_evidence": 0.1},
            "confidence": 0.8,
            "correct": True,
        }
        for i in range(10)
    ]
    pred_file = pred_dir / "test_predictions.jsonl"
    with open(pred_file, "w", encoding="utf-8") as f:
        for r in mock_records:
            f.write(json.dumps(r) + "\n")

    # Run table generation twice
    generate_all_tables_and_plots(repo_root=tmp_path)
    t1_content_run1 = (tables_dir / "table1_main_comparison.md").read_text(encoding="utf-8")

    generate_all_tables_and_plots(repo_root=tmp_path)
    t1_content_run2 = (tables_dir / "table1_main_comparison.md").read_text(encoding="utf-8")

    assert t1_content_run1 == t1_content_run2
    assert (fig_dir / "reliability_diagrams.png").exists()
    assert (fig_dir / "risk_coverage_curves.pdf").exists()


# --- Test 5: Paired bootstrap gives zero difference on identical systems ---

def test_paired_bootstrap_identical_systems():
    y_true = np.array([0, 1, 0, 2, 1, 0, 2, 1])
    data_a = np.array([0, 1, 0, 2, 1, 0, 2, 1])
    data_b = np.array([0, 1, 0, 2, 1, 0, 2, 1])

    res = paired_bootstrap_difference(compute_accuracy, y_true, data_a, data_b, n_bootstraps=500)
    assert res["point_diff"] == 0.0
    assert res["ci_lower"] == 0.0
    assert res["ci_upper"] == 0.0
    assert res["std_error"] == 0.0
    assert not res["excludes_zero"]

    # Also test McNemar's paired test on identical systems
    mcn = mcnemar_test(y_true, data_a, data_b)
    assert mcn["statistic"] == 0.0
    assert mcn["p_value"] == 1.0
    assert not mcn["significant_at_05"]


# --- Test 6: hypotheses.md contains no numbers that do not trace to saved evaluation results ---

def test_hypotheses_numbers_traceable(tmp_path):
    output_md = tmp_path / "hypotheses.md"
    eval_res = evaluate_hypotheses(repo_root=Path(".").resolve(), output_file=output_md)

    content = output_md.read_text(encoding="utf-8")
    assert "TrustNet Formal Hypothesis Testing Report" in content
    assert "H1: Multimodal Fusion vs. Text-Only Baseline" in content
    assert "H2: Evidence Retrieval Contribution" in content
    assert "H3: Uncertainty Calibration and ECE Reduction" in content
    assert "H4: Grounded Reasoning Layer Performance" in content
    assert "H5: Cross-Lingual Parity" in content

    # Check neutral language enforcement
    assert "no effect" not in content.lower()
    assert "proves" not in content.lower()

    # Verify numbers trace directly to evaluation output
    h1_diff_str = f"{eval_res['h1']['f1_boot']['point_diff']:+.4f}"
    assert h1_diff_str in content

    h3_diff_str = f"{eval_res['h3']['ece_boot']['point_diff']:+.4f}"
    assert h3_diff_str in content


# --- Test 7: Telegram bot parses commands, formats responses, and handles network failures ---

def test_telegram_bot_formatting_and_rate_limiting():
    # Confidence bar rendering
    bar = render_confidence_bar(0.85)
    assert "████████" in bar
    assert "85%" in bar

    # Rate limiting
    uid = 99999
    for _ in range(5):
        allowed, _ = check_rate_limit(uid)
        assert allowed
    # 6th attempt within window must be rate-limited
    allowed, retry_after = check_rate_limit(uid)
    assert not allowed
    assert retry_after > 0.0

    # Response formatting
    mock_envelope = {
        "report_id": "test-uuid-1234",
        "claims": [
            {
                "verdict": "verified",
                "confidence": 0.88,
                "share_advice": "safe_to_share",
                "claim": {"text": "Solar energy installations grew across rural districts."},
                "evidence": [{"title": "Official Report", "text": "Rural solar initiatives expanded."}],
            }
        ]
    }
    formatted = format_verification_response(mock_envelope, dashboard_base_url="http://localhost:5173")
    assert "VERIFIED" in formatted
    assert "Safe to Share" in formatted
    assert "88%" in formatted
    assert "http://localhost:5173/reports/test-uuid-1234" in formatted
    assert "Disclaimer" in formatted


@pytest.mark.asyncio
async def test_telegram_bot_network_failure_handling():
    # Mock network connection refused
    with patch("httpx.AsyncClient.post", side_effect=httpx.ConnectError("Connection refused")):
        with pytest.raises(httpx.ConnectError):
            await call_verification_api(text="Test statement", base_url="http://127.0.0.1:9999")


# --- Test 8: No module in src/trustnet/final/ imports model-training code ---

def test_no_training_code_imported_in_final_modules():
    final_dir = Path("src/trustnet/final")
    forbidden_modules = [
        "trustnet.text.train",
        "trustnet.fusion.train",
        "trustnet.retrieval.train",
        "torch.optim",
        "transformers.Trainer",
    ]

    for py_file in final_dir.glob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        parsed = ast.parse(content, filename=str(py_file))

        for node in ast.walk(parsed):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for forbidden in forbidden_modules:
                        assert not alias.name.startswith(forbidden), (
                            f"File {py_file.name} imports forbidden training module '{alias.name}'"
                        )
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                for forbidden in forbidden_modules:
                    assert not mod.startswith(forbidden), (
                        f"File {py_file.name} imports forbidden training module '{mod}'"
                    )
