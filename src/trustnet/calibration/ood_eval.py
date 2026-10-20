"""
Out-of-Distribution (OOD) Evaluation for TrustNet Stage 7.

Evaluates calibration, uncertainty, and risk-coverage on hand-labeled
Indian socio-cultural scenarios (data/eval/indian_demo_set.jsonl).

CRITICAL CONSTRAINTS:
- This dataset is strictly held out: NEVER used for model training, temperature scaling,
  or abstention threshold fitting.
- Clearly flags and logs results as OUT-OF-DISTRIBUTION and SMALL SAMPLE (N <= 10).
"""

import argparse
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from trustnet.calibration.abstain import Abstainer
from trustnet.calibration.metrics import (
    adaptive_expected_calibration_error,
    compute_aurc,
    compute_brier_score,
    compute_confidently_wrong_rate,
    compute_coverage_at_fixed_error_rate,
    compute_nll,
    compute_reliability_diagram_data,
    compute_risk_coverage_curve,
    expected_calibration_error,
    plot_reliability_diagram,
    plot_risk_coverage_curve,
)
from trustnet.calibration.temperature import TemperatureScaler
from trustnet.data.loader import load_jsonl
from trustnet.fusion.predict import MultimodalPredictor
from trustnet.schemas import Claim, Evidence, Verdict

logger = logging.getLogger("ood_eval")

CLASS_NAMES = ["verified", "contradicted", "insufficient_evidence"]
LABEL2ID = {l: i for i, l in enumerate(CLASS_NAMES)}


def run_ood_evaluation(
    ood_file: str = "data/eval/indian_demo_set.jsonl",
    config_path: str = "configs/stage7_uncertainty.yaml",
    temperature_path: str = "outputs/stage7/temperature.json",
    abstainer_path: str = "outputs/stage7/abstainer.json",
    output_dir: str = "outputs/stage7",
) -> Dict[str, Any]:
    """
    Evaluates calibrated predictor on the OOD Indian Demo Set.
    """
    ood_path = Path(ood_file)
    if not ood_path.exists():
        raise FileNotFoundError(f"OOD dataset not found: {ood_path}")

    records = load_jsonl(ood_path)
    n_samples = len(records)
    logger.info(f"Loaded {n_samples} OOD records from {ood_path}")

    # Load configuration
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    target_error_rates = [float(x) for x in cfg.get("target_error_rates", [0.05, 0.10, 0.15])]
    conf_wrong_cutoff = float(cfg.get("confidently_wrong_cutoff", 0.8))
    n_bins = int(cfg.get("n_bins", 15))

    out_base = Path(output_dir)
    plots_dir = out_base / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Initialize predictor from saved Stage 6/7 artifacts
    predictor = MultimodalPredictor.from_artifacts(
        temperature_path=temperature_path,
        abstainer_path=abstainer_path,
    )

    y_true = []
    uncal_probs_list = []
    cal_probs_list = []
    reports = []

    for r in records:
        cid = r.get("claim_id") or r.get("id")
        text = r.get("claim_text") or r.get("claim") or ""
        lbl_str = r.get("label4") or r.get("verdict") or "insufficient_evidence"
        y_true.append(LABEL2ID.get(lbl_str, 2))

        claim_obj = Claim(id=cid, text=text)
        report = predictor.predict_claim_report(
            claim=claim_obj,
            image_path=r.get("image_path"),
            dataset="indian_demo",
        )
        reports.append(report)

        # Extract probability vectors
        raw_p = [report.raw_probabilities.get(c, 0.33) for c in CLASS_NAMES]
        cal_p = [report.calibrated_probabilities.get(c, 0.33) for c in CLASS_NAMES]
        uncal_probs_list.append(raw_p)
        cal_probs_list.append(cal_p)

    y_arr = np.array(y_true, dtype=np.int64)
    p_uncal = np.array(uncal_probs_list, dtype=np.float64)
    p_cal = np.array(cal_probs_list, dtype=np.float64)

    # Normalize rows
    p_uncal = p_uncal / np.sum(p_uncal, axis=1, keepdims=True)
    p_cal = p_cal / np.sum(p_cal, axis=1, keepdims=True)

    # Compute metrics for both uncalibrated and calibrated
    results: Dict[str, Any] = {
        "dataset": "indian_demo_set",
        "description": "Out-of-distribution hand-labeled Indian socio-cultural claims",
        "is_out_of_distribution": True,
        "sample_size": n_samples,
        "warning": "SMALL SAMPLE SIZE (N <= 10). Metric values are indicative and have high variance.",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    for name, p_mat in [("uncalibrated", p_uncal), ("calibrated", p_cal)]:
        preds = np.argmax(p_mat, axis=1)
        acc = float(np.mean(preds == y_arr))
        ece = expected_calibration_error(p_mat, y_arr, n_bins=min(n_bins, n_samples))
        ada_ece = adaptive_expected_calibration_error(p_mat, y_arr, n_bins=min(n_bins, n_samples))
        nll = compute_nll(p_mat, y_arr)
        brier = compute_brier_score(p_mat, y_arr)
        conf_wrong = compute_confidently_wrong_rate(p_mat, y_arr, cutoff=conf_wrong_cutoff)

        covs, risks = compute_risk_coverage_curve(p_mat, y_arr)
        aurc = compute_aurc(covs, risks)

        abstention = {}
        for terr in target_error_rates:
            cov = compute_coverage_at_fixed_error_rate(p_mat, y_arr, terr)
            abstention[f"target_err_{int(terr*100):02d}pct"] = {
                "target_error_rate": terr,
                "coverage": round(float(cov), 4),
            }

        results[name] = {
            "accuracy": round(acc, 4),
            "ece": round(float(ece), 4),
            "adaptive_ece": round(float(ada_ece), 4),
            "nll": round(float(nll), 4),
            "brier_score": round(float(brier), 4),
            "aurc": round(float(aurc), 4),
            "confidently_wrong_rate": round(float(conf_wrong), 4),
            "abstention": abstention,
        }

    # Plot reliability diagram and risk-coverage curve for calibrated
    diag = compute_reliability_diagram_data(p_cal, y_arr, n_bins=min(5, n_samples))
    plot_reliability_diagram(
        diag,
        save_path=plots_dir / "reliability_ood_calibrated.png",
        title="OOD Reliability: Indian Demo Set",
    )

    covs_cal, risks_cal = compute_risk_coverage_curve(p_cal, y_arr)
    plot_risk_coverage_curve(
        covs_cal,
        risks_cal,
        aurc=results["calibrated"]["aurc"],
        save_path=plots_dir / "risk_coverage_ood_calibrated.png",
        title="OOD Risk-Coverage: Indian Demo Set",
    )

    # Save to results files
    ood_json_path = out_base / "ood_results.json"
    with open(ood_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved OOD evaluation results to {ood_json_path}")

    # Generate Markdown summary
    md_lines = [
        "# TrustNet Stage 7: Out-of-Distribution (OOD) Evaluation",
        "",
        "> [!IMPORTANT]",
        f"> **Notice:** Evaluated on `{ood_file}` (N = {n_samples}). This dataset is strictly **out-of-distribution**",
        "> and hand-labeled for Indian multimodal scenarios. It was **never used for fitting** models,",
        "> temperature scaling, or abstention thresholds.",
        "",
        "| Metric | Uncalibrated | Calibrated (Temperature) | Difference / Gain |",
        "|---|---|---|---|",
        f"| **ECE** | {results['uncalibrated']['ece']:.4f} | {results['calibrated']['ece']:.4f} | "
        f"{results['calibrated']['ece'] - results['uncalibrated']['ece']:+.4f} |",
        f"| **NLL** | {results['uncalibrated']['nll']:.4f} | {results['calibrated']['nll']:.4f} | "
        f"{results['calibrated']['nll'] - results['uncalibrated']['nll']:+.4f} |",
        f"| **Brier Score** | {results['uncalibrated']['brier_score']:.4f} | {results['calibrated']['brier_score']:.4f} | "
        f"{results['calibrated']['brier_score'] - results['uncalibrated']['brier_score']:+.4f} |",
        f"| **AURC** | {results['uncalibrated']['aurc']:.4f} | {results['calibrated']['aurc']:.4f} | "
        f"{results['calibrated']['aurc'] - results['uncalibrated']['aurc']:+.4f} |",
        f"| **Confidently Wrong Rate** | {results['uncalibrated']['confidently_wrong_rate']:.4f} | "
        f"{results['calibrated']['confidently_wrong_rate']:.4f} | "
        f"{results['calibrated']['confidently_wrong_rate'] - results['uncalibrated']['confidently_wrong_rate']:+.4f} |",
        "",
        "### Selective Abstention Coverage on OOD Data",
        "",
        "| Target Error Rate | Uncalibrated Coverage | Calibrated Coverage |",
        "|---|---|---|",
    ]

    for terr in target_error_rates:
        tkey = f"target_err_{int(terr*100):02d}pct"
        u_cov = results["uncalibrated"]["abstention"][tkey]["coverage"]
        c_cov = results["calibrated"]["abstention"][tkey]["coverage"]
        md_lines.append(f"| {terr:.0%} error | {u_cov:.2%} | {c_cov:.2%} |")

    md_report = "\n".join(md_lines)
    ood_md_path = out_base / "ood_results.md"
    with open(ood_md_path, "w", encoding="utf-8") as f:
        f.write(md_report)
    logger.info(f"Saved OOD Markdown report to {ood_md_path}")

    print("\n" + md_report + "\n")
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate on OOD Indian Demo Set")
    parser.add_argument("--ood_file", type=str, default="data/eval/indian_demo_set.jsonl")
    parser.add_argument("--config", type=str, default="configs/stage7_uncertainty.yaml")
    parser.add_argument("--temperature_path", type=str, default="outputs/stage7/temperature.json")
    parser.add_argument("--abstainer_path", type=str, default="outputs/stage7/abstainer.json")
    parser.add_argument("--output_dir", type=str, default="outputs/stage7")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    run_ood_evaluation(
        ood_file=args.ood_file,
        config_path=args.config,
        temperature_path=args.temperature_path,
        abstainer_path=args.abstainer_path,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
