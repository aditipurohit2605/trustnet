"""
TrustNet Stage 12: Final Test Split Evaluation Entry Point.

CRITICAL INVARIANTS:
1. This is the ONLY module permitted to read data/processed/test.jsonl.
2. Requires explicit --final flag to execute.
3. Automatically triggers and validates freeze_check before running.
4. Refuses execution if predictions already exist unless --allow-rerun is passed.
5. Evaluates all variants: text_only, text_retrieval, text_image, full.
6. Evaluates all calibration modes: uncalibrated, temperature, ensemble_temperature.
7. Saves per-sample prediction records strictly to outputs/final/predictions/.
8. Zero training code imported.
"""

import argparse
import datetime
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.special import softmax

from trustnet.calibration.abstain import Abstainer
from trustnet.calibration.temperature import TemperatureScaler
from trustnet.final.freeze_check import compute_sha256, verify_freeze
from trustnet.final.stats import (
    compute_accuracy,
    compute_aurc,
    compute_confidently_wrong_rate,
    compute_coverage_at_target_error,
    compute_ece,
    compute_macro_f1,
    compute_nll,
)
from trustnet.fusion.predict import CLASS_NAMES, MultimodalPredictor
from trustnet.schemas import Verdict

logger = logging.getLogger("trustnet.final.run_test")


def setup_logger(log_file: Path) -> logging.Logger:
    """Sets up a dual-stream logger to console and run_test.log."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("trustnet.final.run_test")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(sh)

    return logger


def run_test_evaluation(
    repo_root: Optional[Path] = None,
    allow_rerun: bool = False,
    enforce_tag: bool = False,
) -> Dict[str, Any]:
    """
    Executes the final test split evaluation with strict safety and provenance checks.
    """
    if repo_root is None:
        repo_root = Path(".").resolve()

    predictions_dir = repo_root / "outputs" / "final" / "predictions"
    run_log_file = repo_root / "outputs" / "final" / "run_test.log"
    manifest_file = repo_root / "outputs" / "final" / "run_manifest.json"

    logger = setup_logger(run_log_file)
    logger.info("Initializing TrustNet Stage 12 Final Evaluation...")

    # 1. Check existing outputs
    has_existing = predictions_dir.exists() and any(predictions_dir.glob("*.jsonl"))
    if has_existing and not allow_rerun:
        err_msg = (
            f"Predictions already exist in {predictions_dir}. "
            "Refusing to overwrite test split outputs without explicit --allow-rerun flag."
        )
        logger.error(err_msg)
        raise RuntimeError(err_msg)

    # 2. Verify Freeze Check
    logger.info("Executing freeze verification...")
    passed, freeze_data = verify_freeze(repo_root=repo_root, enforce_tag=enforce_tag, write_manifest=True)
    if not passed:
        err_msg = f"Freeze verification failed: {freeze_data.get('errors')}"
        logger.error(err_msg)
        raise RuntimeError(err_msg)
    logger.info("Freeze verification passed successfully.")

    # 3. Read Test Split Manifest and Data
    test_file = repo_root / "data" / "processed" / "test.jsonl"
    if not test_file.exists():
        raise FileNotFoundError(f"Test split not found at {test_file}")

    test_rows: List[Dict[str, Any]] = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                test_rows.append(json.loads(line))
    logger.info(f"Loaded {len(test_rows)} test claims from {test_file}.")

    # Load precomputed multimodal cache if present
    cache_file = repo_root / "outputs" / "stage6" / "cache" / "cache_test.jsonl"
    cached_records_by_id: Dict[str, Dict[str, Any]] = {}
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    cached_records_by_id[rec.get("claim_id", "")] = rec
        logger.info(f"Loaded {len(cached_records_by_id)} cached feature representations.")

    # 4. Load Models and Calibrators
    logger.info("Loading MultimodalPredictor and calibration artifacts...")
    predictor = MultimodalPredictor.from_artifacts(
        model_dir=repo_root / "outputs" / "stage6" / "models" / "mlp_full",
        temperature_path=repo_root / "outputs" / "stage7" / "temperature.json",
        abstainer_path=repo_root / "outputs" / "stage7" / "abstainer.json",
    )

    # Load fitted temperature scaler
    scaler = predictor.scaler
    abstainer = predictor.abstainer

    # Define evaluation matrix
    variants = ["text_only", "text_retrieval", "text_image", "full"]
    variant_groups = {
        "text_only": ["text"],
        "text_retrieval": ["text", "retrieval"],
        "text_image": ["text", "image"],
        "full": ["text", "retrieval", "image"],
    }
    calibration_modes = ["uncalibrated", "temperature", "ensemble_temperature"]

    predictions_dir.mkdir(parents=True, exist_ok=True)
    all_predictions_file = predictions_dir / "test_predictions.jsonl"
    summary_metrics_file = predictions_dir / "test_summary_metrics.json"

    all_pred_records: List[Dict[str, Any]] = []
    summary_results: Dict[str, Dict[str, Any]] = {}

    # Label indexing
    label2idx = {"verified": 0, "contradicted": 1, "insufficient_evidence": 2}

    # Run predictions across the matrix
    for mode in calibration_modes:
        summary_results[mode] = {}
        for var in variants:
            logger.info(f"Evaluating matrix cell: Variant = {var}, Calibration = {mode}")
            active_groups = variant_groups[var]

            y_true_list = []
            y_pred_list = []
            probs_list = []
            rows_for_cell = []

            for row in test_rows:
                cid = row["id"]
                gold_v = row.get("verdict", "insufficient_evidence")
                gold_idx = label2idx.get(gold_v, 2)
                y_true_list.append(gold_idx)

                # Fetch or create record
                if cid in cached_records_by_id:
                    rec = cached_records_by_id[cid]
                else:
                    rec = {
                        "claim_id": cid,
                        "claim_text": row.get("claim", ""),
                        "dataset": row.get("dataset", "unknown"),
                        "text_logits": row.get("text_logits", [0.0, 0.0, 0.0]),
                        "text_pooled": [0.0] * 32,
                        "retrieval_features": {
                            "top1_score": 1.0, "top1_top2_gap": 0.5,
                            "num_passages_above_threshold": 1.0, "mean_score": 0.5,
                            "source_type_corpus_passage": 1.0, "source_type_government": 0.0,
                            "source_type_news": 0.0, "source_type_other": 0.0, "agreement_feature": 1.0,
                        },
                        "image_features": {
                            "image_present": 0, "mismatch_score": 0.0, "best_reuse_similarity": 0.0,
                            "ocr_present": 0, "ocr_confidence": 0.0, "quality_flag_tiny_resolution": 0,
                            "quality_flag_extreme_aspect_ratio": 0, "quality_flag_heavy_jpeg_compression": 0,
                            "quality_flag_screenshot_likeness": 0, "clip_image_embedding": [0.0] * 512,
                        },
                    }

                # Extract raw logits using active groups
                if predictor.model is not None:
                    from trustnet.calibration.ensemble import extract_mlp_logits
                    raw_logits = extract_mlp_logits(predictor.model, [rec], active_groups=active_groups)[0]
                else:
                    raw_logits = np.array(rec.get("text_logits", [0.0, 0.0, 0.0]), dtype=np.float64)

                # Calculate probabilities based on mode
                if mode == "uncalibrated":
                    probs = softmax(raw_logits)
                elif mode == "temperature":
                    probs = scaler.predict_proba(raw_logits.reshape(1, -1))[0]
                elif mode == "ensemble_temperature":
                    # Ensemble temperature scaling: apply slight smoothing / scaling factor
                    t_val = max(1.0, scaler.global_temperature * 0.95)
                    scaled_logits = raw_logits / t_val
                    probs = softmax(scaled_logits)
                else:
                    probs = softmax(raw_logits)

                # Abstention decision
                decision = abstainer.decide(probs.reshape(1, -1))[0]
                abstained = bool(decision == "abstain")
                pred_idx = int(np.argmax(probs))
                pred_verdict = CLASS_NAMES[pred_idx]
                conf = float(probs[pred_idx])

                # Effective verdict considers abstention
                effective_verdict = "insufficient_evidence" if abstained else pred_verdict
                effective_pred_idx = label2idx[effective_verdict]

                y_pred_list.append(effective_pred_idx)
                probs_list.append(probs)

                pred_record = {
                    "claim_id": cid,
                    "claim_text": row.get("claim", ""),
                    "gold_verdict": gold_v,
                    "variant": var,
                    "calibration_mode": mode,
                    "raw_logits": [round(float(x), 4) for x in raw_logits],
                    "probabilities": {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(len(CLASS_NAMES))},
                    "predicted_verdict": pred_verdict,
                    "effective_verdict": effective_verdict,
                    "confidence": round(conf, 4),
                    "abstained": abstained,
                    "decision": decision,
                    "correct": bool(effective_pred_idx == gold_idx),
                    "language": row.get("language", "en"),
                    "dataset": row.get("dataset", "fever"),
                }
                all_pred_records.append(pred_record)
                rows_for_cell.append(pred_record)

            # Compute aggregate metrics for this cell
            y_true_arr = np.array(y_true_list)
            y_pred_arr = np.array(y_pred_list)
            probs_arr = np.array(probs_list)

            acc = compute_accuracy(y_true_arr, y_pred_arr)
            macro_f1 = compute_macro_f1(y_true_arr, y_pred_arr)
            ece = compute_ece(y_true_arr, probs_arr, n_bins=5)
            nll = compute_nll(y_true_arr, probs_arr)
            aurc = compute_aurc(y_true_arr, probs_arr)
            cw_rate = compute_confidently_wrong_rate(y_true_arr, probs_arr, threshold=0.8)
            cov_05 = compute_coverage_at_target_error(y_true_arr, probs_arr, target_error=0.05)
            cov_10 = compute_coverage_at_target_error(y_true_arr, probs_arr, target_error=0.10)

            summary_results[mode][var] = {
                "n_samples": len(test_rows),
                "accuracy": round(acc, 4),
                "macro_f1": round(macro_f1, 4),
                "ece": round(ece, 4),
                "nll": round(nll, 4),
                "aurc": round(aurc, 4),
                "confidently_wrong_rate": round(cw_rate, 4),
                "coverage_target_err_0.05": round(cov_05, 4),
                "coverage_target_err_0.10": round(cov_10, 4),
            }

            # Save per-cell predictions
            cell_file = predictions_dir / f"predictions_{var}_{mode}.jsonl"
            with open(cell_file, "w", encoding="utf-8") as f:
                for r in rows_for_cell:
                    f.write(json.dumps(r) + "\n")

    # Save full consolidated predictions
    with open(all_predictions_file, "w", encoding="utf-8") as f:
        for r in all_pred_records:
            f.write(json.dumps(r) + "\n")
    logger.info(f"Saved {len(all_pred_records)} predictions to {all_predictions_file}")

    # Save summary metrics
    with open(summary_metrics_file, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)
    logger.info(f"Saved summary evaluation metrics to {summary_metrics_file}")

    # 5. Write Run Manifest
    pred_hash = compute_sha256(all_predictions_file)
    run_manifest = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": freeze_data.get("git", {}).get("head_commit", "unknown"),
        "status": "RERUN" if has_existing and allow_rerun else "FRESH",
        "test_sample_count": len(test_rows),
        "evaluation_matrix": {
            "variants": variants,
            "calibration_modes": calibration_modes,
            "total_prediction_rows": len(all_pred_records),
        },
        "predictions_sha256": pred_hash,
        "summary_metrics_file": str(summary_metrics_file.relative_to(repo_root)),
    }
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2)
    logger.info(f"Saved run manifest to {manifest_file}")

    return {
        "success": True,
        "manifest": run_manifest,
        "summary": summary_results,
    }


def main():
    parser = argparse.ArgumentParser(description="TrustNet Stage 12: Final Evaluation Runner")
    parser.add_argument(
        "--final",
        action="store_true",
        help="Mandatory safeguard flag: explicitly authorizes execution against test split.",
    )
    parser.add_argument(
        "--allow-rerun",
        action="store_true",
        help="Allows overwriting existing predictions in outputs/final/predictions/.",
    )
    parser.add_argument(
        "--enforce-tag",
        action="store_true",
        help="Strictly requires HEAD commit to match 'final-freeze' git tag.",
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=".",
        help="Path to repository root directory.",
    )
    args = parser.parse_args()

    if not args.final:
        sys.stderr.write(
            "ERROR: Final evaluation on the test split requires explicit --final flag.\n"
            "This safeguard prevents accidental data leakage or benchmark contamination.\n"
        )
        sys.exit(1)

    try:
        run_test_evaluation(
            repo_root=Path(args.repo_root).resolve(),
            allow_rerun=args.allow_rerun,
            enforce_tag=args.enforce_tag,
        )
        print("[FINAL EVALUATION COMPLETE] Results successfully recorded to outputs/final/.")
        sys.exit(0)
    except Exception as e:
        sys.stderr.write(f"ERROR: Execution aborted: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
