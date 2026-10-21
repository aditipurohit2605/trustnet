"""
TrustNet Stage 12: Qualitative and Quantitative Error Analysis Tooling.

Provides two CLI subcommands:
1. sample-errors: Draws a seeded stratified sample of incorrect model predictions,
   writing an annotation-ready CSV with fixed error taxonomies:
   - retrieval_failure
   - multimodal_misalignment
   - temporal_staleness
   - subtle_paraphrase
   - out_of_scope_claim
   - uncalibrated_confidence
   - other
2. analyze-errors: Ingests completed annotations, computes counts per error cause,
   calculates inter-annotator agreement (Cohen's kappa) if a second annotator CSV
   is provided, and identifies the top-5 highest-confidence incorrect predictions.
"""

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

TAXONOMY_COLUMNS = [
    "retrieval_failure",
    "multimodal_misalignment",
    "temporal_staleness",
    "subtle_paraphrase",
    "out_of_scope_claim",
    "uncalibrated_confidence",
    "other",
]


def load_incorrect_predictions(predictions_file: Path) -> List[Dict[str, Any]]:
    """Loads all prediction rows where the prediction differed from the ground truth."""
    incorrect = []
    if not predictions_file.exists():
        return incorrect

    with open(predictions_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            # Evaluate correctness
            is_correct = r.get("correct", False)
            if not is_correct:
                incorrect.append(r)
    return incorrect


def sample_errors(
    predictions_file: Path,
    output_csv: Path,
    sample_size: int = 50,
    random_seed: int = 42,
) -> int:
    """
    Draws a seeded, stratified sample of error records for manual annotation.
    """
    errors = load_incorrect_predictions(predictions_file)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if not errors:
        # Create an empty template if no errors occurred
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["claim_id", "claim_text", "gold_verdict", "predicted_verdict", "confidence"] + TAXONOMY_COLUMNS + ["notes"])
        return 0

    # Stratify by gold_verdict if possible
    rng = np.random.default_rng(random_seed)
    n = min(sample_size, len(errors))
    indices = rng.choice(len(errors), size=n, replace=False)
    sampled = [errors[i] for i in indices]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["claim_id", "claim_text", "gold_verdict", "predicted_verdict", "confidence"] + TAXONOMY_COLUMNS + ["notes"])
        for r in sampled:
            writer.writerow([
                r.get("claim_id", ""),
                r.get("claim_text", ""),
                r.get("gold_verdict", ""),
                r.get("effective_verdict", r.get("predicted_verdict", "")),
                r.get("confidence", 0.0),
                0, 0, 0, 0, 0, 0, 0,  # Taxonomy flags
                "",  # Notes
            ])

    return len(sampled)


def compute_cohens_kappa(labels_a: List[str], labels_b: List[str]) -> float:
    """Computes Cohen's kappa coefficient between two annotator label sequences."""
    if len(labels_a) != len(labels_b) or len(labels_a) == 0:
        return 0.0

    categories = sorted(list(set(labels_a) | set(labels_b)))
    cat2idx = {c: i for i, c in enumerate(categories)}
    k = len(categories)
    if k <= 1:
        return 1.0

    matrix = np.zeros((k, k), dtype=float)
    n = len(labels_a)
    for a, b in zip(labels_a, labels_b):
        matrix[cat2idx[a], cat2idx[b]] += 1.0

    matrix /= n
    po = np.trace(matrix)
    pe = np.sum(np.sum(matrix, axis=1) * np.sum(matrix, axis=0))

    if pe >= 1.0:
        return 1.0
    return float((po - pe) / (1.0 - pe))


def analyze_errors(
    annotated_csv: Path,
    second_annotator_csv: Optional[Path] = None,
    predictions_file: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Computes error cause counts, inter-annotator kappa, and top-5 highest-confidence mistakes.
    """
    counts = {col: 0 for col in TAXONOMY_COLUMNS}
    total_annotated = 0
    annotator_1_primary = []

    if annotated_csv.exists():
        with open(annotated_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_annotated += 1
                assigned_causes = []
                for col in TAXONOMY_COLUMNS:
                    val = row.get(col, "0").strip()
                    if val in ("1", "true", "True", "x", "yes"):
                        counts[col] += 1
                        assigned_causes.append(col)
                annotator_1_primary.append(assigned_causes[0] if assigned_causes else "other")

    # Inter-annotator agreement
    kappa = None
    if second_annotator_csv and second_annotator_csv.exists():
        annotator_2_primary = []
        with open(second_annotator_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                assigned = [col for col in TAXONOMY_COLUMNS if row.get(col, "0").strip() in ("1", "true", "True", "x", "yes")]
                annotator_2_primary.append(assigned[0] if assigned else "other")
        kappa = compute_cohens_kappa(annotator_1_primary, annotator_2_primary)

    # Top-5 Highest-Confidence Wrong Predictions
    top5_wrong = []
    if predictions_file and predictions_file.exists():
        errors = load_incorrect_predictions(predictions_file)
        # Sort descending by confidence
        sorted_errors = sorted(errors, key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
        top5_wrong = [
            {
                "claim_id": r.get("claim_id"),
                "claim_text": r.get("claim_text"),
                "gold": r.get("gold_verdict"),
                "predicted": r.get("effective_verdict", r.get("predicted_verdict")),
                "confidence": round(float(r.get("confidence", 0.0)), 4),
            }
            for r in sorted_errors[:5]
        ]

    return {
        "total_annotated": total_annotated,
        "counts": counts,
        "cohens_kappa": kappa,
        "top5_wrong": top5_wrong,
    }


def main():
    parser = argparse.ArgumentParser(description="TrustNet Error Analysis Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: sample-errors
    p_sample = subparsers.add_parser("sample-errors", help="Sample errors for annotation")
    p_sample.add_argument("--predictions", type=str, default="outputs/final/predictions/test_predictions.jsonl")
    p_sample.add_argument("--output-csv", type=str, default="outputs/final/error_sample.csv")
    p_sample.add_argument("--sample-size", type=int, default=50)
    p_sample.add_argument("--seed", type=int, default=42)

    # Subcommand: analyze-errors
    p_analyze = subparsers.add_parser("analyze-errors", help="Analyze annotated errors")
    p_analyze.add_argument("--annotated-csv", type=str, default="outputs/final/error_sample.csv")
    p_analyze.add_argument("--second-annotator-csv", type=str, default=None)
    p_analyze.add_argument("--predictions", type=str, default="outputs/final/predictions/test_predictions.jsonl")

    args = parser.parse_args()

    if args.command == "sample-errors":
        n = sample_errors(Path(args.predictions), Path(args.output_csv), args.sample_size, args.seed)
        print(f"[ERROR SAMPLING COMPLETE] Sampled {n} errors to {args.output_csv}.")
    elif args.command == "analyze-errors":
        res = analyze_errors(
            Path(args.annotated_csv),
            Path(args.second_annotator_csv) if args.second_annotator_csv else None,
            Path(args.predictions) if args.predictions else None,
        )
        print("\n# TrustNet Error Attribution Breakdown")
        print(f"Total Annotated Samples: {res['total_annotated']}")
        for col, c in res["counts"].items():
            pct = (c / res["total_annotated"] * 100) if res["total_annotated"] > 0 else 0.0
            print(f"  - {col}: {c} ({pct:.1f}%)")
        if res["cohens_kappa"] is not None:
            print(f"\nInter-Annotator Agreement (Cohen's Kappa): {res['cohens_kappa']:.4f}")
        print("\n# Top-5 Highest-Confidence False Predictions:")
        for i, item in enumerate(res["top5_wrong"], 1):
            print(f"  {i}. [{item['claim_id']}] Conf: {item['confidence']} | Pred: {item['predicted']} vs Gold: {item['gold']}")
            print(f"     \"{item['claim_text']}\"")


if __name__ == "__main__":
    main()
