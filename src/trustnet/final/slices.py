"""
TrustNet Stage 12: Subgroup and Demographic Slice Analysis.

CRITICAL INVARIANTS:
1. Evaluates performance across slices:
   - Dataset source (FEVER, Snopes, etc.)
   - Language form (native Hindi, Hinglish, English)
   - Image presence (with image vs. text-only)
   - Claim length (short <=15 words, medium 16-30 words, long >30 words)
   - Evidence availability (with evidence vs. unevidenced)
2. Statistical Guard: Any slice with N < 30 is explicitly flagged as "too small to interpret".
3. Outputs structured outputs/final/slices.json and publication-ready outputs/final/slices.md.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from trustnet.final.stats import compute_accuracy, compute_ece, compute_macro_f1


def categorize_length(text: str) -> str:
    """Categorizes text length based on word count."""
    wc = len(text.split())
    if wc <= 15:
        return "short (<=15 words)"
    elif wc <= 30:
        return "medium (16-30 words)"
    return "long (>30 words)"


def run_slice_analysis(
    predictions_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    repo_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Computes performance metrics across predefined subgroups and slices.
    """
    if repo_root is None:
        repo_root = Path(".").resolve()
    if predictions_path is None:
        predictions_path = repo_root / "outputs" / "final" / "predictions" / "test_predictions.jsonl"
    if output_dir is None:
        output_dir = repo_root / "outputs" / "final"

    records = []
    if predictions_path.exists():
        with open(predictions_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    # Use calibrated full model predictions by default
                    if r.get("variant") == "full" and r.get("calibration_mode") == "temperature":
                        records.append(r)

    # Fallback to all records if full/temperature filter is empty
    if not records and predictions_path.exists():
        with open(predictions_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    # Define slice groupings
    slice_definitions = {
        "Dataset Source": lambda r: r.get("dataset", "unknown"),
        "Language Form": lambda r: r.get("language", "en"),
        "Image Presence": lambda r: "with_image" if r.get("image_present", 0) > 0 else "text_only",
        "Claim Length": lambda r: categorize_length(r.get("claim_text", "")),
        "Evidence Availability": lambda r: "evidenced" if r.get("has_evidence", True) else "unevidenced",
    }

    label2idx = {"verified": 0, "contradicted": 1, "insufficient_evidence": 2}
    slices_results: Dict[str, Dict[str, Any]] = {}

    for group_name, extractor in slice_definitions.items():
        slices_results[group_name] = {}
        # Partition records
        partitions: Dict[str, List[Dict[str, Any]]] = {}
        for r in records:
            key = extractor(r)
            partitions.setdefault(key, []).append(r)

        for cat_name, cat_records in partitions.items():
            n = len(cat_records)
            too_small = n < 30

            if n > 0:
                y_true = np.array([label2idx.get(r["gold_verdict"], 2) for r in cat_records])
                y_pred = np.array([label2idx.get(r.get("effective_verdict", r.get("predicted_verdict")), 2) for r in cat_records])
                probs = np.array([
                    [r["probabilities"].get("verified", 0.33),
                     r["probabilities"].get("contradicted", 0.33),
                     r["probabilities"].get("insufficient_evidence", 0.34)]
                    for r in cat_records
                ])

                acc = compute_accuracy(y_true, y_pred)
                f1 = compute_macro_f1(y_true, y_pred)
                ece = compute_ece(y_true, probs, n_bins=min(5, max(2, n)))
            else:
                acc = 0.0
                f1 = 0.0
                ece = 0.0

            slices_results[group_name][cat_name] = {
                "n_samples": n,
                "accuracy": round(acc, 4),
                "macro_f1": round(f1, 4),
                "ece": round(ece, 4),
                "flag": "too small to interpret" if too_small else "interpretable",
            }

    # Write JSON output
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "slices.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(slices_results, f, indent=2)

    # Write Markdown output
    md_path = output_dir / "slices.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# TrustNet Subgroup and Slice Analysis\n\n")
        f.write("> **Statistical Caution**: Per evaluation protocol, any slice with $N < 30$ is marked as **too small to interpret**.\n\n")
        for group_name, cats in slices_results.items():
            f.write(f"### {group_name}\n\n")
            f.write("| Subgroup Slice | N Samples | Accuracy | Macro-F1 | ECE | Statistical Status |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: | :--- |\n")
            for cat_name, metrics in cats.items():
                status_str = f"⚠️ *{metrics['flag']}*" if metrics["n_samples"] < 30 else "✅ *Interpretable*"
                f.write(f"| {cat_name} | {metrics['n_samples']} | {metrics['accuracy']:.3f} | {metrics['macro_f1']:.3f} | {metrics['ece']:.3f} | {status_str} |\n")
            f.write("\n")

    return slices_results


if __name__ == "__main__":
    run_slice_analysis()
    print("[SLICE ANALYSIS COMPLETE] Recorded to outputs/final/slices.md and slices.json.")
