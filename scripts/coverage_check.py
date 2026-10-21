"""
Evaluation Script: Retrieval Coverage Verification on Indian Demo Set.

Evaluates and reports the fraction of claims in data/eval/indian_demo_set.jsonl
that have at least one retrieved passage above the configured relevance threshold (0.50).

Strictly non-mutating: does NOT change any model, threshold, verdict or schema.

Usage:
    python scripts/coverage_check.py [--threshold 0.50] [--config configs/stage3_retrieval.yaml]
"""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

# Reconfigure stdout and stderr for UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure trustnet package is importable
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from trustnet.retrieval.retriever import Retriever
from trustnet.schemas import Evidence

# Configure logging with standard output formatting
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("coverage_check")


def load_demo_claims(dataset_path: Path) -> List[Dict[str, Any]]:
    """Loads claims from the target JSONL file, validating basic required fields."""
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Target evaluation dataset not found at: {dataset_path}")

    claims = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                rec = json.loads(line_str)
                claims.append(rec)
            except json.JSONDecodeError as err:
                logger.warning(f"Skipping malformed JSON at line {line_no} in {dataset_path}: {err}")
    return claims


def run_coverage_check(
    dataset_path: Path = Path("data/eval/indian_demo_set.jsonl"),
    config_path: Path = Path("configs/stage3_retrieval.yaml"),
    override_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Evaluates evidence retrieval coverage for all claims in the target dataset.
    
    A claim is defined as 'covered' if and only if at least one retrieved passage
    achieves a cross-encoder / fused relevance score >= relevance_threshold.
    """
    print("=" * 90)
    print("TRUSTNET RETRIEVAL COVERAGE CHECK: INDIAN DEMO SET")
    print("=" * 90)
    print(f"Dataset Path:    {dataset_path}")
    print(f"Retrieval Config: {config_path}")

    # Initialize the Stage 3 multi-mode retriever
    retriever = Retriever(config_or_path=config_path)

    # Resolve the relevance threshold from config or CLI override
    # Non-obvious: retrieval config specifies relevance_threshold=0.50 by default
    if override_threshold is not None:
        threshold = float(override_threshold)
    else:
        threshold = float(retriever.config.get("relevance_threshold", 0.50))

    print(f"Relevance Threshold: {threshold:.4f}\n")

    claims = load_demo_claims(dataset_path)
    total_claims = len(claims)
    if total_claims == 0:
        print("No claims found in dataset. Exiting with coverage 0.0%.")
        return {"total": 0, "covered": 0, "fraction": 0.0, "details": []}

    print(f"Loaded {total_claims} claims. Running hybrid rerank retrieval...\n")

    results = []
    covered_count = 0

    print(f"{'Claim ID':<10} | {'Lang':<5} | {'Top Score':<10} | {'>= Thresh':<10} | {'Covered?':<9} | {'Claim Excerpt':<35}")
    print("-" * 90)

    for rec in claims:
        cid = str(rec.get("claim_id") or rec.get("id") or "unknown")
        text = str(rec.get("claim_text") or rec.get("claim") or "")
        lang = "hi" if any("\u0900" <= ch <= "\u097f" for ch in text) else "en"

        # Execute claim verification search using hybrid_rerank mode
        # Non-obvious: Retriever.search executes BM25, Dense FAISS, RRF fusion, and cross-encoder rerank
        try:
            evidence_items: List[Evidence] = retriever.search(claim_text=text, mode="hybrid_rerank")
            scores = [float(e.score) for e in evidence_items if e.score is not None]
            top_score = max(scores) if scores else 0.0
            passages_above = sum(1 for s in scores if s >= threshold)
            is_covered = (passages_above >= 1)
        except Exception as e:
            logger.error(f"Retrieval failed for claim [{cid}]: {e}")
            top_score = 0.0
            passages_above = 0
            is_covered = False

        if is_covered:
            covered_count += 1

        status_str = "YES" if is_covered else "NO"
        excerpt = (text[:32] + "...") if len(text) > 32 else text
        print(f"{cid:<10} | {lang:<5} | {top_score:<10.4f} | {passages_above:<10} | {status_str:<9} | {excerpt:<35}")

        results.append({
            "claim_id": cid,
            "claim_text": text,
            "language": lang,
            "top_score": top_score,
            "passages_above_threshold": passages_above,
            "is_covered": is_covered,
        })

    print("-" * 90)

    coverage_fraction = covered_count / total_claims if total_claims > 0 else 0.0
    coverage_pct = coverage_fraction * 100.0

    print(f"\nFinal Summary Statistics:")
    print(f"  Total Claims Evaluated:               {total_claims}")
    print(f"  Claims with Passage >= {threshold:.2f}:       {covered_count}")
    print(f"  Coverage Fraction:                    {coverage_fraction:.4f} ({coverage_pct:.2f}%)")
    print("=" * 90 + "\n")

    return {
        "total_claims": total_claims,
        "covered_claims": covered_count,
        "coverage_fraction": round(coverage_fraction, 4),
        "coverage_percentage": round(coverage_pct, 2),
        "threshold": threshold,
        "details": results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Verify retrieval coverage for claims in data/eval/indian_demo_set.jsonl."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/eval/indian_demo_set.jsonl",
        help="Path to evaluation JSONL claims file.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stage3_retrieval.yaml",
        help="Path to Stage 3 retrieval configuration YAML.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional relevance score threshold override (default: from config, 0.50).",
    )
    args = parser.parse_args()

    run_coverage_check(
        dataset_path=Path(args.dataset),
        config_path=Path(args.config),
        override_threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
