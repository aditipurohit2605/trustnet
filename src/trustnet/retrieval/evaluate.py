"""
CLI evaluation script for Stage 3 retrieval engine.

Evaluates lexical, dense, hybrid, and reranked retrieval quality at the document level
on claims with gold evidence. Computes Recall@k, MRR, nDCG@10, and average query latency.
Enforces an explicit safety guard prohibiting test split evaluation without --final.
"""

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set

from trustnet.data.loader import load_claims
from trustnet.retrieval.retriever import Retriever
from trustnet.schemas import ClaimExample

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def compute_ndcg_at_k(retrieved_doc_ids: List[str], gold_doc_ids: Set[str], k: int = 10) -> float:
    """
    Computes Normalized Discounted Cumulative Gain at rank k (binary relevance).
    """
    if not gold_doc_ids:
        return 0.0

    dcg = 0.0
    for idx, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        if doc_id in gold_doc_ids:
            dcg += 1.0 / math.log2(idx + 1)

    ideal_hits = min(k, len(gold_doc_ids))
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))

    return dcg / idcg if idcg > 0.0 else 0.0


def evaluate_split(
    retriever: Retriever,
    claims: List[ClaimExample],
    modes: List[str] = ("bm25", "dense", "hybrid", "hybrid_rerank"),
    max_k: int = 100,
) -> Dict[str, Any]:
    """
    Evaluates retrieval modes across all claims that have gold evidence.
    """
    recall_cutoffs = [1, 5, 10, 20, 50, 100]
    metrics_by_mode: Dict[str, Dict[str, Any]] = {}

    # Filter claims that HAVE gold evidence
    eval_claims = [c for c in claims if len(c.evidence_doc_ids) > 0]
    num_claims = len(eval_claims)

    if num_claims == 0:
        logger.warning("No claims with gold evidence found in the specified split.")
        return {}

    logger.info(f"Evaluating {num_claims} claims with gold evidence across modes: {modes}...")

    for mode in modes:
        logger.info(f"Running evaluation for mode: '{mode}'...")
        recalls_at_k: Dict[int, List[float]] = {k: [] for k in recall_cutoffs}
        mrrs: List[float] = []
        ndcgs_10: List[float] = []
        latencies_ms: List[float] = []

        for claim in eval_claims:
            gold_set = set(str(gid) for gid in claim.evidence_doc_ids)

            t0 = time.perf_counter()
            # Retrieve candidates up to max_k
            results = retriever.search(claim.claim, mode=mode, top_k=max_k)
            t1 = time.perf_counter()

            latencies_ms.append((t1 - t0) * 1000.0)
            retrieved_doc_ids = [r.doc_id for r in results]

            # 1. Recall@k
            for k in recall_cutoffs:
                top_k_retrieved = set(retrieved_doc_ids[:k])
                num_hits = len(top_k_retrieved.intersection(gold_set))
                recalls_at_k[k].append(num_hits / len(gold_set))

            # 2. MRR (Mean Reciprocal Rank of first gold hit)
            rr = 0.0
            for rank_idx, doc_id in enumerate(retrieved_doc_ids, start=1):
                if doc_id in gold_set:
                    rr = 1.0 / rank_idx
                    break
            mrrs.append(rr)

            # 3. nDCG@10
            ndcgs_10.append(compute_ndcg_at_k(retrieved_doc_ids, gold_set, k=10))

        # Aggregate averages
        mode_metrics: Dict[str, float] = {
            f"Recall@{k}": float(sum(recalls_at_k[k]) / num_claims)
            for k in recall_cutoffs
        }
        mode_metrics["MRR"] = float(sum(mrrs) / num_claims)
        mode_metrics["nDCG@10"] = float(sum(ndcgs_10) / num_claims)
        mode_metrics["avg_latency_ms"] = float(sum(latencies_ms) / num_claims)

        metrics_by_mode[mode] = mode_metrics

    return metrics_by_mode


def format_markdown_table(metrics_by_mode: Dict[str, Dict[str, Any]], split_name: str) -> str:
    """Formats evaluation results into a clean GitHub Flavored Markdown table."""
    headers = [
        "Mode", "Recall@1", "Recall@5", "Recall@10", "Recall@20",
        "Recall@50", "Recall@100", "MRR", "nDCG@10", "Latency (ms)"
    ]
    lines = [
        f"### Retrieval Engine Evaluation ({split_name.upper()} split)",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for mode, m in metrics_by_mode.items():
        row = [
            f"`{mode}`",
            f"{m['Recall@1']:.3f}",
            f"{m['Recall@5']:.3f}",
            f"{m['Recall@10']:.3f}",
            f"{m['Recall@20']:.3f}",
            f"{m['Recall@50']:.3f}",
            f"{m['Recall@100']:.3f}",
            f"{m['MRR']:.3f}",
            f"{m['nDCG@10']:.3f}",
            f"{m['avg_latency_ms']:.1f}",
        ]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="TrustNet Stage 3 Retrieval Evaluation CLI")
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Split to evaluate on (default: 'val').",
    )
    parser.add_argument(
        "--final",
        action="store_true",
        default=False,
        help="Required safety guard flag to evaluate on the 'test' split.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stage3_retrieval.yaml",
        help="Path to retrieval configuration YAML.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/stage3",
        help="Output directory for metrics and tables.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["bm25", "dense", "hybrid", "hybrid_rerank"],
        help="Retrieval modes to evaluate.",
    )

    args = parser.parse_args()

    # SAFETY GUARD: Strict prohibition against evaluating test split without --final
    if args.split.lower() == "test" and not args.final:
        logger.error(
            "GUARD TRIGGERED: Attempted to run evaluation on 'test' split without the '--final' flag. "
            "To prevent data leakage and benchmark contamination, test set evaluation is strictly guarded."
        )
        sys.exit(
            "ERROR: Cannot run evaluation on 'test' without --final flag. "
            "Perform hyperparameter tuning and model comparison strictly on 'val'."
        )

    split_file = Path(f"data/processed/{args.split}.jsonl")
    if not split_file.is_file():
        raise FileNotFoundError(f"Split file does not exist: {split_file}")

    logger.info(f"Loading split: {split_file}...")
    claims = load_claims(split_file)

    retriever = Retriever(config_or_path=args.config)
    for m in args.modes:
        retriever._ensure_indices_loaded(m)

    metrics = evaluate_split(retriever, claims, modes=args.modes)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Write JSON metrics
    json_path = out_dir / f"retrieval_metrics_{args.split}.json"
    payload = {
        "split": args.split,
        "num_evidenced_claims": len([c for c in claims if len(c.evidence_doc_ids) > 0]),
        "metrics": metrics,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Saved evaluation metrics JSON to: {json_path}")

    # 2. Write Markdown table
    md_table = format_markdown_table(metrics, args.split)
    md_path = out_dir / f"retrieval_metrics_{args.split}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_table + "\n")
    logger.info(f"Saved evaluation markdown table to: {md_path}")

    print("\n" + md_table + "\n")


if __name__ == "__main__":
    main()
