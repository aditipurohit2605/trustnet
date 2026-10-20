"""
CLI evaluation script for Stage 4 claim extraction.

Evaluates ClaimExtractor over data/eval/extraction_gold.jsonl using a two-step semantic match
with dense embeddings, flags borderline pairs to outputs/stage4/needs_review.jsonl,
computes precision, recall, empty-output accuracy, and latency, and outputs metrics.json
along with a detailed per-message comparison file.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from trustnet.schemas import ExtractedClaim
from trustnet.text.extraction.extractor import ClaimExtractor

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def compute_pairwise_cosine_similarity(
    texts_a: List[str],
    texts_b: List[str],
    embed_model: SentenceTransformer,
) -> np.ndarray:
    """
    Computes a pairwise cosine similarity matrix between two lists of texts.
    Returns matrix of shape (len(texts_a), len(texts_b)).
    """
    if not texts_a or not texts_b:
        return np.zeros((len(texts_a), len(texts_b)), dtype=np.float32)

    embs_a = embed_model.encode(texts_a, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
    embs_b = embed_model.encode(texts_b, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)

    # Unit vectors -> inner product is cosine similarity
    sim_matrix = np.matmul(embs_a, embs_b.T)
    return sim_matrix


def evaluate_extraction(
    gold_records: List[Dict[str, Any]],
    extractor: ClaimExtractor,
    embed_model: SentenceTransformer,
    sim_threshold: float = 0.85,
    borderline_min: float = 0.70,
    output_dir: Path = Path("outputs/stage4"),
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Runs extraction and evaluates claim precision, recall, empty accuracy, and latency.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    total_gold_claims = 0
    total_pred_claims = 0
    total_true_positives = 0

    gold_empty_messages_count = 0
    correct_empty_messages_count = 0

    borderline_pairs: List[Dict[str, Any]] = []
    comparisons: List[Dict[str, Any]] = []
    latencies: List[float] = []
    malformed_count = 0

    for idx, record in enumerate(gold_records, start=1):
        msg_id = record.get("id", f"msg_{idx}")
        raw_text = record["text"]
        gold_claims_data = record.get("gold_claims", [])
        gold_claim_texts = [gc["text"] for gc in gold_claims_data]
        total_gold_claims += len(gold_claim_texts)

        if len(gold_claim_texts) == 0:
            gold_empty_messages_count += 1

        # Time extraction latency
        t0 = time.perf_counter()
        result = extractor.extract(raw_text)
        t1 = time.perf_counter()
        latencies.append(t1 - t0)

        if result.error:
            malformed_count += 1

        pred_claim_texts = [c.text for c in result.claims]
        total_pred_claims += len(pred_claim_texts)

        if len(gold_claim_texts) == 0 and len(pred_claim_texts) == 0:
            correct_empty_messages_count += 1

        # Match predicted claims against gold claims using dense embeddings
        matched_gold_indices = set()
        msg_matches = []

        if pred_claim_texts and gold_claim_texts:
            sim_matrix = compute_pairwise_cosine_similarity(pred_claim_texts, gold_claim_texts, embed_model)

            for p_idx, pred_text in enumerate(pred_claim_texts):
                best_g_idx = int(np.argmax(sim_matrix[p_idx]))
                best_sim = float(sim_matrix[p_idx, best_g_idx])
                gold_text = gold_claim_texts[best_g_idx]

                if best_sim >= sim_threshold:
                    if best_g_idx not in matched_gold_indices:
                        matched_gold_indices.add(best_g_idx)
                        total_true_positives += 1
                        msg_matches.append({
                            "pred": pred_text,
                            "gold": gold_text,
                            "similarity": best_sim,
                            "status": "matched",
                        })
                elif best_sim >= borderline_min:
                    # Step 2: Flag borderline pairs for manual review
                    borderline_record = {
                        "message_id": msg_id,
                        "original_text": raw_text,
                        "pred_claim": pred_text,
                        "gold_claim": gold_text,
                        "cosine_similarity": round(best_sim, 4),
                        "note": "Borderline semantic similarity - requires human judgment.",
                    }
                    borderline_pairs.append(borderline_record)
                    msg_matches.append({
                        "pred": pred_text,
                        "gold": gold_text,
                        "similarity": best_sim,
                        "status": "borderline",
                    })
                else:
                    msg_matches.append({
                        "pred": pred_text,
                        "gold": gold_text,
                        "similarity": best_sim,
                        "status": "unmatched",
                    })

        comparisons.append({
            "id": msg_id,
            "text": raw_text,
            "gold_claims": gold_claim_texts,
            "pred_claims": pred_claim_texts,
            "matches": msg_matches,
            "latency_seconds": round(t1 - t0, 3),
            "error": result.error,
        })

    # Metric computations
    precision = (total_true_positives / total_pred_claims) if total_pred_claims > 0 else 0.0
    recall = (total_true_positives / total_gold_claims) if total_gold_claims > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    empty_accuracy = (
        (correct_empty_messages_count / gold_empty_messages_count)
        if gold_empty_messages_count > 0 else 1.0
    )
    avg_claims_per_msg = (total_pred_claims / len(gold_records)) if gold_records else 0.0
    avg_latency = (sum(latencies) / len(latencies)) if latencies else 0.0
    malformed_rate = (malformed_count / len(gold_records)) if gold_records else 0.0

    metrics = {
        "total_messages": len(gold_records),
        "total_gold_claims": total_gold_claims,
        "total_predicted_claims": total_pred_claims,
        "true_positives": total_true_positives,
        "claim_precision": round(precision, 4),
        "claim_recall": round(recall, 4),
        "claim_f1": round(f1, 4),
        "correct_empty_output_share": round(empty_accuracy, 4),
        "average_claims_per_message": round(avg_claims_per_msg, 2),
        "average_latency_seconds": round(avg_latency, 3),
        "malformed_output_rate": round(malformed_rate, 4),
        "borderline_pairs_count": len(borderline_pairs),
        "similarity_threshold": sim_threshold,
        "borderline_min": borderline_min,
    }

    return metrics, borderline_pairs, comparisons


def main() -> None:
    parser = argparse.ArgumentParser(description="TrustNet Stage 4 Claim Extraction Evaluator")
    parser.add_argument(
        "--gold_path",
        type=str,
        default="data/eval/extraction_gold.jsonl",
        help="Path to gold extraction evaluation JSONL file.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stage4_extraction.yaml",
        help="Path to extraction configuration YAML.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/stage4",
        help="Directory where metrics.json and review files will be written.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Cosine similarity threshold for automated claim match.",
    )
    parser.add_argument(
        "--borderline_min",
        type=float,
        default=0.70,
        help="Lower threshold below which pairs are considered unmatched.",
    )
    parser.add_argument(
        "--dense_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Dense embedding model used for semantic match scoring.",
    )

    args = parser.parse_args()

    gold_file = Path(args.gold_path)
    if not gold_file.is_file():
        raise FileNotFoundError(f"Gold evaluation file not found: {gold_file}")

    gold_records = []
    with open(gold_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                gold_records.append(json.loads(line))

    logger.info(f"Loaded {len(gold_records)} gold evaluation messages from {gold_file}.")

    logger.info(f"Initializing ClaimExtractor with config: {args.config}...")
    extractor = ClaimExtractor(config_or_path=args.config)

    logger.info(f"Loading dense embedding model for semantic evaluation: {args.dense_model}...")
    embed_model = SentenceTransformer(args.dense_model)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics, borderline_pairs, comparisons = evaluate_extraction(
        gold_records=gold_records,
        extractor=extractor,
        embed_model=embed_model,
        sim_threshold=args.threshold,
        borderline_min=args.borderline_min,
        output_dir=output_dir,
    )

    # 1. Save metrics.json
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved extraction evaluation metrics to: {metrics_path}")

    # 2. Save borderline pairs needing human review
    borderline_path = output_dir / "needs_review.jsonl"
    with open(borderline_path, "w", encoding="utf-8") as f:
        for bp in borderline_pairs:
            f.write(json.dumps(bp, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(borderline_pairs)} borderline pairs to: {borderline_path}")

    # 3. Save detailed per-message comparisons
    comparison_path = output_dir / "comparison.jsonl"
    with open(comparison_path, "w", encoding="utf-8") as f:
        for comp in comparisons:
            f.write(json.dumps(comp, ensure_ascii=False) + "\n")
    logger.info(f"Saved per-message comparison details to: {comparison_path}")

    print("\n" + "=" * 50)
    print("STAGE 4 CLAIM EXTRACTION EVALUATION RESULTS")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:30s}: {v}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
