"""
Multilingual retrieval evaluation for TrustNet Stage 9.

Evaluates retrieval quality (Recall@1, 5, 10, 20, 50, MRR, nDCG@10, and latency)
for English, Hindi, and Hinglish queries against:
(a) English corpus (data/corpus/corpus_v1.jsonl)
(b) India corpus (data/corpus/corpus_india.jsonl)

Across strategies:
- native: queries searched as-is (leverages multilingual dense representations like BGE-M3).
- translate_test: non-English queries translated to English before retrieval.
- hybrid: fuses candidate lists from native and translated queries via Reciprocal Rank Fusion (RRF).

Includes safety guards: dev split (30% seed 42) evaluated unless --final is specified.
"""

import argparse
import json
import logging
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

from trustnet.lang.detect import detect_language
from trustnet.lang.normalize import normalize_text
from trustnet.lang.translate import Translator
from trustnet.retrieval.fusion import reciprocal_rank_fusion
from trustnet.retrieval.retriever import Retriever
from trustnet.schemas import Evidence

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def compute_ndcg_at_k(retrieved_doc_ids: List[str], gold_doc_ids: Set[str], k: int = 10) -> float:
    """Computes Normalized Discounted Cumulative Gain at rank k (binary relevance)."""
    if not gold_doc_ids:
        return 0.0

    dcg = 0.0
    for idx, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        if doc_id in gold_doc_ids:
            dcg += 1.0 / math.log2(idx + 1)

    ideal_hits = min(k, len(gold_doc_ids))
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0.0 else 0.0


def load_parallel_claims(
    path: Path,
    is_final: bool = False,
    dev_ratio: float = 0.3,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Loads parallel evaluation triplets (en, hi, hinglish) from JSONL.
    Enforces split discipline: without --final, restricts to dev subset (30% fixed seed).
    """
    if not path.is_file():
        raise FileNotFoundError(f"Parallel evaluation set not found at: {path}")

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if not is_final:
        # Fixed deterministic pseudo-random dev sample
        import random
        rng = random.Random(seed)
        shuffled = list(records)
        rng.shuffle(shuffled)
        dev_size = max(1, int(len(shuffled) * dev_ratio))
        logger.info(f"Dev mode active (--final not set): evaluating on {dev_size}/{len(records)} claims.")
        return shuffled[:dev_size]

    logger.info(f"FINAL mode active (--final passed): evaluating on full parallel benchmark of {len(records)} claims.")
    return records


def get_retriever_for_corpus(
    corpus_path: Path,
    index_dir: Path,
    passages_path: Path,
    modes: Optional[List[str]] = None,
) -> Retriever:
    """Initializes or builds a Retriever instance configured for a specific corpus."""
    base_cfg_path = Path("configs/stage3_retrieval.yaml")
    cfg: Dict[str, Any] = {}
    if base_cfg_path.is_file():
        with open(base_cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    cfg["corpus_path"] = str(corpus_path)
    cfg["index_dir"] = str(index_dir)
    cfg["passages_path"] = str(passages_path)

    retriever = Retriever(config_or_path=cfg)
    # Ensure BM25 index is initialized
    retriever.build_or_load_bm25()

    needs_dense = modes is None or any(m in ("dense", "hybrid", "hybrid_rerank") for m in modes)
    if needs_dense:
        try:
            retriever.build_or_load_dense()
        except Exception as exc:
            logger.warning(
                f"Could not load/build dense index for {corpus_path}: {exc}. "
                "Dense/hybrid queries will fall back to BM25."
            )
    return retriever


def evaluate_query(
    query_text: str,
    lang_form: str,
    strategy: str,
    retriever: Retriever,
    retrieval_mode: str,
    top_k: int = 50,
) -> Tuple[List[str], float]:
    """
    Executes retrieval under the chosen strategy and measures query execution latency.
    Returns: (list_of_retrieved_doc_ids, latency_ms)
    """
    t0 = time.perf_counter()

    # Graceful fallback to BM25 if dense index is not loaded
    effective_mode = retrieval_mode
    if effective_mode in ("dense", "hybrid", "hybrid_rerank"):
        if retriever.dense_index is None or retriever.dense_index.index is None:
            effective_mode = "bm25"

    if strategy == "native":
        # Search query text directly
        ev_items = retriever.search(query_text, mode=effective_mode, top_k=top_k)
        retrieved_ids = [e.doc_id for e in ev_items]

    elif strategy == "translate_test":
        # If non-English, translate to English first
        if lang_form == "en":
            effective_query = query_text
        else:
            translator = Translator.get_instance()
            # Normalize text before translation
            norm_res = normalize_text(query_text)
            effective_query = translator.translate(norm_res.normalized, src_lang="hi", tgt_lang="en")

        ev_items = retriever.search(effective_query, mode=effective_mode, top_k=top_k)
        retrieved_ids = [e.doc_id for e in ev_items]

    elif strategy == "hybrid":
        # Retrieve both native and translated candidates, then fuse via RRF
        ev_native = retriever.search(query_text, mode=effective_mode, top_k=top_k)

        if lang_form == "en":
            translated_query = query_text
        else:
            translator = Translator.get_instance()
            norm_res = normalize_text(query_text)
            translated_query = translator.translate(norm_res.normalized, src_lang="hi", tgt_lang="en")

        ev_trans = retriever.search(translated_query, mode=effective_mode, top_k=top_k)

        # Convert to passage dicts for RRF
        list1 = [{"doc_id": e.doc_id, "score": e.score or 1.0, "passage_id": e.passage_id, "text": e.text} for e in ev_native]
        list2 = [{"doc_id": e.doc_id, "score": e.score or 1.0, "passage_id": e.passage_id, "text": e.text} for e in ev_trans]

        fused = reciprocal_rank_fusion([list1, list2], rrf_k=60, top_k=top_k, key_field="doc_id")
        retrieved_ids = [p["doc_id"] for p in fused]

    else:
        raise ValueError(f"Unknown retrieval strategy: '{strategy}'")

    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000.0
    return retrieved_ids, latency_ms


def run_multilingual_retrieval_eval(
    parallel_set_path: Path = Path("data/eval/parallel_set.jsonl"),
    english_corpus_path: Path = Path("data/corpus/corpus_v1.jsonl"),
    india_corpus_path: Path = Path("data/corpus/corpus_india.jsonl"),
    output_dir: Path = Path("outputs/stage9"),
    is_final: bool = False,
    modes: Optional[List[str]] = None,
    strategies: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Runs full multilingual retrieval evaluation across all query forms, corpora, and strategies.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    claims = load_parallel_claims(parallel_set_path, is_final=is_final)

    modes = modes or ["bm25", "dense", "hybrid", "hybrid_rerank"]
    strategies = strategies or ["native", "translate_test", "hybrid"]
    recall_cutoffs = [1, 5, 10, 20, 50]

    corpora_specs = {
        "english_corpus": {
            "corpus_path": english_corpus_path,
            "index_dir": output_dir / "index_en",
            "passages_path": output_dir / "passages_en.jsonl",
        },
        "india_corpus": {
            "corpus_path": india_corpus_path,
            "index_dir": output_dir / "index_india",
            "passages_path": output_dir / "passages_india.jsonl",
        },
    }

    full_results: Dict[str, Any] = {
        "metadata": {
            "num_claims": len(claims),
            "is_final": is_final,
            "modes": modes,
            "strategies": strategies,
        },
        "corpora": {},
    }

    for corpus_key, c_info in corpora_specs.items():
        logger.info(f"\n================ Evaluating against {corpus_key} ================")
        retriever = get_retriever_for_corpus(
            c_info["corpus_path"],
            c_info["index_dir"],
            c_info["passages_path"],
            modes=modes,
        )

        corpus_eval: Dict[str, Any] = {}

        for strategy in strategies:
            corpus_eval[strategy] = {}

            for lang_form in ["en", "hi", "hinglish"]:
                corpus_eval[strategy][lang_form] = {}

                for mode in modes:
                    recalls: Dict[int, List[float]] = {k: [] for k in recall_cutoffs}
                    mrrs: List[float] = []
                    ndcgs: List[float] = []
                    latencies: List[float] = []

                    for claim in claims:
                        gold_set = set(claim.get("gold_evidence_ids", []))
                        if not gold_set:
                            continue

                        query_text = claim.get(lang_form, "")
                        retrieved_ids, lat_ms = evaluate_query(
                            query_text=query_text,
                            lang_form=lang_form,
                            strategy=strategy,
                            retriever=retriever,
                            retrieval_mode=mode,
                            top_k=50,
                        )

                        latencies.append(lat_ms)

                        # Recall@k
                        for k in recall_cutoffs:
                            top_k_hits = set(retrieved_ids[:k]).intersection(gold_set)
                            recalls[k].append(len(top_k_hits) / len(gold_set))

                        # MRR
                        rr = 0.0
                        for rank_idx, did in enumerate(retrieved_ids, start=1):
                            if did in gold_set:
                                rr = 1.0 / rank_idx
                                break
                        mrrs.append(rr)

                        # nDCG@10
                        ndcgs.append(compute_ndcg_at_k(retrieved_ids, gold_set, k=10))

                    num_eval = len(mrrs)
                    if num_eval > 0:
                        metrics_summary = {
                            "num_evaluated": num_eval,
                            "recall_at_1": float(sum(recalls[1]) / num_eval),
                            "recall_at_5": float(sum(recalls[5]) / num_eval),
                            "recall_at_10": float(sum(recalls[10]) / num_eval),
                            "recall_at_20": float(sum(recalls[20]) / num_eval),
                            "recall_at_50": float(sum(recalls[50]) / num_eval),
                            "mrr": float(sum(mrrs) / num_eval),
                            "ndcg_at_10": float(sum(ndcgs) / num_eval),
                            "mean_latency_ms": float(sum(latencies) / len(latencies)) if latencies else 0.0,
                        }
                    else:
                        metrics_summary = {
                            "num_evaluated": 0,
                            "recall_at_1": 0.0,
                            "recall_at_5": 0.0,
                            "recall_at_10": 0.0,
                            "recall_at_20": 0.0,
                            "recall_at_50": 0.0,
                            "mrr": 0.0,
                            "ndcg_at_10": 0.0,
                            "mean_latency_ms": 0.0,
                        }

                    corpus_eval[strategy][lang_form][mode] = metrics_summary

        full_results["corpora"][corpus_key] = corpus_eval

    # Write results.json
    results_path = output_dir / "retrieval_multilingual.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)
    logger.info(f"Multilingual retrieval evaluation saved to: {results_path}")

    # Print summary markdown table
    _print_summary_table(full_results)
    return full_results


def _print_summary_table(results: Dict[str, Any]) -> None:
    """Renders a clean Markdown table summarizing multilingual retrieval performance."""
    print("\n### Multilingual Retrieval Evaluation Summary\n")
    print("| Corpus | Strategy | Lang | Mode | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 | Latency (ms) |")
    print("|---|---|---|---|---|---|---|---|---|---|")

    for corpus, strat_dict in results.get("corpora", {}).items():
        for strat, lang_dict in strat_dict.items():
            for lang, mode_dict in lang_dict.items():
                for mode, m in mode_dict.items():
                    print(
                        f"| {corpus} | {strat} | {lang} | {mode} | "
                        f"{m['recall_at_1']:.3f} | {m['recall_at_5']:.3f} | {m['recall_at_10']:.3f} | "
                        f"{m['mrr']:.3f} | {m['ndcg_at_10']:.3f} | {m['mean_latency_ms']:.1f} |"
                    )
    print("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multilingual Retrieval Evaluation")
    parser.add_argument("--parallel_set", type=str, default="data/eval/parallel_set.jsonl", help="Parallel set path")
    parser.add_argument("--final", action="store_true", help="Evaluate full benchmark (safety guard)")
    parser.add_argument("--output_dir", type=str, default="outputs/stage9", help="Output directory")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["bm25", "dense", "hybrid", "hybrid_rerank"],
        help="Retrieval modes to evaluate (e.g. --modes bm25 or --modes bm25 dense)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_multilingual_retrieval_eval(
        parallel_set_path=Path(args.parallel_set),
        output_dir=Path(args.output_dir),
        is_final=args.final,
        modes=args.modes,
    )
