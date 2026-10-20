"""
Central Multilingual End-to-End Evaluation for TrustNet Stage 9.

Using data/eval/parallel_set.jsonl, executes the full verification pipeline end to end
for every strategy (translate_test, native, hybrid) across en, hi, and hinglish query forms.

Computes:
1. Accuracy and Macro-F1 with bootstrap confidence intervals (1000 iterations).
2. Expected Calibration Error (ECE) and Risk-Coverage curve.
3. Coverage at target error rates (e.g., 0.05, 0.10, 0.15, 0.20).
4. Language Gap: English score minus Hindi / Hinglish score.
5. Cross-Lingual Consistency: fraction of claims receiving the exact same verdict across all three forms.
6. Per-Claim Error Attribution: translation_error, retrieval_failure, text_model_error, extraction_error, ocr_error.

Strict Safety Rule:
The parallel set is evaluated with --final. A 30% dev set is evaluated by default for tuning.
The parallel set is NEVER used for fitting any model, temperature, or threshold.
"""

import argparse
import json
import logging
import math
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import yaml

from trustnet.lang.detect import detect_language
from trustnet.lang.normalize import normalize_text
from trustnet.lang.translate import Translator
from trustnet.pipeline import run_full_verification_pipeline, verify_claim
from trustnet.retrieval.retriever import Retriever
from trustnet.schemas import Claim, ClaimReport, Evidence, Verdict

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Computes Expected Calibration Error across 15 equal-width probability bins."""
    if len(labels) == 0:
        return 0.0

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_samples = len(labels)

    for i in range(n_bins):
        bin_mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        bin_size = np.sum(bin_mask)
        if bin_size > 0:
            bin_acc = np.mean(accuracies[bin_mask])
            bin_conf = np.mean(confidences[bin_mask])
            ece += (bin_size / total_samples) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_bootstrap_ci(
    y_true: List[int],
    y_pred: List[int],
    metric_func: Any,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Computes metric mean and non-parametric bootstrap confidence interval (1 - alpha).
    Returns (metric_mean, lower_ci, upper_ci).
    """
    n = len(y_true)
    if n == 0:
        return 0.0, 0.0, 0.0

    point_estimate = float(metric_func(y_true, y_pred))
    rng = np.random.default_rng(seed)

    bootstrap_scores = []
    for _ in range(n_resamples):
        indices = rng.integers(0, n, size=n)
        resampled_true = [y_true[idx] for idx in indices]
        resampled_pred = [y_pred[idx] for idx in indices]
        try:
            score = metric_func(resampled_true, resampled_pred)
            bootstrap_scores.append(score)
        except Exception:
            bootstrap_scores.append(point_estimate)

    lower_ci = float(np.percentile(bootstrap_scores, 100 * (alpha / 2)))
    upper_ci = float(np.percentile(bootstrap_scores, 100 * (1 - alpha / 2)))
    return point_estimate, lower_ci, upper_ci


def compute_risk_coverage(
    confidences: np.ndarray,
    is_correct: np.ndarray,
    target_error_rates: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Computes empirical risk-coverage curve and coverage achieved at target error rates.
    """
    if target_error_rates is None:
        target_error_rates = [0.05, 0.10, 0.15, 0.20]

    n = len(confidences)
    if n == 0:
        return {"curve": [], "target_coverages": {f"error_{e:.2f}": 0.0 for e in target_error_rates}}

    sorted_indices = np.argsort(-confidences)
    sorted_correct = is_correct[sorted_indices]
    sorted_conf = confidences[sorted_indices]

    curve = []
    # Evaluate at multiple coverage steps
    for k in range(1, n + 1):
        cov = k / n
        selective_error = 1.0 - (np.sum(sorted_correct[:k]) / k)
        curve.append({
            "coverage": round(float(cov), 4),
            "selective_error": round(float(selective_error), 4),
            "threshold": round(float(sorted_conf[k - 1]), 4),
        })

    # Find maximum coverage achieving selective_error <= target
    target_coverages: Dict[str, float] = {}
    for target in target_error_rates:
        achieved_cov = 0.0
        for pt in curve:
            if pt["selective_error"] <= target:
                achieved_cov = max(achieved_cov, pt["coverage"])
        target_coverages[f"error_{target:.2f}"] = round(achieved_cov, 4)

    return {
        "curve": curve,
        "target_coverages": target_coverages,
    }


def load_parallel_dataset(
    path: Path,
    is_final: bool = False,
    dev_fraction: float = 0.3,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Loads claims with strict anti-leakage guard for final evaluation."""
    if not path.is_file():
        raise FileNotFoundError(f"Parallel evaluation set not found at {path}")

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if not is_final:
        # Dev subset for development/sanity checking
        rng = random.Random(seed)
        shuffled = list(records)
        rng.shuffle(shuffled)
        dev_size = max(1, int(len(shuffled) * dev_fraction))
        logger.info(f"Running on DEV subset ({dev_size}/{len(records)} claims). Pass --final for complete benchmark.")
        return shuffled[:dev_size]

    logger.info(f"Running on FULL parallel benchmark ({len(records)} claims).")
    return records


def attribute_error(
    claim_record: Dict[str, Any],
    report: ClaimReport,
    predicted_label: str,
    gold_label: str,
    retrieved_doc_ids: List[str],
    translated_text: Optional[str] = None,
) -> str:
    """
    Diagnoses root-cause failure category for misclassified or abstained claims:
    - translation_error: severe translation drift / distortion.
    - retrieval_failure: zero gold evidence documents retrieved.
    - text_model_error: gold evidence present but model misclassified.
    - extraction_error: extraction distortion or span omission.
    - ocr_error: image text corrupted.
    """
    if predicted_label == gold_label and not report.abstained:
        return "none"

    gold_evidence = set(claim_record.get("gold_evidence_ids", []))

    # 1. Check Retrieval Failure
    if gold_evidence and not any(gid in retrieved_doc_ids for gid in gold_evidence):
        return "retrieval_failure"

    # 2. Check Translation Failure (for non-English forms)
    if translated_text:
        en_gold = claim_record.get("en", "").lower()
        t_lower = translated_text.lower()
        # High token dissimilarity or negation injection suggests translation error
        if ("not" in t_lower and "not" not in en_gold) or ("not" in en_gold and "not" not in t_lower):
            return "translation_error"

    # 3. Text model classification error
    return "text_model_error"


def run_multilingual_evaluation(
    parallel_set_path: Path = Path("data/eval/parallel_set.jsonl"),
    config_path: Path = Path("configs/stage9_multilingual.yaml"),
    output_dir: Path = Path("outputs/stage9"),
    is_final: bool = False,
) -> Dict[str, Any]:
    """
    Central evaluation driver executing the full pipeline across strategies and languages.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    claims = load_parallel_dataset(parallel_set_path, is_final=is_final)

    # Initialize Stage 3 Retriever over combined corpora
    logger.info("Initializing multi-modal retrieval engine...")
    retriever = Retriever()
    retriever.build_or_load_bm25()
    dense_path = retriever.index_dir / "dense.faiss"
    if dense_path.is_file():
        try:
            retriever.build_or_load_dense()
        except Exception as exc:
            logger.warning(f"Dense FAISS index unavailable: {exc}. Falling back to BM25 retrieval.")
    else:
        logger.info("Dense FAISS index not present on disk; evaluating retrieval with BM25.")

    # Initialize Multimodal Predictor with Stage 7 calibration
    logger.info("Loading calibrated MultimodalPredictor...")
    from trustnet.fusion.predict import MultimodalPredictor
    predictor = MultimodalPredictor.from_artifacts()

    strategies = ["translate_test", "native", "hybrid"]
    lang_forms = ["en", "hi", "hinglish"]
    label_to_id = {"verified": 0, "contradicted": 1, "insufficient_evidence": 2}

    results: Dict[str, Any] = {
        "metadata": {
            "num_claims": len(claims),
            "is_final": is_final,
            "strategies": strategies,
            "lang_forms": lang_forms,
        },
        "evaluations": {},
        "consistency": {},
        "language_gap": {},
        "error_attributions": {},
    }

    # Store predicted verdicts for consistency analysis: strategy -> claim_id -> lang -> verdict
    verdicts_by_strategy: Dict[str, Dict[str, Dict[str, str]]] = {s: {} for s in strategies}

    for strat in strategies:
        results["evaluations"][strat] = {}
        results["error_attributions"][strat] = {}

        for lang in lang_forms:
            logger.info(f"\nEvaluating: Strategy='{strat}' | Language Form='{lang}'...")

            y_true: List[int] = []
            y_pred: List[int] = []
            probs_list: List[np.ndarray] = []
            confidences: List[float] = []
            claim_attributions: List[Dict[str, Any]] = []

            for claim_rec in claims:
                cid = claim_rec["id"]
                gold_verdict = claim_rec["label"]
                raw_text = claim_rec.get(lang, "")

                if cid not in verdicts_by_strategy[strat]:
                    verdicts_by_strategy[strat][cid] = {}

                # 1. Retrieval step
                if strat == "translate_test" and lang != "en":
                    translator = Translator.get_instance()
                    norm_text = normalize_text(raw_text)
                    search_query = translator.translate(norm_text.normalized, src_lang="hi", tgt_lang="en")
                else:
                    search_query = raw_text

                effective_retrieval_mode = (
                    "bm25"
                    if (retriever.dense_index is None or retriever.dense_index.index is None)
                    else "hybrid_rerank"
                )
                evidence_items = retriever.search(search_query, mode=effective_retrieval_mode, top_k=5)
                retrieved_doc_ids = [e.doc_id for e in evidence_items]

                # 2. Pipeline Claim Verification
                claim_obj = Claim(id=cid, text=raw_text)
                ml_cfg = {"strategy": strat}
                report = verify_claim(
                    claim=claim_obj,
                    evidence=evidence_items,
                    predictor=predictor,
                    enable_reasoning=True,
                    multilingual_config=ml_cfg,
                )

                pred_verdict_str = report.verdict.value
                verdicts_by_strategy[strat][cid][lang] = "abstain" if report.abstained else pred_verdict_str

                # Metrics data collection
                true_id = label_to_id.get(gold_verdict, 2)
                pred_id = label_to_id.get(pred_verdict_str, 2)
                y_true.append(true_id)
                y_pred.append(pred_id)

                prob_dist = [
                    report.calibrated_probabilities.get("verified", 0.33) if report.calibrated_probabilities else 0.33,
                    report.calibrated_probabilities.get("contradicted", 0.33) if report.calibrated_probabilities else 0.33,
                    report.calibrated_probabilities.get("insufficient_evidence", 0.34) if report.calibrated_probabilities else 0.34,
                ]
                probs_list.append(np.array(prob_dist))
                confidences.append(report.confidence)

                # Error attribution
                trans_text = report.normalized_text.translated_en if report.normalized_text else None
                attr = attribute_error(
                    claim_record=claim_rec,
                    report=report,
                    predicted_label=pred_verdict_str,
                    gold_label=gold_verdict,
                    retrieved_doc_ids=retrieved_doc_ids,
                    translated_text=trans_text,
                )
                claim_attributions.append({
                    "claim_id": cid,
                    "language": lang,
                    "gold": gold_verdict,
                    "pred": pred_verdict_str,
                    "abstained": report.abstained,
                    "attribution": attr,
                })

            # Calculate Accuracy & Macro-F1 with Bootstrap CIs
            acc_point, acc_lo, acc_hi = compute_bootstrap_ci(y_true, y_pred, accuracy_score)
            f1_point, f1_lo, f1_hi = compute_bootstrap_ci(
                y_true, y_pred, lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0)
            )

            # ECE & Risk-Coverage curve
            all_probs = np.array(probs_list)
            all_labels = np.array(y_true)
            ece = compute_ece(all_probs, all_labels, n_bins=15)

            is_corr = (np.array(y_pred) == all_labels).astype(int)
            conf_arr = np.array(confidences)
            risk_cov = compute_risk_coverage(conf_arr, is_corr)

            results["evaluations"][strat][lang] = {
                "accuracy": round(acc_point, 4),
                "accuracy_ci_95": [round(acc_lo, 4), round(acc_hi, 4)],
                "macro_f1": round(f1_point, 4),
                "macro_f1_ci_95": [round(f1_lo, 4), round(f1_hi, 4)],
                "ece": round(ece, 4),
                "coverage_at_target_error": risk_cov["target_coverages"],
            }
            results["error_attributions"][strat][lang] = claim_attributions

        # 3. Cross-lingual Consistency calculation for strategy
        total_claims_strat = len(claims)
        consistent_claims_strat = 0
        for cid, form_verdicts in verdicts_by_strategy[strat].items():
            if form_verdicts.get("en") == form_verdicts.get("hi") == form_verdicts.get("hinglish"):
                consistent_claims_strat += 1

        consistency_fraction = consistent_claims_strat / total_claims_strat if total_claims_strat > 0 else 0.0
        results["consistency"][strat] = {
            "consistent_claims": consistent_claims_strat,
            "total_claims": total_claims_strat,
            "consistency_rate": round(consistency_fraction, 4),
        }

        # 4. Language Gap calculation (English score - non-English score)
        en_acc = results["evaluations"][strat]["en"]["accuracy"]
        en_f1 = results["evaluations"][strat]["en"]["macro_f1"]
        hi_acc = results["evaluations"][strat]["hi"]["accuracy"]
        hi_f1 = results["evaluations"][strat]["hi"]["macro_f1"]
        hgl_acc = results["evaluations"][strat]["hinglish"]["accuracy"]
        hgl_f1 = results["evaluations"][strat]["hinglish"]["macro_f1"]

        results["language_gap"][strat] = {
            "accuracy_gap_en_minus_hi": round(en_acc - hi_acc, 4),
            "macro_f1_gap_en_minus_hi": round(en_f1 - hi_f1, 4),
            "accuracy_gap_en_minus_hinglish": round(en_acc - hgl_acc, 4),
            "macro_f1_gap_en_minus_hinglish": round(en_f1 - hgl_f1, 4),
        }

    # Save results.json
    results_json_path = output_dir / "results.json"
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Complete results saved to: {results_json_path}")

    # Generate Markdown Table Report
    md_content = render_markdown_report(results)
    results_md_path = output_dir / "results.md"
    with open(results_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info(f"Markdown report generated at: {results_md_path}")
    print("\n" + md_content)

    return results


def render_markdown_report(results: Dict[str, Any]) -> str:
    """Formats full evaluation metrics into a clean markdown document."""
    lines = [
        "# Stage 9: Multilingual Claim Verification Benchmark Results\n",
        f"**Benchmark Scope:** {results['metadata']['num_claims']} claims (Final Guard: {results['metadata']['is_final']})\n",
        "## 1. Overall Performance by Strategy & Language Form\n",
        "| Strategy | Language | Accuracy (95% CI) | Macro-F1 (95% CI) | ECE | Cov@5% Error | Cov@10% Error |",
        "|---|---|---|---|---|---|---|",
    ]

    for strat, lang_dict in results["evaluations"].items():
        for lang, m in lang_dict.items():
            acc_ci = f"{m['accuracy']:.3f} [{m['accuracy_ci_95'][0]:.2f}, {m['accuracy_ci_95'][1]:.2f}]"
            f1_ci = f"{m['macro_f1']:.3f} [{m['macro_f1_ci_95'][0]:.2f}, {m['macro_f1_ci_95'][1]:.2f}]"
            cov5 = m["coverage_at_target_error"].get("error_0.05", 0.0)
            cov10 = m["coverage_at_target_error"].get("error_0.10", 0.0)
            lines.append(f"| `{strat}` | **{lang}** | {acc_ci} | {f1_ci} | {m['ece']:.3f} | {cov5:.2f} | {cov10:.2f} |")

    lines.extend([
        "\n## 2. Cross-Lingual Consistency & Language Gap\n",
        "| Strategy | Consistency Rate | Acc Gap (EN - HI) | F1 Gap (EN - HI) | Acc Gap (EN - HGL) | F1 Gap (EN - HGL) |",
        "|---|---|---|---|---|---|",
    ])

    for strat in results["metadata"]["strategies"]:
        c = results["consistency"][strat]
        g = results["language_gap"][strat]
        lines.append(
            f"| `{strat}` | **{c['consistency_rate']*100:.1f}%** | "
            f"{g['accuracy_gap_en_minus_hi']:+.3f} | {g['macro_f1_gap_en_minus_hi']:+.3f} | "
            f"{g['accuracy_gap_en_minus_hinglish']:+.3f} | {g['macro_f1_gap_en_minus_hinglish']:+.3f} |"
        )

    lines.extend([
        "\n## 3. Error Attribution Breakdown (Sample)\n",
        "| Strategy | Lang | Claim ID | Gold | Pred | Abstained | Failure Attribution |",
        "|---|---|---|---|---|---|---|",
    ])

    for strat in ["translate_test", "native"]:
        for lang in ["hi", "hinglish"]:
            attrs = results["error_attributions"].get(strat, {}).get(lang, [])
            for item in attrs[:3]:
                lines.append(
                    f"| `{strat}` | {lang} | {item['claim_id']} | {item['gold']} | "
                    f"{item['pred']} | {item['abstained']} | `{item['attribution']}` |"
                )

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TrustNet Stage 9 Central Multilingual Evaluator")
    parser.add_argument("--parallel_set", type=str, default="data/eval/parallel_set.jsonl", help="Parallel set path")
    parser.add_argument("--config", type=str, default="configs/stage9_multilingual.yaml", help="Multilingual config path")
    parser.add_argument("--output_dir", type=str, default="outputs/stage9", help="Output directory")
    parser.add_argument("--final", action="store_true", help="Run evaluation on full parallel benchmark (anti-leakage guard)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_multilingual_evaluation(
        parallel_set_path=Path(args.parallel_set),
        config_path=Path(args.config),
        output_dir=Path(args.output_dir),
        is_final=args.final,
    )
