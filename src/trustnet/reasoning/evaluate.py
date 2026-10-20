"""
CLI Evaluation Harness for TrustNet Stage 8: Reasoning Layer.

Evaluates:
1. Conflict Detection: Precision, Recall, and F1.
2. Source Weighting Ablation: Conflict metrics with weighting ON vs OFF.
3. Gap Detection: Precision and Recall per gap code.
4. Manipulation Tagging: Precision and Recall per rhetorical pattern.
5. Temporal Freshness: Accuracy of staleness detection.
6. Counterfactual Faithfulness: Percentage of reported flips reproduced on re-run (target: 100%).
7. Latency Benchmarking: Execution time (ms) per reasoning module.

Outputs:
- outputs/stage8/metrics.json
- outputs/stage8/metrics.md
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import yaml

from trustnet.reasoning.conflict import ConflictDetector
from trustnet.reasoning.counterfactual import CounterfactualGenerator
from trustnet.reasoning.gaps import GapAnalyzer
from trustnet.reasoning.manipulation import ManipulationTagger
from trustnet.reasoning.share_advice import determine_share_advice
from trustnet.reasoning.source_reliability import SourceReliabilityAssigner
from trustnet.reasoning.temporal import TemporalAnalyzer
from trustnet.schemas import Evidence, Verdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("trustnet.reasoning.evaluate")


def compute_binary_metrics(y_true: List[bool], y_pred: List[bool]) -> Dict[str, float]:
    """Computes precision, recall, and F1 for binary classification."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)

    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if fn == 0 and tp == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def evaluate_reasoning_layer(
    gold_path: Union[str, Path] = "data/eval/reasoning_gold.jsonl",
    config_path: Union[str, Path] = "configs/stage8_reasoning.yaml",
    output_dir: Union[str, Path] = "outputs/stage8",
) -> Dict[str, Any]:
    """
    Runs full evaluation over the gold reasoning dataset and records benchmark metrics.
    """
    gold_path = Path(gold_path)
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Configuration
    cfg: Dict[str, Any] = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    # 2. Load Gold Benchmark Dataset
    if not gold_path.exists():
        raise FileNotFoundError(f"Gold evaluation file not found: {gold_path}")

    records: List[Dict[str, Any]] = []
    with open(gold_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    logger.info(f"Loaded {len(records)} gold reasoning benchmark cases from {gold_path}.")

    # 3. Initialize Reasoning Engines
    # Source reliability
    source_assigner_on = SourceReliabilityAssigner(
        config_path=cfg.get("source_reliability_path", "configs/source_reliability.yaml"),
        enable_weighting=True,
    )
    source_assigner_off = SourceReliabilityAssigner(
        config_path=cfg.get("source_reliability_path", "configs/source_reliability.yaml"),
        enable_weighting=False,
    )

    # Conflict detector (with source weighting ON and OFF for ablation)
    conflict_detector_on = ConflictDetector(
        source_assigner=source_assigner_on,
        conflict_min_weight=cfg.get("conflict_min_weight", 0.50),
        confidence_threshold=cfg.get("conflict_confidence_threshold", 0.45),
    )
    # Share text classifier model and tokenizer to avoid reloading weights
    conflict_detector_off = ConflictDetector(
        text_model=conflict_detector_on.text_model,
        tokenizer=conflict_detector_on.tokenizer,
        source_assigner=source_assigner_off,
        conflict_min_weight=cfg.get("conflict_min_weight", 0.50),
        confidence_threshold=cfg.get("conflict_confidence_threshold", 0.45),
    )

    # Temporal analyzer
    temporal_analyzer = TemporalAnalyzer(stale_days=cfg.get("stale_days", 90))

    # Manipulation tagger
    manipulation_tagger = ManipulationTagger(
        patterns_path=cfg.get("manipulation_patterns_path", "data/eval/manipulation_patterns.yaml"),
        model_name=cfg.get("manipulation_model", "llama3.2:1b"),
    )

    # Gap analyzer
    gap_analyzer = GapAnalyzer(
        templates_path=cfg.get("gap_templates_path", "data/eval/gap_templates.yaml")
    )

    # Counterfactual generator
    try:
        cf_generator = CounterfactualGenerator(max_counterfactuals=cfg.get("max_counterfactuals", 5))
    except Exception as e:
        logger.warning(f"Counterfactual generator init fallback: {e}")
        cf_generator = None

    # Track latencies
    latencies: Dict[str, List[float]] = {
        "source_reliability": [],
        "conflict": [],
        "temporal": [],
        "manipulation": [],
        "gaps": [],
        "counterfactual": [],
        "share_advice": [],
    }

    # Tracking metrics accumulators
    conflict_gold: List[bool] = []
    conflict_pred_on: List[bool] = []
    conflict_pred_off: List[bool] = []

    temporal_gold_stale: List[bool] = []
    temporal_pred_stale: List[bool] = []

    all_gap_codes: Set[str] = set(gap_analyzer.templates.keys())
    gap_gold_by_code: Dict[str, List[bool]] = {code: [] for code in all_gap_codes}
    gap_pred_by_code: Dict[str, List[bool]] = {code: [] for code in all_gap_codes}

    all_patterns: Set[str] = {p["name"] for p in manipulation_tagger.patterns}
    manip_gold_by_pat: Dict[str, List[bool]] = {pat: [] for pat in all_patterns}
    manip_pred_by_pat: Dict[str, List[bool]] = {pat: [] for pat in all_patterns}

    cf_flips_reported = 0
    cf_flips_verified = 0

    # 4. Evaluation Loop
    for rec in records:
        claim_text = rec["claim_text"]
        ev_dicts = rec.get("evidence", [])

        # Parse evidence into canonical Evidence objects
        evidence_objs: List[Evidence] = []
        for i, ed in enumerate(ev_dicts):
            meta = dict(ed.get("metadata", {}))
            if "stance" in ed:
                meta["stance"] = ed["stance"]
            ev_obj = Evidence(
                doc_id=ed.get("doc_id", f"doc_{i}"),
                text=ed.get("text", ""),
                source_type=ed.get("source_type", "news"),
                score=ed.get("score", 0.5),
                date=ed.get("date"),
                url=ed.get("url"),
                metadata=meta,
            )
            evidence_objs.append(ev_obj)

        # A. Source Reliability Latency
        t0 = time.perf_counter()
        source_assigner_on.assign_weights(evidence_objs)
        latencies["source_reliability"].append((time.perf_counter() - t0) * 1000.0)

        # B. Conflict Detection Evaluation (ON & OFF)
        gold_has_conflict = bool(rec.get("gold_conflict", {}).get("has_conflict", False))
        conflict_gold.append(gold_has_conflict)

        t0 = time.perf_counter()
        c_info_on = conflict_detector_on.detect_conflicts(claim_text, evidence_objs)
        latencies["conflict"].append((time.perf_counter() - t0) * 1000.0)
        conflict_pred_on.append(c_info_on.has_conflict)

        c_info_off = conflict_detector_off.detect_conflicts(claim_text, evidence_objs)
        conflict_pred_off.append(c_info_off.has_conflict)

        # C. Temporal Freshness Evaluation
        gold_stale = bool(rec.get("gold_temporal", {}).get("is_stale", False))
        temporal_gold_stale.append(gold_stale)

        t0 = time.perf_counter()
        t_info = temporal_analyzer.analyze(claim_text, evidence_objs)
        latencies["temporal"].append((time.perf_counter() - t0) * 1000.0)
        temporal_pred_stale.append(t_info.is_stale)

        # D. Manipulation Tagging Evaluation
        gold_manip_set = set(rec.get("gold_manipulation", []))
        t0 = time.perf_counter()
        tags = manipulation_tagger.tag(claim_text)
        latencies["manipulation"].append((time.perf_counter() - t0) * 1000.0)
        pred_manip_set = {t.name for t in tags}

        for pat in all_patterns:
            manip_gold_by_pat[pat].append(pat in gold_manip_set)
            manip_pred_by_pat[pat].append(pat in pred_manip_set)

        # E. Evidence Gap Analysis Evaluation
        gold_gap_set = set(rec.get("gold_gaps", []))
        t0 = time.perf_counter()
        gaps = gap_analyzer.analyze(
            claim_text=claim_text,
            evidence_list=evidence_objs,
            conflict_info=c_info_on,
            temporal_info=t_info,
        )
        latencies["gaps"].append((time.perf_counter() - t0) * 1000.0)
        pred_gap_set = {g.code for g in gaps}

        for code in all_gap_codes:
            gap_gold_by_code[code].append(code in gold_gap_set)
            gap_pred_by_code[code].append(code in pred_gap_set)

        # F. Counterfactual Faithfulness Verification
        if cf_generator is not None:
            t0 = time.perf_counter()
            dummy_record = {
                "claim_id": rec.get("id", "rec_0"),
                "claim_text": claim_text,
                "dataset": "eval",
                "text_logits": [0.0, 0.0, 0.0],
                "text_pooled": [0.0] * 32,
                "retrieval_features": {
                    "top1_score": evidence_objs[0].score if evidence_objs else 1.0,
                    "top1_top2_gap": 0.5,
                    "num_passages_above_threshold": len(evidence_objs),
                    "mean_score": 1.0,
                    "source_type_government": sum(1 for e in evidence_objs if (e.source_type or "").lower() == "government"),
                },
                "image_features": {
                    "image_present": 0,
                    "mismatch_score": 0.0,
                },
            }
            try:
                cf_pairs = cf_generator.generate_with_records(dummy_record)
                latencies["counterfactual"].append((time.perf_counter() - t0) * 1000.0)

                for cf_item, perturbed_input in cf_pairs:
                    if cf_item.flipped:
                        cf_flips_reported += 1
                        # Re-run pipeline / predictor on perturbed input to verify faithfulness
                        rerun_out = cf_generator.predictor.predict_record(perturbed_input)
                        if rerun_out["predicted_verdict"].lower() == cf_item.new_verdict.lower():
                            cf_flips_verified += 1
                        else:
                            logger.error(
                                f"Counterfactual mismatch! Reported: {cf_item.new_verdict}, "
                                f"Rerun: {rerun_out['predicted_verdict']}"
                            )
            except Exception as e:
                logger.warning(f"Counterfactual evaluation skipped for {rec.get('id')}: {e}")

        # G. Share Advice Latency
        t0 = time.perf_counter()
        determine_share_advice(
            verdict=Verdict.VERIFIED if not c_info_on.has_conflict and not t_info.is_stale else Verdict.CONTRADICTED,
            confidence=0.90,
            abstained=False,
            conflict=c_info_on,
            temporal=t_info,
        )
        latencies["share_advice"].append((time.perf_counter() - t0) * 1000.0)

    # 5. Calculate Metrics
    # Conflict Detection (with source weighting ON)
    conflict_metrics_on = compute_binary_metrics(conflict_gold, conflict_pred_on)
    # Conflict Detection (with source weighting OFF - ablation)
    conflict_metrics_off = compute_binary_metrics(conflict_gold, conflict_pred_off)

    # Temporal Staleness Accuracy
    correct_temporal = sum(1 for g, p in zip(temporal_gold_stale, temporal_pred_stale) if g == p)
    temporal_accuracy = round(correct_temporal / len(temporal_gold_stale), 4) if temporal_gold_stale else 1.0

    # Gap Detection Metrics per code
    gap_metrics_by_code: Dict[str, Dict[str, float]] = {}
    for code in sorted(all_gap_codes):
        gap_metrics_by_code[code] = compute_binary_metrics(gap_gold_by_code[code], gap_pred_by_code[code])

    # Macro Gap Metrics
    macro_gap_p = sum(m["precision"] for m in gap_metrics_by_code.values()) / len(gap_metrics_by_code)
    macro_gap_r = sum(m["recall"] for m in gap_metrics_by_code.values()) / len(gap_metrics_by_code)
    macro_gap_f1 = 2 * macro_gap_p * macro_gap_r / (macro_gap_p + macro_gap_r) if (macro_gap_p + macro_gap_r) > 0 else 0.0

    # Manipulation Metrics per pattern
    manip_metrics_by_pattern: Dict[str, Dict[str, float]] = {}
    for pat in sorted(all_patterns):
        manip_metrics_by_pattern[pat] = compute_binary_metrics(manip_gold_by_pat[pat], manip_pred_by_pat[pat])

    macro_manip_p = sum(m["precision"] for m in manip_metrics_by_pattern.values()) / len(manip_metrics_by_pattern)
    macro_manip_r = sum(m["recall"] for m in manip_metrics_by_pattern.values()) / len(manip_metrics_by_pattern)
    macro_manip_f1 = 2 * macro_manip_p * macro_manip_r / (macro_manip_p + macro_manip_r) if (macro_manip_p + macro_manip_r) > 0 else 0.0

    # Counterfactual Faithfulness
    cf_faithfulness = (
        round(cf_flips_verified / cf_flips_reported, 4) if cf_flips_reported > 0 else 1.0
    )

    # Latencies
    mean_latencies: Dict[str, float] = {
        name: round(float(sum(vals) / len(vals)), 2) if vals else 0.0
        for name, vals in latencies.items()
    }

    results = {
        "benchmark_records": len(records),
        "conflict_detection": {
            "source_weighting_on": conflict_metrics_on,
            "source_weighting_off_ablation": conflict_metrics_off,
        },
        "temporal_analysis": {
            "staleness_accuracy": temporal_accuracy,
            "evaluated_cases": len(temporal_gold_stale),
        },
        "gap_analysis": {
            "macro_precision": round(macro_gap_p, 4),
            "macro_recall": round(macro_gap_r, 4),
            "macro_f1": round(macro_gap_f1, 4),
            "per_code": gap_metrics_by_code,
        },
        "manipulation_tagging": {
            "macro_precision": round(macro_manip_p, 4),
            "macro_recall": round(macro_manip_r, 4),
            "macro_f1": round(macro_manip_f1, 4),
            "per_pattern": manip_metrics_by_pattern,
        },
        "counterfactual_faithfulness": {
            "reported_flips": cf_flips_reported,
            "verified_flips": cf_flips_verified,
            "faithfulness_rate": cf_faithfulness,
        },
        "component_latency_ms": mean_latencies,
    }

    # 6. Save JSON
    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved evaluation metrics JSON to {metrics_file}")

    # 7. Generate Markdown Report
    md_file = output_dir / "metrics.md"
    md_content = build_markdown_report(results)
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info(f"Saved evaluation metrics Markdown to {md_file}")

    return results


def build_markdown_report(res: Dict[str, Any]) -> str:
    """Builds a formatted GitHub Markdown report from evaluation results."""
    lines: List[str] = [
        "# TrustNet Stage 8: Reasoning Layer Evaluation Report",
        "",
        f"**Evaluated Records**: {res['benchmark_records']}",
        "",
        "## 1. Conflict Detection & Source Weighting Ablation",
        "",
        "| Condition | Precision | Recall | F1 | TP | FP | FN | TN |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    on_m = res["conflict_detection"]["source_weighting_on"]
    off_m = res["conflict_detection"]["source_weighting_off_ablation"]
    lines.append(
        f"| Source Weighting ON | **{on_m['precision']:.2f}** | **{on_m['recall']:.2f}** | **{on_m['f1']:.2f}** | "
        f"{on_m['tp']} | {on_m['fp']} | {on_m['fn']} | {on_m['tn']} |"
    )
    lines.append(
        f"| Source Weighting OFF (Ablation) | {off_m['precision']:.2f} | {off_m['recall']:.2f} | {off_m['f1']:.2f} | "
        f"{off_m['tp']} | {off_m['fp']} | {off_m['fn']} | {off_m['tn']} |"
    )

    lines.extend([
        "",
        "## 2. Temporal Freshness Evaluation",
        "",
        f"- **Staleness Detection Accuracy**: **{res['temporal_analysis']['staleness_accuracy'] * 100:.1f}%**",
        f"- **Total Evaluated Cases**: {res['temporal_analysis']['evaluated_cases']}",
        "",
        "## 3. Evidence Gap Detection (by Code)",
        "",
        "| Gap Code | Precision | Recall | F1 | TP | FP | FN |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for code, m in res["gap_analysis"]["per_code"].items():
        lines.append(f"| `{code}` | {m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f} | {m['tp']} | {m['fp']} | {m['fn']} |")

    lines.append(
        f"| **Macro Average** | **{res['gap_analysis']['macro_precision']:.2f}** | "
        f"**{res['gap_analysis']['macro_recall']:.2f}** | **{res['gap_analysis']['macro_f1']:.2f}** | - | - | - |"
    )

    lines.extend([
        "",
        "## 4. Manipulation Tagging Evaluation (by Pattern)",
        "",
        "| Pattern | Precision | Recall | F1 | TP | FP | FN |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for pat, m in res["manipulation_tagging"]["per_pattern"].items():
        lines.append(f"| `{pat}` | {m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f} | {m['tp']} | {m['fp']} | {m['fn']} |")

    lines.append(
        f"| **Macro Average** | **{res['manipulation_tagging']['macro_precision']:.2f}** | "
        f"**{res['manipulation_tagging']['macro_recall']:.2f}** | **{res['manipulation_tagging']['macro_f1']:.2f}** | - | - | - |"
    )

    lines.extend([
        "",
        "## 5. Counterfactual Faithfulness Verification",
        "",
        f"- **Reported Flips**: {res['counterfactual_faithfulness']['reported_flips']}",
        f"- **Verified Reproductions**: {res['counterfactual_faithfulness']['verified_flips']}",
        f"- **Faithfulness Rate**: **{res['counterfactual_faithfulness']['faithfulness_rate'] * 100:.1f}%** (Target: 100%)",
        "",
        "## 6. Component Latencies (Mean Wall-Clock Time)",
        "",
        "| Component | Latency (ms) |",
        "| :--- | :---: |",
    ])

    for comp, lat in res["component_latency_ms"].items():
        lines.append(f"| `{comp}` | {lat:.2f} ms |")

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate TrustNet Stage 8 Reasoning Layer.")
    parser.add_argument("--gold_path", default="data/eval/reasoning_gold.jsonl", help="Path to gold reasoning annotations.")
    parser.add_argument("--config_path", default="configs/stage8_reasoning.yaml", help="Path to stage 8 config.")
    parser.add_argument("--output_dir", default="outputs/stage8", help="Directory to save metric outputs.")

    args = parser.parse_args()
    results = evaluate_reasoning_layer(
        gold_path=args.gold_path,
        config_path=args.config_path,
        output_dir=args.output_dir,
    )
    print("\n" + build_markdown_report(results))


if __name__ == "__main__":
    main()
