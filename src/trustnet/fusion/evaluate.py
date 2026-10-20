"""
Comprehensive evaluation CLI for TrustNet Stage 6 Multimodal Fusion.

Evaluates:
- Models: LightGBM GBDT and PyTorch MLP
- Variants: text_only, text_retrieval, text_image, full, claim_only,
            shuffled_image (control), shuffled_evidence (control)
- Metrics: Accuracy, Macro-F1, per-class Precision/Recall/F1, Confusion Matrix
- Feature Importances (for LightGBM)
- Enforces strict safety: requires '--final' flag for any test split evaluation.

Outputs:
- outputs/stage6/metrics.json
- outputs/stage6/metrics.md (Markdown comparative report)
"""

import argparse
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import yaml

from trustnet.data.loader import load_jsonl
from trustnet.fusion.ablations import ALL_VARIANTS, apply_variant_transform
from trustnet.fusion.gbm import LightGBMFusionClassifier
from trustnet.fusion.mlp import MLPFusionTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("evaluate_stage6")

CLASS_NAMES = ["verified", "contradicted", "insufficient_evidence"]


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Computes accuracy, macro-F1, per-class metrics, and confusion matrix.
    """
    acc = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    p_per, r_per, f1_per, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0
    )

    per_class = {}
    for idx, cname in enumerate(CLASS_NAMES):
        per_class[cname] = {
            "precision": round(float(p_per[idx]), 4),
            "recall": round(float(r_per[idx]), 4),
            "f1": round(float(f1_per[idx]), 4),
            "support": int(support[idx]),
        }

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(float(f1_macro), 4),
        "precision_macro": round(float(p_macro), 4),
        "recall_macro": round(float(r_macro), 4),
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def load_cached_split(cache_dir: Path, split: str) -> List[Dict[str, Any]]:
    """Loads all cached records for a given split."""
    master_file = cache_dir / f"cache_{split}.jsonl"
    if master_file.exists():
        return load_jsonl(master_file)

    split_dir = cache_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Cache directory not found: {split_dir}")

    records = []
    for f in split_dir.glob("*.json"):
        with open(f, "r", encoding="utf-8") as fp:
            records.append(json.load(fp))
    return records


def generate_markdown_report(
    eval_results: Dict[str, Dict[str, Any]],
    split: str,
    feature_importances: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generates a GitHub-flavored Markdown table comparing all models and ablation variants.
    """
    lines = [
        f"# TrustNet Stage 6: Multimodal Fusion Evaluation ({split.upper()} Split)",
        "",
        "## 1. System Variants and Ablation Comparison",
        "",
        "| Model | Variant | Accuracy | Macro-F1 | Precision | Recall |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |",
    ]

    for model_name, variants_data in eval_results.items():
        for variant_name, m in variants_data.items():
            lines.append(
                f"| **{model_name.upper()}** | `{variant_name}` | "
                f"{m['accuracy']:.4f} | **{m['macro_f1']:.4f}** | "
                f"{m['precision_macro']:.4f} | {m['recall_macro']:.4f} |"
            )

    # Add LightGBM Feature Importances table
    if feature_importances:
        lines.extend([
            "",
            "## 2. LightGBM Top Feature Importances (Full Model)",
            "",
            "| Rank | Feature Name | Importance (Gain) | Split Count |",
            "| :---: | :--- | :---: | :---: |",
        ])
        sorted_feats = sorted(
            feature_importances.items(),
            key=lambda item: item[1]["gain"],
            reverse=True,
        )
        for rank, (fname, d) in enumerate(sorted_feats[:10], start=1):
            lines.append(f"| {rank} | `{fname}` | {d['gain']:.4f} | {d['split_count']} |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="TrustNet Stage 6 Multimodal Fusion Evaluation")
    parser.add_argument("--config", type=str, default="configs/stage6_fusion.yaml", help="Config file path")
    parser.add_argument("--cache_dir", type=str, default="outputs/stage6/cache", help="Cache directory")
    parser.add_argument("--output_dir", type=str, default="outputs/stage6", help="Output directory")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Evaluation split")
    parser.add_argument("--final", action="store_true", help="Flag strictly required to evaluate on test split")
    parser.add_argument("--models", nargs="+", default=["gbm", "mlp"], choices=["gbm", "mlp"], help="Models to evaluate")
    args = parser.parse_args()

    # Safety guard: require --final flag for test split evaluation
    if args.split == "test" and not args.final:
        raise PermissionError(
            "Evaluating on the test split requires the '--final' CLI flag. "
            "Do not evaluate on the test split during intermediate development."
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)

    # Load configuration
    cfg: Dict[str, Any] = {}
    if Path(args.config).exists():
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    seed = cfg.get("seed", 42)
    lgb_params = cfg.get("lightgbm", {})
    mlp_params = cfg.get("mlp", {})

    # Load cache records
    logger.info("Loading cached records for train_fusion and validation...")
    train_fusion_records = load_cached_split(cache_dir, "train_fusion")
    val_records = load_cached_split(cache_dir, "val")

    if args.split == "test":
        eval_records_base = load_cached_split(cache_dir, "test")
    else:
        eval_records_base = val_records

    all_results: Dict[str, Dict[str, Any]] = {}
    best_gbm_importances: Optional[Dict[str, Any]] = None

    # Evaluate each requested model
    for model_name in args.models:
        all_results[model_name] = {}

        for variant in ALL_VARIANTS:
            logger.info(f"Evaluating [{model_name.upper()}] with variant '{variant}' on {args.split}...")

            # Apply variant transformations (e.g. group masks or shuffled controls)
            t_train, active_groups = apply_variant_transform(train_fusion_records, variant, seed=seed)
            t_eval, _ = apply_variant_transform(eval_records_base, variant, seed=seed)
            t_val, _ = apply_variant_transform(val_records, variant, seed=seed)

            y_true = np.array([r.get("label_idx", 2) for r in t_eval], dtype=np.int64)

            if model_name == "gbm":
                clf = LightGBMFusionClassifier(params=lgb_params, seed=seed)
                clf.fit(
                    train_records=t_train,
                    val_records=t_val,
                    active_groups=active_groups,
                    class_weights_mode=cfg.get("class_weights_mode", "balanced"),
                )
                preds = clf.predict(t_eval, active_groups=active_groups)

                if variant == "full":
                    best_gbm_importances = clf.feature_importances
                    clf.save(out_dir / "models" / "gbm_full")

            elif model_name == "mlp":
                trainer = MLPFusionTrainer(
                    hidden_sizes=mlp_params.get("hidden_sizes", [128, 64]),
                    dropout=mlp_params.get("dropout", 0.2),
                    modality_dropout_image=cfg.get("modality_dropout_image", 0.3),
                    modality_dropout_evidence=cfg.get("modality_dropout_evidence", 0.3),
                    lr=mlp_params.get("learning_rate", 0.001),
                    weight_decay=mlp_params.get("weight_decay", 1e-4),
                    epochs=mlp_params.get("epochs", 30),
                    batch_size=mlp_params.get("batch_size", 4),
                    early_stopping_patience=mlp_params.get("early_stopping_patience", 5),
                    seed=seed,
                )
                trainer.fit(
                    train_records=t_train,
                    val_records=t_val,
                    active_groups=active_groups,
                    class_weights_mode=cfg.get("class_weights_mode", "balanced"),
                )
                preds = trainer.predict(t_eval, active_groups=active_groups)

                if variant == "full":
                    trainer.save(out_dir / "models" / "mlp_full")

            metrics = compute_classification_metrics(y_true, preds)
            all_results[model_name][variant] = metrics

    # Format outputs
    final_output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "split": args.split,
        "is_final": args.final,
        "results": all_results,
        "lightgbm_feature_importances": best_gbm_importances,
    }

    # Save metrics.json
    metrics_file = out_dir / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    logger.info(f"Saved evaluation metrics to {metrics_file}")

    # Save Markdown report
    report_md = generate_markdown_report(all_results, args.split, feature_importances=best_gbm_importances)
    report_file = out_dir / "metrics.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_md)
    logger.info(f"Saved Markdown report to {report_file}")

    print("\n" + report_md + "\n")


if __name__ == "__main__":
    main()
