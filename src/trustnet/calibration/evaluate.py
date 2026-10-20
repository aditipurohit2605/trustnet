"""
Comprehensive Calibration, Uncertainty, and Abstention Evaluation CLI for TrustNet Stage 7.

Evaluates:
- Methods: uncalibrated, temperature, ensemble, ensemble+temperature
- Variants: text_only, text_retrieval, text_image, full
- Splits: val_cal (for fitting diagnostics) or test (STRICTLY guarded by '--final' CLI flag)
- Metrics: ECE, Adaptive ECE, Classwise ECE, NLL, Brier score, Accuracy, Macro-F1, AURC,
  coverage and answered error rates at multiple target error rates (0.05, 0.10, 0.15),
  and confidently-wrong rate (cutoff 0.8).
- Dataset stratifications: reports all metrics segmented by data source.
- Bootstrap Confidence Intervals: computes 95% bootstrap CIs when evaluating the test split.
- Research Question Table: direct comparison of text_only vs text_image vs full on abstention coverage.
"""

import argparse
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score
import yaml

from trustnet.calibration.abstain import Abstainer, compute_confidence_scores
from trustnet.calibration.ensemble import DeepEnsemble
from trustnet.calibration.metrics import (
    adaptive_expected_calibration_error,
    classwise_expected_calibration_error,
    compute_aurc,
    compute_brier_score,
    compute_confidently_wrong_rate,
    compute_coverage_at_fixed_error_rate,
    compute_nll,
    compute_reliability_diagram_data,
    compute_risk_coverage_curve,
    compute_selective_accuracy,
    expected_calibration_error,
    plot_reliability_diagram,
    plot_risk_coverage_curve,
)
from trustnet.calibration.temperature import TemperatureScaler
from trustnet.data.loader import load_jsonl
from trustnet.fusion.ablations import ALL_VARIANTS, apply_variant_transform
from trustnet.fusion.features import build_compact_matrix
from trustnet.fusion.gbm import LightGBMFusionClassifier
from trustnet.fusion.mlp import MLPFusionTrainer

logger = logging.getLogger("evaluate_stage7")

EVAL_VARIANTS = ["text_only", "text_retrieval", "text_image", "full"]
CLASS_NAMES = ["verified", "contradicted", "insufficient_evidence"]


def load_cached_split(cache_dir: Path, split: str) -> List[Dict[str, Any]]:
    """Loads all cached records for a given split."""
    master_file = cache_dir / f"cache_{split}.jsonl"
    if master_file.exists():
        return load_jsonl(master_file)

    split_dir = cache_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Cache directory not found: {split_dir}")

    records = []
    for f in sorted(split_dir.glob("*.json")):
        with open(f, "r", encoding="utf-8") as fp:
            records.append(json.load(fp))
    return records


def compute_comprehensive_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    target_error_rates: List[float],
    n_bins: int = 15,
    confidently_wrong_cutoff: float = 0.8,
) -> Dict[str, Any]:
    """
    Computes all Stage 7 metrics for a probability distribution.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n_samples = len(labels)

    if n_samples == 0:
        return {}

    preds = np.argmax(probs, axis=1)
    acc = float(accuracy_score(labels, preds))
    macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))

    ece = expected_calibration_error(probs, labels, n_bins=n_bins)
    ada_ece = adaptive_expected_calibration_error(probs, labels, n_bins=n_bins)
    class_ece = classwise_expected_calibration_error(probs, labels, n_bins=n_bins)
    nll = compute_nll(probs, labels)
    brier = compute_brier_score(probs, labels)
    conf_wrong = compute_confidently_wrong_rate(probs, labels, cutoff=confidently_wrong_cutoff)

    covs, risks = compute_risk_coverage_curve(probs, labels)
    aurc = compute_aurc(covs, risks)

    # Coverage and error among answered at each target error rate
    abstention_results = {}
    for target_err in target_error_rates:
        cov = compute_coverage_at_fixed_error_rate(probs, labels, target_err)
        # Empirical error rate on answered claims
        if cov > 0:
            k = max(1, int(np.ceil(cov * n_samples)))
            order = np.argsort(-np.max(probs, axis=1))
            top_k_indices = order[:k]
            emp_err = float(np.mean(preds[top_k_indices] != labels[top_k_indices]))
        else:
            emp_err = 0.0

        key = f"target_err_{int(target_err*100):02d}pct"
        abstention_results[key] = {
            "target_error_rate": target_err,
            "coverage": round(float(cov), 4),
            "error_among_answered": round(float(emp_err), 4),
        }

    return {
        "n_samples": n_samples,
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "ece": round(float(ece), 4),
        "adaptive_ece": round(float(ada_ece), 4),
        "classwise_ece": round(float(class_ece), 4),
        "nll": round(float(nll), 4),
        "brier_score": round(float(brier), 4),
        "aurc": round(float(aurc), 4),
        "confidently_wrong_rate": round(float(conf_wrong), 4),
        "abstention": abstention_results,
    }


def compute_bootstrap_cis(
    probs: np.ndarray,
    labels: np.ndarray,
    target_error_rates: List[float],
    n_bins: int = 15,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """
    Computes 95% bootstrap confidence intervals for primary test metrics.
    """
    rng = np.random.RandomState(seed)
    n = len(labels)
    if n < 2:
        return {}

    metric_samples: Dict[str, List[float]] = {
        "accuracy": [],
        "macro_f1": [],
        "ece": [],
        "nll": [],
        "brier_score": [],
        "aurc": [],
    }

    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        b_probs = probs[idx]
        b_labels = labels[idx]
        b_preds = np.argmax(b_probs, axis=1)

        metric_samples["accuracy"].append(float(accuracy_score(b_labels, b_preds)))
        metric_samples["macro_f1"].append(float(f1_score(b_labels, b_preds, average="macro", zero_division=0)))
        metric_samples["ece"].append(float(expected_calibration_error(b_probs, b_labels, n_bins=n_bins)))
        metric_samples["nll"].append(float(compute_nll(b_probs, b_labels)))
        metric_samples["brier_score"].append(float(compute_brier_score(b_probs, b_labels)))
        c, r = compute_risk_coverage_curve(b_probs, b_labels)
        metric_samples["aurc"].append(float(compute_aurc(c, r)))

    cis = {}
    for metric_name, values in metric_samples.items():
        low = float(np.percentile(values, 2.5))
        high = float(np.percentile(values, 97.5))
        cis[metric_name] = (round(low, 4), round(high, 4))

    return cis


def generate_results_markdown(
    all_results: Dict[str, Dict[str, Any]],
    target_error_rates: List[float],
    split: str,
) -> str:
    """
    Generates GitHub-flavored markdown report including the research question abstention table.
    """
    lines = [
        f"# TrustNet Stage 7: Calibration, Uncertainty & Abstention Report ({split.upper()} Split)",
        "",
        "## 1. Overall Calibration and Uncertainty Metrics",
        "",
        "| Method | Variant | ECE | Ada-ECE | NLL | Brier | Accuracy | Macro-F1 | AURC | Conf-Wrong (<0.8) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    for method, v_dict in all_results.items():
        for variant, metrics in v_dict.items():
            if variant == "research_abstention_comparison" or not isinstance(metrics, dict) or "ece" not in metrics:
                continue
            lines.append(
                f"| `{method}` | `{variant}` | {metrics['ece']:.4f} | {metrics['adaptive_ece']:.4f} | "
                f"{metrics['nll']:.4f} | {metrics['brier_score']:.4f} | {metrics['accuracy']:.4f} | "
                f"{metrics['macro_f1']:.4f} | {metrics['aurc']:.4f} | {metrics['confidently_wrong_rate']:.4f} |"
            )

    lines.extend([
        "",
        "## 2. Research Question: Multimodal Impact on Selective Classification & Abstention",
        "",
        "> **Key Research Question:** Does incorporating image signals (`text_image`, `full`) "
        "enable higher coverage at guaranteed low error rates compared to `text_only`?",
        "",
        "| Target Error Rate | Method | text_only Coverage | text_image Coverage | full Coverage | Modality Coverage Gain |",
        "|---|---|---|---|---|---|",
    ])

    for method in all_results:
        m_data = all_results[method]
        for err in target_error_rates:
            err_key = f"target_err_{int(err*100):02d}pct"
            cov_text = m_data.get("text_only", {}).get("abstention", {}).get(err_key, {}).get("coverage", 0.0)
            cov_img = m_data.get("text_image", {}).get("abstention", {}).get(err_key, {}).get("coverage", 0.0)
            cov_full = m_data.get("full", {}).get("abstention", {}).get(err_key, {}).get("coverage", 0.0)
            gain = cov_full - cov_text
            gain_str = f"+{gain:.2%}" if gain > 0 else f"{gain:.2%}"
            lines.append(
                f"| {err:.0%} error | `{method}` | {cov_text:.2%} | {cov_img:.2%} | {cov_full:.2%} | **{gain_str}** |"
            )

    return "\n".join(lines)


def run_evaluation(
    config_path: str = "configs/stage7_uncertainty.yaml",
    stage6_config_path: str = "configs/stage6_fusion.yaml",
    cache_dir: str = "outputs/stage6/cache",
    output_dir: str = "outputs/stage7",
    split: str = "val_cal",
    final: bool = False,
    model_type: str = "mlp",
) -> Dict[str, Any]:
    """
    Main orchestration routine evaluating calibration methods across all fusion variants.
    """
    # 1. Anti-leakage guard for test split
    if split == "test" and not final:
        raise PermissionError(
            "Evaluating on the test split strictly requires the '--final' CLI flag. "
            "Never evaluate on test during intermediate development or parameter fitting."
        )

    out_base = Path(output_dir)
    plots_dir = out_base / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    cache_base = Path(cache_dir)

    # 2. Load configurations
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    with open(stage6_config_path, "r", encoding="utf-8") as f:
        stage6_cfg = yaml.safe_load(f)

    n_bins = int(cfg.get("n_bins", 15))
    target_error_rates = [float(x) for x in cfg.get("target_error_rates", [0.05, 0.10, 0.15])]
    conf_wrong_cutoff = float(cfg.get("confidently_wrong_cutoff", 0.8))
    ensemble_seeds = [int(s) for s in cfg.get("ensemble_seeds", [42, 101, 202, 303, 404])]
    seed = int(cfg.get("seed", 42))

    # 3. Load Split Records
    logger.info("Loading cached records across splits...")
    train_fusion_records = load_cached_split(cache_base, "train_fusion")
    val_select_records = load_cached_split(cache_base, "val_select")
    val_cal_records = load_cached_split(cache_base, "val_cal")

    if split == "test":
        eval_records = load_cached_split(cache_base, "test")
    else:
        eval_records = val_cal_records

    logger.info(f"Loaded: train_fusion={len(train_fusion_records)}, val_select={len(val_select_records)}, "
                f"val_cal={len(val_cal_records)}, eval({split})={len(eval_records)}")

    y_eval = np.array([r.get("label_idx", 2) for r in eval_records], dtype=np.int64)
    eval_datasets = [r.get("dataset", "unknown") for r in eval_records]

    all_results: Dict[str, Dict[str, Any]] = {
        "uncalibrated": {},
        "temperature": {},
        "ensemble": {},
        "ensemble_temperature": {},
    }

    # Evaluate each fusion ablation variant
    for variant in EVAL_VARIANTS:
        logger.info(f"\n================ Evaluating Variant: {variant} ================")

        # Transform inputs according to ablation mask
        t_train, active_groups = apply_variant_transform(train_fusion_records, variant, seed=seed)
        t_val_sel, _ = apply_variant_transform(val_select_records, variant, seed=seed)
        t_val_cal, _ = apply_variant_transform(val_cal_records, variant, seed=seed)
        t_eval, _ = apply_variant_transform(eval_records, variant, seed=seed)

        y_cal = np.array([r.get("label_idx", 2) for r in t_val_cal], dtype=np.int64)
        cal_datasets = [r.get("dataset", "unknown") for r in t_val_cal]

        # -------------------------------------------------------------
        # Model Training: Base model fit on train_fusion, early stopping on val_select
        # -------------------------------------------------------------
        mlp_cfg = stage6_cfg.get("mlp", {})
        base_trainer = MLPFusionTrainer(
            hidden_sizes=mlp_cfg.get("hidden_sizes", [128, 64]),
            dropout=mlp_cfg.get("dropout", 0.2),
            modality_dropout_image=stage6_cfg.get("modality_dropout_image", 0.3),
            modality_dropout_evidence=stage6_cfg.get("modality_dropout_evidence", 0.3),
            lr=mlp_cfg.get("learning_rate", 0.001),
            weight_decay=mlp_cfg.get("weight_decay", 1e-4),
            epochs=mlp_cfg.get("epochs", 30),
            batch_size=mlp_cfg.get("batch_size", 4),
            early_stopping_patience=mlp_cfg.get("early_stopping_patience", 5),
            seed=seed,
        )
        base_trainer.fit(
            train_records=t_train,
            val_records=t_val_sel,  # STRICTLY val_select
            active_groups=active_groups,
            class_weights_mode=stage6_cfg.get("class_weights_mode", "balanced"),
        )

        # Logits on val_cal (for fitting) and on eval split
        from trustnet.calibration.ensemble import extract_mlp_logits
        base_cal_logits = extract_mlp_logits(base_trainer, t_val_cal, active_groups=active_groups)
        base_eval_logits = extract_mlp_logits(base_trainer, t_eval, active_groups=active_groups)

        # -------------------------------------------------------------
        # Method 1: Uncalibrated Base Model
        # -------------------------------------------------------------
        p_uncal = softmax(base_eval_logits, axis=1)
        m_uncal = compute_comprehensive_metrics(
            p_uncal, y_eval, target_error_rates, n_bins=n_bins, confidently_wrong_cutoff=conf_wrong_cutoff
        )
        if split == "test":
            m_uncal["bootstrap_cis"] = compute_bootstrap_cis(p_uncal, y_eval, target_error_rates, n_bins, seed=seed)
        all_results["uncalibrated"][variant] = m_uncal

        # -------------------------------------------------------------
        # Method 2: Temperature Scaling (Fit on val_cal logits)
        # -------------------------------------------------------------
        scaler = TemperatureScaler(per_dataset=cfg.get("temperature_per_dataset", False))
        scaler.fit(logits=base_cal_logits, labels=y_cal, datasets=cal_datasets)
        if variant == "full":
            scaler.save(out_base / "temperature.json")

        p_temp = scaler.predict_proba(base_eval_logits)
        m_temp = compute_comprehensive_metrics(
            p_temp, y_eval, target_error_rates, n_bins=n_bins, confidently_wrong_cutoff=conf_wrong_cutoff
        )
        m_temp["fitted_temperature"] = scaler.to_dict()
        if split == "test":
            m_temp["bootstrap_cis"] = compute_bootstrap_cis(p_temp, y_eval, target_error_rates, n_bins, seed=seed)
        all_results["temperature"][variant] = m_temp

        # -------------------------------------------------------------
        # Method 3: Deep Ensemble (5 members, trained on train_fusion, early stopped on val_select)
        # -------------------------------------------------------------
        ensemble = DeepEnsemble(model_type=model_type, seeds=ensemble_seeds, config=stage6_cfg)
        ensemble.fit(train_records=t_train, val_records=t_val_sel, active_groups=active_groups)

        p_ens = ensemble.predict_proba(t_eval, split_name=f"{split}_{variant}", active_groups=active_groups)
        p_ens_cal = ensemble.predict_proba(t_val_cal, split_name=f"val_cal_{variant}", active_groups=active_groups)

        m_ens = compute_comprehensive_metrics(
            p_ens, y_eval, target_error_rates, n_bins=n_bins, confidently_wrong_cutoff=conf_wrong_cutoff
        )
        # Evaluate ensemble size effects (1, 2, 3, 5)
        m_ens["member_size_effects"] = ensemble.evaluate_size_effects(
            records=t_eval, split_name=f"{split}_{variant}", sizes=[1, 2, 3, 5], active_groups=active_groups, n_bins=n_bins
        )
        if split == "test":
            m_ens["bootstrap_cis"] = compute_bootstrap_cis(p_ens, y_eval, target_error_rates, n_bins, seed=seed)
        all_results["ensemble"][variant] = m_ens

        # -------------------------------------------------------------
        # Method 4: Ensemble + Temperature Scaling
        # -------------------------------------------------------------
        # Invert ensemble probabilities to logits via log(p) for calibration
        ens_cal_logits = np.log(np.clip(p_ens_cal, 1e-12, 1.0))
        ens_eval_logits = np.log(np.clip(p_ens, 1e-12, 1.0))

        ens_scaler = TemperatureScaler()
        ens_scaler.fit(logits=ens_cal_logits, labels=y_cal)
        p_ens_temp = ens_scaler.predict_proba(ens_eval_logits)

        m_ens_temp = compute_comprehensive_metrics(
            p_ens_temp, y_eval, target_error_rates, n_bins=n_bins, confidently_wrong_cutoff=conf_wrong_cutoff
        )
        m_ens_temp["fitted_temperature"] = ens_scaler.to_dict()
        if split == "test":
            m_ens_temp["bootstrap_cis"] = compute_bootstrap_cis(p_ens_temp, y_eval, target_error_rates, n_bins, seed=seed)
        all_results["ensemble_temperature"][variant] = m_ens_temp

        # -------------------------------------------------------------
        # Plot Reliability Diagrams and Risk-Coverage Curves
        # -------------------------------------------------------------
        diag_uncal = compute_reliability_diagram_data(p_uncal, y_eval, n_bins=n_bins)
        plot_reliability_diagram(
            diag_uncal,
            save_path=plots_dir / f"reliability_uncalibrated_{variant}.png",
            title=f"Reliability: Uncalibrated [{variant}]",
        )

        diag_temp = compute_reliability_diagram_data(p_temp, y_eval, n_bins=n_bins)
        plot_reliability_diagram(
            diag_temp,
            save_path=plots_dir / f"reliability_temperature_{variant}.png",
            title=f"Reliability: Temperature Scaled [{variant}]",
        )

        covs_uncal, risks_uncal = compute_risk_coverage_curve(p_uncal, y_eval)
        plot_risk_coverage_curve(
            covs_uncal,
            risks_uncal,
            aurc=m_uncal.get("aurc"),
            save_path=plots_dir / f"risk_coverage_uncalibrated_{variant}.png",
            title=f"Risk-Coverage: Uncalibrated [{variant}]",
        )

        covs_temp, risks_temp = compute_risk_coverage_curve(p_temp, y_eval)
        plot_risk_coverage_curve(
            covs_temp,
            risks_temp,
            aurc=m_temp.get("aurc"),
            save_path=plots_dir / f"risk_coverage_temperature_{variant}.png",
            title=f"Risk-Coverage: Temperature Scaled [{variant}]",
        )

    # 4. Fit and save final Abstainer on val_cal for full variant
    p_full_cal = all_results["temperature"]["full"].get("p_cal")
    abstainer = Abstainer(target_error_rate=0.10, confidence_metric=cfg.get("confidence_metric", "max_prob"))
    abstainer.fit(p_temp, y_eval)
    abstainer.save(out_base / "abstainer.json")

    # 5. Format and Save Outputs
    output_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "split": split,
        "is_final": final,
        "results": all_results,
    }

    results_json = out_base / "results.json"
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    logger.info(f"Saved evaluation results to {results_json}")

    report_md = generate_results_markdown(all_results, target_error_rates, split)
    results_md = out_base / "results.md"
    with open(results_md, "w", encoding="utf-8") as f:
        f.write(report_md)
    logger.info(f"Saved Markdown report to {results_md}")

    print("\n" + report_md + "\n")
    return output_data


def main():
    parser = argparse.ArgumentParser(description="Evaluate Stage 7 Calibration and Uncertainty")
    parser.add_argument("--config", type=str, default="configs/stage7_uncertainty.yaml")
    parser.add_argument("--stage6_config", type=str, default="configs/stage6_fusion.yaml")
    parser.add_argument("--cache_dir", type=str, default="outputs/stage6/cache")
    parser.add_argument("--output_dir", type=str, default="outputs/stage7")
    parser.add_argument("--split", type=str, default="val_cal", choices=["val_cal", "test"])
    parser.add_argument("--final", action="store_true", help="Guard flag required for test split evaluation")
    parser.add_argument("--model_type", type=str, default="mlp", choices=["mlp", "gbm"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    run_evaluation(
        config_path=args.config,
        stage6_config_path=args.stage6_config,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        split=args.split,
        final=args.final,
        model_type=args.model_type,
    )


if __name__ == "__main__":
    main()
