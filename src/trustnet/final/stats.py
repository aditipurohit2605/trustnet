"""
TrustNet Stage 12: Statistical Inference and Significance Testing.

Provides:
1. Paired bootstrap confidence intervals for differences in Macro-F1, Accuracy, ECE, NLL,
   Coverage, AURC, and Confidently-Wrong rates.
2. McNemar's exact paired test for classification accuracy comparisons.
3. Multi-seed aggregation (mean and standard deviation).
4. Multiple comparison adjustments (Bonferroni and Holm-Bonferroni step-down).
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import scipy.stats as stats
from sklearn.metrics import f1_score, accuracy_score, log_loss, brier_score_loss


def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Macro-F1 across all classes."""
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes classification accuracy."""
    return float(accuracy_score(y_true, y_pred))


def compute_nll(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Computes categorical cross-entropy / negative log likelihood."""
    # Clip for numerical stability
    probs_clipped = np.clip(probs, 1e-12, 1.0 - 1e-12)
    # Re-normalize rows
    probs_clipped = probs_clipped / probs_clipped.sum(axis=-1, keepdims=True)
    return float(log_loss(y_true, probs_clipped, labels=list(range(probs.shape[-1]))))


def compute_ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error with equal-width binning."""
    confidences = np.max(probs, axis=-1)
    predictions = np.argmax(probs, axis=-1)
    accuracies = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    if n == 0:
        return 0.0

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def compute_aurc(y_true: np.ndarray, probs: np.ndarray) -> float:
    """
    Computes Area Under the Risk-Coverage curve (AURC).
    Sorts samples by confidence descending, computing empirical risk at each coverage.
    """
    n = len(y_true)
    if n == 0:
        return 0.0

    confidences = np.max(probs, axis=-1)
    predictions = np.argmax(probs, axis=-1)
    correct = (predictions == y_true).astype(float)

    # Sort descending by confidence
    order = np.argsort(-confidences)
    sorted_correct = correct[order]

    coverages = np.arange(1, n + 1) / n
    cumulative_errors = np.cumsum(1.0 - sorted_correct)
    risks = cumulative_errors / np.arange(1, n + 1)

    # Approximate integral using trapezoid rule
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(trapz_fn(risks, coverages))


def compute_confidently_wrong_rate(
    y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.8
) -> float:
    """Proportion of claims where confidence exceeds threshold but prediction is incorrect."""
    if len(y_true) == 0:
        return 0.0
    confidences = np.max(probs, axis=-1)
    predictions = np.argmax(probs, axis=-1)
    high_conf_wrong = (confidences >= threshold) & (predictions != y_true)
    return float(np.mean(high_conf_wrong))


def compute_coverage_at_target_error(
    y_true: np.ndarray, probs: np.ndarray, target_error: float = 0.05
) -> float:
    """Computes maximum coverage achieving empirical error <= target_error."""
    n = len(y_true)
    if n == 0:
        return 0.0

    confidences = np.max(probs, axis=-1)
    predictions = np.argmax(probs, axis=-1)
    correct = (predictions == y_true).astype(float)

    order = np.argsort(-confidences)
    sorted_correct = correct[order]

    max_cov = 0.0
    for k in range(1, n + 1):
        error_at_k = 1.0 - (np.sum(sorted_correct[:k]) / k)
        if error_at_k <= target_error:
            max_cov = k / n

    return float(max_cov)


def paired_bootstrap_difference(
    metric_fn: Callable[..., float],
    y_true: np.ndarray,
    data_a: np.ndarray,
    data_b: np.ndarray,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Computes paired bootstrap confidence interval for difference: metric(A) - metric(B).

    GUARANTEE: If data_a and data_b are identical, every bootstrap sample difference
    is identically zero, returning diff=0.0, ci_lower=0.0, ci_upper=0.0.
    """
    # Strict invariance check: identical systems yield exact zero difference
    if np.array_equal(data_a, data_b):
        return {
            "point_diff": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "std_error": 0.0,
            "excludes_zero": False,
        }

    n = len(y_true)
    rng = np.random.default_rng(random_state)
    point_a = metric_fn(y_true, data_a)
    point_b = metric_fn(y_true, data_b)
    point_diff = point_a - point_b

    diffs = np.empty(n_bootstraps, dtype=float)
    for i in range(n_bootstraps):
        idx = rng.integers(0, n, n)
        val_a = metric_fn(y_true[idx], data_a[idx])
        val_b = metric_fn(y_true[idx], data_b[idx])
        diffs[i] = val_a - val_b

    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(diffs, 100 * (alpha / 2)))
    ci_upper = float(np.percentile(diffs, 100 * (1.0 - alpha / 2)))
    std_error = float(np.std(diffs))

    # Interval excludes zero if both bounds are on the same side of zero
    excludes_zero = (ci_lower > 0) or (ci_upper < 0)

    return {
        "point_diff": float(point_diff),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std_error": std_error,
        "excludes_zero": bool(excludes_zero),
    }


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> Dict[str, Any]:
    """
    Performs McNemar's paired test of classification accuracy differences.
    Uses exact binomial test when discordant pairs < 25, otherwise continuity-corrected chi-square.
    """
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    # Contingency table counts
    n00 = int(np.sum((~correct_a) & (~correct_b)))
    n01 = int(np.sum((~correct_a) & correct_b))  # A wrong, B right
    n10 = int(np.sum(correct_a & (~correct_b)))  # A right, B wrong
    n11 = int(np.sum(correct_a & correct_b))

    discordant = n01 + n10
    if discordant == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "n01_b_only": n01,
            "n10_a_only": n10,
            "method": "exact_binomial",
            "significant_at_05": False,
        }

    if discordant < 25:
        # Exact binomial test under null p=0.5
        res = stats.binomtest(min(n01, n10), discordant, 0.5, alternative="two-sided")
        p_val = float(res.pvalue)
        stat = float(min(n01, n10))
        method = "exact_binomial"
    else:
        # Continuity-corrected McNemar chi-square
        stat = float(((abs(n01 - n10) - 1.0) ** 2) / discordant)
        p_val = float(stats.chi2.sf(stat, df=1))
        method = "chi2_continuity_corrected"

    return {
        "statistic": stat,
        "p_value": p_val,
        "n01_b_only": n01,
        "n10_a_only": n10,
        "method": method,
        "significant_at_05": p_val < 0.05,
    }


def holm_bonferroni_adjustment(p_values: Sequence[float]) -> List[float]:
    """
    Computes Holm-Bonferroni step-down adjusted p-values.
    Guarantees family-wise error rate control without excessive conservative distortion.
    """
    m = len(p_values)
    if m == 0:
        return []

    # Sort p-values with original indices
    sorted_pairs = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m

    running_max = 0.0
    for rank, (orig_idx, p_val) in enumerate(sorted_pairs):
        multiplier = m - rank
        adj_p = min(1.0, p_val * multiplier)
        running_max = max(running_max, adj_p)
        adjusted[orig_idx] = running_max

    return adjusted


def multi_seed_summary(metric_values: Sequence[float]) -> Dict[str, float]:
    """Computes mean, standard deviation, and sample size across multiple seeds."""
    arr = np.asarray(metric_values, dtype=float)
    if len(arr) == 0:
        return {"mean": 0.0, "std": 0.0, "n_seeds": 0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "n_seeds": len(arr),
    }
