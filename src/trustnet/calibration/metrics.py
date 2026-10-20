"""
Calibration, Uncertainty, and Abstention Metrics implemented from scratch with NumPy.

Provides:
- Expected Calibration Error (ECE, equal-width bins)
- Adaptive-bin ECE (equal-frequency quantile bins)
- Class-wise ECE (one-vs-rest ECE averaged across classes)
- Negative Log-Likelihood (NLL) / Cross Entropy
- Brier Score (multiclass mean squared error)
- Confidently-Wrong Rate (confidence >= threshold but wrong)
- Risk-Coverage Curve and AURC (Area Under Risk-Coverage curve)
- Selective Accuracy at fixed coverage
- Coverage at fixed target error rate
- Reliability diagram data extraction and Matplotlib plotting
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import numpy as np


def compute_nll(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-15) -> float:
    """
    Computes Negative Log-Likelihood (NLL) / multiclass cross-entropy loss.
    
    Args:
        probs: Array of shape (N, K) containing predicted class probabilities.
        labels: Array of shape (N,) containing integer ground-truth class labels.
        eps: Small epsilon for numerical clipping to prevent log(0).
        
    Returns:
        Mean negative log-likelihood across all N samples.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n_samples = len(labels)
    if n_samples == 0:
        return 0.0

    # Clip probabilities to [eps, 1 - eps]
    probs = np.clip(probs, eps, 1.0 - eps)
    # Extract probability assigned to the true class
    true_class_probs = probs[np.arange(n_samples), labels]
    return float(-np.mean(np.log(true_class_probs)))


def compute_brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Computes multiclass Brier score: mean squared error between probability vectors and one-hot labels.
    
    Args:
        probs: Array of shape (N, K) with class probabilities.
        labels: Array of shape (N,) with ground-truth class labels.
        
    Returns:
        Multiclass Brier score.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n_samples, n_classes = probs.shape
    if n_samples == 0:
        return 0.0

    # One-hot encode ground truth labels
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n_samples), labels] = 1.0

    # Sum of squared differences over classes, averaged over samples
    squared_errors = np.sum((probs - one_hot) ** 2, axis=1)
    return float(np.mean(squared_errors))


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Computes Expected Calibration Error (ECE) using equal-width confidence bins.
    
    Confidence is defined as the maximum predicted class probability: c_i = max_k p_{ik}.
    Prediction is defined as the argmax class: y_hat_i = argmax_k p_{ik}.
    
    Args:
        probs: Array of shape (N, K) with class probabilities.
        labels: Array of shape (N,) with integer class labels.
        n_bins: Number of equal-width bins in [0, 1].
        
    Returns:
        Scalar ECE value in [0, 1].
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n_samples = len(labels)
    if n_samples == 0:
        return 0.0

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]
        
        # Include lower bound for first bin
        if i == 0:
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences > lower) & (confidences <= upper)

        bin_count = np.sum(in_bin)
        if bin_count > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            weight = bin_count / n_samples
            ece += weight * np.abs(bin_acc - bin_conf)

    return float(ece)


def adaptive_expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Computes Adaptive-bin ECE where bin edges are defined by empirical quantiles
    of confidence, ensuring each bin contains approximately equal sample counts.
    
    Args:
        probs: Array of shape (N, K) with class probabilities.
        labels: Array of shape (N,) with integer class labels.
        n_bins: Number of equal-frequency bins.
        
    Returns:
        Scalar Adaptive ECE value.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n_samples = len(labels)
    if n_samples == 0:
        return 0.0

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(np.float64)

    # Sort by confidence
    sorted_idx = np.argsort(confidences)
    sorted_conf = confidences[sorted_idx]
    sorted_acc = accuracies[sorted_idx]

    # Effective number of bins capped at sample count
    effective_bins = min(n_bins, n_samples)
    if effective_bins <= 0:
        return 0.0

    # Split into equal-sized chunks
    bin_chunks = np.array_split(np.arange(n_samples), effective_bins)
    ece = 0.0

    for chunk in bin_chunks:
        if len(chunk) == 0:
            continue
        bin_acc = np.mean(sorted_acc[chunk])
        bin_conf = np.mean(sorted_conf[chunk])
        weight = len(chunk) / n_samples
        ece += weight * np.abs(bin_acc - bin_conf)

    return float(ece)


def classwise_expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Computes Class-wise ECE: evaluates one-vs-rest binary ECE for each class
    individually and averages across all classes.
    
    Args:
        probs: Array of shape (N, K) with class probabilities.
        labels: Array of shape (N,) with integer class labels.
        n_bins: Number of equal-width bins.
        
    Returns:
        Macro-averaged Class-wise ECE across all classes.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n_samples, n_classes = probs.shape
    if n_samples == 0:
        return 0.0

    class_eces = []
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    for k in range(n_classes):
        class_conf = probs[:, k]
        class_binary_labels = (labels == k).astype(np.float64)

        class_ece = 0.0
        for i in range(n_bins):
            lower = bin_edges[i]
            upper = bin_edges[i + 1]
            if i == 0:
                in_bin = (class_conf >= lower) & (class_conf <= upper)
            else:
                in_bin = (class_conf > lower) & (class_conf <= upper)

            bin_count = np.sum(in_bin)
            if bin_count > 0:
                bin_acc = np.mean(class_binary_labels[in_bin])
                bin_conf = np.mean(class_conf[in_bin])
                weight = bin_count / n_samples
                class_ece += weight * np.abs(bin_acc - bin_conf)

        class_eces.append(class_ece)

    return float(np.mean(class_eces))


def compute_confidently_wrong_rate(
    probs: np.ndarray,
    labels: np.ndarray,
    cutoff: float = 0.8,
) -> float:
    """
    Computes the fraction of predictions that are confidently wrong (confidence >= cutoff, but incorrect).
    
    Args:
        probs: Array of shape (N, K) with class probabilities.
        labels: Array of shape (N,) with integer class labels.
        cutoff: Confidence threshold (default: 0.8).
        
    Returns:
        Scalar rate in [0, 1].
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n_samples = len(labels)
    if n_samples == 0:
        return 0.0

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    wrong = predictions != labels
    confidently_wrong = wrong & (confidences >= cutoff)

    return float(np.mean(confidently_wrong))


def compute_risk_coverage_curve(
    probs: np.ndarray,
    labels: np.ndarray,
    confidence_metric: str = "max_prob",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the empirical risk-coverage curve by ordering predictions descending by confidence.
    
    At coverage k/N (retaining the top k most confident predictions):
    Risk is the selective error rate on those k samples: (1 - selective_accuracy).
    
    Args:
        probs: Array of shape (N, K) with class probabilities.
        labels: Array of shape (N,) with integer class labels.
        confidence_metric: 'max_prob' or 'normalized_entropy'.
        
    Returns:
        (coverages, risks): Tuple of 1D arrays of length N.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n_samples, n_classes = probs.shape
    if n_samples == 0:
        return np.array([0.0]), np.array([0.0])

    if confidence_metric == "normalized_entropy":
        # Normalized entropy: H / log(K) in [0, 1]. Higher confidence = lower entropy
        eps = 1e-15
        p = np.clip(probs, eps, 1.0)
        entropy = -np.sum(p * np.log(p), axis=1)
        max_entropy = np.log(n_classes) if n_classes > 1 else 1.0
        confidences = 1.0 - (entropy / max_entropy)
    else:
        confidences = np.max(probs, axis=1)

    predictions = np.argmax(probs, axis=1)
    errors = (predictions != labels).astype(np.float64)

    # Sort descending by confidence
    order = np.argsort(-confidences)
    sorted_errors = errors[order]

    # Cumulative errors at each sample retention level k = 1 .. N
    cum_errors = np.cumsum(sorted_errors)
    k_values = np.arange(1, n_samples + 1, dtype=np.float64)

    coverages = k_values / n_samples
    risks = cum_errors / k_values

    return coverages, risks


def compute_aurc(coverages: np.ndarray, risks: np.ndarray) -> float:
    """
    Computes Area Under the Risk-Coverage Curve (AURC) via numerical trapezoidal integration.
    Lower AURC indicates superior uncertainty estimation / selective classification.
    
    Args:
        coverages: Monotonically increasing array of coverage values in (0, 1].
        risks: Corresponding risk (error rate) values.
        
    Returns:
        Scalar AURC value in [0, 1].
    """
    coverages = np.asarray(coverages, dtype=np.float64)
    risks = np.asarray(risks, dtype=np.float64)
    if len(coverages) < 2:
        return float(risks[0]) if len(risks) > 0 else 0.0

    # If first coverage is not 0, extrapolate risk to coverage=0 (same as first risk)
    if coverages[0] > 0.0:
        x = np.concatenate(([0.0], coverages))
        y = np.concatenate(([risks[0]], risks))
    else:
        x = coverages
        y = risks

    # Trapezoid rule integration
    dx = np.diff(x)
    trap = 0.5 * (y[:-1] + y[1:]) * dx
    return float(np.sum(trap))


def compute_selective_accuracy(
    probs: np.ndarray,
    labels: np.ndarray,
    target_coverage: float,
    confidence_metric: str = "max_prob",
) -> float:
    """
    Computes accuracy on the top target_coverage fraction of most confident claims.
    
    Args:
        probs: Array of shape (N, K).
        labels: Array of shape (N,).
        target_coverage: Desired retention coverage in (0, 1] (e.g. 0.8).
        confidence_metric: 'max_prob' or 'normalized_entropy'.
        
    Returns:
        Accuracy achieved on the retained subset.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n_samples = len(labels)
    if n_samples == 0:
        return 0.0

    target_coverage = float(np.clip(target_coverage, 0.0, 1.0))
    k = max(1, int(np.ceil(target_coverage * n_samples)))

    if confidence_metric == "normalized_entropy":
        eps = 1e-15
        p = np.clip(probs, eps, 1.0)
        entropy = -np.sum(p * np.log(p), axis=1)
        max_entropy = np.log(probs.shape[1]) if probs.shape[1] > 1 else 1.0
        confidences = 1.0 - (entropy / max_entropy)
    else:
        confidences = np.max(probs, axis=1)

    order = np.argsort(-confidences)
    top_k_indices = order[:k]

    predictions = np.argmax(probs[top_k_indices], axis=1)
    correct = (predictions == labels[top_k_indices]).astype(np.float64)
    return float(np.mean(correct))


def compute_coverage_at_fixed_error_rate(
    probs: np.ndarray,
    labels: np.ndarray,
    target_error_rate: float,
    confidence_metric: str = "max_prob",
) -> float:
    """
    Finds the maximum coverage (fraction of answered claims) achievable such that
    the selective error rate on answered claims is <= target_error_rate.
    
    Args:
        probs: Array of shape (N, K).
        labels: Array of shape (N,).
        target_error_rate: Desired maximum error tolerance (e.g. 0.05, 0.10).
        confidence_metric: 'max_prob' or 'normalized_entropy'.
        
    Returns:
        Coverage fraction in [0, 1].
    """
    coverages, risks = compute_risk_coverage_curve(
        probs=probs,
        labels=labels,
        confidence_metric=confidence_metric,
    )
    valid_mask = risks <= target_error_rate
    if not np.any(valid_mask):
        return 0.0

    # Max coverage where risk condition holds
    max_cov = np.max(coverages[valid_mask])
    return float(max_cov)


def compute_reliability_diagram_data(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> Dict[str, Any]:
    """
    Extracts bin statistics for plotting reliability diagrams.
    
    Returns:
        Dictionary with bin_edges, bin_centers, bin_accuracies,
        bin_confidences, bin_counts, and ece.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n_samples = len(labels)

    confidences = np.max(probs, axis=1) if n_samples > 0 else np.array([])
    predictions = np.argmax(probs, axis=1) if n_samples > 0 else np.array([])
    accuracies = (predictions == labels).astype(np.float64) if n_samples > 0 else np.array([])

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]
        if i == 0:
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences > lower) & (confidences <= upper)

        count = int(np.sum(in_bin))
        bin_counts.append(count)
        if count > 0:
            bin_accs.append(float(np.mean(accuracies[in_bin])))
            bin_confs.append(float(np.mean(confidences[in_bin])))
        else:
            bin_accs.append(0.0)
            bin_confs.append(float(bin_centers[i]))

    ece = expected_calibration_error(probs, labels, n_bins=n_bins)

    return {
        "n_samples": n_samples,
        "n_bins": n_bins,
        "bin_edges": bin_edges.tolist(),
        "bin_centers": bin_centers.tolist(),
        "bin_accuracies": bin_accs,
        "bin_confidences": bin_confs,
        "bin_counts": bin_counts,
        "ece": round(float(ece), 4),
    }


def plot_reliability_diagram(
    diagram_data: Dict[str, Any],
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Reliability Diagram",
) -> plt.Figure:
    """
    Renders and optionally saves a publication-grade reliability diagram.
    """
    centers = np.array(diagram_data["bin_centers"])
    accs = np.array(diagram_data["bin_accuracies"])
    confs = np.array(diagram_data["bin_confidences"])
    counts = np.array(diagram_data["bin_counts"])
    ece = diagram_data.get("ece", 0.0)
    n_bins = diagram_data.get("n_bins", 15)
    width = 1.0 / n_bins

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    # Top plot: Calibration bars and perfect diagonal
    ax1.plot([0, 1], [0, 1], "--", color="#718096", label="Perfect Calibration")

    # Filter out empty bins for bar plotting
    non_empty = counts > 0
    if np.any(non_empty):
        # Accuracy bars
        ax1.bar(
            centers[non_empty],
            accs[non_empty],
            width=width * 0.85,
            color="#3182ce",
            alpha=0.75,
            edgecolor="#2b6cb0",
            label="Empirical Accuracy",
        )
        # Gap bars
        gaps = np.abs(accs[non_empty] - confs[non_empty])
        ax1.bar(
            centers[non_empty],
            gaps,
            bottom=np.minimum(accs[non_empty], confs[non_empty]),
            width=width * 0.85,
            color="#e53e3e",
            alpha=0.4,
            edgecolor="#c53030",
            hatch="//",
            label="Calibration Gap",
        )

    ax1.set_ylabel("Accuracy", fontsize=11, fontweight="bold")
    ax1.set_ylim([0.0, 1.05])
    ax1.set_title(f"{title} (ECE = {ece:.4f})", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Bottom plot: Sample count distribution per bin
    ax2.bar(
        centers,
        counts,
        width=width * 0.85,
        color="#4a5568",
        alpha=0.7,
        edgecolor="#2d3748",
    )
    ax2.set_xlabel("Confidence", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Count", fontsize=11, fontweight="bold")
    ax2.set_xlim([0.0, 1.0])
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)

    return fig


def plot_risk_coverage_curve(
    coverages: np.ndarray,
    risks: np.ndarray,
    aurc: Optional[float] = None,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Risk-Coverage Curve",
) -> plt.Figure:
    """
    Renders and optionally saves a Risk-Coverage curve.
    """
    if aurc is None:
        aurc = compute_aurc(coverages, risks)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(coverages, risks, color="#d69e2e", lw=2.5, label=f"Selective Risk (AURC={aurc:.4f})")
    ax.fill_between(coverages, 0, risks, color="#ecc94b", alpha=0.25)

    ax.set_xlabel("Coverage (Fraction of Claims Answered)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Risk (Error Rate on Answered)", fontsize=11, fontweight="bold")
    ax.set_xlim([0.0, 1.02])
    ax.set_ylim([0.0, max(0.5, float(np.max(risks)) * 1.1) if len(risks) > 0 else 1.0])
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)

    return fig
