"""
Selective Classification and Abstention for TrustNet Stage 7.

Implements an Abstainer that fits an optimal confidence threshold on val_cal
to satisfy a user-specified target error rate on answered claims.

Supports two confidence scoring functions:
1. Max Probability: S(p) = max_k p_k
2. Normalized Certainty: S(p) = 1.0 - (H(p) / log(K))
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

CLASS_NAMES = ["verified", "contradicted", "insufficient_evidence"]


def compute_confidence_scores(
    probs: np.ndarray,
    metric: str = "max_prob",
) -> np.ndarray:
    """
    Computes scalar confidence scores for predicted probability distributions.
    
    Args:
        probs: Array of shape (N, K).
        metric: 'max_prob' or 'normalized_entropy'.
        
    Returns:
        1D array of length N with scores in [0, 1].
    """
    probs = np.asarray(probs, dtype=np.float64)
    if len(probs) == 0:
        return np.array([])

    if metric == "normalized_entropy":
        eps = 1e-15
        p = np.clip(probs, eps, 1.0)
        # Normalize sum to 1 in case of precision issues
        p = p / np.sum(p, axis=1, keepdims=True)
        entropy = -np.sum(p * np.log(p), axis=1)
        max_entropy = np.log(probs.shape[1]) if probs.shape[1] > 1 else 1.0
        scores = 1.0 - (entropy / max_entropy)
        return np.clip(scores, 0.0, 1.0)

    # Default to max probability
    return np.max(probs, axis=1)


class Abstainer:
    """
    Confidence-based abstention decision module.
    
    Fits a confidence threshold theta on val_cal such that empirical error rate
    on answered samples is <= target_error_rate.
    """

    def __init__(
        self,
        target_error_rate: float = 0.10,
        confidence_metric: str = "max_prob",
        class_names: Optional[List[str]] = None,
    ):
        self.target_error_rate = float(target_error_rate)
        self.confidence_metric = confidence_metric
        self.class_names = class_names or CLASS_NAMES
        self.threshold: float = 0.5
        self.fitted: bool = False
        self.val_cal_coverage: float = 1.0
        self.val_cal_error: float = 0.0

    def fit(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        target_error_rate: Optional[float] = None,
    ) -> "Abstainer":
        """
        Picks the minimum confidence threshold on val_cal that achieves
        error_rate <= target_error_rate on answered samples.
        """
        if target_error_rate is not None:
            self.target_error_rate = float(target_error_rate)

        probs = np.asarray(probs, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)
        n_samples = len(labels)

        if n_samples == 0:
            logger.warning("Empty calibration data provided to Abstainer.fit; setting threshold to 0.5.")
            self.fitted = True
            return self

        confidences = compute_confidence_scores(probs, metric=self.confidence_metric)
        predictions = np.argmax(probs, axis=1)
        errors = (predictions != labels).astype(np.float64)

        # Sort candidates by confidence descending
        sorted_indices = np.argsort(-confidences)
        sorted_confs = confidences[sorted_indices]
        sorted_errors = errors[sorted_indices]

        # Scan candidate thresholds
        # Unique thresholds to evaluate
        unique_confs = np.unique(sorted_confs)[::-1]
        best_threshold = 1.0  # Conservative fallback
        best_cov = 0.0
        best_err = 0.0
        found = False

        for theta in unique_confs:
            answered_mask = confidences >= theta
            n_answered = np.sum(answered_mask)
            if n_answered == 0:
                continue

            emp_error = np.sum(errors[answered_mask]) / n_answered
            coverage = n_answered / n_samples

            if emp_error <= self.target_error_rate:
                # We want maximum coverage (lowest threshold satisfying error condition)
                if coverage >= best_cov:
                    best_threshold = float(theta)
                    best_cov = float(coverage)
                    best_err = float(emp_error)
                    found = True

        if not found:
            # If even the single most confident sample does not satisfy target error,
            # set threshold above maximum confidence to abstain on everything
            best_threshold = float(np.max(confidences) + 1e-4) if len(confidences) > 0 else 1.0
            best_cov = 0.0
            best_err = 0.0

        self.threshold = best_threshold
        self.val_cal_coverage = best_cov
        self.val_cal_error = best_err
        self.fitted = True

        logger.info(
            f"Fitted Abstainer ({self.confidence_metric}): "
            f"target_error={self.target_error_rate}, threshold={self.threshold:.4f}, "
            f"val_cal_cov={self.val_cal_coverage:.2%}, val_cal_err={self.val_cal_error:.2%}"
        )
        return self

    def decide(self, probs: np.ndarray) -> List[str]:
        """
        Determines verdicts for predictions: predicted class label or 'abstain'.
        
        Args:
            probs: Array of shape (N, K) or 1D array of shape (K,).
            
        Returns:
            List of decision strings: class name or 'abstain'.
        """
        probs = np.asarray(probs, dtype=np.float64)
        single_sample = False
        if probs.ndim == 1:
            probs = probs.reshape(1, -1)
            single_sample = True

        confidences = compute_confidence_scores(probs, metric=self.confidence_metric)
        predictions = np.argmax(probs, axis=1)

        decisions = []
        for conf, pred_idx in zip(confidences, predictions):
            if conf >= self.threshold:
                cname = self.class_names[pred_idx] if pred_idx < len(self.class_names) else str(pred_idx)
                decisions.append(cname)
            else:
                decisions.append("abstain")

        return decisions if not single_sample else decisions

    def to_dict(self) -> Dict[str, Any]:
        """Serializes Abstainer configuration and fitted threshold."""
        return {
            "fitted": self.fitted,
            "confidence_metric": self.confidence_metric,
            "target_error_rate": round(self.target_error_rate, 4),
            "threshold": round(self.threshold, 4),
            "val_cal_coverage": round(self.val_cal_coverage, 4),
            "val_cal_error": round(self.val_cal_error, 4),
            "class_names": self.class_names,
        }

    def save(self, filepath: Union[str, Path]) -> None:
        """Saves threshold and settings to a JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved abstention configuration to {filepath}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "Abstainer":
        """Loads Abstainer from a JSON file."""
        filepath = Path(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        abstainer = cls(
            target_error_rate=data.get("target_error_rate", 0.10),
            confidence_metric=data.get("confidence_metric", "max_prob"),
            class_names=data.get("class_names", CLASS_NAMES),
        )
        abstainer.threshold = float(data.get("threshold", 0.5))
        abstainer.val_cal_coverage = float(data.get("val_cal_coverage", 1.0))
        abstainer.val_cal_error = float(data.get("val_cal_error", 0.0))
        abstainer.fitted = data.get("fitted", True)
        return abstainer


def compare_confidence_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    target_error_rate: float = 0.10,
) -> Dict[str, Dict[str, float]]:
    """
    Fits and compares abstention performance using max_prob vs normalized_entropy.
    """
    res = {}
    for metric in ["max_prob", "normalized_entropy"]:
        abstainer = Abstainer(target_error_rate=target_error_rate, confidence_metric=metric)
        abstainer.fit(probs, labels)
        res[metric] = {
            "threshold": round(abstainer.threshold, 4),
            "val_cal_coverage": round(abstainer.val_cal_coverage, 4),
            "val_cal_error": round(abstainer.val_cal_error, 4),
        }
    return res
