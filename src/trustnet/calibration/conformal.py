"""
Split Conformal Prediction for TrustNet Stage 7.

Computes finite-sample distribution-free prediction sets with guaranteed marginal
coverage >= 1 - alpha on validation calibration (val_cal) data.

Method:
- Non-conformity score s_i = 1 - p_{i, y_i}
- Quantile cutoff q_hat = np.quantile(scores, ceil((n+1)(1-alpha))/n)
- Prediction set C(p) = { k : 1 - p_k <= q_hat } = { k : p_k >= 1 - q_hat }
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

CLASS_NAMES = ["verified", "contradicted", "insufficient_evidence"]


class SplitConformalPredictor:
    """
    Split Conformal Prediction set generator for multiclass classification.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        class_names: Optional[List[str]] = None,
    ):
        self.alpha = float(alpha)
        self.class_names = class_names or CLASS_NAMES
        self.q_hat: float = 1.0
        self.fitted: bool = False
        self.n_cal_samples: int = 0

    def fit(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        alpha: Optional[float] = None,
    ) -> "SplitConformalPredictor":
        """
        Fits conformal cutoff on val_cal probabilities and labels.
        """
        if alpha is not None:
            self.alpha = float(alpha)

        probs = np.asarray(probs, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)
        n = len(labels)
        self.n_cal_samples = n

        if n == 0:
            logger.warning("Empty calibration data; setting q_hat = 1.0 (empty or trivial sets).")
            self.fitted = True
            return self

        # Extract true class probabilities
        true_probs = probs[np.arange(n), labels]
        # Non-conformity scores: higher score means less conforming (more surprising)
        scores = 1.0 - true_probs

        # Finite-sample adjusted quantile level
        q_level = min(1.0, float(np.ceil((n + 1) * (1.0 - self.alpha)) / n))
        self.q_hat = float(np.quantile(scores, q_level, method="higher" if hasattr(np, "quantile") else "linear"))
        self.fitted = True

        logger.info(
            f"Fitted SplitConformalPredictor (alpha={self.alpha}): "
            f"q_level={q_level:.4f}, q_hat={self.q_hat:.4f} across {n} samples."
        )
        return self

    def predict_sets(
        self,
        probs: np.ndarray,
    ) -> List[List[str]]:
        """
        Constructs prediction set for each probability distribution.
        
        Args:
            probs: Array of shape (N, K).
            
        Returns:
            List of lists of class names included in the conformal prediction set.
        """
        probs = np.asarray(probs, dtype=np.float64)
        if probs.ndim == 1:
            probs = probs.reshape(1, -1)

        threshold = 1.0 - self.q_hat
        prediction_sets = []

        for p in probs:
            included_indices = np.where(p >= threshold)[0]
            # If empty due to numerical boundary, guarantee at least the argmax class
            if len(included_indices) == 0:
                included_indices = [np.argmax(p)]

            class_set = [
                self.class_names[idx] if idx < len(self.class_names) else str(idx)
                for idx in included_indices
            ]
            prediction_sets.append(class_set)

        return prediction_sets

    def evaluate(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Evaluates empirical coverage and average set size on evaluation data.
        """
        probs = np.asarray(probs, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)
        n = len(labels)
        if n == 0:
            return {"coverage": 0.0, "avg_set_size": 0.0}

        pred_sets = self.predict_sets(probs)
        covered = 0
        set_sizes = []

        for idx, pset in enumerate(pred_sets):
            true_class_name = self.class_names[labels[idx]]
            if true_class_name in pset:
                covered += 1
            set_sizes.append(len(pset))

        empirical_coverage = covered / n
        avg_set_size = float(np.mean(set_sizes))

        return {
            "alpha": self.alpha,
            "target_coverage": round(1.0 - self.alpha, 4),
            "empirical_coverage": round(empirical_coverage, 4),
            "avg_set_size": round(avg_set_size, 4),
            "q_hat": round(self.q_hat, 4),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alpha": round(self.alpha, 4),
            "q_hat": round(self.q_hat, 4),
            "fitted": self.fitted,
            "n_cal_samples": self.n_cal_samples,
            "class_names": self.class_names,
        }

    def save(self, filepath: Union[str, Path]) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved conformal predictor to {filepath}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "SplitConformalPredictor":
        filepath = Path(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        pred = cls(
            alpha=data.get("alpha", 0.1),
            class_names=data.get("class_names", CLASS_NAMES),
        )
        pred.q_hat = float(data.get("q_hat", 1.0))
        pred.fitted = data.get("fitted", True)
        pred.n_cal_samples = int(data.get("n_cal_samples", 0))
        return pred
