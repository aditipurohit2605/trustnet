"""
Temperature Scaling for TrustNet Stage 7.

Fits temperature parameter T > 0 on validation calibration (val_cal) logits by minimizing
Negative Log-Likelihood (NLL) via SciPy's L-BFGS-B optimizer.

Supports:
- Global temperature scaling (single T for all predictions)
- Optional per-dataset temperature scaling (T_d per dataset, with fallback to T_global)
- JSON serialization and deserialization
- Strict anti-leakage guarantee: never fits on test data.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

logger = logging.getLogger(__name__)


def _nll_loss(t: float, logits: np.ndarray, labels: np.ndarray) -> float:
    """
    Computes cross-entropy / NLL for a given temperature t > 0.
    """
    t = max(t, 1e-4)
    scaled_logits = logits / t
    # Stable log-softmax using scipy logsumexp
    log_denominators = logsumexp(scaled_logits, axis=1)
    true_class_logits = scaled_logits[np.arange(len(labels)), labels]
    loss = -np.mean(true_class_logits - log_denominators)
    return float(loss)


class TemperatureScaler:
    """
    Learns and applies temperature scaling to soften or sharpen model logits.
    Supports global, per-dataset, and optional per-language-form temperature scaling.
    """

    def __init__(
        self,
        per_dataset: bool = False,
        per_language: bool = False,
        default_temp: float = 1.0,
    ):
        self.per_dataset = per_dataset
        self.per_language = per_language
        self.default_temp = float(default_temp)
        self.global_temperature: float = float(default_temp)
        self.dataset_temperatures: Dict[str, float] = {}
        self.language_temperatures: Dict[str, float] = {}
        self.fitted: bool = False

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        datasets: Optional[List[str]] = None,
        languages: Optional[List[str]] = None,
        per_dataset: Optional[bool] = None,
        per_language: Optional[bool] = None,
    ) -> "TemperatureScaler":
        """
        Fits global temperature and optional per-dataset or per-language temperatures on val_cal logits.
        
        Args:
            logits: Array of shape (N, K) containing unscaled prediction logits.
            labels: Array of shape (N,) containing ground truth class indices.
            datasets: Optional list of dataset names of length N.
            languages: Optional list of language form codes (e.g. 'en', 'hi', 'hinglish') of length N.
            per_dataset: Override self.per_dataset flag if specified.
            per_language: Override self.per_language flag if specified.
        """
        logits = np.asarray(logits, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)
        n_samples = len(labels)
        if n_samples == 0:
            logger.warning("Empty calibration data provided to TemperatureScaler.fit; using default 1.0.")
            self.fitted = True
            return self

        if per_dataset is not None:
            self.per_dataset = per_dataset
        if per_language is not None:
            self.per_language = per_language

        # 1. Fit Global Temperature using SciPy L-BFGS-B
        logger.info(f"Fitting global temperature scaling on {n_samples} val_cal samples...")
        res = minimize(
            fun=_nll_loss,
            x0=[1.0],
            args=(logits, labels),
            method="L-BFGS-B",
            bounds=[(0.01, 50.0)],
        )
        self.global_temperature = float(res.x[0])
        logger.info(f"Learned global temperature T = {self.global_temperature:.4f} (NLL: {res.fun:.4f})")

        # 2. Fit Per-Dataset Temperatures if enabled
        if self.per_dataset and datasets is not None:
            self.dataset_temperatures = {}
            unique_datasets = sorted(list(set(datasets)))
            for d in unique_datasets:
                mask = np.array([ds == d for ds in datasets])
                d_logits = logits[mask]
                d_labels = labels[mask]
                if len(d_labels) >= 2 and len(np.unique(d_labels)) > 1:
                    d_res = minimize(
                        fun=_nll_loss,
                        x0=[self.global_temperature],
                        args=(d_logits, d_labels),
                        method="L-BFGS-B",
                        bounds=[(0.01, 50.0)],
                    )
                    self.dataset_temperatures[d] = float(d_res.x[0])
                    logger.info(f"Dataset '{d}' temperature T = {self.dataset_temperatures[d]:.4f}")
                else:
                    logger.info(f"Insufficient samples for dataset '{d}'; falling back to global T.")
                    self.dataset_temperatures[d] = self.global_temperature

        # 3. Fit Per-Language Temperatures if enabled (behind per_language_temperature flag)
        if self.per_language and languages is not None:
            self.language_temperatures = {}
            unique_langs = sorted(list(set(languages)))
            for lang in unique_langs:
                mask = np.array([l == lang for l in languages])
                l_logits = logits[mask]
                l_labels = labels[mask]
                if len(l_labels) >= 2 and len(np.unique(l_labels)) > 1:
                    l_res = minimize(
                        fun=_nll_loss,
                        x0=[self.global_temperature],
                        args=(l_logits, l_labels),
                        method="L-BFGS-B",
                        bounds=[(0.01, 50.0)],
                    )
                    self.language_temperatures[lang] = float(l_res.x[0])
                    logger.info(f"Language '{lang}' temperature T = {self.language_temperatures[lang]:.4f}")
                else:
                    logger.info(f"Insufficient samples for language '{lang}'; falling back to global T.")
                    self.language_temperatures[lang] = self.global_temperature

        self.fitted = True
        return self

    def scale_logits(
        self,
        logits: np.ndarray,
        dataset: Optional[str] = None,
        language: Optional[str] = None,
    ) -> np.ndarray:
        """
        Scales logits by the fitted temperature parameter.
        """
        logits = np.asarray(logits, dtype=np.float64)
        temp = self.get_temperature(dataset=dataset, language=language)
        return logits / temp

    def predict_proba(
        self,
        logits: np.ndarray,
        dataset: Optional[str] = None,
        language: Optional[str] = None,
    ) -> np.ndarray:
        """
        Applies temperature scaling and softmax normalization to produce calibrated probabilities.
        """
        scaled = self.scale_logits(logits, dataset=dataset, language=language)
        return softmax(scaled, axis=1)

    def get_temperature(
        self,
        dataset: Optional[str] = None,
        language: Optional[str] = None,
    ) -> float:
        """
        Retrieves the appropriate temperature for a given language form or dataset,
        with fallback to global T.
        """
        if self.per_language and language and language in self.language_temperatures:
            return self.language_temperatures[language]
        if self.per_dataset and dataset and dataset in self.dataset_temperatures:
            return self.dataset_temperatures[dataset]
        return self.global_temperature

    def to_dict(self) -> Dict[str, Any]:
        """Serializes temperature scaling parameters."""
        return {
            "fitted": self.fitted,
            "per_dataset": self.per_dataset,
            "per_language": self.per_language,
            "global_temperature": round(self.global_temperature, 4),
            "dataset_temperatures": {k: round(v, 4) for k, v in self.dataset_temperatures.items()},
            "language_temperatures": {k: round(v, 4) for k, v in self.language_temperatures.items()},
        }

    def save(self, filepath: Union[str, Path]) -> None:
        """Saves temperature parameters to a JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved temperature parameters to {filepath}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "TemperatureScaler":
        """Loads temperature scaling parameters from a JSON file."""
        filepath = Path(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        scaler = cls(
            per_dataset=data.get("per_dataset", False),
            per_language=data.get("per_language", False),
        )
        scaler.fitted = data.get("fitted", False)
        scaler.global_temperature = float(data.get("global_temperature", 1.0))
        scaler.dataset_temperatures = {
            k: float(v) for k, v in data.get("dataset_temperatures", {}).items()
        }
        scaler.language_temperatures = {
            k: float(v) for k, v in data.get("language_temperatures", {}).items()
        }
        return scaler
