"""
LightGBM Multimodal Fusion Model for TrustNet Stage 6.

Features:
- Multiclass GBDT classifier trained strictly on train_fusion.
- Early stopping monitored exclusively on validation Macro-F1 score.
- Saves model weights, feature names, and readable feature importance attributions.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import lightgbm as lgb
import numpy as np
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

from trustnet.fusion.features import build_compact_matrix

logger = logging.getLogger(__name__)


def macro_f1_eval(preds: np.ndarray, train_data: lgb.Dataset) -> Tuple[str, float, bool]:
    """
    Custom LightGBM evaluation metric for multiclass Macro-F1 score.
    Higher score is better (is_higher_better=True).
    """
    labels = train_data.get_label().astype(int)
    # preds has shape [N * num_class] or [N, num_class]
    num_class = len(np.unique(labels))
    if preds.ndim == 1 and num_class > 1:
        preds = preds.reshape(num_class, -1).T

    pred_classes = np.argmax(preds, axis=1)
    score = f1_score(labels, pred_classes, average="macro", zero_division=0)
    return "macro_f1", float(score), True


class LightGBMFusionClassifier:
    """
    Multimodal gradient boosted decision tree classifier.
    """

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ):
        self.seed = seed
        default_params = {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "None",  # Use custom macro_f1 metric
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 15,
            "max_depth": 4,
            "min_data_in_leaf": 1,
            "feature_fraction": 0.9,
            "seed": self.seed,
            "verbosity": -1,
        }
        if params:
            default_params.update(params)
        self.params = default_params

        self.booster: Optional[lgb.Booster] = None
        self.feature_names: List[str] = []
        self.feature_importances: Dict[str, float] = {}

    def fit(
        self,
        train_records: List[Dict[str, Any]],
        val_records: List[Dict[str, Any]],
        active_groups: Optional[List[str]] = None,
        n_estimators: int = 100,
        early_stopping_rounds: int = 15,
        class_weights_mode: Optional[str] = "balanced",
    ) -> "LightGBMFusionClassifier":
        """
        Fits the LightGBM model on train_fusion, tracking early stopping on val macro-F1.
        """
        X_train, y_train, feature_names = build_compact_matrix(train_records, active_groups=active_groups)
        X_val, y_val, _ = build_compact_matrix(val_records, active_groups=active_groups)
        self.feature_names = feature_names

        # Compute sample weights if balanced mode is active
        sample_weights = None
        if class_weights_mode == "balanced" and len(np.unique(y_train)) > 1:
            classes = np.unique(y_train)
            weights = compute_class_weight("balanced", classes=classes, y=y_train)
            weight_map = {c: w for c, w in zip(classes, weights)}
            sample_weights = np.array([weight_map[label] for label in y_train], dtype=np.float32)

        # Create LightGBM datasets
        dtrain = lgb.Dataset(
            X_train,
            label=y_train,
            feature_name=feature_names,
            weight=sample_weights,
            free_raw_data=False,
        )
        dval = lgb.Dataset(
            X_val,
            label=y_val,
            feature_name=feature_names,
            reference=dtrain,
            free_raw_data=False,
        )

        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),  # Silent
        ]

        logger.info(f"Training LightGBM on {len(X_train)} samples with {len(feature_names)} features...")
        self.booster = lgb.train(
            self.params,
            dtrain,
            num_boost_round=n_estimators,
            valid_sets=[dval],
            valid_names=["val"],
            feval=macro_f1_eval,
            callbacks=callbacks,
        )

        # Compute and record feature importances (split and gain)
        importance_gain = self.booster.feature_importance(importance_type="gain")
        importance_split = self.booster.feature_importance(importance_type="split")

        self.feature_importances = {}
        for name, gain, split in zip(feature_names, importance_gain, importance_split):
            self.feature_importances[name] = {
                "gain": round(float(gain), 4),
                "split_count": int(split),
            }

        logger.info(f"LightGBM training complete. Best iteration: {self.booster.best_iteration}")
        return self

    def predict_proba(
        self,
        records_or_matrix: Union[List[Dict[str, Any]], np.ndarray],
        active_groups: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Returns predicted probability distribution [N, num_classes]."""
        if self.booster is None:
            raise ValueError("Model is not fitted.")

        if isinstance(records_or_matrix, list):
            X, _, _ = build_compact_matrix(records_or_matrix, active_groups=active_groups)
        else:
            X = records_or_matrix

        preds = self.booster.predict(X)
        if preds.ndim == 1:
            # Binary fallback
            preds = np.vstack([1.0 - preds, preds]).T
        return preds

    def predict(
        self,
        records_or_matrix: Union[List[Dict[str, Any]], np.ndarray],
        active_groups: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Returns class predictions [N]."""
        probs = self.predict_proba(records_or_matrix, active_groups=active_groups)
        return np.argmax(probs, axis=1)

    def save(self, output_dir: Union[str, Path]) -> None:
        """Saves model, feature names, and importance dictionary to disk."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        if self.booster is not None:
            self.booster.save_model(str(out_path / "lightgbm_model.txt"))

        with open(out_path / "feature_names.json", "w", encoding="utf-8") as f:
            json.dump(self.feature_names, f, indent=2)

        with open(out_path / "feature_importances.json", "w", encoding="utf-8") as f:
            json.dump(self.feature_importances, f, indent=2)

        logger.info(f"Saved LightGBM model and importances to {out_path}")

    @classmethod
    def load(cls, model_dir: Union[str, Path]) -> "LightGBMFusionClassifier":
        """Loads a saved LightGBM model from disk."""
        in_path = Path(model_dir)
        model_file = in_path / "lightgbm_model.txt"
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        classifier = cls()
        classifier.booster = lgb.Booster(model_file=str(model_file))

        feat_file = in_path / "feature_names.json"
        if feat_file.exists():
            with open(feat_file, "r", encoding="utf-8") as f:
                classifier.feature_names = json.load(f)

        imp_file = in_path / "feature_importances.json"
        if imp_file.exists():
            with open(imp_file, "r", encoding="utf-8") as f:
                classifier.feature_importances = json.load(f)

        return classifier
