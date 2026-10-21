"""
Deep Ensembling for Multimodal Fusion in TrustNet Stage 7.

Trains multiple instances of Stage 6 fusion models (MLP and LightGBM) across
specified random seeds (e.g. 42, 101, 202, 303, 404), saving per-seed logits,
and aggregating predictions via probability averaging.

Evaluates and reports ensemble size effects for 1, 2, 3, and 5 members on:
- ECE (Equal-width)
- Adaptive ECE
- NLL
- Brier Score
- Accuracy & Macro-F1
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.special import softmax
import torch
from torch.utils.data import DataLoader

from trustnet.calibration.metrics import (
    adaptive_expected_calibration_error,
    compute_brier_score,
    compute_nll,
    expected_calibration_error,
)
from trustnet.fusion.gbm import LightGBMFusionClassifier
from trustnet.fusion.mlp import MLPFusionTrainer, MultimodalDataset

logger = logging.getLogger(__name__)


def extract_mlp_logits(
    trainer: MLPFusionTrainer,
    records: List[Dict[str, Any]],
    active_groups: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Extracts raw unnormalized logits [N, 3] from a trained PyTorch MLP model.
    """
    trainer.model.eval()
    ds = MultimodalDataset(records, active_groups=None)
    loader = DataLoader(ds, batch_size=trainer.batch_size, shuffle=False)
    all_logits = []

    device = next(trainer.model.parameters()).device if hasattr(trainer.model, "parameters") else trainer.device
    with torch.no_grad():
        for batch in loader:
            compact = batch["compact"].to(device)
            clip = batch["clip"].to(device)
            text_pooled = batch["text_pooled"].to(device)

            if active_groups is not None:
                from trustnet.fusion.features import FEATURE_GROUPS, FEATURE_NAMES
                inactive_indices = []
                for g, fnames in FEATURE_GROUPS.items():
                    if g not in active_groups:
                        for fn in fnames:
                            inactive_indices.append(FEATURE_NAMES.index(fn))
                if inactive_indices:
                    compact[:, inactive_indices] = 0.0
                if "image" not in active_groups:
                    clip.zero_()
                if "text" not in active_groups:
                    text_pooled.zero_()

            logits = trainer.model(compact, clip, text_pooled)
            all_logits.append(logits.cpu().numpy())

    return np.concatenate(all_logits, axis=0) if all_logits else np.empty((0, 3))


def extract_gbm_logits(
    classifier: LightGBMFusionClassifier,
    records: List[Dict[str, Any]],
    active_groups: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Extracts raw margin scores / logits [N, 3] from a trained LightGBM booster.
    """
    if classifier.booster is None:
        raise ValueError("LightGBM booster is not fitted.")
    from trustnet.fusion.features import build_compact_matrix
    X, _, _ = build_compact_matrix(records, active_groups=active_groups)
    raw_scores = classifier.booster.predict(X, raw_score=True)
    if raw_scores.ndim == 1:
        raw_scores = np.vstack([-raw_scores, raw_scores]).T
    return raw_scores


class DeepEnsemble:
    """
    Manages an ensemble of trained models across multiple random seeds.
    """

    def __init__(
        self,
        model_type: str = "mlp",
        seeds: Optional[List[int]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.model_type = model_type.lower()
        self.seeds = seeds or [42, 101, 202, 303, 404]
        self.config = config or {}
        self.models: List[Any] = []
        self.seed_logits_cache: Dict[int, Dict[str, np.ndarray]] = {}

    def fit(
        self,
        train_records: List[Dict[str, Any]],
        val_records: List[Dict[str, Any]],
        active_groups: Optional[List[str]] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> "DeepEnsemble":
        """
        Trains model members for each seed using val_records (strictly val_select) for early stopping.
        """
        self.models = []
        out_base = Path(output_dir) if output_dir else None

        for seed in self.seeds:
            logger.info(f"Training ensemble member [{self.model_type.upper()}] with seed {seed}...")
            if self.model_type == "mlp":
                mlp_cfg = self.config.get("mlp", {})
                trainer = MLPFusionTrainer(
                    hidden_sizes=mlp_cfg.get("hidden_sizes", [128, 64]),
                    dropout=mlp_cfg.get("dropout", 0.2),
                    modality_dropout_image=self.config.get("modality_dropout_image", 0.3),
                    modality_dropout_evidence=self.config.get("modality_dropout_evidence", 0.3),
                    lr=mlp_cfg.get("learning_rate", 0.001),
                    weight_decay=mlp_cfg.get("weight_decay", 1e-4),
                    epochs=mlp_cfg.get("epochs", 30),
                    batch_size=mlp_cfg.get("batch_size", 4),
                    early_stopping_patience=mlp_cfg.get("early_stopping_patience", 5),
                    seed=seed,
                )
                trainer.fit(
                    train_records=train_records,
                    val_records=val_records,
                    active_groups=active_groups,
                    class_weights_mode=self.config.get("class_weights_mode", "balanced"),
                )
                self.models.append((seed, trainer))
                if out_base:
                    trainer.save(out_base / f"mlp_seed_{seed}")

            elif self.model_type == "gbm":
                gbm_params = dict(self.config.get("lightgbm", {}))
                gbm_params["random_state"] = seed
                clf = LightGBMFusionClassifier(params=gbm_params, seed=seed)
                clf.fit(
                    train_records=train_records,
                    val_records=val_records,
                    active_groups=active_groups,
                    class_weights_mode=self.config.get("class_weights_mode", "balanced"),
                )
                self.models.append((seed, clf))
                if out_base:
                    clf.save(out_base / f"gbm_seed_{seed}")
            else:
                raise ValueError(f"Unsupported model_type: {self.model_type}")

        return self

    def predict_seed_logits(
        self,
        records: List[Dict[str, Any]],
        split_name: str = "eval",
        active_groups: Optional[List[str]] = None,
        save_dir: Optional[Union[str, Path]] = None,
    ) -> List[Tuple[int, np.ndarray]]:
        """
        Computes or retrieves logits for each seed.
        """
        results = []
        save_path = Path(save_dir) if save_dir else None

        for seed, model in self.models:
            if seed in self.seed_logits_cache and split_name in self.seed_logits_cache[seed]:
                logits = self.seed_logits_cache[seed][split_name]
            else:
                if self.model_type == "mlp":
                    logits = extract_mlp_logits(model, records, active_groups=active_groups)
                else:
                    logits = extract_gbm_logits(model, records, active_groups=active_groups)

                self.seed_logits_cache.setdefault(seed, {})[split_name] = logits

            if save_path:
                save_path.mkdir(parents=True, exist_ok=True)
                np.save(save_path / f"logits_{self.model_type}_seed_{seed}_{split_name}.npy", logits)

            results.append((seed, logits))

        return results

    def predict_proba(
        self,
        records: List[Dict[str, Any]],
        n_members: Optional[int] = None,
        split_name: str = "eval",
        active_groups: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Averages probabilities across the first n_members seeds.
        """
        seed_logits = self.predict_seed_logits(
            records=records,
            split_name=split_name,
            active_groups=active_groups,
        )
        if n_members is not None:
            seed_logits = seed_logits[:n_members]

        all_probs = [softmax(logits, axis=1) for _, logits in seed_logits]
        return np.mean(all_probs, axis=0)

    def evaluate_size_effects(
        self,
        records: List[Dict[str, Any]],
        split_name: str = "eval",
        sizes: Optional[List[int]] = None,
        active_groups: Optional[List[str]] = None,
        n_bins: int = 15,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Evaluates ensemble effects for subsets of members (e.g. 1, 2, 3, 5).
        """
        if sizes is None:
            sizes = [1, 2, 3, 5]
        sizes = [s for s in sizes if s <= len(self.models)]

        labels = np.array([r.get("label_idx", 2) for r in records], dtype=np.int64)
        report: Dict[int, Dict[str, Any]] = {}

        for k in sizes:
            probs = self.predict_proba(
                records=records,
                n_members=k,
                split_name=split_name,
                active_groups=active_groups,
            )
            preds = np.argmax(probs, axis=1)
            acc = float(np.mean(preds == labels))
            from sklearn.metrics import f1_score
            f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
            ece = expected_calibration_error(probs, labels, n_bins=n_bins)
            ada_ece = adaptive_expected_calibration_error(probs, labels, n_bins=n_bins)
            nll = compute_nll(probs, labels)
            brier = compute_brier_score(probs, labels)

            report[k] = {
                "members": k,
                "accuracy": round(acc, 4),
                "macro_f1": round(f1, 4),
                "ece": round(float(ece), 4),
                "adaptive_ece": round(float(ada_ece), 4),
                "nll": round(float(nll), 4),
                "brier": round(float(brier), 4),
            }

        return report
