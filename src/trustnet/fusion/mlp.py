"""
PyTorch Multimodal MLP Fusion Head for TrustNet Stage 6.

Architecture:
- Inputs:
  1. Compact tabular features (25-dim text, retrieval, and image signals).
  2. Dense OpenCLIP image embedding (512-dim), projected to lower dimension via linear layer.
  3. Dense Stage 2 pooled text embedding, projected to lower dimension via linear layer.
- Fuses projected representations into deep MLP layers with LayerNorm, GELU, and Dropout.
- Modality Dropout during training:
  - Randomly drops visual signals (zeros features, sets image_present=0, zeros CLIP embedding).
  - Randomly drops retrieval evidence signals (zeros retrieval features).
- Monitors validation Macro-F1 strictly for early stopping and best checkpoint selection.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from trustnet.fusion.features import FEATURE_GROUPS, FEATURE_NAMES, build_compact_matrix

logger = logging.getLogger(__name__)


class MultimodalDataset(Dataset):
    """Dataset providing compact features, projected visual embeddings, and text embeddings."""

    def __init__(
        self,
        records: List[Dict[str, Any]],
        active_groups: Optional[List[str]] = None,
    ):
        self.records = records
        self.X, self.y, self.feature_names = build_compact_matrix(records, active_groups=active_groups)

        # Extract CLIP embeddings and pooled text embeddings
        clip_list = []
        text_pooled_list = []

        for r in records:
            img_feats = r.get("image_features", {})
            clip_emb = img_feats.get("clip_image_embedding", [0.0] * 512)
            clip_list.append(clip_emb)

            t_emb = r.get("text_pooled", [0.0] * 768)
            text_pooled_list.append(t_emb)

        self.clip_matrix = np.array(clip_list, dtype=np.float32)
        self.text_pooled_matrix = np.array(text_pooled_list, dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            "compact": torch.tensor(self.X[idx], dtype=torch.float32),
            "clip": torch.tensor(self.clip_matrix[idx], dtype=torch.float32),
            "text_pooled": torch.tensor(self.text_pooled_matrix[idx], dtype=torch.float32),
            "label": torch.tensor(self.y[idx], dtype=torch.long),
        }


class MLPFusionModel(nn.Module):
    """
    Multimodal neural fusion head combining tabular signals and deep dense embeddings.
    """

    def __init__(
        self,
        compact_dim: int = 25,
        clip_dim: int = 512,
        text_pooled_dim: int = 768,
        clip_proj_dim: int = 64,
        text_proj_dim: int = 64,
        hidden_sizes: Optional[List[int]] = None,
        num_classes: int = 3,
        dropout: float = 0.2,
        modality_dropout_image: float = 0.3,
        modality_dropout_evidence: float = 0.3,
        feature_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.compact_dim = compact_dim
        self.feature_names = feature_names or list(FEATURE_NAMES)
        self.modality_dropout_image = modality_dropout_image
        self.modality_dropout_evidence = modality_dropout_evidence

        # Projection layers for high-dimensional embeddings
        self.clip_projector = nn.Sequential(
            nn.Linear(clip_dim, clip_proj_dim),
            nn.LayerNorm(clip_proj_dim),
            nn.GELU(),
        )

        self.text_projector = nn.Sequential(
            nn.Linear(text_pooled_dim, text_proj_dim),
            nn.LayerNorm(text_proj_dim),
            nn.GELU(),
        )

        # Total fused input dimension
        fused_dim = compact_dim + clip_proj_dim + text_proj_dim
        hidden_sizes = hidden_sizes or [128, 64]

        # Multi-layer perceptron trunk
        layers: List[nn.Module] = []
        in_dim = fused_dim
        for h_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        self.mlp_trunk = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_dim, num_classes)

        # Indices for modality dropout masking
        self.img_indices = [
            i for i, name in enumerate(self.feature_names)
            if name in FEATURE_GROUPS["image"]
        ]
        self.ret_indices = [
            i for i, name in enumerate(self.feature_names)
            if name in FEATURE_GROUPS["retrieval"]
        ]

    def forward(
        self,
        compact: torch.Tensor,
        clip: torch.Tensor,
        text_pooled: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with optional modality dropout during training.
        """
        compact_feat = compact.clone()
        clip_feat = clip.clone()

        # Apply modality dropout only during training mode
        if self.training:
            batch_size = compact.shape[0]

            # 1. Image Modality Dropout
            if self.img_indices and self.modality_dropout_image > 0.0:
                img_mask = torch.rand(batch_size, device=compact.device) < self.modality_dropout_image
                if img_mask.any():
                    compact_feat[img_mask, :][:, self.img_indices] = 0.0
                    clip_feat[img_mask] = 0.0

            # 2. Retrieval Modality Dropout
            if self.ret_indices and self.modality_dropout_evidence > 0.0:
                ret_mask = torch.rand(batch_size, device=compact.device) < self.modality_dropout_evidence
                if ret_mask.any():
                    compact_feat[ret_mask, :][:, self.ret_indices] = 0.0

        # Project dense embeddings
        proj_clip = self.clip_projector(clip_feat)
        proj_text = self.text_projector(text_pooled)

        # Concatenate into fused multimodal representation
        fused = torch.cat([compact_feat, proj_clip, proj_text], dim=-1)

        # Pass through MLP
        hidden = self.mlp_trunk(fused)
        logits = self.classifier(hidden)
        return logits


class MLPFusionTrainer:
    """
    Orchestrates PyTorch MLP training with early stopping on validation Macro-F1.
    """

    def __init__(
        self,
        hidden_sizes: Optional[List[int]] = None,
        dropout: float = 0.2,
        modality_dropout_image: float = 0.3,
        modality_dropout_evidence: float = 0.3,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        epochs: int = 30,
        batch_size: int = 4,
        early_stopping_patience: int = 5,
        seed: int = 42,
        device: str = "auto",
    ):
        self.hidden_sizes = hidden_sizes or [128, 64]
        self.dropout = dropout
        self.modality_dropout_image = modality_dropout_image
        self.modality_dropout_evidence = modality_dropout_evidence
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = early_stopping_patience
        self.seed = seed

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model: Optional[MLPFusionModel] = None
        self.best_macro_f1: float = 0.0
        self.feature_names: List[str] = []

    def fit(
        self,
        train_records: List[Dict[str, Any]],
        val_records: List[Dict[str, Any]],
        active_groups: Optional[List[str]] = None,
        class_weights_mode: Optional[str] = "balanced",
    ) -> "MLPFusionTrainer":
        """
        Trains the MLP on train_fusion with validation Macro-F1 checkpointing.
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        train_ds = MultimodalDataset(train_records, active_groups=active_groups)
        val_ds = MultimodalDataset(val_records, active_groups=active_groups)

        self.feature_names = train_ds.feature_names
        compact_dim = train_ds.X.shape[1]
        clip_dim = train_ds.clip_matrix.shape[1]
        text_pooled_dim = train_ds.text_pooled_matrix.shape[1]

        # Instantiate model
        self.model = MLPFusionModel(
            compact_dim=compact_dim,
            clip_dim=clip_dim,
            text_pooled_dim=text_pooled_dim,
            hidden_sizes=self.hidden_sizes,
            dropout=self.dropout,
            modality_dropout_image=self.modality_dropout_image,
            modality_dropout_evidence=self.modality_dropout_evidence,
            feature_names=self.feature_names,
        ).to(self.device)

        # Loss function with optional balanced class weights
        criterion = nn.CrossEntropyLoss()
        if class_weights_mode == "balanced" and len(np.unique(train_ds.y)) > 1:
            classes = np.unique(train_ds.y)
            w = compute_class_weight("balanced", classes=classes, y=train_ds.y)
            weight_tensor = torch.ones(3, dtype=torch.float32)
            for c, weight_val in zip(classes, w):
                weight_tensor[c] = float(weight_val)
            criterion = nn.CrossEntropyLoss(weight=weight_tensor.to(self.device))

        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        best_weights = None
        best_val_f1 = -1.0
        patience_counter = 0

        logger.info(f"Training PyTorch MLP on {len(train_ds)} samples with {compact_dim} compact features...")

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss = 0.0

            for batch in train_loader:
                compact = batch["compact"].to(self.device)
                clip = batch["clip"].to(self.device)
                text_pooled = batch["text_pooled"].to(self.device)
                labels = batch["label"].to(self.device)

                optimizer.zero_grad()
                logits = self.model(compact, clip, text_pooled)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Evaluate on validation split
            val_f1 = self.evaluate_macro_f1(val_ds)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                logger.info(f"Early stopping triggered at epoch {epoch}. Best Val Macro-F1: {best_val_f1:.4f}")
                break

        if best_weights is not None:
            self.model.load_state_dict(best_weights)
            self.best_macro_f1 = best_val_f1

        logger.info(f"MLP training complete. Best Val Macro-F1: {self.best_macro_f1:.4f}")
        return self

    def evaluate_macro_f1(self, dataset: MultimodalDataset) -> float:
        """Evaluates Macro-F1 score on a dataset without dropout."""
        self.model.eval()
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in loader:
                compact = batch["compact"].to(self.device)
                clip = batch["clip"].to(self.device)
                text_pooled = batch["text_pooled"].to(self.device)
                labels = batch["label"]

                logits = self.model(compact, clip, text_pooled)
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(labels.numpy())

        return float(f1_score(all_targets, all_preds, average="macro", zero_division=0))

    def predict_proba(
        self,
        records: List[Dict[str, Any]],
        active_groups: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Computes predicted probabilities for records."""
        self.model.eval()
        ds = MultimodalDataset(records, active_groups=active_groups)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
        all_probs = []

        with torch.no_grad():
            for batch in loader:
                compact = batch["compact"].to(self.device)
                clip = batch["clip"].to(self.device)
                text_pooled = batch["text_pooled"].to(self.device)

                logits = self.model(compact, clip, text_pooled)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                all_probs.append(probs)

        return np.concatenate(all_probs, axis=0) if all_probs else np.empty((0, 3))

    def predict(
        self,
        records: List[Dict[str, Any]],
        active_groups: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Computes discrete class predictions."""
        probs = self.predict_proba(records, active_groups=active_groups)
        return np.argmax(probs, axis=1)

    def save(self, output_dir: Union[str, Path]) -> None:
        """Saves model weights and config to disk."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), out_path / "mlp_model.pt")

        config = {
            "compact_dim": self.model.compact_dim,
            "hidden_sizes": self.hidden_sizes,
            "dropout": self.dropout,
            "best_macro_f1": float(self.best_macro_f1),
            "feature_names": self.feature_names,
        }
        with open(out_path / "mlp_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Saved PyTorch MLP model to {out_path}")
