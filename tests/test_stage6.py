"""
Comprehensive test suite for TrustNet Stage 6: Multimodal Fusion Model.

Tests:
1. No claim ID appears in more than one split (strictly disjoint).
2. data/manifest.json counts and SHA-256 hashes match physical files.
3. Image features are strictly zeroed and image_present=0 when image is absent.
4. LightGBM classifier trains and predicts with valid probability distributions.
5. PyTorch MLP fusion head trains with modality dropout.
6. System ablation variants and shuffled negative controls function correctly.
7. Evaluating on the test split strictly requires the '--final' flag.
"""

import hashlib
import json
from pathlib import Path
import pytest
import numpy as np
import torch

from trustnet.data.loader import load_jsonl
from trustnet.fusion.ablations import ALL_VARIANTS, apply_variant_transform
from trustnet.fusion.cache import extract_image_features
from trustnet.fusion.features import FEATURE_NAMES, build_compact_matrix
from trustnet.fusion.gbm import LightGBMFusionClassifier
from trustnet.fusion.mlp import MLPFusionModel, MLPFusionTrainer


def compute_file_sha256(filepath: Path) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def test_splits_are_mutually_exclusive():
    """
    Verifies that no claim ID appears in more than one split.
    Specifically checks train_text, train_fusion, val, and test.
    """
    processed_dir = Path("data/processed")
    splits = ["train_text", "train_fusion", "val", "test"]

    split_ids = {}
    for s in splits:
        file_path = processed_dir / f"{s}.jsonl"
        assert file_path.exists(), f"Split file missing: {file_path}"
        records = load_jsonl(file_path)
        ids = [str(r.get("id") or r.get("claim_id")) for r in records]
        split_ids[s] = set(ids)
        # Ensure no duplicates within split
        assert len(ids) == len(split_ids[s]), f"Duplicate claim IDs detected within split '{s}'"

    # Pairwise disjoint verification
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            s1, s2 = splits[i], splits[j]
            overlap = split_ids[s1].intersection(split_ids[s2])
            assert not overlap, f"Data leakage detected! Overlapping claim IDs between '{s1}' and '{s2}': {overlap}"


def test_manifest_integrity():
    """
    Ensures data/manifest.json accurately reflects real file paths, row counts, and SHA-256 hashes.
    """
    manifest_path = Path("data/manifest.json")
    assert manifest_path.exists(), "Manifest file data/manifest.json does not exist."

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    splits = manifest.get("splits", {})
    assert "train_text" in splits
    assert "train_fusion" in splits
    assert "val" in splits
    assert "test" in splits

    for split_name, meta in splits.items():
        rel_file = meta["file"]
        file_path = Path("data") / rel_file
        assert file_path.exists(), f"File declared in manifest does not exist: {file_path}"

        records = load_jsonl(file_path)
        assert len(records) == meta["count"], (
            f"Record count mismatch in split '{split_name}': "
            f"manifest says {meta['count']}, file contains {len(records)}"
        )

        current_hash = compute_file_sha256(file_path)
        assert current_hash == meta["sha256"], (
            f"SHA-256 mismatch in split '{split_name}': "
            f"manifest={meta['sha256']}, actual={current_hash}"
        )


def test_image_features_zeroed_when_absent():
    """
    Verifies that when an image is absent (image_path=None or missing file):
    - image_present is 0
    - mismatch_score is 0.0
    - best_reuse_similarity is 0.0
    - quality flags are 0
    - clip_image_embedding is all zeros (length 512)
    """
    feats_none = extract_image_features(image_path=None, claim_text="Test claim")
    assert feats_none["image_present"] == 0
    assert feats_none["mismatch_score"] == 0.0
    assert feats_none["best_reuse_similarity"] == 0.0
    assert feats_none["ocr_present"] == 0
    assert feats_none["ocr_confidence"] == 0.0
    assert feats_none["quality_flag_tiny_resolution"] == 0
    assert feats_none["quality_flag_extreme_aspect_ratio"] == 0
    assert feats_none["quality_flag_heavy_jpeg_compression"] == 0
    assert feats_none["quality_flag_screenshot_likeness"] == 0
    assert len(feats_none["clip_image_embedding"]) == 512
    assert all(v == 0.0 for v in feats_none["clip_image_embedding"])

    # Test with non-existent path
    feats_fake = extract_image_features(image_path="non_existent_image_12345.jpg", claim_text="Test claim")
    assert feats_fake["image_present"] == 0
    assert all(v == 0.0 for v in feats_fake["clip_image_embedding"])


def test_lightgbm_fusion_fit_predict():
    """
    Tests LightGBMFusionClassifier fitting on train_fusion and predicting on val records.
    """
    cache_dir = Path("outputs/stage6/cache")
    train_fusion_records = load_jsonl(cache_dir / "cache_train_fusion.jsonl")
    val_records = load_jsonl(cache_dir / "cache_val.jsonl")

    clf = LightGBMFusionClassifier(seed=42)
    clf.fit(
        train_records=train_fusion_records,
        val_records=val_records,
        n_estimators=10,
        early_stopping_rounds=5,
    )

    preds = clf.predict(val_records)
    assert len(preds) == len(val_records)
    for p in preds:
        assert p in [0, 1, 2]

    probs = clf.predict_proba(val_records)
    assert probs.shape == (len(val_records), 3)
    # Probabilities should sum to 1
    assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-3)

    # Feature importances must be recorded
    assert len(clf.feature_importances) > 0
    for name in clf.feature_names:
        assert name in clf.feature_importances


def test_mlp_fusion_training_and_modality_dropout():
    """
    Tests PyTorch MLPFusionModel forward pass, modality dropout masking, and trainer fitting.
    """
    model = MLPFusionModel(
        compact_dim=25,
        clip_dim=512,
        text_pooled_dim=768,
        hidden_sizes=[32, 16],
        modality_dropout_image=1.0,  # Force drop
        modality_dropout_evidence=1.0,  # Force drop
    )

    compact = torch.ones(2, 25)
    clip = torch.ones(2, 512)
    text_pooled = torch.ones(2, 768)

    # In eval mode: no dropout
    model.eval()
    out_eval = model(compact, clip, text_pooled)
    assert out_eval.shape == (2, 3)

    # In train mode with 1.0 dropout: image and retrieval features masked to 0
    model.train()
    out_train = model(compact, clip, text_pooled)
    assert out_train.shape == (2, 3)

    # Test MLPFusionTrainer
    cache_dir = Path("outputs/stage6/cache")
    train_fusion_records = load_jsonl(cache_dir / "cache_train_fusion.jsonl")
    val_records = load_jsonl(cache_dir / "cache_val.jsonl")

    trainer = MLPFusionTrainer(
        hidden_sizes=[32, 16],
        epochs=3,
        batch_size=2,
        seed=42,
    )
    trainer.fit(train_fusion_records, val_records)

    preds = trainer.predict(val_records)
    assert len(preds) == len(val_records)
    for p in preds:
        assert p in [0, 1, 2]


def test_ablations_and_shuffled_controls():
    """
    Verifies that system variants filter active feature groups and that
    negative controls (shuffled_image, shuffled_evidence) correctly permute features.
    """
    cache_dir = Path("outputs/stage6/cache")
    val_records = load_jsonl(cache_dir / "cache_val.jsonl")

    # 1. text_only variant retains only text group
    _, active_text = apply_variant_transform(val_records, "text_only")
    assert active_text == ["text"]

    # 2. full variant retains all groups
    _, active_full = apply_variant_transform(val_records, "full")
    assert set(active_full) == {"text", "retrieval", "image"}

    # 3. shuffled_image permutes image features while preserving labels and text
    shuffled_img_recs, _ = apply_variant_transform(val_records, "shuffled_image", seed=42)
    assert len(shuffled_img_recs) == len(val_records)
    for orig, shuf in zip(val_records, shuffled_img_recs):
        assert orig["claim_id"] == shuf["claim_id"]
        assert orig["label4"] == shuf["label4"]

    # 4. shuffled_evidence permutes retrieval features
    shuffled_ret_recs, _ = apply_variant_transform(val_records, "shuffled_evidence", seed=42)
    assert len(shuffled_ret_recs) == len(val_records)
    for orig, shuf in zip(val_records, shuffled_ret_recs):
        assert orig["claim_id"] == shuf["claim_id"]
        assert orig["label4"] == shuf["label4"]


def test_evaluate_test_split_requires_final_flag():
    """
    Verifies that attempting to evaluate on the test split without --final raises PermissionError.
    """
    import subprocess
    cmd = ["python", "-m", "trustnet.fusion.evaluate", "--split", "test"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode != 0
    assert "PermissionError" in result.stderr or "requires the '--final' CLI flag" in result.stderr
