"""
Feature vector construction and inspection for TrustNet Stage 6 Multimodal Fusion.

Provides:
- Compact, interpretable tabular feature matrix construction for LightGBM and MLP.
- Named feature registry for transparent feature importance attribution.
- Modality dropout transformations for robust training under missing inputs.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


FEATURE_NAMES = [
    # 1. Text Model Logits and Calibrated Softmax Probabilities
    "text_logit_verified",
    "text_logit_contradicted",
    "text_logit_insufficient",
    "text_prob_verified",
    "text_prob_contradicted",
    "text_prob_insufficient",
    "text_prob_margin",
    # 2. Stage 3 Retrieval and Evidence Features
    "retrieval_top1_score",
    "retrieval_top1_top2_gap",
    "retrieval_num_passages_above_threshold",
    "retrieval_mean_score",
    "retrieval_source_corpus_passage",
    "retrieval_source_government",
    "retrieval_source_news",
    "retrieval_source_other",
    "retrieval_agreement_feature",
    # 3. Stage 5 Image and Provenance Signals
    "image_present",
    "image_mismatch_score",
    "image_best_reuse_similarity",
    "image_ocr_present",
    "image_ocr_confidence",
    "quality_tiny_resolution",
    "quality_extreme_aspect_ratio",
    "quality_heavy_jpeg_compression",
    "quality_screenshot_likeness",
]

FEATURE_GROUPS = {
    "text": [
        "text_logit_verified",
        "text_logit_contradicted",
        "text_logit_insufficient",
        "text_prob_verified",
        "text_prob_contradicted",
        "text_prob_insufficient",
        "text_prob_margin",
    ],
    "retrieval": [
        "retrieval_top1_score",
        "retrieval_top1_top2_gap",
        "retrieval_num_passages_above_threshold",
        "retrieval_mean_score",
        "retrieval_source_corpus_passage",
        "retrieval_source_government",
        "retrieval_source_news",
        "retrieval_source_other",
        "retrieval_agreement_feature",
    ],
    "image": [
        "image_present",
        "image_mismatch_score",
        "image_best_reuse_similarity",
        "image_ocr_present",
        "image_ocr_confidence",
        "quality_tiny_resolution",
        "quality_extreme_aspect_ratio",
        "quality_heavy_jpeg_compression",
        "quality_screenshot_likeness",
    ],
}


def softmax(x: np.ndarray) -> np.ndarray:
    """Computes numerically stable softmax over a 1D or 2D array."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def extract_compact_vector(record: Dict[str, Any]) -> Tuple[List[float], List[str]]:
    """
    Extracts the compact 25-dimensional feature vector from a single cached record.
    """
    # 1. Text features
    logits = np.array(record.get("text_logits", [0.0, 0.0, 0.0]), dtype=np.float32)
    probs = softmax(logits)
    sorted_probs = np.sort(probs)
    prob_margin = float(sorted_probs[-1] - sorted_probs[-2]) if len(sorted_probs) >= 2 else 0.0

    # 2. Retrieval features
    ret = record.get("retrieval_features", {})

    # 3. Image features
    img = record.get("image_features", {})

    feature_values = [
        float(logits[0]),
        float(logits[1]),
        float(logits[2]),
        float(probs[0]),
        float(probs[1]),
        float(probs[2]),
        float(prob_margin),
        float(ret.get("top1_score", 0.0)),
        float(ret.get("top1_top2_gap", 0.0)),
        float(ret.get("num_passages_above_threshold", 0.0)),
        float(ret.get("mean_score", 0.0)),
        float(ret.get("source_type_corpus_passage", 0.0)),
        float(ret.get("source_type_government", 0.0)),
        float(ret.get("source_type_news", 0.0)),
        float(ret.get("source_type_other", 0.0)),
        float(ret.get("agreement_feature", 0.0)),
        float(img.get("image_present", 0)),
        float(img.get("mismatch_score", 0.0)),
        float(img.get("best_reuse_similarity", 0.0)),
        float(img.get("ocr_present", 0)),
        float(img.get("ocr_confidence", 0.0)),
        float(img.get("quality_flag_tiny_resolution", 0)),
        float(img.get("quality_flag_extreme_aspect_ratio", 0)),
        float(img.get("quality_flag_heavy_jpeg_compression", 0)),
        float(img.get("quality_flag_screenshot_likeness", 0)),
    ]

    return feature_values, FEATURE_NAMES


def build_compact_matrix(
    cached_records: List[Dict[str, Any]],
    active_groups: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Constructs compact feature matrix X and target vector y from cached records.

    Args:
        cached_records: List of cached record dictionaries.
        active_groups: List of feature groups to retain ('text', 'retrieval', 'image').
                       If None, retains all groups.

    Returns:
        X (np.ndarray of shape [N, D]), y (np.ndarray of shape [N]), selected_feature_names
    """
    if not cached_records:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32), np.empty((0,), dtype=np.int64), FEATURE_NAMES

    X_list = []
    y_list = []

    for r in cached_records:
        feat_vals, _ = extract_compact_vector(r)
        X_list.append(feat_vals)
        y_list.append(int(r.get("label_idx", 2)))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    # Filter features based on active groups if specified
    if active_groups is not None:
        selected_indices = []
        selected_names = []
        for g in active_groups:
            if g in FEATURE_GROUPS:
                for fname in FEATURE_GROUPS[g]:
                    idx = FEATURE_NAMES.index(fname)
                    selected_indices.append(idx)
                    selected_names.append(fname)

        if selected_indices:
            X = X[:, selected_indices]
            return X, y, selected_names

    return X, y, list(FEATURE_NAMES)


def apply_modality_dropout(
    X: np.ndarray,
    feature_names: List[str],
    p_image: float = 0.3,
    p_evidence: float = 0.3,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Applies modality dropout to training feature vectors:
    - Randomly drops image features with probability p_image (zeroing features and setting image_present=0).
    - Randomly drops retrieval features with probability p_evidence (zeroing retrieval features).
    """
    X_out = X.copy()
    rng = np.random.RandomState(seed)
    n_samples = len(X_out)

    img_indices = [i for i, name in enumerate(feature_names) if name in FEATURE_GROUPS["image"]]
    ret_indices = [i for i, name in enumerate(feature_names) if name in FEATURE_GROUPS["retrieval"]]

    # Image modality dropout
    if img_indices and p_image > 0.0:
        mask_img = rng.rand(n_samples) < p_image
        X_out[mask_img, :][:, img_indices] = 0.0

    # Evidence modality dropout
    if ret_indices and p_evidence > 0.0:
        mask_ret = rng.rand(n_samples) < p_evidence
        X_out[mask_ret, :][:, ret_indices] = 0.0

    return X_out
