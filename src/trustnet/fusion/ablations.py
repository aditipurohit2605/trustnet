"""
System variants and ablation controls for TrustNet Stage 6.

Variants:
1. text_only: text classifier signals only (logits, probabilities).
2. text_retrieval: text classifier + Stage 3 retrieval signals.
3. text_image: text classifier + Stage 5 visual signals.
4. full: complete multimodal system (text + retrieval + vision).
5. claim_only: claim-only baseline features.
6. shuffled_image (negative control): permutes image feature rows across the dataset
   with a fixed seed, breaking the true image-claim correspondence while preserving
   the marginal feature distributions.
7. shuffled_evidence (negative control): permutes retrieval feature rows across the
   dataset with a fixed seed, breaking evidence-claim alignment.

Guarantees:
Every variant is trained and evaluated with the identical code path, hyperparameters,
seeds, and split definitions.
"""

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

ALL_VARIANTS = [
    "text_only",
    "text_retrieval",
    "text_image",
    "full",
    "claim_only",
    "shuffled_image",
    "shuffled_evidence",
]


def apply_variant_transform(
    records: List[Dict[str, Any]],
    variant: str,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], Optional[List[str]]]:
    """
    Transforms cached records and returns (modified_records, active_feature_groups).
    """
    variant = variant.strip().lower()
    if variant not in ALL_VARIANTS:
        raise ValueError(f"Unknown variant: '{variant}'. Must be one of {ALL_VARIANTS}")

    # Work on shallow copy of record list
    records_out = [dict(r) for r in records]

    if variant == "text_only":
        return records_out, ["text"]

    elif variant == "text_retrieval":
        return records_out, ["text", "retrieval"]

    elif variant == "text_image":
        return records_out, ["text", "image"]

    elif variant == "full":
        return records_out, ["text", "retrieval", "image"]

    elif variant == "claim_only":
        return records_out, ["text"]

    elif variant == "shuffled_image":
        # Permute image features across samples with fixed RNG seed
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(records_out))
        img_feats = [r.get("image_features", {}) for r in records]
        shuffled_img_feats = [img_feats[i] for i in perm]

        for idx, r in enumerate(records_out):
            r_copy = dict(r)
            r_copy["image_features"] = shuffled_img_feats[idx]
            records_out[idx] = r_copy

        logger.info(f"Applied shuffled_image control with seed={seed} (permuted {len(records_out)} rows).")
        return records_out, ["text", "retrieval", "image"]

    elif variant == "shuffled_evidence":
        # Permute retrieval features across samples with fixed RNG seed
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(records_out))
        ret_feats = [r.get("retrieval_features", {}) for r in records]
        shuffled_ret_feats = [ret_feats[i] for i in perm]

        for idx, r in enumerate(records_out):
            r_copy = dict(r)
            r_copy["retrieval_features"] = shuffled_ret_feats[idx]
            records_out[idx] = r_copy

        logger.info(f"Applied shuffled_evidence control with seed={seed} (permuted {len(records_out)} rows).")
        return records_out, ["text", "retrieval", "image"]

    return records_out, ["text", "retrieval", "image"]
