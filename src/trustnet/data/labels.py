"""
Canonical label mappings for fact-checking datasets to TrustNet Verdict enum.

Supported dataset taxonomies:
- FEVER (SUPPORTS, REFUTES, NOT ENOUGH INFO)
- MOCHEG (true, false, mixture, no_evidence)
- Factify (Support_Multimodal, Support_Text, Refute, Insufficient_Text, etc.)
- NewsCLIPpings (matching, not_matching)
"""

from typing import Dict, Optional, Union
from trustnet.schemas import Verdict

# Canonical FEVER mapping
FEVER_LABEL_MAP: Dict[str, Verdict] = {
    "SUPPORTS": Verdict.VERIFIED,
    "REFUTES": Verdict.CONTRADICTED,
    "NOT ENOUGH INFO": Verdict.INSUFFICIENT,
    "verified": Verdict.VERIFIED,
    "contradicted": Verdict.CONTRADICTED,
    "insufficient_evidence": Verdict.INSUFFICIENT,
}

# Canonical MOCHEG mapping
MOCHEG_LABEL_MAP: Dict[str, Verdict] = {
    "true": Verdict.VERIFIED,
    "mostly_true": Verdict.VERIFIED,
    "false": Verdict.CONTRADICTED,
    "mostly_false": Verdict.CONTRADICTED,
    "mixture": Verdict.INSUFFICIENT,
    "no_evidence": Verdict.INSUFFICIENT,
    "nei": Verdict.INSUFFICIENT,
}

# Canonical Factify mapping
FACTIFY_LABEL_MAP: Dict[str, Verdict] = {
    "support_multimodal": Verdict.VERIFIED,
    "support_text": Verdict.VERIFIED,
    "insufficient_multimodal": Verdict.INSUFFICIENT,
    "insufficient_text": Verdict.INSUFFICIENT,
    "refute": Verdict.CONTRADICTED,
}

# Canonical NewsCLIPpings mapping
NEWSCLIPPINGS_LABEL_MAP: Dict[str, Verdict] = {
    "matching": Verdict.VERIFIED,
    "not_matching": Verdict.CONTRADICTED,
}

DATASET_LABEL_MAPS: Dict[str, Dict[str, Verdict]] = {
    "fever": FEVER_LABEL_MAP,
    "mocheg": MOCHEG_LABEL_MAP,
    "factify": FACTIFY_LABEL_MAP,
    "newsclippings": NEWSCLIPPINGS_LABEL_MAP,
}


def map_label_to_verdict(raw_label: str, dataset: str = "fever") -> Verdict:
    """
    Maps a dataset-specific raw string label to the canonical Verdict enum.

    Args:
        raw_label: Raw string label from the dataset.
        dataset: Dataset name ('fever', 'mocheg', 'factify', 'newsclippings').

    Returns:
        Canonical Verdict enum.
    """
    clean_ds = dataset.strip().lower()
    mapping = DATASET_LABEL_MAPS.get(clean_ds, FEVER_LABEL_MAP)

    # First check exact match in the dataset mapping
    if raw_label in mapping:
        return mapping[raw_label]

    # Try case-insensitive and normalized check
    norm_key = str(raw_label).strip().lower()
    for k, v in mapping.items():
        if k.lower() == norm_key:
            return v

    # Fallback to general Verdict.normalize
    return Verdict.normalize(raw_label)
