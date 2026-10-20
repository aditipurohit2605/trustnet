"""
Reciprocal Rank Fusion (RRF) module for hybrid retrieval.

Implements the standard Cormack et al. (2009) reciprocal rank fusion formula:
    RRF_score(item) = sum_{system} [ 1 / (rrf_k + rank) ]
where rank is 1-indexed (1 for top result, 2 for second, etc.).
"""

from typing import Any, Dict, List, Optional, Union


def reciprocal_rank_fusion(
    ranked_lists: List[List[Union[Dict[str, Any], str]]],
    rrf_k: int = 60,
    top_k: Optional[int] = None,
    key_field: str = "passage_id",
) -> List[Dict[str, Any]]:
    """
    Combines multiple ranked lists using Reciprocal Rank Fusion (RRF).
    
    Args:
        ranked_lists: A list of candidate lists from different retrievers (e.g., BM25 and Dense).
                      Each list contains item dictionaries or string identifiers.
        rrf_k: Constant smoothing parameter (standard default: 60).
        top_k: Maximum number of fused items to return. If None, returns all unique candidates.
        key_field: Dictionary field name used as the unique item identifier.
        
    Returns:
        List of fused item dictionaries, sorted descending by 'score' (the RRF score).
    """
    if rrf_k < 0:
        raise ValueError(f"rrf_k must be non-negative, got {rrf_k}")

    # Accumulator: item_key -> total RRF score
    scores: Dict[str, float] = {}
    # Item metadata storage: item_key -> item dictionary
    item_lookup: Dict[str, Dict[str, Any]] = {}

    for ranking in ranked_lists:
        for rank_1_indexed, item in enumerate(ranking, start=1):
            # Extract unique key and normalize item representation
            if isinstance(item, dict):
                key = str(item.get(key_field) or item.get("id") or item.get("doc_id"))
                item_dict = dict(item)
            else:
                key = str(item)
                item_dict = {key_field: key, "id": key}

            if not key:
                continue

            # Standard RRF formula: 1 / (rrf_k + rank)
            contribution = 1.0 / (rrf_k + rank_1_indexed)
            scores[key] = scores.get(key, 0.0) + contribution

            # Cache the item metadata dictionary if not already stored
            if key not in item_lookup:
                item_lookup[key] = item_dict

    # Sort all candidate items descending by their accumulated RRF score
    sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)

    if top_k is not None and top_k > 0:
        sorted_keys = sorted_keys[:top_k]

    fused_results = []
    for k in sorted_keys:
        res = dict(item_lookup[k])
        res["score"] = scores[k]
        fused_results.append(res)

    return fused_results
