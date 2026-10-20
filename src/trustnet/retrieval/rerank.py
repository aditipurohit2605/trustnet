"""
Cross-encoder reranking module using SentenceTransformers CrossEncoder.

Scores (query, passage) pairs using deep bidirectional cross-attention
(e.g., BAAI/bge-reranker-v2-m3), reorders candidate pools, and selects
the top-scoring final passages.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class Reranker:
    """
    Cross-encoder reranker for scoring (claim, passage) pairs.
    
    Attributes:
        model_name: Hugging Face model identifier (e.g. 'BAAI/bge-reranker-v2-m3').
        device: Device for inference ('cuda' or 'cpu').
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: Optional[CrossEncoder] = None

    def _load_model(self) -> CrossEncoder:
        """Loads and caches the CrossEncoder model."""
        if self._model is None:
            logger.info(f"Loading CrossEncoder reranker '{self.model_name}' on {self.device}...")
            # Initialize CrossEncoder on target device
            self._model = CrossEncoder(
                self.model_name,
                device=self.device,
            )
        return self._model

    def rerank(
        self,
        query: str,
        passages: List[Dict[str, Any]],
        top_k: int = 5,
        batch_size: int = 16,
    ) -> List[Dict[str, Any]]:
        """
        Reranks a list of candidate passages against a query.
        
        Args:
            query: The search query text (claim).
            passages: Candidate passage dictionaries (must contain 'text').
            top_k: Number of highest-ranked passages to return.
            batch_size: Batch size for cross-encoder inference.
            
        Returns:
            List of passage dicts with updated 'score' field, ordered descending by cross-encoder score.
        """
        if not passages:
            return []

        model = self._load_model()
        # Construct input pairs: (query, passage_text)
        pairs = [[query, p["text"]] for p in passages]

        logger.debug(f"Reranking {len(pairs)} pairs with batch_size={batch_size}...")
        raw_scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        scores = np.array(raw_scores, dtype=np.float32)

        # Handle edge case where single pair returns 0-dim scalar
        if scores.ndim == 0:
            scores = np.array([float(scores)])

        # Rank pairs descending by cross-encoder score
        ranked_indices = np.argsort(scores)[::-1]
        num_to_take = min(top_k, len(passages)) if top_k > 0 else len(passages)
        top_indices = ranked_indices[:num_to_take]

        reranked_results = []
        for idx in top_indices:
            res = dict(passages[idx])
            res["score"] = float(scores[idx])
            reranked_results.append(res)

        return reranked_results
