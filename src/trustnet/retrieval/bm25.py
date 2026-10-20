"""
BM25 lexical retrieval module using rank_bm25.

Provides index building over chunked passages, lowercase regex tokenization,
score computation, disk serialization, and top-k search.
"""

import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from rank_bm25 import BM25Okapi


def tokenize_text(text: str) -> List[str]:
    r"""
    Standard lowercase alphanumeric tokenization for lexical search.
    
    Extracts contiguous word characters (\w+) converted to lowercase.
    """
    if not text:
        return []
    return re.findall(r"\b\w+\b", text.lower())


class BM25Index:
    """
    Lexical retrieval index powered by BM25Okapi over passages.
    
    Attributes:
        passages: The list of indexed passage records.
        bm25: The underlying rank_bm25 BM25Okapi model.
    """

    def __init__(self, passages: Optional[List[Dict[str, Any]]] = None):
        self.passages: List[Dict[str, Any]] = passages or []
        self.bm25: Optional[BM25Okapi] = None
        if self.passages:
            self._build_index()

    def _build_index(self) -> None:
        """Tokenizes passage texts and initializes BM25Okapi."""
        tokenized_corpus = [tokenize_text(p["text"]) for p in self.passages]
        # Safeguard: if a passage produces zero tokens, insert a placeholder token
        safe_corpus = [tokens if tokens else ["<empty>"] for tokens in tokenized_corpus]
        self.bm25 = BM25Okapi(safe_corpus)

    def build(self, passages: List[Dict[str, Any]]) -> "BM25Index":
        """
        Builds the BM25 index from a list of passage dictionaries.
        """
        self.passages = passages
        self._build_index()
        return self

    def search(self, query: str, top_k: int = 100) -> List[Dict[str, Any]]:
        """
        Searches the passage index for a query string.
        
        Args:
            query: The search query (e.g. claim text).
            top_k: Maximum number of passage matches to return.
            
        Returns:
            List of passage dicts with added 'score' field, ordered descending by score.
        """
        if self.bm25 is None or not self.passages:
            return []

        query_tokens = tokenize_text(query)
        if not query_tokens:
            # Query has no tokens: return empty or unranked top_k with 0.0 scores
            return []

        # Compute BM25 relevance scores across all passages
        scores = self.bm25.get_scores(query_tokens)
        scores = np.array(scores, dtype=np.float32)

        # Retrieve indices of top-k scores in descending order
        num_candidates = min(top_k, len(scores))
        if num_candidates <= 0:
            return []

        # Argsort descending
        top_indices = np.argsort(scores)[::-1][:num_candidates]

        results = []
        for idx in top_indices:
            p = dict(self.passages[idx])
            p["score"] = float(scores[idx])
            results.append(p)

        return results

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Serializes the index and passage metadata to disk.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "passages": self.passages,
            "bm25": self.bm25,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "BM25Index":
        """
        Deserializes a BM25Index from disk.
        """
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"BM25 index not found at: {path}")

        with open(path, "rb") as f:
            payload = pickle.load(f)

        instance = cls(passages=payload["passages"])
        instance.bm25 = payload["bm25"]
        return instance
