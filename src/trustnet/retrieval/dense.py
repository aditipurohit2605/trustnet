"""
Dense semantic retrieval module using SentenceTransformers and FAISS.

Embeds passages on GPU using normalized embeddings (with FP16 when CUDA is available),
builds a FAISS IndexFlatIP inner-product index, caches embeddings to disk to prevent
redundant computation, and performs semantic vector search.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class DenseIndex:
    """
    Dense vector search index powered by SentenceTransformers and FAISS IndexFlatIP.
    
    Attributes:
        model_name: Hugging Face model identifier (e.g., 'BAAI/bge-m3').
        passages: The list of indexed passage records.
        index: The underlying FAISS IndexFlatIP index.
        device: PyTorch device ('cuda' or 'cpu').
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        passages: Optional[List[Dict[str, Any]]] = None,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.passages: List[Dict[str, Any]] = passages or []
        self.index: Optional[faiss.IndexFlatIP] = None
        self._model: Optional[SentenceTransformer] = None

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def _load_model(self) -> SentenceTransformer:
        """Loads and caches the SentenceTransformer model on the target device."""
        if self._model is None:
            logger.info(f"Loading dense embedding model '{self.model_name}' on {self.device}...")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            # If on GPU, utilize FP16 precision for improved memory efficiency and throughput
            if self.device == "cuda":
                self._model = self._model.half()
        return self._model

    def build(
        self,
        passages: List[Dict[str, Any]],
        batch_size: int = 32,
        cache_path: Optional[Union[str, Path]] = None,
    ) -> "DenseIndex":
        """
        Embeds passages, builds the FAISS inner-product index, and caches embeddings.
        
        Args:
            passages: List of passage dictionaries.
            batch_size: Batch size for sentence embedding computation.
            cache_path: Optional path to an .npy file for caching raw embeddings.
            
        Returns:
            The populated DenseIndex instance.
        """
        self.passages = passages
        if not self.passages:
            logger.warning("Empty passages list provided to DenseIndex.build().")
            return self

        embeddings = None
        cache_file = Path(cache_path) if cache_path else None

        # Check if valid cached embeddings exist on disk
        if cache_file and cache_file.is_file():
            try:
                cached_data = np.load(cache_file)
                if cached_data.shape[0] == len(self.passages):
                    logger.info(f"Loaded {len(cached_data)} cached embeddings from: {cache_file}")
                    embeddings = cached_data
                else:
                    logger.warning(
                        f"Embedding cache size mismatch ({cached_data.shape[0]} vs {len(self.passages)} passages). "
                        "Recomputing embeddings..."
                    )
            except Exception as e:
                logger.warning(f"Failed to load embedding cache from {cache_file}: {e}. Recomputing...")

        # Compute embeddings if not found in cache
        if embeddings is None:
            model = self._load_model()
            texts = [p["text"] for p in self.passages]
            logger.info(f"Computing normalized embeddings for {len(texts)} passages (batch_size={batch_size})...")

            # normalize_embeddings=True guarantees L2 unit vectors, so inner product == cosine similarity
            raw_embs = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            embeddings = np.ascontiguousarray(raw_embs, dtype=np.float32)

            # Persist computed embeddings to cache
            if cache_file:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_file, embeddings)
                logger.info(f"Saved computed embeddings to cache: {cache_file}")

        # Construct FAISS inner-product index
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype(np.float32))
        logger.info(f"Built FAISS IndexFlatIP with {self.index.ntotal} vectors (dimension={dim}).")
        return self

    def search(self, query: str, top_k: int = 100) -> List[Dict[str, Any]]:
        """
        Searches the dense FAISS index for the top-k most semantically similar passages.
        
        Args:
            query: The search query text (claim).
            top_k: Maximum number of matches to retrieve.
            
        Returns:
            List of passage dictionaries augmented with their cosine similarity 'score'.
        """
        if self.index is None or not self.passages:
            return []

        model = self._load_model()
        # Encode query as a normalized unit vector
        q_emb = model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)

        num_results = min(top_k, self.index.ntotal)
        if num_results <= 0:
            return []

        scores, indices = self.index.search(q_emb, num_results)

        results = []
        for rank_idx, passage_idx in enumerate(indices[0]):
            if passage_idx < 0 or passage_idx >= len(self.passages):
                continue
            item = dict(self.passages[passage_idx])
            item["score"] = float(scores[0][rank_idx])
            results.append(item)

        return results

    def save(self, index_dir: Union[str, Path]) -> None:
        """
        Saves the FAISS index, passage metadata, and model identifier to a directory.
        """
        dir_path = Path(index_dir)
        dir_path.mkdir(parents=True, exist_ok=True)

        if self.index is not None:
            faiss_file = dir_path / "dense.faiss"
            faiss.write_index(self.index, str(faiss_file))

        metadata_file = dir_path / "dense_metadata.json"
        metadata = {
            "model_name": self.model_name,
            "passages": self.passages,
        }
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False)

    @classmethod
    def load(cls, index_dir: Union[str, Path], device: Optional[str] = None) -> "DenseIndex":
        """
        Loads the FAISS index and metadata from a directory.
        """
        dir_path = Path(index_dir)
        metadata_file = dir_path / "dense_metadata.json"
        faiss_file = dir_path / "dense.faiss"

        if not metadata_file.is_file():
            raise FileNotFoundError(f"Dense metadata not found at: {metadata_file}")
        if not faiss_file.is_file():
            raise FileNotFoundError(f"FAISS index file not found at: {faiss_file}")

        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        instance = cls(
            model_name=metadata.get("model_name", "BAAI/bge-m3"),
            passages=metadata.get("passages", []),
            device=device,
        )
        instance.index = faiss.read_index(str(faiss_file))
        return instance
