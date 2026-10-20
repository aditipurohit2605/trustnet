"""
Unified Retriever class orchestrating lexical (BM25), dense (FAISS), hybrid (RRF),
and cross-encoder reranking retrieval modes.

Collapses multi-passage document representations using Max-Score Passage Pooling
and returns structured Evidence Pydantic objects.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from trustnet.retrieval.bm25 import BM25Index
from trustnet.retrieval.chunking import chunk_corpus
from trustnet.retrieval.dense import DenseIndex
from trustnet.retrieval.fusion import reciprocal_rank_fusion
from trustnet.retrieval.rerank import Reranker
from trustnet.schemas import Evidence

logger = logging.getLogger(__name__)


def collapse_passages_to_documents(
    passages: List[Dict[str, Any]],
    top_k: int = 5,
) -> List[Evidence]:
    """
    Sensibly collapses multiple retrieved passages belonging to the same document.
    
    Strategy: Max-Score Passage Pooling
    1. Group retrieved passages by parent `doc_id`.
    2. For each document, retain the single passage that achieved the highest relevance score.
       Why: The highest-scoring passage represents the strongest, most salient piece of evidence
       without dilution from surrounding less-relevant sentences, and prevents a single document
       from occupying multiple slots in the top-k list.
    3. Order the resulting unique documents descending by their maximum passage score.
    4. Truncate to top_k and construct structured Evidence objects with score, text, and passage_id.
    """
    if not passages:
        return []

    # Map doc_id -> best passage dict
    doc_best_passage: Dict[str, Dict[str, Any]] = {}

    for p in passages:
        doc_id = str(p.get("doc_id") or "")
        if not doc_id:
            continue

        score = float(p.get("score", 0.0))
        if doc_id not in doc_best_passage:
            doc_best_passage[doc_id] = p
        else:
            # Keep passage with strictly higher score
            if score > float(doc_best_passage[doc_id].get("score", float("-inf"))):
                doc_best_passage[doc_id] = p

    # Sort documents descending by their maximum passage score
    sorted_passages = sorted(
        doc_best_passage.values(),
        key=lambda item: float(item.get("score", 0.0)),
        reverse=True,
    )

    if top_k > 0:
        sorted_passages = sorted_passages[:top_k]

    evidence_list: List[Evidence] = []
    for sp in sorted_passages:
        ev = Evidence(
            doc_id=str(sp["doc_id"]),
            text=str(sp["text"]),
            score=float(sp.get("score", 0.0)),
            passage_id=str(sp.get("passage_id", "")),
            source=sp.get("title"),
            source_type="corpus_passage",
        )
        evidence_list.append(ev)

    return evidence_list


class Retriever:
    """
    Multi-modal retrieval engine supporting 'bm25', 'dense', 'hybrid', and 'hybrid_rerank'.
    
    Attributes:
        config: Parsed configuration dictionary.
        bm25_index: BM25 lexical index over passages.
        dense_index: FAISS dense embedding index over passages.
        reranker: Cross-encoder reranking model.
    """

    SUPPORTED_MODES = ("bm25", "dense", "hybrid", "hybrid_rerank")

    def __init__(
        self,
        config_or_path: Union[str, Path, Dict[str, Any]] = "configs/stage3_retrieval.yaml",
        bm25_index: Optional[BM25Index] = None,
        dense_index: Optional[DenseIndex] = None,
        reranker: Optional[Reranker] = None,
    ):
        if isinstance(config_or_path, (str, Path)):
            cfg_path = Path(config_or_path)
            if not cfg_path.is_file():
                raise FileNotFoundError(f"Retrieval config file not found: {cfg_path}")
            with open(cfg_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
        elif isinstance(config_or_path, dict):
            self.config = config_or_path
        else:
            raise TypeError(f"Invalid config type: {type(config_or_path)}")

        self.dense_model_name: str = self.config.get("dense_model", "BAAI/bge-m3")
        self.reranker_model_name: str = self.config.get("reranker_model", "BAAI/bge-reranker-v2-m3")

        self.chunk_size_sentences: int = int(self.config.get("chunk_size_sentences", 3))
        self.chunk_stride: int = int(self.config.get("chunk_stride", 2))

        self.bm25_top_k: int = int(self.config.get("bm25_top_k", 100))
        self.dense_top_k: int = int(self.config.get("dense_top_k", 100))
        self.rrf_k: int = int(self.config.get("rrf_k", 60))
        self.fused_top_k: int = int(self.config.get("fused_top_k", 50))
        self.final_top_k: int = int(self.config.get("final_top_k", 5))

        self.embed_batch_size: int = int(self.config.get("embed_batch_size", 32))
        self.rerank_batch_size: int = int(self.config.get("rerank_batch_size", 16))

        self.corpus_path = Path(self.config.get("corpus_path", "data/corpus/corpus_v1.jsonl"))
        self.passages_path = Path(self.config.get("passages_path", "outputs/stage3/passages.jsonl"))
        self.index_dir = Path(self.config.get("index_dir", "outputs/stage3/index"))
        self.cache_embeddings: bool = bool(self.config.get("cache_embeddings", True))

        self.bm25_index: Optional[BM25Index] = bm25_index
        self.dense_index: Optional[DenseIndex] = dense_index
        self.reranker: Optional[Reranker] = reranker

    def _get_or_create_passages(self, force_rebuild: bool = False) -> List[Dict[str, Any]]:
        """Loads existing passages from disk or chunks the corpus."""
        import json
        if self.passages_path.is_file() and not force_rebuild:
            logger.info(f"Loading chunked passages from {self.passages_path}...")
            passages = []
            with open(self.passages_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        passages.append(json.loads(line))
            return passages

        logger.info(f"Chunking corpus from {self.corpus_path}...")
        return chunk_corpus(
            self.corpus_path,
            output_path=self.passages_path,
            chunk_size_sentences=self.chunk_size_sentences,
            chunk_stride=self.chunk_stride,
        )

    def build_or_load_bm25(
        self,
        force_rebuild: bool = False,
        passages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Loads BM25 index from disk or builds from passages."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        bm25_path = self.index_dir / "bm25.pkl"

        if not force_rebuild and bm25_path.is_file() and self.bm25_index is None:
            logger.info(f"Loading BM25 index from {bm25_path}...")
            self.bm25_index = BM25Index.load(bm25_path)
            return

        if self.bm25_index is None or force_rebuild:
            if passages is None:
                passages = self._get_or_create_passages(force_rebuild=force_rebuild)
            logger.info(f"Building BM25 index over {len(passages)} passages...")
            self.bm25_index = BM25Index(passages)
            self.bm25_index.save(bm25_path)

    def build_or_load_dense(
        self,
        force_rebuild: bool = False,
        passages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Loads Dense FAISS index from disk or builds from passages."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        dense_faiss_path = self.index_dir / "dense.faiss"

        if not force_rebuild and dense_faiss_path.is_file() and self.dense_index is None:
            logger.info(f"Loading Dense FAISS index from {self.index_dir}...")
            self.dense_index = DenseIndex.load(self.index_dir)
            return

        if self.dense_index is None or force_rebuild:
            if passages is None:
                passages = self._get_or_create_passages(force_rebuild=force_rebuild)
            cache_path = (self.index_dir / "embeddings.npy") if self.cache_embeddings else None
            logger.info(f"Building Dense FAISS index over {len(passages)} passages...")
            self.dense_index = DenseIndex(model_name=self.dense_model_name, passages=passages)
            self.dense_index.build(
                passages=passages,
                batch_size=self.embed_batch_size,
                cache_path=cache_path,
            )
            self.dense_index.save(self.index_dir)

    def build_or_load_indices(
        self,
        force_rebuild: bool = False,
        passages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Loads existing BM25 and Dense indices from disk, or builds them from corpus/passages."""
        if passages is None:
            passages = self._get_or_create_passages(force_rebuild=force_rebuild)
        self.build_or_load_bm25(force_rebuild=force_rebuild, passages=passages)
        self.build_or_load_dense(force_rebuild=force_rebuild, passages=passages)
        logger.info("Indices built and loaded successfully.")

    def _ensure_indices_loaded(self, mode: str) -> None:
        """Helper to ensure required indexes for the given mode are initialized."""
        if mode == "bm25":
            if self.bm25_index is None:
                self.build_or_load_bm25()
        elif mode == "dense":
            if self.dense_index is None:
                self.build_or_load_dense()
        elif mode in ("hybrid", "hybrid_rerank"):
            if self.bm25_index is None:
                self.build_or_load_bm25()
            if self.dense_index is None:
                self.build_or_load_dense()

    def _get_reranker(self) -> Reranker:
        """Lazy-instantiates the cross-encoder reranker."""
        if self.reranker is None:
            self.reranker = Reranker(model_name=self.reranker_model_name)
        return self.reranker

    def search(
        self,
        claim_text: str,
        mode: str = "hybrid_rerank",
        top_k: Optional[int] = None,
    ) -> List[Evidence]:
        """
        Executes claim verification retrieval under the specified search mode.
        
        Args:
            claim_text: Factual statement to verify against the corpus.
            mode: One of 'bm25', 'dense', 'hybrid', 'hybrid_rerank'.
            top_k: Desired number of collapsed Evidence objects to return.
                   Defaults to final_top_k from config.
                   
        Returns:
            List of Evidence Pydantic objects containing doc_id, text, score, and passage_id.
        """
        mode = mode.lower().strip()
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(f"Unknown retrieval mode: '{mode}'. Supported modes: {self.SUPPORTED_MODES}")

        self._ensure_indices_loaded(mode)
        target_top_k = top_k if top_k is not None else self.final_top_k

        # 1. BM25 Lexical Retrieval
        if mode == "bm25":
            assert self.bm25_index is not None
            raw_passages = self.bm25_index.search(claim_text, top_k=self.bm25_top_k)
            return collapse_passages_to_documents(raw_passages, top_k=target_top_k)

        # 2. Dense Semantic Retrieval
        elif mode == "dense":
            assert self.dense_index is not None
            raw_passages = self.dense_index.search(claim_text, top_k=self.dense_top_k)
            return collapse_passages_to_documents(raw_passages, top_k=target_top_k)

        # 3. Hybrid Retrieval (RRF of BM25 + Dense)
        elif mode == "hybrid":
            assert self.bm25_index is not None and self.dense_index is not None
            bm25_candidates = self.bm25_index.search(claim_text, top_k=self.bm25_top_k)
            dense_candidates = self.dense_index.search(claim_text, top_k=self.dense_top_k)

            fused_passages = reciprocal_rank_fusion(
                ranked_lists=[bm25_candidates, dense_candidates],
                rrf_k=self.rrf_k,
                top_k=self.fused_top_k,
            )
            return collapse_passages_to_documents(fused_passages, top_k=target_top_k)

        # 4. Hybrid + Cross-Encoder Reranking
        elif mode == "hybrid_rerank":
            assert self.bm25_index is not None and self.dense_index is not None
            bm25_candidates = self.bm25_index.search(claim_text, top_k=self.bm25_top_k)
            dense_candidates = self.dense_index.search(claim_text, top_k=self.dense_top_k)

            fused_passages = reciprocal_rank_fusion(
                ranked_lists=[bm25_candidates, dense_candidates],
                rrf_k=self.rrf_k,
                top_k=self.fused_top_k,
            )

            # Rerank the fused candidate passages using cross-attention
            reranker = self._get_reranker()
            reranked_passages = reranker.rerank(
                query=claim_text,
                passages=fused_passages,
                top_k=self.fused_top_k,
                batch_size=self.rerank_batch_size,
            )

            return collapse_passages_to_documents(reranked_passages, top_k=target_top_k)

        return []
