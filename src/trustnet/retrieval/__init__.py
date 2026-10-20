"""
Retrieval engine module for TrustNet.
"""

from trustnet.retrieval.bm25 import BM25Index
from trustnet.retrieval.chunking import chunk_corpus, chunk_document
from trustnet.retrieval.dense import DenseIndex
from trustnet.retrieval.fusion import reciprocal_rank_fusion
from trustnet.retrieval.rerank import Reranker
from trustnet.retrieval.retriever import Retriever, collapse_passages_to_documents

__all__ = [
    "Retriever",
    "BM25Index",
    "DenseIndex",
    "Reranker",
    "chunk_corpus",
    "chunk_document",
    "reciprocal_rank_fusion",
    "collapse_passages_to_documents",
]
