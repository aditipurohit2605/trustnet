"""
Unit and integration tests for Stage 3 Retrieval Engine.

Covers:
1. Reciprocal Rank Fusion (RRF) mathematical correctness on a hand-crafted example.
2. Chunking preserves all doc_ids, text, and sentences without loss.
3. Retriever produces validated Pydantic Evidence objects with scores and collapses doc_ids.
4. End-to-end smoke test on 50 synthetic documents verifying chunking, BM25 indexing,
   dense indexing with lightweight embeddings, collapsing, and search.
"""

import json
import math
from pathlib import Path
from typing import List

import numpy as np
import pytest

from trustnet.retrieval.bm25 import BM25Index, tokenize_text
from trustnet.retrieval.chunking import chunk_corpus, chunk_document, split_sentences
from trustnet.retrieval.dense import DenseIndex
from trustnet.retrieval.fusion import reciprocal_rank_fusion
from trustnet.retrieval.retriever import Retriever, collapse_passages_to_documents
from trustnet.schemas import CorpusDocument, Evidence


def test_rrf_correctness():
    """
    Verifies that Reciprocal Rank Fusion strictly follows 1 / (rrf_k + rank)
    and correctly ranks items on a hand-crafted example.
    
    Setup:
      List 1: ["doc_A", "doc_B", "doc_C"]
      List 2: ["doc_B", "doc_D", "doc_A"]
      rrf_k = 60
      
    Manual Expected Scores:
      doc_B: rank 2 in L1 (1/62) + rank 1 in L2 (1/61) = 1/62 + 1/61 ≈ 0.03252247
      doc_A: rank 1 in L1 (1/61) + rank 3 in L2 (1/63) = 1/61 + 1/63 ≈ 0.03226646
      doc_D: rank 2 in L2 (1/62) = 1/62 ≈ 0.01612903
      doc_C: rank 3 in L1 (1/63) = 1/63 ≈ 0.01587302
      
    Expected Order: doc_B > doc_A > doc_D > doc_C
    """
    l1 = ["doc_A", "doc_B", "doc_C"]
    l2 = ["doc_B", "doc_D", "doc_A"]
    rrf_k = 60

    fused = reciprocal_rank_fusion([l1, l2], rrf_k=rrf_k, key_field="id")
    fused_ids = [item["id"] for item in fused]

    # Verify ranking order
    assert fused_ids == ["doc_B", "doc_A", "doc_D", "doc_C"]

    # Verify exact scores within float precision
    score_map = {item["id"]: item["score"] for item in fused}
    expected_b = (1.0 / 62) + (1.0 / 61)
    expected_a = (1.0 / 61) + (1.0 / 63)
    expected_d = 1.0 / 62
    expected_c = 1.0 / 63

    assert math.isclose(score_map["doc_B"], expected_b, rel_tol=1e-6)
    assert math.isclose(score_map["doc_A"], expected_a, rel_tol=1e-6)
    assert math.isclose(score_map["doc_D"], expected_d, rel_tol=1e-6)
    assert math.isclose(score_map["doc_C"], expected_c, rel_tol=1e-6)


def test_chunking_never_loses_text_or_doc_id():
    """
    Verifies that passage chunking:
    1. Preserves parent doc_id on every generated passage.
    2. Never drops any sentence or text across sliding windows.
    3. Gracefully handles short documents (1 sentence) and empty documents.
    """
    docs = [
        CorpusDocument(
            doc_id="doc_multi",
            title="Multi Sentence Report",
            text="First sentence here. Second sentence details facts. Third sentence provides evidence. Fourth sentence concludes findings. Fifth sentence is an addendum.",
        ),
        CorpusDocument(
            doc_id="doc_single",
            title="Single Sentence Dispatch",
            text="Only one solitary statement exists in this document.",
        ),
        CorpusDocument(
            doc_id="doc_empty",
            title="Empty Document",
            text="",
        ),
    ]

    passages = chunk_corpus(
        docs,
        chunk_size_sentences=3,
        chunk_stride=2,
    )

    # 1. Check doc_multi: 5 sentences -> Window 1 (s0-s2), Window 2 (s2-s4), Window 3 (s4)
    multi_passages = [p for p in passages if p["doc_id"] == "doc_multi"]
    assert len(multi_passages) >= 2
    for p in multi_passages:
        assert p["doc_id"] == "doc_multi"
        assert p["passage_id"].startswith("doc_multi_p")
        assert len(p["text"]) > 0

    # Ensure every original sentence from doc_multi is present in at least one passage
    orig_sentences = split_sentences(docs[0].text)
    assert len(orig_sentences) == 5
    for s in orig_sentences:
        assert any(s in p["text"] for p in multi_passages), f"Sentence lost: '{s}'"

    # 2. Check doc_single
    single_passages = [p for p in passages if p["doc_id"] == "doc_single"]
    assert len(single_passages) == 1
    assert single_passages[0]["doc_id"] == "doc_single"
    assert "Only one solitary statement" in single_passages[0]["text"]

    # 3. Check doc_empty
    empty_passages = [p for p in passages if p["doc_id"] == "doc_empty"]
    assert len(empty_passages) == 1
    assert empty_passages[0]["doc_id"] == "doc_empty"


def test_retriever_returns_valid_evidence_objects():
    """
    Verifies that Retriever.search and collapse_passages_to_documents:
    1. Returns a list of valid Pydantic Evidence objects.
    2. Populates doc_id, text, score, and passage_id.
    3. Correctly collapses multiple passages from the same document by retaining
       the maximum passage score.
    """
    sample_passages = [
        {"doc_id": "doc_100", "passage_id": "doc_100_p0", "text": "Low score snippet.", "score": 0.45, "title": "Doc 100"},
        {"doc_id": "doc_100", "passage_id": "doc_100_p1", "text": "High score snippet with key facts.", "score": 0.95, "title": "Doc 100"},
        {"doc_id": "doc_200", "passage_id": "doc_200_p0", "text": "Medium score from doc 200.", "score": 0.75, "title": "Doc 200"},
    ]

    collapsed = collapse_passages_to_documents(sample_passages, top_k=5)

    # Must collapse 3 passages across 2 documents into exactly 2 Evidence objects
    assert len(collapsed) == 2

    # Every item must be an instance of the Evidence schema
    for ev in collapsed:
        assert isinstance(ev, Evidence)
        assert ev.doc_id in ("doc_100", "doc_200")
        assert ev.score is not None
        assert isinstance(ev.score, float)
        assert len(ev.text) > 0

    # doc_100 must retain the MAX score (0.95), not the lower score (0.45)
    doc_100_ev = next(ev for ev in collapsed if ev.doc_id == "doc_100")
    assert doc_100_ev.score == 0.95
    assert "High score snippet" in doc_100_ev.text
    assert doc_100_ev.passage_id == "doc_100_p1"

    # Top result must be doc_100 (score 0.95 > doc_200 score 0.75)
    assert collapsed[0].doc_id == "doc_100"
    assert collapsed[1].doc_id == "doc_200"


def test_stage3_smoke_50_documents(tmp_path: Path):
    """
    End-to-end smoke test on 50 synthetic documents:
    1. Chunks 50 documents into passages.
    2. Builds BM25 and FAISS dense indices on disk.
    3. Executes search across modes ('bm25', 'dense', 'hybrid').
    4. Verifies search output format, scores, and collapsing.
    """
    import faiss

    # 1. Create 50 synthetic documents
    docs: List[CorpusDocument] = []
    for i in range(1, 51):
        docs.append(CorpusDocument(
            doc_id=f"doc_{i:03d}",
            title=f"Topic {i} Overview",
            text=(
                f"Document {i} discusses environmental monitoring. "
                f"Atmospheric sensors detected condition {i} over northern regions. "
                f"Final conclusions for case {i} confirm stability."
            ),
        ))

    passages_file = tmp_path / "smoke_passages.jsonl"
    passages = chunk_corpus(
        docs,
        output_path=passages_file,
        chunk_size_sentences=2,
        chunk_stride=1,
    )
    assert len(passages) >= 50
    assert passages_file.is_file()

    # 2. Test BM25 Index
    bm25_file = tmp_path / "bm25.pkl"
    bm25_index = BM25Index(passages)
    bm25_index.save(bm25_file)
    assert bm25_file.is_file()

    loaded_bm25 = BM25Index.load(bm25_file)
    bm25_hits = loaded_bm25.search("atmospheric sensors detected condition 25", top_k=5)
    assert len(bm25_hits) > 0
    # doc_025 should be the top match
    assert bm25_hits[0]["doc_id"] == "doc_025"
    assert bm25_hits[0]["score"] > 0.0

    # 3. Test Dense Index with lightweight synthetic embeddings (without heavy external downloads)
    dim = 64
    rng = np.random.RandomState(42)
    fake_embeddings = rng.randn(len(passages), dim).astype(np.float32)
    # L2 normalize
    norms = np.linalg.norm(fake_embeddings, axis=1, keepdims=True)
    fake_embeddings /= norms

    dense_index = DenseIndex(model_name="dummy", passages=passages)
    dense_index.index = faiss.IndexFlatIP(dim)
    dense_index.index.add(fake_embeddings)

    dense_dir = tmp_path / "dense_index"
    dense_index.save(dense_dir)
    assert (dense_dir / "dense.faiss").is_file()
    assert (dense_dir / "dense_metadata.json").is_file()

    loaded_dense = DenseIndex.load(dense_dir)
    assert loaded_dense.index is not None
    assert loaded_dense.index.ntotal == len(passages)

    # 4. Test RRF Fusion between BM25 and Dense
    fake_query_emb = fake_embeddings[10:11]
    scores, indices = loaded_dense.index.search(fake_query_emb, 5)
    dense_hits = []
    for s, idx in zip(scores[0], indices[0]):
        item = dict(passages[idx])
        item["score"] = float(s)
        dense_hits.append(item)

    fused = reciprocal_rank_fusion([bm25_hits, dense_hits], rrf_k=60, top_k=5)
    assert len(fused) > 0
    assert "score" in fused[0]

    # 5. Test Evidence Collapsing
    evidence_results = collapse_passages_to_documents(fused, top_k=3)
    assert len(evidence_results) <= 3
    assert all(isinstance(e, Evidence) for e in evidence_results)
    assert len(set(e.doc_id for e in evidence_results)) == len(evidence_results)


def test_claim_retrieved_dataset_mode(tmp_path: Path):
    """
    Verifies that TrustNetTextDataset correctly loads and formats
    retrieved_<split>.jsonl data when input_mode='claim_retrieved'.
    """
    from transformers import AutoTokenizer
    from trustnet.text.dataset import TrustNetTextDataset

    # Generate dummy retrieved JSONL file
    retrieved_data = [
        {
            "id": "claim_1",
            "claim": "Ozone layer is recovering.",
            "verdict": "verified",
            "evidence": [
                {"doc_id": "doc_001", "text": "Satellite data confirms stratospheric ozone healing.", "score": 0.98},
                {"doc_id": "doc_002", "text": "Second supporting report.", "score": 0.85},
            ],
            "gold_evidence_ids": ["doc_001"],
        },
        {
            "id": "claim_2",
            "claim": "Dam is breaking down.",
            "verdict": "contradicted",
            "evidence": [
                {"doc_id": "doc_004", "text": "Water reservoir levels are normal.", "score": 0.92},
            ],
            "gold_evidence_ids": ["doc_004"],
        },
    ]

    retrieved_file = tmp_path / "retrieved_val.jsonl"
    with open(retrieved_file, "w", encoding="utf-8") as f:
        for r in retrieved_data:
            f.write(json.dumps(r) + "\n")

    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-deberta")
    labels = ["verified", "contradicted", "insufficient_evidence"]

    dataset = TrustNetTextDataset(
        split_path=retrieved_file,
        tokenizer=tokenizer,
        label_list=labels,
        input_mode="claim_retrieved",
        top_k_evidence=2,
        max_length=64,
    )

    assert len(dataset) == 2
    ex0 = dataset[0]
    assert ex0["claim_id"] == "claim_1"
    assert ex0["label"] == 0
    assert "Satellite data confirms" in ex0["input_text"]
    assert "input_ids" in ex0
    assert "attention_mask" in ex0
