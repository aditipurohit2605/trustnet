"""
Passage chunking module for the TrustNet retrieval engine.

Splits knowledge corpus documents into overlapping passages using a sliding window
over sentences, preserving document lineage (doc_id) and generating unique passage IDs.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from trustnet.data.loader import load_corpus
from trustnet.schemas import CorpusDocument


def split_sentences(text: str) -> List[str]:
    """
    Splits text into sentences using regex boundary detection.
    
    Splits on punctuation marks (., !, ?) followed by whitespace, preserving
    the punctuation mark with the preceding sentence.
    """
    text = text.strip()
    if not text:
        return []
    # Split after sentence-terminating punctuation followed by whitespace
    raw_sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    return sentences if sentences else [text]


def chunk_document(
    doc: CorpusDocument,
    chunk_size_sentences: int = 3,
    chunk_stride: int = 2,
) -> List[Dict[str, Any]]:
    """
    Chunks a single CorpusDocument into overlapping passages.
    
    Args:
        doc: The CorpusDocument to chunk.
        chunk_size_sentences: Maximum number of sentences per passage.
        chunk_stride: Number of sentences to slide the window forward.
        
    Returns:
        List of passage dictionaries containing doc_id, passage_id, and text.
    """
    if chunk_size_sentences < 1:
        chunk_size_sentences = 1
    if chunk_stride < 1:
        chunk_stride = 1

    sentences = split_sentences(doc.text)
    if not sentences:
        # Empty document fallback: create a single empty passage
        return [{
            "doc_id": doc.doc_id,
            "passage_id": f"{doc.doc_id}_p0",
            "text": doc.title or "",
            "raw_text": "",
            "title": doc.title,
        }]

    # If document has fewer or equal sentences than chunk_size, output single passage
    if len(sentences) <= chunk_size_sentences:
        passage_text = " ".join(sentences)
        # Prepend title for enriched retrieval context if available
        full_text = f"{doc.title}: {passage_text}" if doc.title else passage_text
        return [{
            "doc_id": doc.doc_id,
            "passage_id": f"{doc.doc_id}_p0",
            "text": full_text,
            "raw_text": passage_text,
            "title": doc.title,
        }]

    passages = []
    passage_idx = 0
    total_sentences = len(sentences)

    # Sliding window over sentences
    for start_idx in range(0, total_sentences, chunk_stride):
        window = sentences[start_idx : start_idx + chunk_size_sentences]
        if not window:
            break

        passage_text = " ".join(window)
        full_text = f"{doc.title}: {passage_text}" if doc.title else passage_text

        passages.append({
            "doc_id": doc.doc_id,
            "passage_id": f"{doc.doc_id}_p{passage_idx}",
            "text": full_text,
            "raw_text": passage_text,
            "title": doc.title,
        })
        passage_idx += 1

        # Terminate when the window reaches or covers the end of the document
        if start_idx + chunk_size_sentences >= total_sentences:
            break

    return passages


def chunk_corpus(
    corpus_source: Union[str, Path, Dict[str, CorpusDocument], List[Dict[str, Any]]],
    output_path: Optional[Union[str, Path]] = None,
    chunk_size_sentences: int = 3,
    chunk_stride: int = 2,
) -> List[Dict[str, Any]]:
    """
    Chunks an entire knowledge corpus into passages and optionally writes to JSONL.
    
    Args:
        corpus_source: Path to corpus JSONL, or dict of CorpusDocuments, or list of dicts.
        output_path: Optional path to write passages JSONL (e.g. outputs/stage3/passages.jsonl).
        chunk_size_sentences: Number of sentences per passage window.
        chunk_stride: Stride between successive passage windows.
        
    Returns:
        List of all chunked passage dictionaries.
    """
    # Load corpus documents into a standardized list
    if isinstance(corpus_source, (str, Path)):
        doc_dict = load_corpus(corpus_source)
        documents = list(doc_dict.values())
    elif isinstance(corpus_source, dict):
        documents = list(corpus_source.values())
    elif isinstance(corpus_source, list):
        documents = []
        for item in corpus_source:
            if isinstance(item, CorpusDocument):
                documents.append(item)
            else:
                documents.append(CorpusDocument(
                    doc_id=str(item.get("doc_id") or item.get("id")),
                    text=str(item.get("text") or ""),
                    title=item.get("title"),
                ))
    else:
        raise TypeError(f"Unsupported corpus_source type: {type(corpus_source)}")

    all_passages: List[Dict[str, Any]] = []
    for doc in documents:
        passages = chunk_document(
            doc,
            chunk_size_sentences=chunk_size_sentences,
            chunk_stride=chunk_stride,
        )
        all_passages.extend(passages)

    # Persist passages to JSONL if output_path is provided
    if output_path is not None:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            for p in all_passages:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

    return all_passages
