"""
Data loading and parsing utilities for TrustNet claims and evidence corpora.
"""

import json
from pathlib import Path
from typing import Dict, Generator, List, Union

from trustnet.schemas import ClaimExample, CorpusDocument


def load_jsonl(file_path: Union[str, Path]) -> List[Dict]:
    """
    Reads a JSONL file and yields each line parsed as a dictionary.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as err:
                raise ValueError(f"Failed to parse JSON on line {line_num} in {path}: {err}") from err
    return records


def load_claims(file_path: Union[str, Path]) -> List[ClaimExample]:
    """
    Loads processed claim records from a JSONL file, converting each into a ClaimExample.
    """
    raw_records = load_jsonl(file_path)
    claims = []
    for item in raw_records:
        # Standardize field names across schema variations
        normalized = {
            "id": item.get("id") or item.get("claim_id") or item.get("_id"),
            "claim": item.get("claim") or item.get("text") or item.get("claim_text"),
            "verdict": item.get("verdict") or item.get("label"),
            "evidence_doc_ids": item.get("evidence_doc_ids") or item.get("gold_evidence_ids") or item.get("doc_ids") or [],
            "metadata": item.get("metadata") or {}
        }
        claims.append(ClaimExample(**normalized))
    return claims


def load_corpus(file_path: Union[str, Path]) -> Dict[str, CorpusDocument]:
    """
    Loads the knowledge corpus from a JSONL file, indexing documents by doc_id.
    """
    raw_records = load_jsonl(file_path)
    corpus = {}
    for item in raw_records:
        doc_id = item.get("doc_id") or item.get("id") or item.get("_id")
        if not doc_id:
            continue
        text = item.get("text") or item.get("contents") or item.get("content") or ""
        title = item.get("title")
        corpus[str(doc_id)] = CorpusDocument(doc_id=str(doc_id), text=text, title=title)
    return corpus
