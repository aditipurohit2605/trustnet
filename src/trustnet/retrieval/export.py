"""
CLI export script for Stage 3 retrieval engine.

Runs retrieval (e.g., hybrid_rerank) across every claim in train, val, and test splits
(including NOT ENOUGH INFO / insufficient evidence claims) and exports structured
JSONL files: data/processed/retrieved_<split>.jsonl with claim ID, label, and top-5 Evidence objects.
These exported files directly feed the Stage 2 text classifier in 'claim_retrieved' mode.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List

from trustnet.data.loader import load_claims
from trustnet.retrieval.retriever import Retriever

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def export_split(
    retriever: Retriever,
    split_name: str,
    data_dir: Path,
    mode: str = "hybrid_rerank",
    top_k: int = 5,
) -> Path:
    """
    Retrieves evidence for every claim in split_name and writes retrieved_<split_name>.jsonl.
    """
    in_file = data_dir / f"{split_name}.jsonl"
    if not in_file.is_file():
        raise FileNotFoundError(f"Source split file not found: {in_file}")

    out_file = data_dir / f"retrieved_{split_name}.jsonl"
    claims = load_claims(in_file)
    logger.info(f"Exporting retrieved evidence for {len(claims)} claims from {in_file} -> {out_file} [Mode: {mode}]...")

    records = []
    for claim_obj in claims:
        # Retrieve top_k Evidence objects
        evidence_list = retriever.search(claim_obj.claim, mode=mode, top_k=top_k)

        # Convert Pydantic Evidence objects to serializable dictionaries
        evidence_dicts = [ev.model_dump() for ev in evidence_list]

        record = {
            "id": claim_obj.id,
            "claim_id": claim_obj.id,
            "claim": claim_obj.claim,
            "text": claim_obj.claim,
            "verdict": claim_obj.verdict.value,
            "label": claim_obj.verdict.value,
            "evidence": evidence_dicts,
            "gold_evidence_ids": claim_obj.evidence_doc_ids,
        }
        records.append(record)

    with open(out_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(f"Successfully exported {len(records)} records to: {out_file}")
    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(description="TrustNet Stage 3 Evidence Export CLI")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Splits to process (default: train val test).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="hybrid_rerank",
        choices=["bm25", "dense", "hybrid", "hybrid_rerank"],
        help="Retrieval mode to use for exporting evidence (default: hybrid_rerank).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of Evidence objects to retrieve per claim (default: 5).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stage3_retrieval.yaml",
        help="Path to retrieval configuration YAML.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="Directory containing source splits and where retrieved_<split>.jsonl will be saved.",
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    retriever = Retriever(config_or_path=args.config)
    retriever._ensure_indices_loaded(args.mode)

    for split in args.splits:
        export_split(
            retriever=retriever,
            split_name=split,
            data_dir=data_dir,
            mode=args.mode,
            top_k=args.top_k,
        )

    logger.info("Evidence export completed for all requested splits.")


if __name__ == "__main__":
    main()
