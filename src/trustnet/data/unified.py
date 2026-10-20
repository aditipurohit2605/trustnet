"""
Unified multimodal dataset builder for TrustNet Stage 6.

Builds standardized schema records across datasets:
Each row in data/processed/unified_{split}.jsonl contains:
- claim_id: unique identifier
- claim_text: statement text
- dataset: dataset origin ('fever', 'mocheg', 'factify', 'newsclippings')
- label4: canonical Verdict enum value ('verified', 'contradicted', 'insufficient_evidence')
- image_path: local file path to associated image, or None (null) for text-only claims

Design rules:
- Uses canonical label mappings from src/trustnet/data/labels.py.
- If a dataset is missing from local disk, skips it cleanly and reports it.
- Never invents labels.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from trustnet.data.labels import map_label_to_verdict
from trustnet.schemas import Verdict

logger = logging.getLogger(__name__)


def build_unified_split(
    split: str,
    processed_dir: Path = Path("data/processed"),
    raw_dir: Path = Path("data/raw"),
) -> List[Dict[str, Any]]:
    """
    Constructs unified records for a single split from all available data sources.
    """
    unified_records: List[Dict[str, Any]] = []
    skipped_datasets: List[str] = []

    # 1. Process FEVER claims from processed_dir
    fever_split_file = processed_dir / f"{split}.jsonl"
    if fever_split_file.exists():
        logger.info(f"Loading FEVER split '{split}' from {fever_split_file}...")
        with open(fever_split_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    raw_lbl = item.get("verdict") or item.get("label") or item.get("label4")
                    verdict_enum = map_label_to_verdict(raw_lbl, dataset="fever")

                    unified_records.append({
                        "claim_id": str(item.get("id") or item.get("claim_id")),
                        "claim_text": str(item.get("claim") or item.get("text") or item.get("claim_text")),
                        "dataset": "fever",
                        "label4": verdict_enum.value,
                        "image_path": item.get("image_path") or None,
                    })
    else:
        logger.warning(f"FEVER split file not found: {fever_split_file}")

    # 2. Check for optional external multimodal datasets
    candidate_datasets = ["mocheg", "factify", "newsclippings"]
    for ds in candidate_datasets:
        ds_path = raw_dir / ds / f"{split}.jsonl"
        if ds_path.exists():
            logger.info(f"Processing available multimodal dataset '{ds}' for split '{split}' from {ds_path}...")
            with open(ds_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        raw_lbl = item.get("label") or item.get("verdict") or item.get("label4")
                        verdict_enum = map_label_to_verdict(raw_lbl, dataset=ds)
                        img_path = item.get("image_path") or item.get("image") or None

                        unified_records.append({
                            "claim_id": str(item.get("claim_id") or item.get("id")),
                            "claim_text": str(item.get("claim_text") or item.get("claim") or item.get("text")),
                            "dataset": ds,
                            "label4": verdict_enum.value,
                            "image_path": str(img_path) if img_path else None,
                        })
        else:
            skipped_datasets.append(ds)

    if skipped_datasets:
        logger.info(f"Split '{split}': Cleanly skipped unavailable datasets: {', '.join(skipped_datasets)}")

    return unified_records


def build_all_unified(
    processed_dir: Path = Path("data/processed"),
    raw_dir: Path = Path("data/raw"),
) -> Dict[str, int]:
    """
    Builds unified_{split}.jsonl files for all splits:
    train_text, train_fusion, val, test (and train).
    """
    splits = ["train_text", "train_fusion", "val", "test", "train"]
    counts = {}

    for split in splits:
        records = build_unified_split(split, processed_dir=processed_dir, raw_dir=raw_dir)
        out_file = processed_dir / f"unified_{split}.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts[split] = len(records)
        logger.info(f"Wrote {len(records)} rows to {out_file}")

    return counts


def main():
    parser = argparse.ArgumentParser(description="Build unified datasets for TrustNet Stage 6")
    parser.add_argument("--processed_dir", type=str, default="data/processed", help="Path to processed directory")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Path to raw datasets directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    counts = build_all_unified(
        processed_dir=Path(args.processed_dir),
        raw_dir=Path(args.raw_dir),
    )
    print("\nUnified dataset build complete. Summary:")
    for split, count in counts.items():
        print(f"  - unified_{split}.jsonl: {count} rows")


if __name__ == "__main__":
    main()
