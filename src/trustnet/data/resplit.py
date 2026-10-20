"""
Stratified claim-level resplitting for TrustNet Stage 6.

Splits the existing training set into:
- train_text (70%): used strictly for retraining text baselines without data leakage
- train_fusion (30%): used strictly for training multimodal fusion heads (LightGBM, MLP)
Keeps validation and test splits untouched.

Guarantees:
- Disjoint splits: no claim_id appears in more than one split.
- Stratified by verdict label and dataset.
- Produces data/manifest.json with verified row counts and SHA-256 hashes.
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def compute_file_sha256(filepath: Path) -> str:
    """Computes SHA-256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_jsonl_records(filepath: Path) -> List[Dict[str, Any]]:
    """Loads all non-empty lines from a JSONL file."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl_records(records: List[Dict[str, Any]], filepath: Path) -> None:
    """Writes records to a JSONL file ensuring atomic directory creation."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def resplit_train(
    processed_dir: Path = Path("data/processed"),
    manifest_path: Path = Path("data/manifest.json"),
    fusion_ratio: float = 0.30,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Splits train.jsonl (and retrieved_train.jsonl) into train_text and train_fusion.
    Updates data/manifest.json with row counts and SHA-256 digests.
    """
    train_file = processed_dir / "train.jsonl"
    val_file = processed_dir / "val.jsonl"
    test_file = processed_dir / "test.jsonl"

    if not train_file.exists():
        raise FileNotFoundError(f"Training split not found: {train_file}")

    train_records = load_jsonl_records(train_file)
    logger.info(f"Loaded {len(train_records)} training records to split.")

    # Build stratification keys: combination of dataset and label
    strat_keys = [
        f"{r.get('dataset', 'fever')}_{r.get('verdict') or r.get('label')}"
        for r in train_records
    ]

    # Perform stratified split
    try:
        text_records, fusion_records = train_test_split(
            train_records,
            test_size=fusion_ratio,
            random_state=seed,
            stratify=strat_keys,
        )
    except ValueError:
        # Fallback if class counts are very small: stratified round-robin
        logger.warning("Stratified train_test_split encountered small class counts; falling back to bucketed split.")
        buckets: Dict[str, List[Dict[str, Any]]] = {}
        for r, key in zip(train_records, strat_keys):
            buckets.setdefault(key, []).append(r)

        text_records, fusion_records = [], []
        for key, items in buckets.items():
            num_fusion = max(1, int(round(len(items) * fusion_ratio))) if len(items) > 1 else 0
            fusion_records.extend(items[:num_fusion])
            text_records.extend(items[num_fusion:])

    # Verify no claim overlap
    text_ids = {r.get("id") or r.get("claim_id") for r in text_records}
    fusion_ids = {r.get("id") or r.get("claim_id") for r in fusion_records}
    overlap = text_ids.intersection(fusion_ids)
    if overlap:
        raise ValueError(f"Data leakage detected! Overlapping claim IDs in splits: {overlap}")

    # Save new split files
    train_text_file = processed_dir / "train_text.jsonl"
    train_fusion_file = processed_dir / "train_fusion.jsonl"

    save_jsonl_records(text_records, train_text_file)
    save_jsonl_records(fusion_records, train_fusion_file)

    logger.info(f"Saved {len(text_records)} records to {train_text_file}")
    logger.info(f"Saved {len(fusion_records)} records to {train_fusion_file}")

    # Also resplit retrieved_train.jsonl if present
    retrieved_train_file = processed_dir / "retrieved_train.jsonl"
    if retrieved_train_file.exists():
        retrieved_train_records = load_jsonl_records(retrieved_train_file)
        ret_text = [r for r in retrieved_train_records if (r.get("id") or r.get("claim_id")) in text_ids]
        ret_fusion = [r for r in retrieved_train_records if (r.get("id") or r.get("claim_id")) in fusion_ids]

        save_jsonl_records(ret_text, processed_dir / "retrieved_train_text.jsonl")
        save_jsonl_records(ret_fusion, processed_dir / "retrieved_train_fusion.jsonl")
        logger.info(f"Saved aligned retrieved_train_text ({len(ret_text)}) and retrieved_train_fusion ({len(ret_fusion)}).")

    # Generate updated manifest.json
    manifest: Dict[str, Any] = {
        "description": "TrustNet Dataset Manifest",
        "seed": seed,
        "splits": {
            "train_text": {
                "file": str(train_text_file.relative_to(processed_dir.parent)).replace("\\", "/"),
                "count": len(text_records),
                "sha256": compute_file_sha256(train_text_file),
            },
            "train_fusion": {
                "file": str(train_fusion_file.relative_to(processed_dir.parent)).replace("\\", "/"),
                "count": len(fusion_records),
                "sha256": compute_file_sha256(train_fusion_file),
            },
            "val": {
                "file": str(val_file.relative_to(processed_dir.parent)).replace("\\", "/"),
                "count": len(load_jsonl_records(val_file)) if val_file.exists() else 0,
                "sha256": compute_file_sha256(val_file) if val_file.exists() else "",
            },
            "test": {
                "file": str(test_file.relative_to(processed_dir.parent)).replace("\\", "/"),
                "count": len(load_jsonl_records(test_file)) if test_file.exists() else 0,
                "sha256": compute_file_sha256(test_file) if test_file.exists() else "",
            },
        },
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Updated dataset manifest at {manifest_path}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Stratified Resplit for TrustNet Stage 6")
    parser.add_argument("--processed_dir", type=str, default="data/processed", help="Path to processed data directory")
    parser.add_argument("--manifest_path", type=str, default="data/manifest.json", help="Path to output manifest.json")
    parser.add_argument("--fusion_ratio", type=float, default=0.30, help="Fraction of train data for train_fusion")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    resplit_train(
        processed_dir=Path(args.processed_dir),
        manifest_path=Path(args.manifest_path),
        fusion_ratio=args.fusion_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
