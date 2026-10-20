"""
Stratified validation split partitioning for TrustNet Stage 7.

Splits the existing validation set (val.jsonl) into:
- val_select (50%): used strictly for early stopping and model hyperparameter selection
- val_cal (50%): used strictly for calibration (temperature scaling, abstention threshold fitting)

Guarantees:
- Disjoint splits: no claim_id overlaps between val_select, val_cal, train_fusion, train_text, and test.
- Stratified by label and dataset.
- Updates data/manifest.json with verified row counts and SHA-256 checksums.
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

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


def resplit_val(
    processed_dir: Path = Path("data/processed"),
    manifest_path: Path = Path("data/manifest.json"),
    split_ratio: float = 0.50,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Splits val.jsonl into val_select and val_cal (50/50).
    Aligns retrieved_val and unified_val splits.
    Updates data/manifest.json.
    """
    val_file = processed_dir / "val.jsonl"
    if not val_file.exists():
        raise FileNotFoundError(f"Validation file not found: {val_file}")

    val_records = load_jsonl_records(val_file)
    logger.info(f"Loaded {len(val_records)} validation records to resplit.")

    # Stratify by label
    strat_keys = [
        f"{r.get('dataset', 'fever')}_{r.get('verdict') or r.get('label') or r.get('label4')}"
        for r in val_records
    ]

    try:
        select_records, cal_records = train_test_split(
            val_records,
            test_size=split_ratio,
            random_state=seed,
            stratify=strat_keys,
        )
    except ValueError:
        # Round-robin allocation if class counts are very small
        logger.warning("Small class counts in val; falling back to bucketed split.")
        buckets: Dict[str, List[Dict[str, Any]]] = {}
        for r, key in zip(val_records, strat_keys):
            buckets.setdefault(key, []).append(r)

        select_records, cal_records = [], []
        for key, items in buckets.items():
            half = max(1, len(items) // 2) if len(items) > 1 else (len(select_records) <= len(cal_records))
            if isinstance(half, bool):
                if half:
                    select_records.extend(items)
                else:
                    cal_records.extend(items)
            else:
                select_records.extend(items[:half])
                cal_records.extend(items[half:])

    select_ids = {str(r.get("id") or r.get("claim_id")) for r in select_records}
    cal_ids = {str(r.get("id") or r.get("claim_id")) for r in cal_records}

    # Verify disjoint splits
    assert not select_ids.intersection(cal_ids), "Overlap between val_select and val_cal!"

    # Save new split files
    val_select_file = processed_dir / "val_select.jsonl"
    val_cal_file = processed_dir / "val_cal.jsonl"
    save_jsonl_records(select_records, val_select_file)
    save_jsonl_records(cal_records, val_cal_file)
    logger.info(f"Saved {len(select_records)} records to {val_select_file}")
    logger.info(f"Saved {len(cal_records)} records to {val_cal_file}")

    # Resplit retrieved_val.jsonl
    ret_val_file = processed_dir / "retrieved_val.jsonl"
    if ret_val_file.exists():
        ret_records = load_jsonl_records(ret_val_file)
        ret_select = [r for r in ret_records if str(r.get("id") or r.get("claim_id")) in select_ids]
        ret_cal = [r for r in ret_records if str(r.get("id") or r.get("claim_id")) in cal_ids]
        save_jsonl_records(ret_select, processed_dir / "retrieved_val_select.jsonl")
        save_jsonl_records(ret_cal, processed_dir / "retrieved_val_cal.jsonl")
        logger.info(f"Saved retrieved_val_select ({len(ret_select)}) and retrieved_val_cal ({len(ret_cal)}).")

    # Resplit unified_val.jsonl
    unified_val_file = processed_dir / "unified_val.jsonl"
    if unified_val_file.exists():
        uni_records = load_jsonl_records(unified_val_file)
        uni_select = [r for r in uni_records if str(r.get("id") or r.get("claim_id")) in select_ids]
        uni_cal = [r for r in uni_records if str(r.get("id") or r.get("claim_id")) in cal_ids]
        save_jsonl_records(uni_select, processed_dir / "unified_val_select.jsonl")
        save_jsonl_records(uni_cal, processed_dir / "unified_val_cal.jsonl")
        logger.info(f"Saved unified_val_select ({len(uni_select)}) and unified_val_cal ({len(uni_cal)}).")

    # Resplit outputs/stage6/cache if it exists
    cache_base_dir = Path("outputs/stage6/cache")
    cache_val_master = cache_base_dir / "cache_val.jsonl"
    if cache_val_master.exists():
        cache_records = load_jsonl_records(cache_val_master)
        c_select = [r for r in cache_records if str(r.get("claim_id")) in select_ids]
        c_cal = [r for r in cache_records if str(r.get("claim_id")) in cal_ids]

        save_jsonl_records(c_select, cache_base_dir / "cache_val_select.jsonl")
        save_jsonl_records(c_cal, cache_base_dir / "cache_val_cal.jsonl")

        # Also populate directory folders val_select and val_cal
        (cache_base_dir / "val_select").mkdir(parents=True, exist_ok=True)
        (cache_base_dir / "val_cal").mkdir(parents=True, exist_ok=True)
        for r in c_select:
            with open(cache_base_dir / "val_select" / f"{r['claim_id']}.json", "w", encoding="utf-8") as fp:
                json.dump(r, fp, indent=2)
        for r in c_cal:
            with open(cache_base_dir / "val_cal" / f"{r['claim_id']}.json", "w", encoding="utf-8") as fp:
                json.dump(r, fp, indent=2)
        logger.info(f"Saved cache_val_select ({len(c_select)}) and cache_val_cal ({len(c_cal)}).")

    # Verify global mutual exclusivity across ALL splits
    train_text_file = processed_dir / "train_text.jsonl"
    train_fusion_file = processed_dir / "train_fusion.jsonl"
    test_file = processed_dir / "test.jsonl"

    all_splits = {
        "train_text": {str(r.get("id") or r.get("claim_id")) for r in load_jsonl_records(train_text_file)},
        "train_fusion": {str(r.get("id") or r.get("claim_id")) for r in load_jsonl_records(train_fusion_file)},
        "val_select": select_ids,
        "val_cal": cal_ids,
        "test": {str(r.get("id") or r.get("claim_id")) for r in load_jsonl_records(test_file)},
    }

    split_names = list(all_splits.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            s1, s2 = split_names[i], split_names[j]
            overlap = all_splits[s1].intersection(all_splits[s2])
            if overlap:
                raise ValueError(f"Cross-split leakage detected between '{s1}' and '{s2}': {overlap}")

    logger.info("Global mutual exclusivity verified across all 5 splits!")

    # Update data/manifest.json
    manifest: Dict[str, Any] = {
        "description": "TrustNet Dataset Manifest",
        "seed": seed,
        "splits": {
            "train_text": {
                "file": "processed/train_text.jsonl",
                "count": len(all_splits["train_text"]),
                "sha256": compute_file_sha256(train_text_file),
            },
            "train_fusion": {
                "file": "processed/train_fusion.jsonl",
                "count": len(all_splits["train_fusion"]),
                "sha256": compute_file_sha256(train_fusion_file),
            },
            "val_select": {
                "file": "processed/val_select.jsonl",
                "count": len(select_records),
                "sha256": compute_file_sha256(val_select_file),
            },
            "val_cal": {
                "file": "processed/val_cal.jsonl",
                "count": len(cal_records),
                "sha256": compute_file_sha256(val_cal_file),
            },
            "val": {
                "file": "processed/val.jsonl",
                "count": len(val_records),
                "sha256": compute_file_sha256(val_file),
            },
            "test": {
                "file": "processed/test.jsonl",
                "count": len(all_splits["test"]),
                "sha256": compute_file_sha256(test_file),
            },
        },
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Updated dataset manifest at {manifest_path}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Resplit validation set for TrustNet Stage 7")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--manifest_path", type=str, default="data/manifest.json")
    parser.add_argument("--split_ratio", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    resplit_val(
        processed_dir=Path(args.processed_dir),
        manifest_path=Path(args.manifest_path),
        split_ratio=args.split_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
