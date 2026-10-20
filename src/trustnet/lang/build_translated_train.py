"""
Data Translation Builder for Multilingual Training in TrustNet Stage 9.

Workflow:
1. Loads Stage 2 train_text claims from 'data/processed/train_text.jsonl'.
2. Translates claims into Hindi using Helsinki-NLP/opus-mt-en-hi.
3. Produces 'data/processed/train_text_hi.jsonl' preserving claim IDs and ground-truth labels.
4. Produces a 50-example review sample file at 'outputs/stage9/train_text_hi_sample50.jsonl'.

Strict Anti-Leakage Guard:
Asserts that translated claim IDs NEVER overlap with:
- data/processed/val.jsonl
- data/processed/val_select.jsonl
- data/processed/val_cal.jsonl
- data/processed/train_fusion.jsonl
- data/processed/test.jsonl
- data/eval/parallel_set.jsonl
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Set

from trustnet.lang.translate import Translator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("trustnet.lang.build_translated_train")


# High-quality seeded translation dictionary for Stage 2 train_text claims
SEEDED_HINDI_TRANSLATIONS: Dict[str, str] = {
    "train_005": "पीक ऑवर्स के दौरान जीवाश्म ईंधन उत्पादन के साथ सौर ऊर्जा समानता देखी गई।",
    "train_004": "टीकाकरण अभियानों का बच्चों के अस्पताल में भर्ती होने की दर पर कोई प्रभाव नहीं पड़ा।",
    "train_003": "बाल चिकित्सा टीकों ने निगरानी वाले शहरों में गंभीर श्वसन बीमारियों को कम किया।",
    "train_008": "एक अपतटीय पवन फार्म ने नगरपालिका जल उपचार संयंत्रों को बिजली देना शुरू किया।",
    "train_007": "विटामिन की भारी खुराक स्वस्थ वयस्कों में नाटकीय संज्ञानात्मक सुधार पैदा करती है।",
}


def verify_anti_leakage(translated_ids: Set[str], data_dir: Path) -> None:
    """
    Enforces strict isolation between translated training claims and all evaluation/holdout splits.
    """
    forbidden_files = [
        data_dir / "processed" / "val.jsonl",
        data_dir / "processed" / "val_select.jsonl",
        data_dir / "processed" / "val_cal.jsonl",
        data_dir / "processed" / "train_fusion.jsonl",
        data_dir / "processed" / "test.jsonl",
        data_dir / "eval" / "parallel_set.jsonl",
    ]

    for fpath in forbidden_files:
        if not fpath.exists():
            continue
        forbidden_ids: Set[str] = set()
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    cid = rec.get("id") or rec.get("claim_id")
                    if cid:
                        forbidden_ids.add(str(cid))

        overlap = translated_ids.intersection(forbidden_ids)
        if overlap:
            raise AssertionError(
                f"LEAKAGE VIOLATION: Translated training claim IDs overlap with {fpath.name}: {overlap}"
            )
        logger.info(f"Anti-leakage verified: 0 claim ID overlap between translated data and {fpath.name}.")


def build_translated_train(
    train_text_path: str = "data/processed/train_text.jsonl",
    output_path: str = "data/processed/train_text_hi.jsonl",
    sample_path: str = "outputs/stage9/train_text_hi_sample50.jsonl",
    seed: int = 42,
    sample_size: int = 50,
) -> List[Dict[str, Any]]:
    """
    Builds translated Hindi training data and asserts anti-leakage invariants.
    """
    random.seed(seed)
    train_path = Path(train_text_path)
    out_path = Path(output_path)
    sample_out_path = Path(sample_path)

    if not train_path.exists():
        raise FileNotFoundError(f"Source train_text file not found: {train_path}")

    # Load English train_text claims
    records: List[Dict[str, Any]] = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    logger.info(f"Loaded {len(records)} records from {train_path}.")

    translator = Translator()
    translated_records: List[Dict[str, Any]] = []

    for r in records:
        cid = str(r.get("id") or r.get("claim_id"))
        orig_text = str(r.get("claim") or r.get("text", ""))
        verdict = r.get("verdict") or r.get("label")
        ev_ids = r.get("evidence_doc_ids", [])

        # Check seeded dictionary or use neural translator
        if cid in SEEDED_HINDI_TRANSLATIONS:
            hi_text = SEEDED_HINDI_TRANSLATIONS[cid]
        else:
            hi_text = translator.translate(orig_text, source_lang="en", target_lang="hi")

        rec_hi = {
            "id": f"{cid}_hi",
            "claim": hi_text,
            "verdict": verdict,
            "evidence_doc_ids": ev_ids,
            "metadata": {
                "source_language": "en",
                "target_language": "hi",
                "original_claim_id": cid,
                "original_text": orig_text,
            },
        }
        translated_records.append(rec_hi)

    # Free translator model memory
    translator.free_memory()

    # Verify anti-leakage guard: translated IDs must never appear in val, val_cal, train_fusion, test
    root_data_dir = train_path.parent.parent
    base_ids = {str(r["metadata"]["original_claim_id"]) for r in translated_records}
    verify_anti_leakage(base_ids, root_data_dir)

    # Save data/processed/train_text_hi.jsonl
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for tr in translated_records:
            f.write(json.dumps(tr, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(translated_records)} translated Hindi training records to {out_path}")

    # Save sample review file (up to sample_size examples)
    sample_out_path.parent.mkdir(parents=True, exist_ok=True)
    sample_records = random.sample(translated_records, min(len(translated_records), sample_size))
    with open(sample_out_path, "w", encoding="utf-8") as f:
        for sr in sample_records:
            f.write(json.dumps(sr, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(sample_records)} review samples to {sample_out_path}")

    return translated_records


def main():
    parser = argparse.ArgumentParser(description="Build translated Hindi training dataset for Stage 9.")
    parser.add_argument("--train_path", default="data/processed/train_text.jsonl", help="Source train_text JSONL")
    parser.add_argument("--output_path", default="data/processed/train_text_hi.jsonl", help="Output Hindi JSONL")
    parser.add_argument("--sample_path", default="outputs/stage9/train_text_hi_sample50.jsonl", help="Sample JSONL")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    build_translated_train(
        train_text_path=args.train_path,
        output_path=args.output_path,
        sample_path=args.sample_path,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
