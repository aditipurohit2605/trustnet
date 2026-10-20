"""
Multimodal feature caching pipeline for TrustNet Stage 6.

Precomputes and caches multimodal representations per row and per split:
1. Stage 2 Text Representation:
   - Evaluates a DeBERTa model trained strictly on train_text in 'claim_retrieved' mode.
   - Extracts 3-class logits [verified, contradicted, insufficient_evidence] and
     pooled hidden representation (from the model pooler / [CLS] token).
2. Stage 3 Retrieval Features:
   - top1_score: highest passage retrieval/reranking score
   - top1_top2_gap: margin between top-1 and top-2 passage scores
   - num_passages_above_threshold: count of passages exceeding relevance threshold (0.5)
   - mean_score: average score across retrieved passages
   - source_type counts: distribution across provenance types
   - agreement_feature: count of top-5 passages whose individual text model predictions
     agree with the top-1 passage prediction.
3. Stage 5 Vision Features:
   - image_present (1 if image exists, 0 otherwise)
   - mismatch_score (calibrated [0, 1] OpenCLIP mismatch score)
   - best_reuse_similarity (top-1 dense visual reuse similarity from image bank)
   - ocr_present (1 if text extracted via OCR, 0 otherwise)
   - ocr_confidence (mean OCR confidence)
   - quality_flags (binary indicators for tiny resolution, extreme aspect ratio,
     heavy JPEG compression, screenshot likeness)
   - clip_image_embedding (512-dim normalized OpenCLIP visual embedding)
   * All image features are strictly 0 if image_present == 0.

Resumability:
- Features for each claim are written to outputs/stage6/cache/{split}/{claim_id}.json.
- If a record already exists, it is loaded immediately without recomputation.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from trustnet.data.labels import map_label_to_verdict
from trustnet.data.loader import load_jsonl
from trustnet.schemas import Verdict
from trustnet.vision.analyzer import ImageAnalyzer

logger = logging.getLogger(__name__)

LABEL_LIST = ["verified", "contradicted", "insufficient_evidence"]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}


class SimpleTextDataset(Dataset):
    """Simple PyTorch dataset for fine-tuning text model on train_text."""

    def __init__(self, records: List[Dict[str, Any]], tokenizer, max_length: int = 128):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        claim = rec.get("claim") or rec.get("claim_text") or ""
        evidence = rec.get("evidence", [])
        ev_text = " ".join([e.get("text", "") for e in evidence[:2]]) if evidence else ""

        encoded = self.tokenizer(
            claim,
            ev_text if ev_text else None,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        raw_lbl = rec.get("verdict") or rec.get("label") or rec.get("label4")
        verdict = map_label_to_verdict(raw_lbl).value
        item["labels"] = torch.tensor(LABEL2ID.get(verdict, 2), dtype=torch.long)
        return item


def train_or_load_text_model(
    train_text_path: Path,
    model_dir: Path,
    base_model_name: str = "hf-internal-testing/tiny-random-deberta",
    epochs: int = 3,
    lr: float = 1e-4,
    device: str = "cpu",
) -> Tuple[Any, Any]:
    """
    Trains a text classifier on train_text only (preventing leakage into train_fusion)
    or loads existing weights if already trained.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    weights_path = model_dir / "model.safetensors"
    bin_path = model_dir / "pytorch_model.bin"

    if weights_path.exists() or bin_path.exists():
        logger.info(f"Loading cached text baseline model from {model_dir}...")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.to(device)
        model.eval()
        return model, tokenizer

    logger.info(f"Training text baseline model on {train_text_path} (train_text only)...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.to(device)

    # Load training records
    records = load_jsonl(train_text_path) if train_text_path.exists() else []
    if records:
        dataset = SimpleTextDataset(records, tokenizer)
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        model.train()
        for epoch in range(epochs):
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad()
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

    model.eval()
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    logger.info(f"Saved trained text baseline model to {model_dir}")
    return model, tokenizer


def extract_retrieval_features(
    claim_text: str,
    evidence: List[Dict[str, Any]],
    text_model: Any,
    tokenizer: Any,
    device: str = "cpu",
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Extracts structured retrieval signals from Stage 3 retrieved evidence.
    """
    if not evidence:
        return {
            "top1_score": 0.0,
            "top1_top2_gap": 0.0,
            "num_passages_above_threshold": 0.0,
            "mean_score": 0.0,
            "source_type_corpus_passage": 0.0,
            "source_type_government": 0.0,
            "source_type_news": 0.0,
            "source_type_other": 0.0,
            "agreement_feature": 0.0,
        }

    scores = [float(e.get("score") or 0.0) for e in evidence]
    top1_score = scores[0] if scores else 0.0
    top1_top2_gap = (scores[0] - scores[1]) if len(scores) >= 2 else 0.0
    above_thresh = sum(1.0 for s in scores if s >= threshold)
    mean_score = float(np.mean(scores)) if scores else 0.0

    # Source type counts
    source_counts = {"corpus_passage": 0.0, "government": 0.0, "news": 0.0, "other": 0.0}
    for e in evidence:
        st = str(e.get("source_type") or "").lower()
        if "gov" in st:
            source_counts["government"] += 1.0
        elif "news" in st or "journal" in st:
            source_counts["news"] += 1.0
        elif "passage" in st or "corpus" in st:
            source_counts["corpus_passage"] += 1.0
        else:
            source_counts["other"] += 1.0

    # Agreement feature: evaluate text model on each passage individually
    preds = []
    with torch.no_grad():
        for e in evidence[:5]:
            p_text = e.get("text", "")
            enc = tokenizer(
                claim_text,
                p_text,
                max_length=128,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            out = text_model(**enc)
            pred_idx = int(torch.argmax(out.logits, dim=-1).item())
            preds.append(pred_idx)

    # Number of passages predicting the same label as the top-1 passage
    top1_pred = preds[0] if preds else 0
    agreement_count = sum(1.0 for p in preds if p == top1_pred)

    return {
        "top1_score": round(top1_score, 4),
        "top1_top2_gap": round(top1_top2_gap, 4),
        "num_passages_above_threshold": float(above_thresh),
        "mean_score": round(mean_score, 4),
        "source_type_corpus_passage": source_counts["corpus_passage"],
        "source_type_government": source_counts["government"],
        "source_type_news": source_counts["news"],
        "source_type_other": source_counts["other"],
        "agreement_feature": float(agreement_count),
    }


def extract_image_features(
    image_path: Optional[str],
    claim_text: str,
    analyzer: Optional[ImageAnalyzer] = None,
) -> Dict[str, Any]:
    """
    Extracts visual features from Stage 5.
    If image_path is None or missing, returns zeroed image features with image_present = 0.
    """
    zero_clip = [0.0] * 512
    if not image_path or not Path(image_path).exists():
        return {
            "image_present": 0,
            "mismatch_score": 0.0,
            "best_reuse_similarity": 0.0,
            "ocr_present": 0,
            "ocr_confidence": 0.0,
            "quality_flag_tiny_resolution": 0,
            "quality_flag_extreme_aspect_ratio": 0,
            "quality_flag_heavy_jpeg_compression": 0,
            "quality_flag_screenshot_likeness": 0,
            "clip_image_embedding": zero_clip,
        }

    # Extract signals using Stage 5 ImageAnalyzer
    try:
        signals = analyzer.analyze(image_path, claim_text=claim_text)
        clip_emb = analyzer.clip_scorer.encode_image(image_path)[0].tolist()

        best_reuse = 0.0
        if signals.reuse_matches:
            best_reuse = float(signals.reuse_matches[0].embedding_similarity)

        ocr_pres = 1 if (signals.ocr_text and signals.ocr_text.strip()) else 0
        ocr_conf = float(signals.ocr_confidence or 0.0)
        mismatch = float(signals.mismatch_score or 0.0)

        qflags = signals.quality_flags or []
        q_tiny = 1 if "tiny_resolution" in qflags else 0
        q_ar = 1 if "extreme_aspect_ratio" in qflags else 0
        q_jpeg = 1 if "heavy_jpeg_compression" in qflags else 0
        q_shot = 1 if "screenshot_likeness" in qflags else 0

        return {
            "image_present": 1,
            "mismatch_score": round(mismatch, 4),
            "best_reuse_similarity": round(best_reuse, 4),
            "ocr_present": ocr_pres,
            "ocr_confidence": round(ocr_conf, 4),
            "quality_flag_tiny_resolution": q_tiny,
            "quality_flag_extreme_aspect_ratio": q_ar,
            "quality_flag_heavy_jpeg_compression": q_jpeg,
            "quality_flag_screenshot_likeness": q_shot,
            "clip_image_embedding": [round(float(x), 5) for x in clip_emb],
        }

    except Exception as e:
        logger.warning(f"Failed to extract image features from {image_path}: {e}")
        return {
            "image_present": 0,
            "mismatch_score": 0.0,
            "best_reuse_similarity": 0.0,
            "ocr_present": 0,
            "ocr_confidence": 0.0,
            "quality_flag_tiny_resolution": 0,
            "quality_flag_extreme_aspect_ratio": 0,
            "quality_flag_heavy_jpeg_compression": 0,
            "quality_flag_screenshot_likeness": 0,
            "clip_image_embedding": zero_clip,
        }


def cache_split_features(
    split: str,
    processed_dir: Path = Path("data/processed"),
    cache_base_dir: Path = Path("outputs/stage6/cache"),
    text_model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    analyzer: Optional[ImageAnalyzer] = None,
    device: str = "cpu",
) -> List[Dict[str, Any]]:
    """
    Computes and caches all features for a split in outputs/stage6/cache/{split}/.
    Skips rows already computed (resumable).
    """
    split_cache_dir = cache_base_dir / split
    split_cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load unified split records
    unified_file = processed_dir / f"unified_{split}.jsonl"
    if not unified_file.exists():
        logger.warning(f"Unified file {unified_file} not found; returning empty cache.")
        return []

    unified_records = load_jsonl(unified_file)

    # 2. Load retrieved evidence file if present
    retrieved_map: Dict[str, List[Dict[str, Any]]] = {}
    candidates = [
        processed_dir / f"retrieved_{split}.jsonl",
        processed_dir / "retrieved_train.jsonl",
    ]
    for rf in candidates:
        if rf.exists():
            for r in load_jsonl(rf):
                cid = str(r.get("id") or r.get("claim_id"))
                if cid not in retrieved_map:
                    retrieved_map[cid] = r.get("evidence", [])

    cached_records: List[Dict[str, Any]] = []

    logger.info(f"Processing feature cache for split '{split}' ({len(unified_records)} rows)...")

    for rec in unified_records:
        claim_id = rec["claim_id"]
        cache_file = split_cache_dir / f"{claim_id}.json"

        # Check if already cached (resumable)
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
                cached_records.append(cached_data)
                continue

        claim_text = rec["claim_text"]
        label4 = rec["label4"]
        dataset = rec["dataset"]
        image_path = rec.get("image_path")
        evidence = retrieved_map.get(claim_id, [])

        # (a) Stage 2 Text Inference (Logits + Pooled representation)
        ev_text = " ".join([e.get("text", "") for e in evidence[:2]]) if evidence else ""
        with torch.no_grad():
            enc = tokenizer(
                claim_text,
                ev_text if ev_text else None,
                max_length=128,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            out = text_model(**enc, output_hidden_states=True)
            text_logits = out.logits[0].cpu().numpy().tolist()

            # Pooled embedding from final layer [CLS] token
            pooled_emb = out.hidden_states[-1][:, 0, :][0].cpu().numpy().tolist()

        # (b) Stage 3 Retrieval features
        ret_features = extract_retrieval_features(
            claim_text=claim_text,
            evidence=evidence,
            text_model=text_model,
            tokenizer=tokenizer,
            device=device,
        )

        # (c) Stage 5 Vision features
        img_features = extract_image_features(
            image_path=image_path,
            claim_text=claim_text,
            analyzer=analyzer,
        )

        # Assemble full cached record
        full_record = {
            "claim_id": claim_id,
            "claim_text": claim_text,
            "dataset": dataset,
            "label4": label4,
            "label_idx": LABEL2ID.get(label4, 2),
            "text_logits": [round(float(x), 4) for x in text_logits],
            "text_pooled": [round(float(x), 5) for x in pooled_emb],
            "retrieval_features": ret_features,
            "image_features": img_features,
        }

        # Save individual row cache
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(full_record, f, indent=2)

        cached_records.append(full_record)

    # Consolidate split cache into single master jsonl
    master_file = cache_base_dir / f"cache_{split}.jsonl"
    with open(master_file, "w", encoding="utf-8") as f:
        for r in cached_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(f"Cached {len(cached_records)} rows for split '{split}' into {split_cache_dir}")
    return cached_records


def build_all_caches(
    processed_dir: Path = Path("data/processed"),
    cache_base_dir: Path = Path("outputs/stage6/cache"),
    text_model_dir: Path = Path("outputs/stage6/text_model_train_text"),
    base_text_model: str = "hf-internal-testing/tiny-random-deberta",
    vision_config: str = "configs/stage5_vision.yaml",
) -> Dict[str, int]:
    """
    Main orchestration routine building caches across train_text, train_fusion, val, test.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Train or load text model trained strictly on train_text
    ret_train_text_file = processed_dir / "retrieved_train_text.jsonl"
    if not ret_train_text_file.exists():
        ret_train_text_file = processed_dir / "train_text.jsonl"

    text_model, tokenizer = train_or_load_text_model(
        train_text_path=ret_train_text_file,
        model_dir=text_model_dir,
        base_model_name=base_text_model,
        device=device,
    )

    # 2. Initialize Stage 5 ImageAnalyzer
    analyzer = ImageAnalyzer(config_path=vision_config)

    # 3. Build cache for each split
    splits = ["train_text", "train_fusion", "val", "test"]
    counts = {}

    for split in splits:
        records = cache_split_features(
            split=split,
            processed_dir=processed_dir,
            cache_base_dir=cache_base_dir,
            text_model=text_model,
            tokenizer=tokenizer,
            analyzer=analyzer,
            device=device,
        )
        counts[split] = len(records)

    return counts


def main():
    parser = argparse.ArgumentParser(description="Precompute and cache multimodal features")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--cache_dir", type=str, default="outputs/stage6/cache")
    parser.add_argument("--text_model_dir", type=str, default="outputs/stage6/text_model_train_text")
    parser.add_argument("--base_text_model", type=str, default="hf-internal-testing/tiny-random-deberta")
    parser.add_argument("--vision_config", type=str, default="configs/stage5_vision.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    counts = build_all_caches(
        processed_dir=Path(args.processed_dir),
        cache_base_dir=Path(args.cache_dir),
        text_model_dir=Path(args.text_model_dir),
        base_text_model=args.base_text_model,
        vision_config=args.vision_config,
    )
    print("\nFeature caching complete. Split counts:")
    for split, count in counts.items():
        print(f"  - {split}: {count} cached rows")


if __name__ == "__main__":
    main()
