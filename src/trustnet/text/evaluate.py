"""
CLI evaluation script for Stage 2 text-only baseline classifier.

Loads a fine-tuned model checkpoint, evaluates performance on a specified split (val/test),
and generates:
1. metrics.json (accuracy, macro-F1, per-class precision/recall/F1, and support)
2. confusion_matrix.png (annotated heatmap plot)
3. predictions.jsonl (claim ID, gold label, predicted label, raw un-normalized logits, and input text)
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend suitable for headless servers / CLI
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from trustnet.text.dataset import DynamicPaddingCollator, TrustNetTextDataset

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def plot_and_save_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Path,
) -> None:
    """
    Renders and saves a clean confusion matrix heatmap.
    
    Includes both raw counts and row-normalized percentages.
    """
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Confusion Matrix",
        ylabel="True Verdict",
        xlabel="Predicted Verdict",
    )

    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    # Annotate each cell with raw count and percentage of true class
    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        row_sum = np.sum(cm[i, :])
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = (count / row_sum * 100) if row_sum > 0 else 0.0
            text_color = "white" if count > thresh else "black"
            ax.text(
                j,
                i,
                f"{count}\n({pct:.1f}%)",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved confusion matrix plot to: {output_path}")


def evaluate(args: argparse.Namespace) -> None:
    """
    Main evaluation pipeline.
    """
    # 1. Load Configuration
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    labels: List[str] = config.get("labels", ["verified", "contradicted", "insufficient_evidence"])
    input_mode = args.input_mode or config.get("input_mode", "claim_only")
    corpus_path = Path(args.corpus_path or config.get("corpus_path", "data/corpus/corpus_v1.jsonl"))
    max_length = int(config.get("max_length", 256))
    max_doc_chars = int(config.get("max_doc_chars", 600))
    seed = int(args.seed or config.get("seed", 42))

    checkpoint_path = Path(args.checkpoint)
    split_path = Path(args.split_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Evaluating checkpoint: {checkpoint_path} on split: {split_path}")
    logger.info(f"Inference device: {device}")

    # 2. Load Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    model.to(device)
    model.eval()

    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for i, lbl in enumerate(labels)}

    # 3. Build Dataset & DataLoader
    dataset = TrustNetTextDataset(
        split_path=split_path,
        tokenizer=tokenizer,
        label_list=labels,
        corpus_path=corpus_path if input_mode == "claim_evidence" else None,
        input_mode=input_mode,
        max_length=max_length,
        max_doc_chars=max_doc_chars,
        seed=seed,
    )

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    collator = DynamicPaddingCollator(pad_token_id=pad_token_id, pad_to_multiple_of=8 if device.type == "cuda" else None)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    # 4. Inference Loop
    all_claim_ids: List[str] = []
    all_input_texts: List[str] = []
    all_gold_labels: List[int] = []
    all_pred_labels: List[int] = []
    all_raw_logits: List[List[float]] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            logits = outputs.logits  # Raw unnormalized model logits

            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            raw_logits_list = logits.cpu().to(torch.float32).tolist()

            all_pred_labels.extend(preds)
            all_raw_logits.extend(raw_logits_list)
            all_gold_labels.extend(batch["labels"].tolist())
            all_claim_ids.extend(batch["claim_ids"])
            all_input_texts.extend(batch["input_texts"])

    # 5. Compute Metrics (if gold labels are present, i.e. not -1)
    has_gold = all(lbl != -1 for lbl in all_gold_labels)
    if has_gold:
        accuracy = float(accuracy_score(all_gold_labels, all_pred_labels))
        macro_f1 = float(f1_score(all_gold_labels, all_pred_labels, average="macro", zero_division=0))
        prec, rec, f1, supp = precision_recall_fscore_support(
            all_gold_labels,
            all_pred_labels,
            labels=list(range(len(labels))),
            zero_division=0,
        )

        per_class_metrics = {}
        for idx, lbl_name in enumerate(labels):
            per_class_metrics[lbl_name] = {
                "precision": float(prec[idx]),
                "recall": float(rec[idx]),
                "f1": float(f1[idx]),
                "support": int(supp[idx]),
            }

        cm = confusion_matrix(all_gold_labels, all_pred_labels, labels=list(range(len(labels))))

        # Plot and save confusion matrix
        plot_and_save_confusion_matrix(cm, labels, output_dir / "confusion_matrix.png")
    else:
        accuracy = None
        macro_f1 = None
        per_class_metrics = {}

    metrics_payload = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class_metrics,
        "total_examples": len(all_claim_ids),
        "split_evaluated": str(split_path),
        "checkpoint": str(checkpoint_path),
        "input_mode": input_mode,
    }

    metrics_json_path = output_dir / "metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    logger.info(f"Saved evaluation metrics to: {metrics_json_path}")

    # 6. Write predictions.jsonl with RAW LOGITS and actual input text used
    predictions_path = output_dir / "predictions.jsonl"
    with open(predictions_path, "w", encoding="utf-8") as f:
        for claim_id, gold_idx, pred_idx, logits, input_text in zip(
            all_claim_ids, all_gold_labels, all_pred_labels, all_raw_logits, all_input_texts
        ):
            record = {
                "claim_id": claim_id,
                "gold_label": id2label.get(gold_idx, "unlabeled") if gold_idx != -1 else "unlabeled",
                "predicted_label": id2label[pred_idx],
                "raw_logits": logits,  # Raw unnormalized float logits
                "input_text": input_text,  # Exact formatted string passed into tokenizer
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"Saved predictions to: {predictions_path}")
    logger.info("Evaluation complete.")


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TrustNet Text Baseline Evaluator")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--split_path", type=str, required=True, help="Path to split JSONL to evaluate")
    parser.add_argument("--config", type=str, default="configs/stage2_baseline.yaml", help="Path to YAML config")
    parser.add_argument("--corpus_path", type=str, default=None, help="Path to corpus JSONL")
    parser.add_argument("--input_mode", type=str, choices=["claim_only", "claim_evidence"], default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/stage2/eval", help="Directory for outputs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    return parser.parse_args(args)


if __name__ == "__main__":
    evaluate(parse_args())
