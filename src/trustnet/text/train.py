"""
CLI training script for Stage 2 text-only baseline classifier.

Fine-tunes Hugging Face sequence classification models (e.g., DeBERTa-v3)
using a plain PyTorch training loop with mixed precision, gradient accumulation,
early stopping based on validation macro-F1, and checkpoint saving.
"""

import argparse
import json
import logging
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    get_linear_schedule_with_warmup,
)

from trustnet.text.dataset import DynamicPaddingCollator, TrustNetTextDataset

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Multilingual sequence classification models (verified HF Hub identifiers)
MULTILINGUAL_MODELS = {
    "mdeberta": "microsoft/mdeberta-v3-base",
    "xlmr": "xlm-roberta-base",
    "muril": "google/muril-base-cased",
}


def set_seed(seed: int = 42) -> None:
    """
    Sets deterministic seeds across Python, NumPy, and PyTorch.
    Ensures reproducibility across runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Enable deterministic algorithms where possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def determine_precision(
    requested: str,
    model_name: str,
    device: torch.device,
) -> Tuple[str, bool, Optional[torch.dtype]]:
    """
    Determines optimal precision (bf16, fp16, or fp32).
    
    Why: DeBERTa-v3 has known numerical instability issues (NaN losses and gradient overflow)
    under standard fp16 due to disentangled attention.
    On Ampere+ GPUs (like RTX 3050), bfloat16 has the same 8-bit dynamic exponent range
    as float32, eliminating the NaN explosion entirely.
    """
    if device.type != "cuda":
        return "fp32", False, None

    requested = requested.lower()

    if requested == "auto" or requested == "bf16":
        if torch.cuda.is_bf16_supported():
            logger.info("Hardware supports BF16. Utilizing native BFloat16 mixed precision.")
            return "bf16", True, torch.bfloat16
        elif requested == "bf16":
            logger.warning("BF16 requested but not supported by hardware. Falling back.")

    if requested == "fp16":
        if "deberta" in model_name.lower():
            logger.warning(
                "CAUTION: DeBERTa-v3 is notoriously unstable under FP16 (frequent NaN loss). "
                "If training collapses, restart with --precision bf16 or --precision fp32."
            )
        return "fp16", True, torch.float16

    # Fallback to standard 32-bit floating point
    logger.info("Running in standard FP32 precision.")
    return "fp32", False, None


def evaluate_model(
    model: PreTrainedModel,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: Optional[torch.dtype],
    labels: List[str],
) -> Dict[str, Any]:
    """
    Runs an evaluation loop over a DataLoader and computes:
    - Cross-entropy loss
    - Accuracy
    - Macro-averaged F1 score
    - Per-class precision, recall, and F1
    """
    model.eval()
    total_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []

    loss_fn = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_labels = batch["labels"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            if use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                    )
                    logits = outputs.logits
                    loss = loss_fn(logits, target_labels)
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                logits = outputs.logits
                loss = loss_fn(logits, target_labels)

            total_loss += loss.item() * len(target_labels)
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(target_labels.cpu().tolist())

    avg_loss = total_loss / len(all_labels) if all_labels else 0.0
    accuracy = accuracy_score(all_labels, all_preds) if all_labels else 0.0
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    # Compute per-class statistics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels,
        all_preds,
        labels=list(range(len(labels))),
        zero_division=0,
    )

    per_class = {}
    for idx, name in enumerate(labels):
        per_class[name] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }

    return {
        "loss": float(avg_loss),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "per_class": per_class,
    }


def train(args: argparse.Namespace) -> None:
    """
    Main training routine executing the plain PyTorch training loop.
    """
    # 1. Load YAML configuration
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Multilingual model resolution
    multilingual_model = getattr(args, "multilingual_model", None)
    if multilingual_model:
        model_name = MULTILINGUAL_MODELS.get(multilingual_model, multilingual_model)
    else:
        model_name = args.model_name or config.get("model_name", "microsoft/deberta-v3-base")

    training_mode = getattr(args, "training_mode", "english_only")
    hi_train_path = Path(getattr(args, "hi_train_path", "data/processed/train_text_hi.jsonl"))
    hi_val_path = Path(getattr(args, "hi_val_path", "data/processed/train_text_hi.jsonl"))

    input_mode = args.input_mode or config.get("input_mode", "claim_only")
    batch_size = args.batch_size or int(config.get("batch_size", 8))
    grad_accum_steps = int(config.get("grad_accum_steps", 2))
    learning_rate = float(args.learning_rate or config.get("learning_rate", 2e-5))
    weight_decay = float(config.get("weight_decay", 0.01))
    warmup_ratio = float(config.get("warmup_ratio", 0.1))
    epochs = int(args.epochs or config.get("epochs", 3))
    patience = int(config.get("early_stopping_patience", 1))
    seed = int(args.seed or config.get("seed", 42))
    max_length = int(config.get("max_length", 256))
    max_doc_chars = int(config.get("max_doc_chars", 600))
    output_dir = Path(args.output_dir or config.get("output_dir", f"outputs/stage2/{input_mode}"))
    labels: List[str] = config.get("labels", ["verified", "contradicted", "insufficient_evidence"])

    train_path = Path(config.get("train_path", "data/processed/train.jsonl"))
    val_path = Path(config.get("val_path", "data/processed/val.jsonl"))
    corpus_path = Path(config.get("corpus_path", "data/corpus/corpus_v1.jsonl"))

    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    precision_mode, use_amp, amp_dtype = determine_precision(args.precision, model_name, device)

    # Optional Weights & Biases initialization
    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=f"{Path(model_name).name}_{input_mode}",
                config={**config, **vars(args)},
            )
            logger.info("Weights & Biases logging initialized.")
        except ImportError:
            logger.warning("Weights & Biases requested (--wandb) but package 'wandb' is not installed.")

    # 2. Initialize Tokenizer & Model
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_name)

    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for i, lbl in enumerate(labels)}

    logger.info(f"Loading sequence classification model: {model_name} with {len(labels)} labels")
    model_cfg = AutoConfig.from_pretrained(
        model_name,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
    )
    model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(model_name, config=model_cfg)
    model.to(device)

    # 3. Build Datasets & DataLoaders with Dynamic Padding
    logger.info(f"Training mode: '{training_mode}' | Model: '{model_name}'")

    if training_mode == "translate_train":
        logger.info(f"Loading English training split from: {train_path} [Mode: {input_mode}]")
        train_ds_en = TrustNetTextDataset(
            split_path=train_path,
            tokenizer=tokenizer,
            label_list=labels,
            corpus_path=corpus_path if input_mode == "claim_evidence" else None,
            input_mode=input_mode,
            max_length=max_length,
            max_doc_chars=max_doc_chars,
            seed=seed,
        )
        logger.info(f"Loading Hindi translated training split from: {hi_train_path} [Mode: {input_mode}]")
        train_ds_hi = TrustNetTextDataset(
            split_path=hi_train_path,
            tokenizer=tokenizer,
            label_list=labels,
            corpus_path=corpus_path if input_mode == "claim_evidence" else None,
            input_mode=input_mode,
            max_length=max_length,
            max_doc_chars=max_doc_chars,
            seed=seed,
        )
        train_dataset = torch.utils.data.ConcatDataset([train_ds_en, train_ds_hi])
        logger.info(f"Combined translate_train dataset created: {len(train_dataset)} total samples ({len(train_ds_en)} en + {len(train_ds_hi)} hi)")

        logger.info(f"Loading validation split from: {val_path} [Mode: {input_mode}]")
        val_dataset = TrustNetTextDataset(
            split_path=val_path,
            tokenizer=tokenizer,
            label_list=labels,
            corpus_path=corpus_path if input_mode == "claim_evidence" else None,
            input_mode=input_mode,
            max_length=max_length,
            max_doc_chars=max_doc_chars,
            seed=seed,
        )
    elif training_mode == "zero_shot_transfer":
        logger.info(f"Zero-shot transfer: training on English {train_path}, validating on Hindi {hi_val_path}")
        train_dataset = TrustNetTextDataset(
            split_path=train_path,
            tokenizer=tokenizer,
            label_list=labels,
            corpus_path=corpus_path if input_mode == "claim_evidence" else None,
            input_mode=input_mode,
            max_length=max_length,
            max_doc_chars=max_doc_chars,
            seed=seed,
        )
        val_dataset = TrustNetTextDataset(
            split_path=hi_val_path,
            tokenizer=tokenizer,
            label_list=labels,
            corpus_path=corpus_path if input_mode == "claim_evidence" else None,
            input_mode=input_mode,
            max_length=max_length,
            max_doc_chars=max_doc_chars,
            seed=seed,
        )
    else:  # "english_only" default
        logger.info(f"Loading training split from: {train_path} [Mode: {input_mode}]")
        train_dataset = TrustNetTextDataset(
            split_path=train_path,
            tokenizer=tokenizer,
            label_list=labels,
            corpus_path=corpus_path if input_mode == "claim_evidence" else None,
            input_mode=input_mode,
            max_length=max_length,
            max_doc_chars=max_doc_chars,
            seed=seed,
        )

        logger.info(f"Loading validation split from: {val_path} [Mode: {input_mode}]")
        val_dataset = TrustNetTextDataset(
            split_path=val_path,
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        drop_last=False,
    )

    # 4. Optimizer and LR Scheduler
    # Filter weight decay for bias and LayerNorm parameters
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)

    total_update_steps = math.ceil(len(train_loader) / grad_accum_steps) * epochs
    warmup_steps = int(total_update_steps * warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )

    scaler = torch.amp.GradScaler('cuda') if (use_amp and precision_mode == "fp16") else None

    # 5. Training Loop
    logger.info(f"Starting training: {epochs} epochs, {total_update_steps} total optimizer steps.")
    best_val_macro_f1 = -1.0
    best_epoch = -1
    patience_counter = 0
    history: List[Dict[str, Any]] = []

    best_checkpoint_dir = output_dir / "best_model"

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_epoch = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_labels = batch["labels"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            if use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=target_labels,
                    )
                    loss = outputs.loss / grad_accum_steps
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=target_labels,
                )
                loss = outputs.loss / grad_accum_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            train_loss_epoch += (loss.item() * grad_accum_steps)

            # Gradient accumulation step
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()

        avg_train_loss = train_loss_epoch / len(train_loader)

        # Validation phase: Select best checkpoint strictly by macro-F1
        val_metrics = evaluate_model(model, val_loader, device, use_amp, amp_dtype, labels)
        val_macro_f1 = val_metrics["macro_f1"]

        logger.info(
            f"Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val Macro-F1: {val_macro_f1:.4f}"
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_macro_f1,
            "val_per_class": val_metrics["per_class"],
        }
        history.append(epoch_record)

        if wandb_run:
            wandb.log(epoch_record)

        # Model selection: ONLY based on validation macro-F1
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            patience_counter = 0

            logger.info(f"New best validation Macro-F1: {best_val_macro_f1:.4f}. Saving checkpoint to {best_checkpoint_dir}")
            best_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_checkpoint_dir)
            tokenizer.save_pretrained(best_checkpoint_dir)
        else:
            patience_counter += 1
            logger.info(f"No improvement in validation Macro-F1 for {patience_counter} epoch(s).")
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after epoch {epoch} (patience={patience}).")
                break

    # Save summary metrics to metrics.json
    summary = {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "history": history,
        "config": {
            "model_name": model_name,
            "multilingual_model": multilingual_model,
            "training_mode": training_mode,
            "input_mode": input_mode,
            "precision": precision_mode,
            "batch_size": batch_size,
            "grad_accum_steps": grad_accum_steps,
            "learning_rate": learning_rate,
            "seed": seed,
        },
    }

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Training completed. Metrics written to: {metrics_path}")
    logger.info(f"Best model checkpoint saved at: {best_checkpoint_dir}")

    if wandb_run:
        wandb.finish()


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TrustNet Text Baseline Trainer")
    parser.add_argument("--config", type=str, default="configs/stage2_baseline.yaml", help="Path to YAML config")
    parser.add_argument("--input_mode", type=str, choices=["claim_only", "claim_evidence"], default=None, help="Input mode")
    parser.add_argument("--model_name", type=str, default=None, help="Hugging Face model checkpoint identifier")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for checkpoints and metrics")
    parser.add_argument("--epochs", type=int, default=None, help="Total training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Per-device batch size")
    parser.add_argument("--learning_rate", type=float, default=None, help="Peak learning rate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--precision", type=str, choices=["auto", "bf16", "fp16", "fp32"], default="auto", help="Precision mode")
    parser.add_argument(
        "--training_mode",
        type=str,
        choices=["english_only", "zero_shot_transfer", "translate_train"],
        default="english_only",
        help="Training mode: english_only, zero_shot_transfer, or translate_train",
    )
    parser.add_argument(
        "--multilingual_model",
        type=str,
        choices=["mdeberta", "xlmr", "muril"],
        default=None,
        help="Multilingual sequence classification model preset",
    )
    parser.add_argument(
        "--hi_train_path",
        type=str,
        default="data/processed/train_text_hi.jsonl",
        help="Path to translated Hindi training split",
    )
    parser.add_argument(
        "--hi_val_path",
        type=str,
        default="data/processed/train_text_hi.jsonl",
        help="Path to Hindi validation split for zero_shot_transfer evaluation",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable optional Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="trustnet", help="Weights & Biases project name")
    return parser.parse_args(args)


if __name__ == "__main__":
    train(parse_args())
