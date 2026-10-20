"""
Unit tests for Stage 2 text baseline classifier.

Tests:
1. Label mapping covers every Verdict in config.
2. claim_only and claim_evidence produce correctly shaped batches with dynamic padding.
3. NEI random evidence selection is 100% deterministic per claim ID and seed.
4. Smoke test: trains for a few steps on 64 synthetic examples on CPU and completes successfully.
"""

import json
from pathlib import Path
from typing import List

import pytest
import torch
import yaml
from transformers import AutoTokenizer

from trustnet.schemas import Verdict
from trustnet.text.dataset import (
    DynamicPaddingCollator,
    TrustNetTextDataset,
    build_label_mapping,
    select_deterministic_nei_doc,
)
from trustnet.text.train import parse_args, train


def test_label_mapping_covers_every_verdict_in_config():
    """Verifies that label mapping covers all Verdict values defined in the configuration."""
    config_path = Path("configs/stage2_baseline.yaml")
    assert config_path.is_file(), f"Config file missing: {config_path}"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    labels = config["labels"]
    label2id, id2label = build_label_mapping(labels)

    # Must cover every item in config
    assert len(label2id) == len(labels)
    assert len(id2label) == len(labels)

    # Every label must map to an integer between 0 and len(labels) - 1
    for idx, label_str in enumerate(labels):
        canonical = Verdict.normalize(label_str).value
        assert canonical in label2id
        assert label2id[canonical] == idx
        assert id2label[idx] == canonical


def test_deterministic_nei_evidence_selection():
    """
    Verifies that NEI random evidence choice is 100% deterministic
    for the same claim ID and seed, but produces variety across different claims.
    """
    corpus_doc_ids = [f"doc_{i:03d}" for i in range(1, 50)]
    seed = 42
    claim_id_1 = "claim_xyz_100"
    claim_id_2 = "claim_abc_200"

    # Repeated calls on the same claim must produce the exact same document ID
    first_choice = select_deterministic_nei_doc(claim_id_1, corpus_doc_ids, seed=seed)
    for _ in range(10):
        assert select_deterministic_nei_doc(claim_id_1, corpus_doc_ids, seed=seed) == first_choice

    # Different claim ID under the same seed should select deterministically
    choice_other_claim = select_deterministic_nei_doc(claim_id_2, corpus_doc_ids, seed=seed)
    assert select_deterministic_nei_doc(claim_id_2, corpus_doc_ids, seed=seed) == choice_other_claim


def test_claim_only_and_claim_evidence_batches(tmp_path: Path):
    """
    Verifies that both claim_only and claim_evidence modes tokenize data
    and produce correctly shaped batched tensors with dynamic padding.
    """
    tiny_model = "hf-internal-testing/tiny-random-deberta"
    tokenizer = AutoTokenizer.from_pretrained(tiny_model)

    # Create dummy corpus
    corpus_file = tmp_path / "corpus.jsonl"
    corpus_data = [
        {"doc_id": "doc_1", "text": "Short evidence text about atmospheric recovery."},
        {"doc_id": "doc_2", "text": "Very long evidence text " + "repeated content " * 100},
    ]
    with open(corpus_file, "w", encoding="utf-8") as f:
        for doc in corpus_data:
            f.write(json.dumps(doc) + "\n")

    # Create dummy split with verified, contradicted, and nei claims
    split_file = tmp_path / "split.jsonl"
    split_data = [
        {"id": "c1", "claim": "Ozone is healing.", "verdict": "verified", "evidence_doc_ids": ["doc_1"]},
        {"id": "c2", "claim": "Dam is breaking.", "verdict": "contradicted", "evidence_doc_ids": ["doc_2"]},
        {"id": "c3", "claim": "Alien invasion reported.", "verdict": "insufficient_evidence", "evidence_doc_ids": []},
    ]
    with open(split_file, "w", encoding="utf-8") as f:
        for c in split_data:
            f.write(json.dumps(c) + "\n")

    labels = ["verified", "contradicted", "insufficient_evidence"]
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    collator = DynamicPaddingCollator(pad_token_id=pad_token_id, pad_to_multiple_of=None)

    # 1. Test claim_only mode
    dataset_claim_only = TrustNetTextDataset(
        split_path=split_file,
        tokenizer=tokenizer,
        label_list=labels,
        input_mode="claim_only",
        max_length=64,
    )
    assert len(dataset_claim_only) == 3

    batch_claim_only = collator([dataset_claim_only[0], dataset_claim_only[1], dataset_claim_only[2]])
    assert "input_ids" in batch_claim_only
    assert "attention_mask" in batch_claim_only
    assert "labels" in batch_claim_only
    assert batch_claim_only["input_ids"].shape[0] == 3
    assert batch_claim_only["input_ids"].ndim == 2
    assert batch_claim_only["attention_mask"].shape == batch_claim_only["input_ids"].shape
    assert batch_claim_only["labels"].tolist() == [0, 1, 2]

    # 2. Test claim_evidence mode
    dataset_claim_evidence = TrustNetTextDataset(
        split_path=split_file,
        tokenizer=tokenizer,
        label_list=labels,
        corpus_path=corpus_file,
        input_mode="claim_evidence",
        max_length=64,
        max_doc_chars=200,
        seed=42,
    )
    assert len(dataset_claim_evidence) == 3

    batch_claim_evidence = collator([dataset_claim_evidence[0], dataset_claim_evidence[1]])
    assert batch_claim_evidence["input_ids"].shape[0] == 2
    assert batch_claim_evidence["input_ids"].ndim == 2
    assert batch_claim_evidence["input_ids"].shape[1] <= 64
    assert batch_claim_evidence["labels"].tolist() == [0, 1]


def test_smoke_training_cpu(tmp_path: Path):
    """
    Smoke test: Generates 64 synthetic examples across 3 classes, trains on CPU
    for 1 epoch with tiny-random-deberta, and verifies checkpoint creation.
    """
    tiny_model = "hf-internal-testing/tiny-random-deberta"

    # Build synthetic corpus
    corpus_file = tmp_path / "corpus_v1.jsonl"
    corpus_records = [
        {"doc_id": f"doc_{i}", "text": f"Knowledge base text paragraph {i} providing context."}
        for i in range(1, 10)
    ]
    with open(corpus_file, "w", encoding="utf-8") as f:
        for rec in corpus_records:
            f.write(json.dumps(rec) + "\n")

    # Build 64 synthetic training examples
    verdicts = ["verified", "contradicted", "insufficient_evidence"]
    train_file = tmp_path / "train.jsonl"
    with open(train_file, "w", encoding="utf-8") as f:
        for i in range(64):
            verdict = verdicts[i % 3]
            evidence = [f"doc_{((i % 9) + 1)}"] if verdict != "insufficient_evidence" else []
            record = {
                "id": f"claim_{i}",
                "claim": f"Synthetic claim number {i} for testing baseline execution.",
                "verdict": verdict,
                "evidence_doc_ids": evidence,
            }
            f.write(json.dumps(record) + "\n")

    # Build 16 synthetic validation examples
    val_file = tmp_path / "val.jsonl"
    with open(val_file, "w", encoding="utf-8") as f:
        for i in range(16):
            verdict = verdicts[i % 3]
            evidence = [f"doc_{((i % 9) + 1)}"] if verdict != "insufficient_evidence" else []
            record = {
                "id": f"val_claim_{i}",
                "claim": f"Validation claim number {i}.",
                "verdict": verdict,
                "evidence_doc_ids": evidence,
            }
            f.write(json.dumps(record) + "\n")

    # Create temporary config
    config_data = {
        "model_name": tiny_model,
        "labels": verdicts,
        "input_mode": "claim_only",
        "nei_evidence": "random_doc",
        "max_length": 32,
        "max_doc_chars": 100,
        "train_path": str(train_file),
        "val_path": str(val_file),
        "corpus_path": str(corpus_file),
        "batch_size": 16,
        "grad_accum_steps": 1,
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "warmup_ratio": 0.0,
        "epochs": 1,
        "early_stopping_patience": 1,
        "seed": 42,
        "precision": "fp32",  # Force FP32 on CPU
        "output_dir": str(tmp_path / "output_smoke"),
    }
    config_file = tmp_path / "smoke_config.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f)

    # Execute training routine
    args = parse_args([])
    args.config = str(config_file)
    args.model_name = tiny_model
    args.input_mode = "claim_only"
    args.output_dir = str(tmp_path / "output_smoke")
    args.epochs = 1
    args.batch_size = 16
    args.precision = "fp32"
    args.wandb = False

    # Run train function
    train(args)

    # Verify outputs
    out_dir = tmp_path / "output_smoke"
    best_model_dir = out_dir / "best_model"
    metrics_file = out_dir / "metrics.json"

    assert best_model_dir.is_dir(), f"Expected checkpoint directory at: {best_model_dir}"
    assert metrics_file.is_file(), f"Expected metrics.json at: {metrics_file}"

    with open(metrics_file, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert "best_val_macro_f1" in metrics
    assert "history" in metrics
    assert len(metrics["history"]) == 1
    assert "val_macro_f1" in metrics["history"][0]

    # Verify evaluation pipeline produces metrics.json, confusion_matrix.png, and predictions.jsonl
    eval_dir = tmp_path / "output_eval"
    from trustnet.text.evaluate import evaluate as run_eval, parse_args as parse_eval_args
    eval_args = parse_eval_args([
        "--checkpoint", str(best_model_dir),
        "--split_path", str(val_file),
        "--config", str(config_file),
        "--output_dir", str(eval_dir),
        "--batch_size", "8",
    ])
    run_eval(eval_args)

    assert (eval_dir / "metrics.json").is_file(), "metrics.json missing from evaluation"
    assert (eval_dir / "confusion_matrix.png").is_file(), "confusion_matrix.png missing from evaluation"
    assert (eval_dir / "predictions.jsonl").is_file(), "predictions.jsonl missing from evaluation"

    with open(eval_dir / "predictions.jsonl", "r", encoding="utf-8") as pf:
        pred_line = json.loads(pf.readline())
        assert "claim_id" in pred_line
        assert "gold_label" in pred_line
        assert "predicted_label" in pred_line
        assert "raw_logits" in pred_line
        assert isinstance(pred_line["raw_logits"], list)
        assert len(pred_line["raw_logits"]) == len(verdicts)
        assert "input_text" in pred_line
