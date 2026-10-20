"""
Dataset loading and tokenization for text-only claim verification baselines.
Supports 'claim_only' and 'claim_evidence' input modes with deterministic NEI evidence selection
and dynamic batch padding.
"""

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from trustnet.schemas import Verdict


def select_deterministic_nei_doc(
    claim_id: str,
    corpus_doc_ids: List[str],
    seed: int = 42,
) -> str:
    """
    Deterministically selects one document ID from corpus_doc_ids for an NEI claim.
    
    Why: Using Python's global random state causes non-deterministic behavior when
    examples are accessed in different orders or by multiple dataloader workers.
    By deriving a dedicated RNG seeded with the MD5 hash of (seed + claim_id), the choice
    is mathematically guaranteed to be identical for the same claim across all runs.
    """
    if not corpus_doc_ids:
        return ""
    # Compute an MD5 digest combining the experiment seed and the unique claim ID
    digest = hashlib.md5(f"{seed}_{claim_id}".encode("utf-8")).hexdigest()
    # Convert hex digest to integer and seed an isolated, thread-safe Random instance
    rng = random.Random(int(digest, 16))
    return rng.choice(corpus_doc_ids)


def build_label_mapping(labels: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Builds bidirectional mappings between canonical Verdict string values and integer IDs.
    
    Validates that each label in the config corresponds to a valid Verdict enum member.
    """
    label2id = {}
    id2label = {}
    for idx, lbl in enumerate(labels):
        # Normalize to canonical Verdict enum value to ensure consistency
        canonical_val = Verdict.normalize(lbl).value
        label2id[canonical_val] = idx
        id2label[idx] = canonical_val
    return label2id, id2label


class TrustNetTextDataset(Dataset):
    """
    PyTorch Dataset for text-only claim verification.
    
    Attributes:
        examples (List[Dict]): Processed examples ready for batching.
        label2id (Dict[str, int]): Mapping from verdict string to class index.
        id2label (Dict[int, str]): Mapping from class index to verdict string.
        input_mode (str): Either 'claim_only' or 'claim_evidence'.
    """

    def __init__(
        self,
        split_path: Union[str, Path],
        tokenizer: PreTrainedTokenizerBase,
        label_list: List[str],
        corpus_path: Optional[Union[str, Path]] = None,
        input_mode: str = "claim_only",
        max_length: int = 256,
        max_doc_chars: int = 600,
        seed: int = 42,
        is_test: bool = False,
    ):
        """
        Args:
            split_path: Path to the split JSONL file (train, val, or test).
            tokenizer: Hugging Face PreTrainedTokenizer instance.
            label_list: List of valid Verdict string values from config.
            corpus_path: Path to corpus_v1.jsonl (required if input_mode == 'claim_evidence').
            input_mode: 'claim_only' or 'claim_evidence'.
            max_length: Maximum combined token sequence length.
            max_doc_chars: Character limit per evidence document before concatenation.
            seed: RNG seed for deterministic NEI evidence document selection.
            is_test: Set to True when loading the test split (for safety checks).
        """
        super().__init__()
        self.split_path = Path(split_path)
        self.tokenizer = tokenizer
        self.input_mode = input_mode.lower()
        self.max_length = max_length
        self.max_doc_chars = max_doc_chars
        self.seed = seed
        self.is_test = is_test

        if self.input_mode not in ("claim_only", "claim_evidence"):
            raise ValueError(f"Unknown input_mode: '{input_mode}'. Must be 'claim_only' or 'claim_evidence'.")

        # Build class index mappings
        self.label2id, self.id2label = build_label_mapping(label_list)

        # Load corpus if evidence mode is requested
        self.corpus: Dict[str, str] = {}
        self.corpus_doc_ids: List[str] = []
        if self.input_mode == "claim_evidence":
            if corpus_path is None:
                raise ValueError("corpus_path must be provided when input_mode is 'claim_evidence'.")
            self._load_corpus(Path(corpus_path))

        # Parse raw jsonl records into tokenized features
        self.examples = self._load_and_process_split()

    def _load_corpus(self, path: Path):
        """Reads corpus documents into an in-memory dictionary keyed by doc_id."""
        if not path.is_file():
            raise FileNotFoundError(f"Corpus file not found at: {path}")

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                doc_id = str(record.get("doc_id") or record.get("id"))
                text = str(record.get("text") or record.get("contents") or "")
                title = record.get("title")
                # Prepend title if present for richer context
                doc_text = f"{title}: {text}" if title else text
                self.corpus[doc_id] = doc_text
        # Sort doc IDs to guarantee identical iteration order across Python environments
        self.corpus_doc_ids = sorted(list(self.corpus.keys()))

    def _load_and_process_split(self) -> List[Dict[str, Any]]:
        """Parses the JSONL split and tokenizes each claim."""
        if not self.split_path.is_file():
            raise FileNotFoundError(f"Split file not found at: {self.split_path}")

        processed = []
        with open(self.split_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)

                # Extract claim ID and text supporting multiple standard naming conventions
                claim_id = str(record.get("id") or record.get("claim_id") or f"claim_{line_idx}")
                claim_text = str(record.get("claim") or record.get("text") or "").strip()

                # Normalize verdict to label ID
                raw_verdict = record.get("verdict") or record.get("label")
                if raw_verdict is None:
                    label_id = -1  # Unlabeled test record
                else:
                    canonical_verdict = Verdict.normalize(raw_verdict).value
                    if canonical_verdict not in self.label2id:
                        raise ValueError(
                            f"Encountered verdict '{raw_verdict}' (normalized: '{canonical_verdict}') on line {line_idx} "
                            f"not present in config labels: {list(self.label2id.keys())}"
                        )
                    label_id = self.label2id[canonical_verdict]

                # Construct model input based on the chosen input mode
                if self.input_mode == "claim_only":
                    # Input is strictly the isolated claim text
                    encoding = self.tokenizer(
                        claim_text,
                        truncation=True,
                        max_length=self.max_length,
                        padding=False,  # Padding handled dynamically by collate_fn
                        return_attention_mask=True,
                    )
                    input_text_repr = claim_text
                else:
                    # 'claim_evidence' mode
                    evidence_doc_ids = record.get("evidence_doc_ids") or record.get("gold_evidence_ids") or []
                    evidence_texts: List[str] = []

                    if evidence_doc_ids:
                        # Gold evidence exists: retrieve and truncate each document
                        for doc_id in evidence_doc_ids:
                            doc_str = str(doc_id)
                            if doc_str in self.corpus:
                                text = self.corpus[doc_str][:self.max_doc_chars].strip()
                                evidence_texts.append(text)
                            else:
                                # Document referenced by ID but missing from corpus
                                evidence_texts.append(f"[Document {doc_str} not in corpus]")
                        evidence_str = " \n ".join(evidence_texts)
                    else:
                        # NOT ENOUGH INFO / insufficient evidence claims:
                        # Sample one random document deterministically based on claim_id & seed
                        chosen_doc_id = select_deterministic_nei_doc(claim_id, self.corpus_doc_ids, self.seed)
                        if chosen_doc_id and chosen_doc_id in self.corpus:
                            evidence_str = self.corpus[chosen_doc_id][:self.max_doc_chars].strip()
                        else:
                            evidence_str = ""

                    # Tokenize pair: (claim, evidence) with truncation='only_second'
                    # Why: This guarantees the claim is NEVER truncated even if evidence is long.
                    encoding = self.tokenizer(
                        claim_text,
                        evidence_str,
                        truncation="only_second",
                        max_length=self.max_length,
                        padding=False,
                        return_attention_mask=True,
                    )
                    input_text_repr = f"{claim_text} [SEP] {evidence_str}" if evidence_str else claim_text

                item = {
                    "claim_id": claim_id,
                    "input_ids": encoding["input_ids"],
                    "attention_mask": encoding["attention_mask"],
                    "label": label_id,
                    "input_text": input_text_repr,
                }
                if "token_type_ids" in encoding:
                    item["token_type_ids"] = encoding["token_type_ids"]

                processed.append(item)

        return processed

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.examples[idx]


class DynamicPaddingCollator:
    """
    Collate function that dynamically pads sequences in a batch to the length of the longest
    sequence in that specific batch, rather than padding all batches to a static max_length.
    
    Why: Dynamic padding significantly reduces wasted computation on pad tokens, increasing
    GPU throughput by up to 2-3x during training and inference.
    """

    def __init__(self, pad_token_id: int, pad_to_multiple_of: Optional[int] = 8):
        """
        Args:
            pad_token_id: Token ID used for padding by the tokenizer.
            pad_to_multiple_of: If set, pads the sequence length to a multiple of this value.
                                (Multiples of 8 activate NVIDIA Tensor Cores for maximum speed).
        """
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_max_len = max(len(x["input_ids"]) for x in batch)

        if self.pad_to_multiple_of is not None and self.pad_to_multiple_of > 0:
            remainder = batch_max_len % self.pad_to_multiple_of
            if remainder != 0:
                batch_max_len += (self.pad_to_multiple_of - remainder)

        batch_input_ids = []
        batch_attention_mask = []
        batch_token_type_ids = []
        has_token_types = "token_type_ids" in batch[0]
        batch_labels = []
        claim_ids = []
        input_texts = []

        for item in batch:
            input_ids = item["input_ids"]
            attn_mask = item["attention_mask"]
            padding_len = batch_max_len - len(input_ids)

            # Pad input_ids with pad_token_id and attention_mask with 0
            padded_input_ids = input_ids + [self.pad_token_id] * padding_len
            padded_attn_mask = attn_mask + [0] * padding_len

            batch_input_ids.append(padded_input_ids)
            batch_attention_mask.append(padded_attn_mask)
            batch_labels.append(item["label"])
            claim_ids.append(item["claim_id"])
            input_texts.append(item.get("input_text", ""))

            if has_token_types:
                token_types = item.get("token_type_ids", [0] * len(input_ids))
                padded_token_types = token_types + [0] * padding_len
                batch_token_type_ids.append(padded_token_types)

        output = {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "claim_ids": claim_ids,
            "input_texts": input_texts,
        }
        if has_token_types:
            output["token_type_ids"] = torch.tensor(batch_token_type_ids, dtype=torch.long)

        return output
