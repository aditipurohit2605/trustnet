"""Data module for TrustNet."""

from trustnet.data.loader import load_claims, load_corpus, load_jsonl

__all__ = ["load_jsonl", "load_claims", "load_corpus"]
