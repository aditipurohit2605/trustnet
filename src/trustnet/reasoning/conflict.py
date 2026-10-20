"""
Conflict Detection Engine for TrustNet Stage 8.

Identifies authentic disagreements and evidentiary contradictions between authoritative sources.
Uses the Stage 2 NLI/claim-passage classifier to label passages as supporting, refuting, or neutral.

Conflict Criteria:
A conflict is flagged if and only if:
1. At least one supporting passage has source_weight >= conflict_min_weight and confidence >= threshold.
2. At least one refuting passage has source_weight >= conflict_min_weight and confidence >= threshold.

Both supporting and contradicting sets are preserved in full with explanatory commentary.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy.special import softmax

from trustnet.reasoning.source_reliability import SourceReliabilityAssigner
from trustnet.schemas import ConflictInfo, Evidence

logger = logging.getLogger(__name__)

# Stance mapping to 3-class indices: 0: verified (supporting), 1: contradicted (refuting), 2: neutral
STANCE_CLASSES = ["supporting", "contradicting", "neutral"]


class ConflictDetector:
    """
    Detects contradictory evidence from authoritative sources for a factual claim.
    """

    _cached_text_model: Optional[Any] = None
    _cached_tokenizer: Optional[Any] = None
    _cached_model_dir: Optional[Path] = None

    def __init__(
        self,
        text_model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        source_assigner: Optional[SourceReliabilityAssigner] = None,
        conflict_min_weight: float = 0.50,
        confidence_threshold: float = 0.45,
        model_dir: Union[str, Path] = "outputs/stage6/text_model_train_text",
        device: str = "auto",
    ):
        self.text_model = text_model
        self.tokenizer = tokenizer
        self.source_assigner = source_assigner or SourceReliabilityAssigner()
        self.conflict_min_weight = float(conflict_min_weight)
        self.confidence_threshold = float(confidence_threshold)
        self.model_dir = Path(model_dir)

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._ensure_model_loaded()

    def _ensure_model_loaded(self) -> None:
        """Loads Stage 2 text model and tokenizer if not provided."""
        if self.text_model is not None and self.tokenizer is not None:
            return

        if ConflictDetector._cached_text_model is not None and ConflictDetector._cached_model_dir == self.model_dir:
            self.text_model = ConflictDetector._cached_text_model
            self.tokenizer = ConflictDetector._cached_tokenizer
            return

        if self.model_dir.exists():
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                logger.info(f"Loading Stage 2 text classifier from {self.model_dir} for conflict detection...")
                self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
                self.text_model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir)).to(self.device)
                self.text_model.eval()
                ConflictDetector._cached_text_model = self.text_model
                ConflictDetector._cached_tokenizer = self.tokenizer
                ConflictDetector._cached_model_dir = self.model_dir
            except Exception as e:
                logger.warning(f"Could not load Stage 2 text model from {self.model_dir}: {e}")

    def classify_passage_stance(
        self,
        claim_text: str,
        passage_text: str,
    ) -> Tuple[str, float]:
        """
        Classifies whether a passage supports, refutes, or is neutral toward the claim.
        
        Returns:
            Tuple of (stance: 'supporting' | 'contradicting' | 'neutral', confidence: float)
        """
        if self.text_model is None or self.tokenizer is None:
            # Simple lexical heuristic fallback if model weights are unavailable
            p_lower = passage_text.lower()
            if any(w in p_lower for w in ["deny", "denies", "false", "hoax", "incorrect", "refutes", "no evidence"]):
                return "contradicting", 0.70
            elif any(w in p_lower for w in ["confirm", "confirms", "verifies", "proven", "true", "accurate", "official"]):
                return "supporting", 0.70
            return "neutral", 0.50

        with torch.no_grad():
            inputs = self.tokenizer(
                claim_text,
                passage_text,
                max_length=128,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            outputs = self.text_model(**inputs)
            logits = outputs.logits[0].cpu().numpy()
            probs = softmax(logits)

            idx = int(np.argmax(probs))
            conf = float(probs[idx])
            stance = STANCE_CLASSES[idx] if idx < len(STANCE_CLASSES) else "neutral"
            return stance, conf

    def detect_conflicts(
        self,
        claim_text: str,
        evidence_list: List[Evidence],
    ) -> ConflictInfo:
        """
        Analyzes evidence passages for genuine evidentiary contradictions.
        
        Args:
            claim_text: The atomic factual claim text.
            evidence_list: Retrieved evidence passages.
            
        Returns:
            ConflictInfo specifying conflict status, supporting passages,
            contradicting passages, and an explanatory summary.
        """
        if not evidence_list:
            return ConflictInfo(
                has_conflict=False,
                supporting=[],
                contradicting=[],
                explanation="No evidence passages available to evaluate conflicts.",
            )

        supporting_passages: List[Evidence] = []
        contradicting_passages: List[Evidence] = []

        for ev in evidence_list:
            # 1. Determine Source Reliability Weight
            st = ev.source_type or "unknown"
            weight, _, _ = self.source_assigner.get_weight(st)

            # 2. Determine Stance (supporting, contradicting, neutral)
            # Check if explicit benchmark stance annotation exists in metadata
            if ev.metadata and "stance" in ev.metadata:
                stance = str(ev.metadata["stance"]).lower()
                conf = 0.90
            else:
                stance, conf = self.classify_passage_stance(claim_text, ev.text)

            # Store computed stance and confidence in evidence metadata
            if ev.metadata is None:
                ev.metadata = {}
            ev.metadata["stance"] = stance
            ev.metadata["stance_confidence"] = round(conf, 4)
            ev.metadata["source_weight"] = round(weight, 4)

            # 3. Filter by Source Weight and Confidence Thresholds
            is_reliable = weight >= self.conflict_min_weight
            is_confident = conf >= self.confidence_threshold

            if is_reliable and is_confident:
                if stance == "supporting":
                    supporting_passages.append(ev)
                elif stance in ("contradicting", "refuting"):
                    contradicting_passages.append(ev)

        # 4. Conflict Condition: Must have BOTH at least 1 reliable supporting AND 1 reliable refuting
        has_conflict = len(supporting_passages) >= 1 and len(contradicting_passages) >= 1

        if has_conflict:
            explanation = (
                f"Contradictory evidence detected: {len(supporting_passages)} authoritative "
                f"supporting passage(s) conflict with {len(contradicting_passages)} authoritative "
                f"refuting passage(s) (minimum source weight >= {self.conflict_min_weight})."
            )
        elif len(supporting_passages) >= 1:
            explanation = f"Consistent evidence: {len(supporting_passages)} reliable passage(s) support the claim with no authoritative contradictions."
        elif len(contradicting_passages) >= 1:
            explanation = f"Consistent evidence: {len(contradicting_passages)} reliable passage(s) refute the claim with no authoritative support."
        else:
            explanation = "Evidence passages do not meet the minimum source reliability threshold to establish or refute the claim definitively."

        return ConflictInfo(
            has_conflict=has_conflict,
            supporting=supporting_passages,
            contradicting=contradicting_passages,
            explanation=explanation,
        )
