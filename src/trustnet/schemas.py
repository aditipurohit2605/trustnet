"""
Core schemas and data models for the TrustNet package.
"""

from enum import Enum
import hashlib
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field


class Verdict(str, Enum):
    """
    Standard verdict labels for claim verification.
    Inherits from str for seamless serialization and string comparison.
    """
    VERIFIED = "verified"
    CONTRADICTED = "contradicted"
    INSUFFICIENT = "insufficient_evidence"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"

    @classmethod
    def normalize(cls, val: Union[str, "Verdict"]) -> "Verdict":
        """
        Normalizes various string representations (e.g. FEVER labels, synonyms)
        to the canonical Verdict enum.
        """
        if isinstance(val, cls):
            return val
        cleaned = str(val).strip().lower().replace(" ", "_").replace("-", "_")
        aliases = {
            "verified": cls.VERIFIED,
            "supports": cls.VERIFIED,
            "supported": cls.VERIFIED,
            "true": cls.VERIFIED,
            "contradicted": cls.CONTRADICTED,
            "refutes": cls.CONTRADICTED,
            "refuted": cls.CONTRADICTED,
            "false": cls.CONTRADICTED,
            "insufficient": cls.INSUFFICIENT,
            "insufficient_evidence": cls.INSUFFICIENT,
            "not_enough_info": cls.INSUFFICIENT,
            "not_enough_information": cls.INSUFFICIENT,
            "nei": cls.INSUFFICIENT,
        }
        if cleaned in aliases:
            return aliases[cleaned]
        return cls(cleaned)


class Claim(BaseModel):
    """Represents a factual claim to be verified."""
    model_config = ConfigDict(populate_by_name=True)

    id: str
    text: str


class ClaimType(str, Enum):
    """Classification taxonomy for extracted claims."""
    EVENT = "event"
    STATISTIC = "statistic"
    QUOTE = "quote"
    ATTRIBUTION = "attribution"
    OTHER = "other"


class ExtractedClaim(BaseModel):
    """Represents an atomic factual statement extracted from unformatted text."""
    model_config = ConfigDict(populate_by_name=True)

    text: str
    claim_type: ClaimType
    checkable: bool
    source_span: str
    language: str = "en"

    def to_claim(self) -> Claim:
        """
        Converts a checkable ExtractedClaim into a canonical Claim instance
        with a deterministic, stable ID derived from the SHA-256 hash of its text.
        """
        if not self.checkable:
            raise ValueError(f"Cannot convert non-checkable claim: '{self.text}'")
        normalized = self.text.strip().lower()
        hash_id = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
        return Claim(id=f"claim_{hash_id}", text=self.text.strip())


class SourceMention(BaseModel):
    """Source attribution or witness mention identified in the text."""
    model_config = ConfigDict(populate_by_name=True)

    text: str
    span: str


class ExtractionResult(BaseModel):
    """Structured output produced by the claim extraction pipeline."""
    model_config = ConfigDict(populate_by_name=True)

    original_text: str
    claims: List[ExtractedClaim] = Field(default_factory=list)
    source_mentions: List[SourceMention] = Field(default_factory=list)
    error: Optional[str] = None


class Evidence(BaseModel):
    """Represents an evidence document or snippet supporting/refuting a claim."""
    model_config = ConfigDict(populate_by_name=True)

    doc_id: str
    text: str
    source: Optional[str] = None
    source_type: Optional[str] = None
    score: Optional[float] = None
    passage_id: Optional[str] = None


class ClaimReport(BaseModel):
    """
    Complete verification report for a claim, containing evidence items,
    the final verdict, confidence score, and any noted gaps.
    """
    model_config = ConfigDict(populate_by_name=True)

    claim: Claim
    evidence: List[Evidence] = Field(default_factory=list)
    verdict: Verdict
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence must be between 0 and 1.")
    gaps: List[str] = Field(default_factory=list)


class ClaimExample(BaseModel):
    """
    Processed claim record used in training, validation, and testing splits.
    Includes gold evidence doc IDs when available.
    """
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(..., alias="claim_id")
    claim: str = Field(..., alias="text")
    verdict: Verdict = Field(..., alias="label")
    evidence_doc_ids: List[str] = Field(default_factory=list, alias="gold_evidence_ids")
    metadata: Optional[Dict[str, Any]] = None


class CorpusDocument(BaseModel):
    """A document stored in the verification knowledge corpus."""
    model_config = ConfigDict(populate_by_name=True)

    doc_id: str
    text: str
    title: Optional[str] = None
