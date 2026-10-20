"""
Core schemas and data models for the TrustNet package.
"""

from enum import Enum
import hashlib
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator


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


class LanguageInfo(BaseModel):
    """Detected language and script metadata for multilingual text processing."""
    model_config = ConfigDict(populate_by_name=True)

    code: str  # e.g., 'en', 'hi', 'hi-Latn'
    script: str  # e.g., 'Devanagari', 'Latin', 'mixed', 'unknown'
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    is_code_mixed: bool = False


class NormalizedText(BaseModel):
    """Structured record tracking text normalization, transliteration, and translation."""
    model_config = ConfigDict(populate_by_name=True)

    original: str
    normalized: str
    translated_en: Optional[str] = None
    steps: List[str] = Field(default_factory=list)


class Claim(BaseModel):
    """Represents a factual claim to be verified."""
    model_config = ConfigDict(populate_by_name=True)

    id: str
    text: str
    language_info: Optional[LanguageInfo] = None
    original_text: Optional[str] = None
    normalized_text: Optional[NormalizedText] = None


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
    metadata: Dict[str, Any] = Field(default_factory=dict)


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


class ShareAdvice(str, Enum):
    """Actionable sharing advice derived from reasoning, calibration, and conflict signals."""
    SAFE_TO_SHARE = "safe_to_share"
    SHARE_WITH_CAUTION = "share_with_caution"
    DO_NOT_SHARE_AS_CONFIRMED = "do_not_share_as_confirmed"


class SourceWeight(BaseModel):
    """Reliability weighting attribution for an evidence source type."""
    model_config = ConfigDict(populate_by_name=True)

    source_type: str
    weight: float
    rationale: str = ""


class ConflictInfo(BaseModel):
    """Detection of contradictory or conflicting evidence across authoritative sources."""
    model_config = ConfigDict(populate_by_name=True)

    has_conflict: bool = False
    supporting: List["Evidence"] = Field(default_factory=list)
    contradicting: List["Evidence"] = Field(default_factory=list)
    explanation: str = ""


class TemporalInfo(BaseModel):
    """Temporal freshness analysis and staleness flags comparing claim to evidence."""
    model_config = ConfigDict(populate_by_name=True)

    claim_time_expressions: List[str] = Field(default_factory=list)
    newest_evidence_date: Optional[str] = None
    oldest_evidence_date: Optional[str] = None
    is_stale: bool = False
    explanation: str = ""


class ManipulationTag(BaseModel):
    """Tagging of psychological, rhetorical, or manipulative tactics in message text."""
    model_config = ConfigDict(populate_by_name=True)

    name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    span: str


class Gap(BaseModel):
    """Specific verification gap identified in the evidence base."""
    model_config = ConfigDict(populate_by_name=True)

    code: str
    description: str
    how_to_check: str


class Counterfactual(BaseModel):
    """Hypothetical input perturbation scenario testing model decision sensitivity."""
    model_config = ConfigDict(populate_by_name=True)

    change_description: str
    original_verdict: Union[Verdict, str]
    new_verdict: Union[Verdict, str]
    flipped: bool


class Evidence(BaseModel):
    """Represents an evidence document or snippet supporting/refuting a claim."""
    model_config = ConfigDict(populate_by_name=True)

    doc_id: str
    text: str
    source: Optional[str] = None
    source_type: Optional[str] = None
    score: Optional[float] = None
    passage_id: Optional[str] = None
    date: Optional[str] = None
    url: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


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
    abstained: bool = Field(default=False, description="Flag indicating if the system abstained from making a definitive verdict.")
    raw_probabilities: Optional[Dict[str, float]] = Field(default=None, description="Uncalibrated model output probability distribution.")
    calibrated_probabilities: Optional[Dict[str, float]] = Field(default=None, description="Calibrated probability distribution for dashboard and reporting.")
    abstention_reason: Optional[str] = Field(default=None, description="Explanation if system abstains.")
    gaps: List[Gap] = Field(default_factory=list)

    # Stage 8 Reasoning Layer Extensions
    source_weights: Optional[List[SourceWeight]] = None
    conflict: Optional[ConflictInfo] = None
    temporal: Optional[TemporalInfo] = None
    manipulation: Optional[List[ManipulationTag]] = None
    counterfactuals: Optional[List[Counterfactual]] = None
    share_advice: Optional[ShareAdvice] = None
    reasoning_notes: List[str] = Field(default_factory=list)

    # Stage 9 Multilingual Extensions
    language_info: Optional[LanguageInfo] = None
    original_text: Optional[str] = None
    normalized_text: Optional[NormalizedText] = None

    @field_validator("gaps", mode="before")
    @classmethod
    def migrate_gaps(cls, v: Any) -> Any:
        """
        Migrates legacy list[str] gap entries to list[Gap] records,
        preserving backward compatibility.
        """
        if isinstance(v, list):
            migrated = []
            for item in v:
                if isinstance(item, str):
                    migrated.append(Gap(code="general_note", description=item, how_to_check="Verify claim evidence directly."))
                elif isinstance(item, dict):
                    migrated.append(Gap(**item))
                elif isinstance(item, Gap):
                    migrated.append(item)
                else:
                    migrated.append(Gap(code="note", description=str(item), how_to_check="Verify claim evidence directly."))
            return migrated
        return v


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


class ReuseMatch(BaseModel):
    """Represents a matched duplicate or near-duplicate image from the image bank."""
    model_config = ConfigDict(populate_by_name=True)

    bank_id: str
    phash_distance: int
    embedding_similarity: float
    known_context: Optional[str] = None
    source: Optional[str] = None


class ImageSignals(BaseModel):
    """
    Extracted visual and provenance signals for an image.
    All fields are optional with defaults to ensure partial or graceful degradation.
    """
    model_config = ConfigDict(populate_by_name=True)

    ocr_text: Optional[str] = None
    ocr_confidence: Optional[float] = None
    ocr_language: Optional[str] = None
    caption: Optional[str] = None
    clip_similarity: Optional[float] = None
    mismatch_score: Optional[float] = None  # 0 to 1, higher = more mismatched
    reuse_matches: List[ReuseMatch] = Field(default_factory=list)
    exif_summary: Dict[str, Any] = Field(default_factory=dict)
    quality_flags: List[str] = Field(default_factory=list)

