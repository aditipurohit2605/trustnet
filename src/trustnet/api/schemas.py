"""
Pydantic schemas and data contracts for TrustNet Stage 10 Backend API.

Reuses existing core domain models (ClaimReport, Verdict, etc.) from `trustnet.schemas`
while defining API-specific request/response contracts, job statuses, and envelopes.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from trustnet.schemas import ClaimReport, SearchCoverage, Verdict


class PipelineVariant(str, Enum):
    """Execution variants supported by the TrustNet verification engine."""
    TEXT_ONLY = "text_only"
    TEXT_RETRIEVAL = "text_retrieval"
    TEXT_IMAGE = "text_image"
    FULL = "full"


class JobStatus(str, Enum):
    """Lifecycle states for background verification jobs."""
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class VerifyOptions(BaseModel):
    """
    User-configurable verification options submitted with a verification request.
    Can be provided as a JSON field in multipart submissions.
    """
    model_config = ConfigDict(populate_by_name=True)

    variant: PipelineVariant = Field(
        default=PipelineVariant.FULL,
        description="Verification variant pipeline to execute (text_only, text_retrieval, text_image, full)."
    )
    language_hint: Optional[str] = Field(
        default=None,
        description="Optional ISO 639-1 code hint (e.g. 'en', 'hi', 'hi-Latn') to bypass automatic language detection."
    )
    enable_reasoning: bool = Field(
        default=True,
        description="Whether to run the Stage 8 multi-factor reasoning layer."
    )
    store_result: bool = Field(
        default=True,
        description="Whether to persist the final verification report in the database."
    )
    store_image: bool = Field(
        default=False,
        description="Whether to retain uploaded images after processing (strictly disallowed if server config prohibits)."
    )


class ReportEnvelope(BaseModel):
    """
    Standard envelope wrapping verification results with provenance, execution timings,
    model version identifiers, and degradation telemetry.
    """
    model_config = ConfigDict(populate_by_name=True)

    report_id: str = Field(description="Unique report identifier (UUID).")
    pipeline_version: str = Field(description="Git commit hash or package release version of the running code.")
    model_versions: Dict[str, str] = Field(
        default_factory=dict,
        description="Dictionary mapping component names (text, retriever, fusion, etc.) to model identifiers or hashes."
    )
    config_hash: str = Field(description="SHA-256 fingerprint of the pipeline configuration active during execution.")
    variant: str = Field(description="Verification pipeline variant that was executed.")
    timings: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-stage latency measurements in seconds (e.g. extraction, retrieval, fusion, reasoning)."
    )
    degraded: List[str] = Field(
        default_factory=list,
        description="List of component names that failed or were skipped gracefully during execution."
    )
    claims: List[ClaimReport] = Field(
        default_factory=list,
        description="List of verified claim reports produced by the pipeline."
    )
    original_text: str = Field(description="Original user-submitted message text.")
    warnings: List[str] = Field(
        default_factory=list,
        description="Operational warnings generated during analysis (e.g. OCR unreadable, source staleness)."
    )
    search_coverage: Optional[SearchCoverage] = Field(
        default=None,
        description="Corpus search execution telemetry across the verification pipeline."
    )


class JobResponse(BaseModel):
    """Status inquiry response for a queued or executing verification job."""
    model_config = ConfigDict(populate_by_name=True)

    job_id: str = Field(description="Unique job identifier.")
    status: JobStatus = Field(description="Current job lifecycle status.")
    stage: Optional[str] = Field(default=None, description="Current executing pipeline stage name.")
    progress: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Progress estimate between 0.0 (queued) and 1.0 (completed)."
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the job was accepted into the queue."
    )
    report: Optional[ReportEnvelope] = Field(
        default=None,
        description="Completed verification report envelope, populated when status == DONE."
    )
    error: Optional[str] = Field(
        default=None,
        description="Error details if the job failed."
    )


class FeedbackIn(BaseModel):
    """
    User feedback submission on a generated claim report.
    Guaranteed: feedback is never automatically used for training or threshold adjustments.
    """
    model_config = ConfigDict(populate_by_name=True)

    report_id: str = Field(description="ID of the report to which this feedback pertains.")
    claim_id: Optional[str] = Field(default=None, description="Optional specific claim ID within the report.")
    suggested_verdict: Optional[str] = Field(
        default=None,
        description="User suggested verdict (e.g. verified, contradicted, insufficient_evidence)."
    )
    comment: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="Free-form user comment explaining the feedback (max 2000 chars)."
    )
    consent_to_store: bool = Field(
        description="Explicit user consent to store feedback. If False, feedback is discarded immediately."
    )


class ErrorResponse(BaseModel):
    """Consistent JSON error response structure returned across all 4xx and 5xx errors."""
    model_config = ConfigDict(populate_by_name=True)

    code: str = Field(description="Machine-readable error classification code (e.g. INVALID_IMAGE, RATE_LIMITED).")
    message: str = Field(description="Human-readable explanation of why the request failed.")
    request_id: Optional[str] = Field(
        default=None,
        description="Unique request tracing ID matching the X-Request-ID response header."
    )
