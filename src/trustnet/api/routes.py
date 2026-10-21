"""
FastAPI Route Handlers and Input Sanitization for TrustNet Stage 10.

Endpoints:
- POST /v1/verify: Asynchronous verification ingestion (multipart text + optional image)
- GET /v1/jobs/{job_id}: Job execution status and progress polling
- GET /v1/reports/{report_id}: Completed ReportEnvelope retrieval
- DELETE /v1/reports/{report_id}: Report deletion and data removal
- POST /v1/feedback: User audit feedback ingestion with consent verification
- GET /v1/health: Readiness probe and component health status
- GET /v1/memory: Real-time GPU and host memory telemetry
"""

import asyncio
import io
import json
import logging
from pathlib import Path
import tempfile
from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, Request, Response, UploadFile, status
from PIL import Image

from trustnet.api.db import JobRecord, ReportRecord
from trustnet.api.feedback import record_user_feedback
from trustnet.api.schemas import (
    ErrorResponse,
    FeedbackIn,
    JobResponse,
    JobStatus,
    ReportEnvelope,
    VerifyOptions,
)
from trustnet.api.security import authenticate_api_key, check_client_rate_limit

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1")


def sniff_image_type(data: bytes) -> Optional[str]:
    """
    Determines image format directly from file magic bytes rather than trusting file extensions.
    Guards against extension spoofing (e.g. arbitrary text files named with a .png extension).
    """
    if len(data) >= 3 and data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if len(data) >= 8 and data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


@router.post(
    "/verify",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
    },
)
async def submit_verification(
    request: Request,
    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    options: Optional[str] = Form(None),
    _auth: Optional[str] = Depends(authenticate_api_key),
):
    """
    Ingests verification request, sanitizes inputs, enforces security constraints,
    and enqueues the job into the single-worker pipeline queue.
    """
    check_client_rate_limit(request)
    cfg: Dict[str, Any] = getattr(request.app.state, "config", {})
    job_manager = getattr(request.app.state, "job_manager", None)

    # 1. Validate that at least one modality (text or image) is provided
    has_text = text is not None and bool(text.strip())
    has_image = image is not None and image.filename is not None and bool(image.filename.strip())

    if not has_text and not has_image:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "EMPTY_INPUT",
                "message": "Verification request must contain either non-empty text or an image payload.",
            },
        )

    # 2. Text length limit validation
    clean_text = text.strip() if text else ""
    max_text_chars = cfg.get("max_text_chars", 10000)
    if len(clean_text) > max_text_chars:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "TEXT_TOO_LONG",
                "message": f"Input text length ({len(clean_text)} chars) exceeds maximum limit of {max_text_chars}.",
            },
        )

    # 3. Image validation and sniffing
    temp_image_path: Optional[str] = None
    if has_image:
        max_bytes = cfg.get("max_image_bytes", 10485760)  # 10 MB default
        image_bytes = await image.read()

        if len(image_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "EMPTY_IMAGE", "message": "Uploaded image file is empty (0 bytes)."},
            )

        if len(image_bytes) > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "code": "IMAGE_TOO_LARGE",
                    "message": f"Image size ({len(image_bytes)} bytes) exceeds maximum limit of {max_bytes} bytes.",
                },
            )

        # Magic byte content-type sniffing (never trust filename extension or client Content-Type header)
        detected_mime = sniff_image_type(image_bytes)
        allowed_mimes = cfg.get("allowed_image_types", ["image/jpeg", "image/png", "image/webp"])

        if detected_mime is None or detected_mime not in allowed_mimes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "code": "INVALID_IMAGE_FORMAT",
                    "message": (
                        f"Unsupported or invalid image payload. Magic byte inspection detected "
                        f"'{detected_mime or 'unknown'}'. Allowed formats: {allowed_mimes}."
                    ),
                },
            )

        # Decompression bomb guard using PIL header inspection without loading pixel data
        max_pixels = cfg.get("max_image_pixels", 25000000)  # 25 MP default
        try:
            with Image.open(io.BytesIO(image_bytes)) as pil_img:
                width, height = pil_img.size
                total_pixels = width * height
                if total_pixels > max_pixels:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail={
                            "code": "DECOMPRESSION_BOMB",
                            "message": (
                                f"Image dimensions {width}x{height} ({total_pixels} pixels) exceed "
                                f"safety limit of {max_pixels} pixels."
                            ),
                        },
                    )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "CORRUPTED_IMAGE", "message": f"Failed to parse image headers: {e}"},
            )

        # Write to ephemeral temporary file on disk for worker processing
        ext = ".jpg" if detected_mime == "image/jpeg" else (".png" if detected_mime == "image/png" else ".webp")
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tf:
            tf.write(image_bytes)
            temp_image_path = tf.name

    # 4. Parse submission options
    verify_opts = VerifyOptions()
    if options:
        try:
            opts_dict = json.loads(options) if isinstance(options, str) else options
            verify_opts = VerifyOptions.model_validate(opts_dict)
        except Exception as e:
            if temp_image_path and Path(temp_image_path).exists():
                Path(temp_image_path).unlink(missing_ok=True)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "INVALID_OPTIONS", "message": f"Invalid verification options JSON: {e}"},
            )

    # 5. Enqueue job into bounded queue
    job_id = await job_manager.submit_job(
        text=clean_text,
        image_path=temp_image_path,
        options=verify_opts,
        temp_image_path=temp_image_path,
    )

    return JobResponse(
        job_id=job_id,
        status=JobStatus.QUEUED,
        stage="queued",
        progress=0.0,
    )


@router.get(
    "/jobs/{job_id}",
    response_model=JobResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_job_status(
    job_id: str,
    request: Request,
    _auth: Optional[str] = Depends(authenticate_api_key),
):
    """Retrieves current execution status, progress, and completed report if available."""
    await asyncio.sleep(0)
    session_factory = getattr(request.app.state, "session_factory", None)
    with session_factory() as session:
        rec = session.query(JobRecord).filter_by(job_id=job_id).first()
        if not rec:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "JOB_NOT_FOUND", "message": f"Verification job '{job_id}' was not found."},
            )

        report_envelope: Optional[ReportEnvelope] = None
        if rec.status == JobStatus.DONE.value and rec.report_id:
            rep_rec = session.query(ReportRecord).filter_by(report_id=rec.report_id).first()
            if rep_rec:
                report_envelope = rep_rec.to_schema()

        return rec.to_schema(report=report_envelope)


@router.get(
    "/reports/{report_id}",
    response_model=ReportEnvelope,
    responses={404: {"model": ErrorResponse}},
)
def get_report(
    report_id: str,
    request: Request,
    _auth: Optional[str] = Depends(authenticate_api_key),
):
    """Retrieves full verification report envelope including evidence, timings, and reasoning notes."""
    session_factory = getattr(request.app.state, "session_factory", None)
    with session_factory() as session:
        rep_rec = session.query(ReportRecord).filter_by(report_id=report_id).first()
        if not rep_rec:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "REPORT_NOT_FOUND", "message": f"Verification report '{report_id}' was not found."},
            )
        return rep_rec.to_schema()


@router.delete(
    "/reports/{report_id}",
    status_code=status.HTTP_200_OK,
    responses={404: {"model": ErrorResponse}},
)
def delete_report(
    report_id: str,
    request: Request,
    _auth: Optional[str] = Depends(authenticate_api_key),
):
    """Deletes stored verification report and cleans up any linked database records."""
    session_factory = getattr(request.app.state, "session_factory", None)
    with session_factory() as session:
        rep_rec = session.query(ReportRecord).filter_by(report_id=report_id).first()
        if not rep_rec:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "REPORT_NOT_FOUND", "message": f"Verification report '{report_id}' was not found."},
            )

        session.delete(rep_rec)
        session.commit()
        logger.info(f"Report '{report_id}' permanently deleted from database.")
        return {"deleted": True, "report_id": report_id}


@router.post(
    "/feedback",
    status_code=status.HTTP_200_OK,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
def submit_feedback(
    request: Request,
    feedback: FeedbackIn,
    _auth: Optional[str] = Depends(authenticate_api_key),
):
    """Submits user audit feedback on a generated report with mandatory consent verification."""
    session_factory = getattr(request.app.state, "session_factory", None)
    return record_user_feedback(feedback, session_factory)


@router.get("/health")
def health_check(request: Request):
    """System liveness and readiness probe for Docker / Kubernetes health checking."""
    registry = getattr(request.app.state, "registry", None)
    readiness_report = registry.readiness() if registry else {"ready": False}
    return {
        "status": "ok",
        "service": "trustnet-api",
        "readiness": readiness_report,
    }


@router.get("/memory")
def get_memory_stats(
    request: Request,
    _auth: Optional[str] = Depends(authenticate_api_key),
):
    """Returns granular per-model and global GPU/CPU memory metrics."""
    registry = getattr(request.app.state, "registry", None)
    if not registry:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "REGISTRY_NOT_READY", "message": "ModelRegistry is not yet initialized."},
        )
    return registry.memory_report()
