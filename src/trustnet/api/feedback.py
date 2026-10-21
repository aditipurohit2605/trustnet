"""
Feedback Ingestion and Consent Management for TrustNet Stage 10.

CORE GOVERNANCE MANDATES:
1. Consent Gating: Feedback records are strictly NEVER written to persistent storage unless
   `consent_to_store` is explicitly True. Submissions with False consent are processed ephemerally
   and immediately discarded.
2. No Autonomous Feedback Loops: Feedback data is NEVER fed into automated model retraining pipelines,
   heuristic weights, or calibration threshold adjustments. Feedback exists solely for asynchronous
   manual audit by human fact-checkers and ML safety researchers.
"""

import logging
from typing import Any, Dict

from fastapi import HTTPException, status
from sqlalchemy.orm import Session, sessionmaker

from trustnet.api.db import FeedbackRecord, ReportRecord
from trustnet.api.schemas import FeedbackIn

logger = logging.getLogger(__name__)


def record_user_feedback(
    feedback: FeedbackIn,
    session_factory: sessionmaker,
) -> Dict[str, Any]:
    """
    Processes human feedback on verification reports adhering strictly to consent constraints.
    """
    # 1. Enforce consent gate
    if not feedback.consent_to_store:
        logger.info(
            f"Feedback for report '{feedback.report_id}' discarded: user did not grant consent_to_store."
        )
        return {
            "stored": False,
            "status": "discarded",
            "message": "Feedback received but discarded because consent_to_store was not granted.",
        }

    # 2. Persist with initial 'unreviewed' status
    with session_factory() as session:
        # Validate that referenced report exists
        report_exists = session.query(ReportRecord).filter_by(report_id=feedback.report_id).first()
        if not report_exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "code": "REPORT_NOT_FOUND",
                    "message": f"Cannot submit feedback: report '{feedback.report_id}' was not found.",
                },
            )

        feedback_entry = FeedbackRecord(
            report_id=feedback.report_id,
            claim_id=feedback.claim_id,
            suggested_verdict=feedback.suggested_verdict,
            comment=feedback.comment,
            consent_to_store=True,
            status="unreviewed",  # Human review queue only; never used for automated model updates
        )
        session.add(feedback_entry)
        session.commit()
        session.refresh(feedback_entry)

        logger.info(
            f"Stored feedback entry #{feedback_entry.id} for report '{feedback.report_id}' [Status: unreviewed]."
        )
        return {
            "stored": True,
            "feedback_id": feedback_entry.id,
            "status": "unreviewed",
            "message": "Feedback successfully recorded for human review.",
        }
