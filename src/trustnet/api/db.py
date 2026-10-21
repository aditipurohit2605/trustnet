"""
Database persistence layer for TrustNet Stage 10.

Configures SQLite in Write-Ahead Logging (WAL) mode with SQLAlchemy,
defining tables for async verification jobs, report envelopes, and user feedback.
Provides lifecycle retention routines and startup crash recovery.
"""

from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    event,
    select,
    delete,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from trustnet.api.schemas import JobResponse, JobStatus, ReportEnvelope

logger = logging.getLogger(__name__)

Base = declarative_base()


class JobRecord(Base):
    """Stores the execution state and lifecycle progression of a verification job."""
    __tablename__ = "jobs"

    job_id = Column(String(64), primary_key=True, index=True)
    status = Column(String(32), nullable=False, default=JobStatus.QUEUED.value, index=True)
    stage = Column(String(64), nullable=True)
    progress = Column(Float, nullable=False, default=0.0)
    variant = Column(String(32), nullable=False, default="full")
    options_json = Column(Text, nullable=True)
    error = Column(Text, nullable=True)
    report_id = Column(String(64), nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc)
    )

    def to_schema(self, report: Optional[ReportEnvelope] = None) -> JobResponse:
        """Serializes DB record to Pydantic JobResponse schema."""
        return JobResponse(
            job_id=self.job_id,
            status=JobStatus(self.status),
            stage=self.stage,
            progress=self.progress,
            created_at=self.created_at,
            report=report,
            error=self.error,
        )


class ReportRecord(Base):
    """Persists completed verification report envelopes alongside metadata for indexing and retrieval."""
    __tablename__ = "reports"

    report_id = Column(String(64), primary_key=True, index=True)
    pipeline_version = Column(String(64), nullable=False)
    config_hash = Column(String(64), nullable=False)
    variant = Column(String(32), nullable=False)
    envelope_json = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)

    def to_schema(self) -> ReportEnvelope:
        """Parses envelope_json back into a Pydantic ReportEnvelope."""
        data = json.loads(self.envelope_json)
        return ReportEnvelope.model_validate(data)


class FeedbackRecord(Base):
    """
    Stores human audit feedback on reports.
    RULE: Never automatically used for model training or autonomous threshold tuning.
    RULE: Only persisted when consent_to_store was explicitly True.
    """
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    report_id = Column(String(64), nullable=False, index=True)
    claim_id = Column(String(64), nullable=True)
    suggested_verdict = Column(String(32), nullable=True)
    comment = Column(Text, nullable=True)
    consent_to_store = Column(Boolean, nullable=False, default=False)
    status = Column(String(32), nullable=False, default="unreviewed")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)


@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """
    Configures SQLite connection settings for high-concurrency production use:
    - WAL mode: allows concurrent readers and a single background writer without blocking.
    - busy_timeout: waits up to 5000ms on lock contention before throwing SQLite busy error.
    - synchronous: NORMAL provides strong durability with reduced disk IOPS overhead in WAL mode.
    """
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA foreign_keys=ON")
    finally:
        cursor.close()


def get_db_engine(db_path: str = "outputs/stage10/trustnet_api.db") -> Engine:
    """Builds and configures SQLAlchemy engine with SQLite WAL pragmas."""
    path_obj = Path(db_path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    sqlite_url = f"sqlite:///{path_obj.resolve()}"
    engine = create_engine(
        sqlite_url,
        connect_args={"check_same_thread": False},
        pool_pre_ping=True,
    )
    return engine


def init_db(engine: Engine) -> None:
    """Initializes schema and tables if they do not already exist."""
    Base.metadata.create_all(bind=engine)
    logger.info("Initialized SQLite database schema with WAL mode.")


def recover_stale_jobs(session: Session) -> int:
    """
    Crash recovery handler run at application startup.
    Finds any jobs that were in 'running' or 'queued' state when the previous server instance crashed,
    and transitions them to 'failed' with a clear diagnostic explanation.
    """
    stale_jobs = session.query(JobRecord).filter(
        JobRecord.status.in_([JobStatus.QUEUED.value, JobStatus.RUNNING.value])
    ).all()

    count = len(stale_jobs)
    for job in stale_jobs:
        job.status = JobStatus.FAILED.value
        job.error = "System restarted while job was executing"
        job.updated_at = datetime.now(timezone.utc)

    if count > 0:
        session.commit()
        logger.warning(f"Crash recovery: marked {count} interrupted jobs as FAILED.")
    return count


def cleanup_old_records(session: Session, retention_days: int = 30) -> Dict[str, int]:
    """
    Purges historical verification reports, jobs, and feedback exceeding the retention window.
    Ensures long-term bounded disk consumption and compliance with data lifecycle requirements.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)

    del_reports = session.query(ReportRecord).filter(ReportRecord.created_at < cutoff).delete()
    del_jobs = session.query(JobRecord).filter(JobRecord.created_at < cutoff).delete()
    del_feedback = session.query(FeedbackRecord).filter(FeedbackRecord.created_at < cutoff).delete()

    session.commit()
    logger.info(
        f"Data retention cleanup: purged {del_reports} reports, "
        f"{del_jobs} jobs, {del_feedback} feedback entries older than {retention_days} days."
    )
    return {
        "deleted_reports": del_reports,
        "deleted_jobs": del_jobs,
        "deleted_feedback": del_feedback,
    }
