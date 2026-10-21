"""
Asynchronous Job Management and Queue Processing for TrustNet Stage 10.

Architectural Guarantees:
1. Strict Single-Worker Constraint: Exactly one GPU worker dequeues and processes verification jobs
   sequentially. Running multiple processes or concurrent GPU threads leads to CUDA memory thrashing,
   CUDA OOM errors, and uncontrolled process crashes.
2. Bounded Queue with Backpressure: If the pending queue reaches capacity, requests are immediately
   rejected with HTTP 429 (QUEUE_FULL) to prevent unconstrained RAM exhaustion.
3. Persistent State in SQLite WAL: All state changes (queued -> running -> done/failed) are persisted
   immediately to SQLite so that server restarts and job histories are preserved.
4. Per-Job Execution Timeout: Jobs exceeding `job_timeout_seconds` are cleanly aborted and marked FAILED.
5. Ephemeral Upload Cleanup: Uploaded image temporary files are unconditionally removed upon job completion
   unless persistent storage is explicitly requested and permitted by server configuration.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import threading
from typing import Any, Callable, Dict, Optional
import uuid

from fastapi import HTTPException, status
from sqlalchemy.orm import sessionmaker

from trustnet.api.db import JobRecord, ReportRecord
from trustnet.api.pipeline_runner import PipelineRunner
from trustnet.api.schemas import JobResponse, JobStatus, ReportEnvelope, VerifyOptions

logger = logging.getLogger(__name__)


@dataclass
class JobTask:
    """Internal task item dispatched into the single-worker asyncio queue."""
    job_id: str
    text: str
    image_path: Optional[str]
    options: VerifyOptions
    temp_image_path: Optional[str] = None


class JobManager:
    """
    Manages job ingestion, bounded FIFO queueing, and single-worker execution with timeout safety.
    """

    def __init__(
        self,
        session_factory: sessionmaker,
        runner: PipelineRunner,
        queue_max_size: int = 50,
        job_timeout_seconds: int = 120,
        allow_store_image: bool = False,
    ):
        self.session_factory = session_factory
        self.runner = runner
        self.queue_max_size = queue_max_size
        self.job_timeout_seconds = job_timeout_seconds
        self.allow_store_image = allow_store_image

        self.queue: asyncio.Queue[JobTask] = asyncio.Queue(maxsize=queue_max_size)
        self.worker_task: Optional[asyncio.Task] = None
        self._is_running: bool = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._ready_event = threading.Event()

    def is_full(self) -> bool:
        """Returns True if the pending job queue has reached its maximum configured limit."""
        if self.queue is None:
            return False
        return self.queue.full()

    async def submit_job(
        self,
        text: str,
        image_path: Optional[str] = None,
        options: Optional[VerifyOptions] = None,
        temp_image_path: Optional[str] = None,
    ) -> str:
        """
        Validates queue headroom, records the job in SQLite as QUEUED, and enqueues for processing.
        Raises HTTP 429 if the queue is full.
        """
        if self.is_full():
            # If rejected due to queue full, clean up temp image file immediately
            if temp_image_path and Path(temp_image_path).exists():
                try:
                    Path(temp_image_path).unlink()
                except Exception:
                    pass
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "code": "QUEUE_FULL",
                    "message": f"Verification job queue is full ({self.queue_max_size} pending). Please retry shortly.",
                },
            )

        job_id = f"job_{uuid.uuid4().hex[:16]}"
        opts = options or VerifyOptions()
        variant_str = opts.variant.value if hasattr(opts.variant, "value") else str(opts.variant)

        # Persist new job record in SQLite
        with self.session_factory() as session:
            record = JobRecord(
                job_id=job_id,
                status=JobStatus.QUEUED.value,
                stage="queued",
                progress=0.0,
                variant=variant_str,
                options_json=opts.model_dump_json(),
                created_at=datetime.now(timezone.utc),
            )
            session.add(record)
            session.commit()

        task = JobTask(
            job_id=job_id,
            text=text,
            image_path=image_path,
            options=opts,
            temp_image_path=temp_image_path,
        )

        if not self._is_running or self._worker_thread is None or not self._worker_thread.is_alive():
            self.start()

        if self._loop and self._loop.is_running():
            fut = asyncio.run_coroutine_threadsafe(self.queue.put(task), self._loop)
            fut.result(timeout=5.0)
        else:
            self.queue.put_nowait(task)

        logger.info(f"Accepted verification job '{job_id}' (Queue depth: {self.queue.qsize()}/{self.queue_max_size}).")
        return job_id

    async def _worker_loop(self) -> None:
        """
        Single-worker sequential execution loop.
        Guarantees that exactly one verification pipeline runs at any given instant on the GPU.
        """
        logger.info("Single GPU worker loop started; awaiting verification jobs.")
        while self._is_running:
            try:
                task: JobTask = await self.queue.get()
            except asyncio.CancelledError:
                break

            try:
                await self._process_task(task)
            except Exception as e:
                logger.error(f"Unexpected fatal error processing job '{task.job_id}': {e}", exc_info=True)
            finally:
                self.queue.task_done()

                # Clean up uploaded image file if ephemeral
                if task.temp_image_path and Path(task.temp_image_path).exists():
                    should_retain = task.options.store_image and self.allow_store_image
                    if not should_retain:
                        try:
                            Path(task.temp_image_path).unlink(missing_ok=True)
                            logger.debug(f"Deleted temporary upload image: {task.temp_image_path}")
                        except Exception as e:
                            logger.warning(f"Could not delete temp image '{task.temp_image_path}': {e}")

    async def _process_task(self, task: JobTask) -> None:
        """Executes a single job with DB progress updates and per-job timeout enforcement."""
        job_id = task.job_id
        logger.info(f"Worker picked up job '{job_id}' (Variant: {task.options.variant}).")

        # Mark RUNNING in DB
        with self.session_factory() as session:
            rec = session.query(JobRecord).filter_by(job_id=job_id).first()
            if rec:
                rec.status = JobStatus.RUNNING.value
                rec.stage = "started"
                rec.progress = 0.05
                rec.updated_at = datetime.now(timezone.utc)
                session.commit()

        def _on_progress(stage: str, progress: float) -> None:
            """Synchronous callback writing incremental stage progress to SQLite."""
            try:
                with self.session_factory() as s:
                    r = s.query(JobRecord).filter_by(job_id=job_id).first()
                    if r and r.status == JobStatus.RUNNING.value:
                        r.stage = stage
                        r.progress = round(progress, 2)
                        r.updated_at = datetime.now(timezone.utc)
                        s.commit()
            except Exception as ex:
                logger.debug(f"Progress update write failed: {ex}")

        # Execute CPU/GPU pipeline within thread pool to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        try:
            envelope: ReportEnvelope = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    self.runner.run_job,
                    task.text,
                    task.image_path,
                    task.options,
                    _on_progress,
                ),
                timeout=float(self.job_timeout_seconds),
            )

            # Persist report if requested
            report_id = envelope.report_id
            with self.session_factory() as session:
                if task.options.store_result:
                    report_rec = ReportRecord(
                        report_id=report_id,
                        pipeline_version=envelope.pipeline_version,
                        config_hash=envelope.config_hash,
                        variant=envelope.variant,
                        envelope_json=envelope.model_dump_json(),
                        created_at=datetime.now(timezone.utc),
                    )
                    session.add(report_rec)

                # Update JobRecord to DONE
                rec = session.query(JobRecord).filter_by(job_id=job_id).first()
                if rec:
                    rec.status = JobStatus.DONE.value
                    rec.stage = "done"
                    rec.progress = 1.0
                    rec.report_id = report_id
                    rec.updated_at = datetime.now(timezone.utc)
                session.commit()

            logger.info(f"Verification job '{job_id}' completed successfully [Report ID: {report_id}].")

        except asyncio.TimeoutError:
            logger.error(f"Job '{job_id}' timed out after {self.job_timeout_seconds}s.")
            with self.session_factory() as session:
                rec = session.query(JobRecord).filter_by(job_id=job_id).first()
                if rec:
                    rec.status = JobStatus.FAILED.value
                    rec.stage = "timed_out"
                    rec.error = f"Job execution exceeded timeout limit of {self.job_timeout_seconds} seconds."
                    rec.updated_at = datetime.now(timezone.utc)
                    session.commit()

        except Exception as e:
            logger.error(f"Job '{job_id}' failed with exception: {e}", exc_info=True)
            with self.session_factory() as session:
                rec = session.query(JobRecord).filter_by(job_id=job_id).first()
                if rec:
                    rec.status = JobStatus.FAILED.value
                    rec.stage = "failed"
                    rec.error = str(e)
                    rec.updated_at = datetime.now(timezone.utc)
                    session.commit()

    def _ensure_worker_running(self) -> None:
        """Ensures the background worker thread and loop are active."""
        if not self._is_running or self._worker_thread is None or not self._worker_thread.is_alive():
            self.start()

    def start(self) -> None:
        """
        Spawns the single GPU worker thread with its dedicated persistent asyncio event loop.
        Guarantees that exactly one verification pipeline runs at any given instant.
        """
        if self._is_running and self._worker_thread and self._worker_thread.is_alive():
            return
        self._is_running = True
        self._ready_event.clear()

        def _thread_main():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self.queue = asyncio.Queue(maxsize=self.queue_max_size)
            self.worker_task = self._loop.create_task(self._worker_loop())
            self._ready_event.set()
            try:
                self._loop.run_forever()
            finally:
                self._loop.close()

        self._worker_thread = threading.Thread(
            target=_thread_main,
            name="TrustNet-GPU-Worker",
            daemon=True,
        )
        self._worker_thread.start()
        self._ready_event.wait(timeout=5.0)
        logger.info("Single GPU worker thread and dedicated asyncio loop active.")

    async def process_one_now(self) -> bool:
        """Manually dequeues and processes a single task. Useful for unit testing."""
        if not self.queue.empty():
            task = await self.queue.get()
            try:
                await self._process_task(task)
            finally:
                self.queue.task_done()
            return True
        return False

    async def shutdown(self) -> None:
        """Gracefully halts the worker task and terminates its dedicated event loop."""
        self._is_running = False
        if self._loop and self._loop.is_running():
            if self.worker_task and not self.worker_task.done():
                self._loop.call_soon_threadsafe(self.worker_task.cancel)
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=3.0)
        logger.info("Single GPU worker successfully shut down.")
