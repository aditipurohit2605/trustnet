"""
FastAPI Application Entrypoint and Lifecycle Orchestration for TrustNet Stage 10.

Features:
1. Lifespan context manager: builds the ModelRegistry once, runs startup crash recovery,
   executes retention cleanup, and starts the dedicated single GPU worker.
2. Request ID tracing middleware: guarantees X-Request-ID propagation across all calls.
3. High-resolution request latency profiling middleware (X-Process-Time-Ms).
4. Privacy-Preserving Structured JSON Logger:
   CRITICAL PRIVACY MANDATE: Logs ONLY HTTP metadata (method, route, status, duration, request ID).
   Message text, image binary payloads, form inputs, and EXIF summaries are NEVER logged.
5. Standardized JSON Error Handler mapping all HTTP errors to ErrorResponse schema.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, Optional, Union
import uuid

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import sessionmaker
import yaml

from trustnet.api.db import (
    cleanup_old_records,
    get_db_engine,
    init_db,
    recover_stale_jobs,
)
from trustnet.api.jobs import JobManager
from trustnet.api.pipeline_runner import PipelineRunner
from trustnet.api.registry import ModelRegistry
from trustnet.api.routes import router as api_router
from trustnet.api.schemas import ErrorResponse
from trustnet.api.security import verify_production_security

# Configure standard logger to output structured JSON
logger = logging.getLogger("trustnet.api")
logging.basicConfig(level=logging.INFO, format="%(message)s")


def load_api_config(config_path: Union[str, Path] = "configs/stage10_api.yaml") -> Dict[str, Any]:
    """Loads backend configuration from YAML file with fallback defaults."""
    p = Path(config_path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    logger.warning(f"Config file '{config_path}' not found; using built-in defaults.")
    return {
        "host": "0.0.0.0",
        "port": 8000,
        "max_text_chars": 10000,
        "max_image_bytes": 10485760,
        "max_image_pixels": 25000000,
        "job_timeout_seconds": 120,
        "queue_max_size": 50,
        "retention_days": 30,
        "rate_limit_per_minute": 60,
        "api_keys_env_var": "TRUSTNET_API_KEYS",
        "db_path": "outputs/stage10/trustnet_api.db",
        "lazy_models": ["translator", "captioner"],
        "lazy_idle_timeout_seconds": 60,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle manager.
    Coordinates single-load model initialization, DB schema migration, crash recovery,
    and single GPU worker startup.
    """
    config = getattr(app.state, "config", None) or load_api_config()
    app.state.config = config

    # 1. Enforce production mode security constraints
    verify_production_security(config)

    # 2. Initialize Database and run WAL startup recovery if not injected
    if not hasattr(app.state, "session_factory") or app.state.session_factory is None:
        db_path = config.get("db_path", "outputs/stage10/trustnet_api.db")
        engine = get_db_engine(db_path)
        init_db(engine)
        session_factory = sessionmaker(bind=engine)
        app.state.db_engine = engine
        app.state.session_factory = session_factory
    else:
        session_factory = app.state.session_factory

    # Crash recovery: mark jobs interrupted by previous server crashes as failed
    with session_factory() as session:
        recover_stale_jobs(session)
        # Daily/startup retention cleanup
        retention_days = config.get("retention_days", 30)
        cleanup_old_records(session, retention_days=retention_days)

    # 3. Build ModelRegistry once if not injected
    if not hasattr(app.state, "registry") or app.state.registry is None:
        lazy_timeout = config.get("lazy_idle_timeout_seconds", 60)
        lazy_models = config.get("lazy_models", ["translator", "captioner"])
        registry = ModelRegistry(
            idle_timeout_seconds=lazy_timeout,
            lazy_model_names=lazy_models,
        )
        registry.load_core_models()
        registry.start_idle_timer()
        app.state.registry = registry

    # 4. Initialize Pipeline Runner and Single-Worker Job Queue if not injected
    if not hasattr(app.state, "runner") or app.state.runner is None:
        runner = PipelineRunner(registry=app.state.registry, config=config)
        app.state.runner = runner

    if not hasattr(app.state, "job_manager") or app.state.job_manager is None:
        job_manager = JobManager(
            session_factory=session_factory,
            runner=app.state.runner,
            queue_max_size=config.get("queue_max_size", 50),
            job_timeout_seconds=config.get("job_timeout_seconds", 120),
            allow_store_image=config.get("allow_store_image", False),
        )
        job_manager.start()
        app.state.job_manager = job_manager

    logger.info("TrustNet API service fully operational and ready to serve traffic.")

    yield

    # Shutdown sequence
    logger.info("Shutting down TrustNet API service...")
    if hasattr(app.state, "job_manager") and app.state.job_manager:
        await app.state.job_manager.shutdown()
    if hasattr(app.state, "registry") and app.state.registry:
        app.state.registry.stop_idle_timer()
    if hasattr(app.state, "db_engine") and app.state.db_engine:
        app.state.db_engine.dispose()
    logger.info("TrustNet API shutdown completed cleanly.")


def create_app(config_path: Union[str, Path] = "configs/stage10_api.yaml") -> FastAPI:
    """Builds and configures the FastAPI application instance."""
    config = load_api_config(config_path)

    app = FastAPI(
        title="TrustNet Verification API",
        description=(
            "Production-grade multimodal factual claim verification API. "
            "Features asynchronous queueing, probability calibration, selective abstention, "
            "and multi-factor reasoning."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.config = config

    # CORS configuration
    cors_origins = config.get("cors_origins", ["*"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Tracing, timing, and privacy-preserving structured logging middleware
    @app.middleware("http")
    async def tracing_and_logging_middleware(request: Request, call_next):
        # 1. Trace ID generation or propagation
        req_id = request.headers.get("X-Request-ID") or f"req_{uuid.uuid4().hex[:16]}"
        request.state.request_id = req_id

        # 2. Timing
        start_time = time.perf_counter()

        # 3. Process downstream
        response: Response = await call_next(request)

        # 4. Latency calculation
        process_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
        response.headers["X-Request-ID"] = req_id
        response.headers["X-Process-Time-Ms"] = str(process_time_ms)

        # 5. PRIVACY MANDATE: Structured log record with ZERO message or binary content
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": req_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": process_time_ms,
            "client_ip": request.client.host if request.client else "unknown",
        }
        logger.info(json.dumps(log_entry))

        return response

    # Standardized Exception Handlers mapping to ErrorResponse schema
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        req_id = getattr(request.state, "request_id", None)
        detail = exc.detail
        if isinstance(detail, dict):
            code = detail.get("code", "HTTP_ERROR")
            message = detail.get("message", str(detail))
        else:
            code = "HTTP_ERROR"
            message = str(detail)

        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(code=code, message=message, request_id=req_id).model_dump(),
            headers={"X-Request-ID": req_id} if req_id else {},
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        req_id = getattr(request.state, "request_id", None)
        logger.error(f"Unhandled exception on request {req_id}: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                code="INTERNAL_SERVER_ERROR",
                message="An unexpected server error occurred during processing.",
                request_id=req_id,
            ).model_dump(),
            headers={"X-Request-ID": req_id} if req_id else {},
        )

    # Attach router
    app.include_router(api_router)

    return app


# Default application object for ASGI servers (e.g. uvicorn src.trustnet.api.app:app)
app = create_app()
