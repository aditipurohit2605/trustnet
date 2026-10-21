"""
Security enforcement, authentication, and threat mitigation for TrustNet Stage 10.

Implements:
1. API-key authentication using constant-time comparison (secrets.compare_digest)
   to protect against side-channel timing attacks.
2. In-memory sliding-window rate limiting per client identifier.
3. Startup guards enforcing API key configuration in production mode.
4. Architectural SSRF (Server-Side Request Forgery) protection policy.
"""

from collections import defaultdict
import logging
import os
import secrets
import time
from typing import Any, Dict, List, Optional, Set

from fastapi import HTTPException, Header, Request, status

logger = logging.getLogger(__name__)

# ==============================================================================
# ARCHITECTURAL SECURITY MANDATE: NO ARBITRARY URL DEREFERENCING (SSRF DEFENSE)
# ==============================================================================
# Why TrustNet NEVER fetches URLs found inside user message text:
#
# Incoming user messages frequently contain web URLs (e.g., hyperlinks to articles,
# forwarded messages, or phishing lures). In modern containerized and cloud architectures
# (AWS, GCP, Azure, Kubernetes), allowing the verification server to issue automated
# HTTP GET/POST requests to arbitrary URLs extracted from untrusted user text exposes
# the application to Server-Side Request Forgery (SSRF) attacks.
#
# A malicious actor could submit URLs pointing to:
# 1. Cloud Instance Metadata Services (IMDS):
#    e.g., `http://169.254.169.254/latest/meta-data/` to harvest IAM instance credentials.
# 2. Local Container & Host loopbacks:
#    e.g., `http://127.0.0.1:8000`, `http://localhost:5432`, `http://localhost:6379` to
#    probe internal ports, databases, or local microservices.
# 3. Internal VPC / Intranet subnets:
#    e.g., `http://10.0.0.0/8`, `http://192.168.0.0/16` to map private infrastructure.
#
# Therefore, TrustNet's pipeline verifies factual claims STRICTLY against pre-indexed,
# vetted knowledge corpora (e.g. Wikipedia/FEVER/fact-check databases) and never initiates
# runtime outbound HTTP calls to user-supplied web links.
# ==============================================================================


class InMemoryRateLimiter:
    """
    Sliding-window request rate limiter indexed by client IP or API key.
    Automatically purges expired request timestamps beyond the window duration.
    """

    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.window_seconds = 60.0
        self._history: Dict[str, List[float]] = defaultdict(list)

    def is_allowed(self, client_id: str) -> bool:
        """
        Evaluates whether a request from client_id is within rate limits.
        Returns True if allowed, False if exceeded.
        """
        now = time.time()
        cutoff = now - self.window_seconds

        # Clean history older than 60 seconds
        timestamps = [t for t in self._history[client_id] if t > cutoff]
        self._history[client_id] = timestamps

        if len(timestamps) >= self.requests_per_minute:
            return False

        self._history[client_id].append(now)
        return True

    def reset(self) -> None:
        """Clears all client request tracking history."""
        self._history.clear()


# Global rate limiter instance
rate_limiter = InMemoryRateLimiter()


def verify_production_security(config: Dict[str, Any]) -> None:
    """
    Startup validation guard.
    In 'production' environment, refuses to start unless at least one valid API key
    is populated via the configured environment variable.
    """
    env = os.environ.get("ENVIRONMENT", os.environ.get("ENV", "development")).lower()
    env_var_name = config.get("api_keys_env_var", "TRUSTNET_API_KEYS")
    raw_keys = os.environ.get(env_var_name, "").strip()

    if env == "production":
        valid_keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
        if not valid_keys:
            raise RuntimeError(
                f"FATAL: Production mode detected (ENVIRONMENT={env}), but no API keys were "
                f"found in environment variable '{env_var_name}'. "
                f"The application refuses to start unprotected in production."
            )
        logger.info(f"Production security check passed: loaded {len(valid_keys)} authorized API key(s).")
    else:
        logger.info(f"Running in '{env}' environment mode.")


def get_authorized_keys(config: Dict[str, Any]) -> Set[str]:
    """Reads and parses comma-separated authorized API keys from the configured environment variable."""
    env_var_name = config.get("api_keys_env_var", "TRUSTNET_API_KEYS")
    raw_keys = os.environ.get(env_var_name, "").strip()
    if not raw_keys:
        return set()
    return {k.strip() for k in raw_keys.split(",") if k.strip()}


def authenticate_api_key(
    request: Request,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> Optional[str]:
    """
    Validates API key header against environment secrets using constant-time comparison.
    If no API keys are configured (e.g. dev/test mode without TRUSTNET_API_KEYS),
    access is permitted.
    """
    cfg = getattr(request.app.state, "config", {})
    authorized_keys = get_authorized_keys(cfg)

    # If no keys are configured in development or test, allow access
    if not authorized_keys:
        return x_api_key

    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "UNAUTHORIZED", "message": "Missing required 'X-API-Key' header."},
        )

    # Constant-time comparison across all authorized keys to prevent timing analysis
    is_valid = any(secrets.compare_digest(x_api_key, valid_key) for valid_key in authorized_keys)

    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "UNAUTHORIZED", "message": "Invalid API key provided."},
        )

    return x_api_key


def check_client_rate_limit(request: Request, config: Optional[Dict[str, Any]] = None) -> None:
    """
    Applies sliding-window rate limiting to the incoming request.
    Uses client host IP or API key as identifier.
    """
    cfg = config or getattr(request.app.state, "config", {})
    limit = cfg.get("rate_limit_per_minute", 60)
    rate_limiter.requests_per_minute = limit

    # Identify client by API key if provided, else fallback to client host IP
    client_ip = request.client.host if request.client else "unknown"
    client_id = request.headers.get("X-API-Key", client_ip)

    if not rate_limiter.is_allowed(client_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "code": "RATE_LIMIT_EXCEEDED",
                "message": f"Rate limit of {limit} requests per minute exceeded. Please retry later.",
            },
        )
