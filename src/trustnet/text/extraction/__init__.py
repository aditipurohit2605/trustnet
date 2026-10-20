"""
Claim extraction module for TrustNet.
"""

from trustnet.text.extraction.extractor import ClaimExtractor
from trustnet.text.extraction.prompts import (
    SYSTEM_PROMPT,
    UNTRUSTED_CONTENT_END,
    UNTRUSTED_CONTENT_START,
    build_extraction_prompt,
    build_user_prompt,
)

__all__ = [
    "ClaimExtractor",
    "build_extraction_prompt",
    "build_user_prompt",
    "SYSTEM_PROMPT",
    "UNTRUSTED_CONTENT_START",
    "UNTRUSTED_CONTENT_END",
]
