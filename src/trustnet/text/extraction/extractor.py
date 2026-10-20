"""
Claim extraction engine using Ollama chat API and Pydantic validation.

Enforces structured JSON generation, automatic retries with graceful error fallback,
and post-processing filters (substring span verification, duplicate dropping, and claim capping).
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import ollama
import yaml

from trustnet.schemas import ExtractionResult
from trustnet.text.extraction.prompts import build_extraction_prompt

logger = logging.getLogger(__name__)


def clean_llm_json_response(raw_text: str) -> str:
    """
    Strips markdown code blocks (e.g. ```json ... ```) or surrounding text
    to extract the raw JSON object string.
    """
    cleaned = raw_text.strip()
    # Match markdown json block if present
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
    if match:
        cleaned = match.group(1).strip()
    return cleaned


class ClaimExtractor:
    """
    Extracts atomic, checkable claims from unformatted text using Ollama LLMs.
    
    Attributes:
        config: Configuration dictionary.
        model_tag: Model name (e.g., 'llama3.2:3b').
        temperature: Generation temperature (default 0.0 for deterministic extraction).
        max_retries: Number of retry attempts on parsing/API failure.
        max_claims_per_message: Maximum number of claims allowed per message.
        timeout_seconds: Request timeout in seconds.
        seed: Random seed for generation reproducibility.
        client: Ollama client instance.
    """

    def __init__(
        self,
        config_or_path: Union[str, Path, Dict[str, Any]] = "configs/stage4_extraction.yaml",
        client: Optional[ollama.Client] = None,
    ):
        if isinstance(config_or_path, (str, Path)):
            cfg_path = Path(config_or_path)
            if not cfg_path.is_file():
                raise FileNotFoundError(f"Extraction config file not found: {cfg_path}")
            with open(cfg_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
        elif isinstance(config_or_path, dict):
            self.config = config_or_path
        else:
            raise TypeError(f"Unsupported config type: {type(config_or_path)}")

        # Configurable model tag - never hardcoded in code
        self.model_tag: str = str(self.config.get("model_tag", "llama3.2:3b"))
        self.temperature: float = float(self.config.get("temperature", 0.0))
        self.max_retries: int = int(self.config.get("max_retries", 2))
        self.max_claims_per_message: int = int(self.config.get("max_claims_per_message", 8))
        self.timeout_seconds: float = float(self.config.get("timeout_seconds", 60.0))
        self.seed: int = int(self.config.get("seed", 42))

        # Initialize Ollama client with configurable timeout
        if client is not None:
            self.client = client
        else:
            self.client = ollama.Client(timeout=self.timeout_seconds)

    def extract(self, text: str) -> ExtractionResult:
        """
        Extracts factual claims from a text string.
        
        Args:
            text: Raw input message to analyze.
            
        Returns:
            ExtractionResult object containing validated claims and source mentions.
            Returns an empty ExtractionResult with error set if retries are exhausted.
        """
        if not text or not text.strip():
            return ExtractionResult(original_text=text, claims=[], source_mentions=[])

        messages = build_extraction_prompt(text)
        last_error: Optional[str] = None
        raw_result: Optional[ExtractionResult] = None

        # Build JSON Schema dictionary to constrain Ollama output format
        json_schema = ExtractionResult.model_json_schema()

        # Retry loop for LLM API calls and JSON schema validation
        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(f"Calling Ollama chat API (attempt {attempt + 1}/{self.max_retries + 1}, model={self.model_tag})...")
                # Parameters used with Ollama Python API:
                # - model: specifies the LLM tag from config
                # - messages: system prompt and user prompt wrapped with untrusted content delimiters
                # - format: Pydantic model JSON schema dict enforcing strict structured JSON output
                # - options: generation hyperparameters (temperature=0.0, seed=42)
                # - stream: False to receive complete response
                response = self.client.chat(
                    model=self.model_tag,
                    messages=messages,
                    format=json_schema,
                    options={
                        "temperature": self.temperature,
                        "seed": self.seed,
                    },
                    stream=False,
                )

                raw_content = response.message.content or ""
                cleaned_json_str = clean_llm_json_response(raw_content)

                parsed_dict = json.loads(cleaned_json_str)
                # Ensure original_text is preserved even if omitted in LLM JSON
                if "original_text" not in parsed_dict or not parsed_dict["original_text"]:
                    parsed_dict["original_text"] = text

                # Validate against Pydantic schema
                raw_result = ExtractionResult.model_validate(parsed_dict)
                break  # Successful extraction

            except Exception as e:
                last_error = f"{type(e).__name__}: {str(e)}"
                logger.warning(f"Extraction attempt {attempt + 1} failed: {last_error}")
                if attempt < self.max_retries:
                    logger.info("Retrying extraction...")

        # If all retries failed, return an empty result with error flag instead of crashing
        if raw_result is None:
            logger.error(f"Extraction exhausted all {self.max_retries + 1} attempts. Returning empty result with error.")
            return ExtractionResult(
                original_text=text,
                claims=[],
                source_mentions=[],
                error=last_error or "Unknown extraction failure",
            )

        # Apply post-processing pipeline
        processed_result = self._post_process(text, raw_result)
        return processed_result

    def _post_process(self, original_text: str, result: ExtractionResult) -> ExtractionResult:
        """
        Applies sanity checks and filters to raw extracted claims:
        1. Drops any claim whose source_span is not an exact substring of the original message.
        2. Drops duplicate claims (case-insensitive normalized comparison).
        3. Caps the total number of claims at max_claims_per_message.
        """
        valid_claims = []
        dropped_span_count = 0
        dropped_dup_count = 0
        seen_claim_texts = set()

        for c in result.claims:
            # 1. Substring Span Check: source_span must be an exact substring of original text
            if not c.source_span or c.source_span not in original_text:
                dropped_span_count += 1
                logger.warning(
                    f"Dropped hallucinated source span: '{c.source_span}' "
                    f"(claim: '{c.text}') is not a substring of original message."
                )
                continue

            # 2. Duplicate Detection: case-insensitive match on normalized text
            norm_claim_text = c.text.strip().lower()
            if norm_claim_text in seen_claim_texts:
                dropped_dup_count += 1
                logger.debug(f"Dropped duplicate claim: '{c.text}'")
                continue
            seen_claim_texts.add(norm_claim_text)

            valid_claims.append(c)

        if dropped_span_count > 0:
            logger.info(f"Total claims dropped due to invalid source_span: {dropped_span_count}")
        if dropped_dup_count > 0:
            logger.info(f"Total duplicate claims dropped: {dropped_dup_count}")

        # 3. Cap claim count at configured limit
        if len(valid_claims) > self.max_claims_per_message:
            logger.info(
                f"Capping claims from {len(valid_claims)} to {self.max_claims_per_message} "
                f"(max_claims_per_message limit)."
            )
            valid_claims = valid_claims[:self.max_claims_per_message]

        # Filter source mentions: spans must also be valid substrings
        valid_mentions = []
        for sm in result.source_mentions:
            if sm.span and sm.span in original_text:
                valid_mentions.append(sm)
            else:
                logger.warning(f"Dropped source mention with invalid span: '{sm.span}'")

        return ExtractionResult(
            original_text=original_text,
            claims=valid_claims,
            source_mentions=valid_mentions,
            error=result.error,
        )
