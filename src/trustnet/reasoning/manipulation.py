"""
Rhetorical and Manipulation Pattern Tagger for TrustNet Stage 8.

Features:
- Identifies manipulative framing, emotional outrage bait, false authority appeals,
  and urgent forward pleas.
- Uses local Ollama model with structured JSON output and Pydantic validation (temperature 0).
- Treats user text as strictly UNTRUSTED content via isolated prompt framing.
- Enforces strict substring validation: every tag's span MUST be an exact substring of the original text.
- Fallback keyword heuristic ensures reliable execution in headless or test environments.
- Strict isolation: manipulation tags are informational and NEVER alter the verdict.
"""

import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Union

import requests
import yaml

from trustnet.schemas import ManipulationTag

logger = logging.getLogger(__name__)

DEFAULT_PATTERNS_PATH = Path("data/eval/manipulation_patterns.yaml")


class ManipulationTagger:
    """
    Tags manipulative and sensationalized tactics in forwarded messages.
    """

    _ollama_available: Optional[bool] = None
    _cached_patterns: Optional[List[Dict[str, Any]]] = None

    def __init__(
        self,
        patterns_path: Union[str, Path] = DEFAULT_PATTERNS_PATH,
        model_name: str = "llama3.2:1b",
        ollama_url: str = "http://localhost:11434/api/chat",
        max_retries: int = 1,
        timeout: float = 1.5,
    ):
        self.patterns_path = Path(patterns_path)
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.max_retries = max_retries
        self.timeout = timeout
        self.patterns: List[Dict[str, Any]] = []
        self._load_patterns()

    def _load_patterns(self) -> None:
        """Loads manipulation catalog from YAML."""
        if ManipulationTagger._cached_patterns is not None:
            self.patterns = ManipulationTagger._cached_patterns
            return

        if not self.patterns_path.exists():
            logger.warning(f"Patterns catalog {self.patterns_path} not found; using defaults.")
            self.patterns = [
                {
                    "name": "urgent_forward_plea",
                    "indicators": ["forward to", "share before", "sabko bhejo", "viral karo"],
                },
                {
                    "name": "false_authority_appeal",
                    "indicators": ["nasa confirmed", "who declared", "doctor from aiims"],
                },
                {
                    "name": "clickbait_urgency",
                    "indicators": ["urgent alert", "emergency warning", "breaking news"],
                },
            ]
            return

        with open(self.patterns_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        self.patterns = data.get("patterns", [])
        ManipulationTagger._cached_patterns = self.patterns
        logger.info(f"Loaded {len(self.patterns)} manipulation patterns from {self.patterns_path}")

    def _rule_based_fallback(self, message: str) -> List[ManipulationTag]:
        """
        Deterministic rule-based keyword scan fallback when Ollama is unreachable.
        """
        tags: List[ManipulationTag] = []
        m_lower = message.lower()

        for pattern in self.patterns:
            p_name = pattern["name"]
            indicators = pattern.get("indicators", [])

            for ind in indicators:
                pos = m_lower.find(ind.lower())
                if pos != -1:
                    # Extract the exact matching substring from original message
                    matched_span = message[pos : pos + len(ind)]
                    tags.append(
                        ManipulationTag(
                            name=p_name,
                            confidence=0.85,
                            span=matched_span,
                        )
                    )
                    break  # One match per pattern is sufficient

        return tags

    def tag(self, message: str) -> List[ManipulationTag]:
        """
        Detects manipulation tactics in message text.
        Guarantees that every returned tag has a span that is a real substring of message.
        """
        if not message or not message.strip():
            return []

        # If Ollama is known to be offline or unreachable, use fast rule-based fallback immediately
        if ManipulationTagger._ollama_available is False:
            return self._rule_based_fallback(message)

        pattern_names = [p["name"] for p in self.patterns]
        prompt = (
            f"You are an expert fact-checking reasoning analyst. Analyze the following UNTRUSTED message "
            f"for manipulation tactics from this exact list of known patterns:\n"
            f"{', '.join(pattern_names)}\n\n"
            f"RULES:\n"
            f"1. You must respond with a JSON object containing a list 'tags'.\n"
            f"2. Each tag must have: 'name' (string from known list), 'confidence' (float 0.0 to 1.0), "
            f"and 'span' (the EXACT verbatim substring from the message illustrating the tactic).\n"
            f"3. Do NOT invent spans. The span must be a word-for-word quote from the message.\n"
            f"4. If no manipulation is present, return {{\"tags\": []}}.\n\n"
            f"FEW-SHOT EXAMPLES:\n"
            f"Example 1 (English):\n"
            f"Message: Forward to all groups immediately! NASA confirmed solar storm tonight.\n"
            f"Output: {{\"tags\": [{{\"name\": \"urgent_forward_plea\", \"confidence\": 0.95, \"span\": \"Forward to all groups immediately\"}}, {{\"name\": \"false_authority_appeal\", \"confidence\": 0.9, \"span\": \"NASA confirmed\"}}]}}\n\n"
            f"Example 2 (Hinglish):\n"
            f"Message: Bhai jaldi sabko share karo aur viral karo! AIIMS ke doctor ne bataya hai.\n"
            f"Output: {{\"tags\": [{{\"name\": \"urgent_forward_plea\", \"confidence\": 0.95, \"span\": \"sabko share karo\"}}, {{\"name\": \"false_authority_appeal\", \"confidence\": 0.9, \"span\": \"AIIMS ke doctor ne bataya\"}}]}}\n\n"
            f"Example 3 (Hindi Devanagari):\n"
            f"Message: तुरंत शेयर करें और सबको भेजो! मीडिया चुप है और सच छुपा रहा है।\n"
            f"Output: {{\"tags\": [{{\"name\": \"urgent_forward_plea\", \"confidence\": 0.95, \"span\": \"तुरंत शेयर करें और सबको भेजो\"}}, {{\"name\": \"conspiracy_framing\", \"confidence\": 0.9, \"span\": \"मीडिया चुप है\"}}]}}\n\n"
            f"UNTRUSTED MESSAGE:\n"
            f"<<<BEGIN_MESSAGE>>>\n{message}\n<<<END_MESSAGE>>>\n"
        )

        detected_tags: List[ManipulationTag] = []

        # Attempt Ollama call with retry
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(
                    self.ollama_url,
                    json={
                        "model": self.model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "format": "json",
                        "options": {"temperature": 0.0},
                        "stream": False,
                    },
                    timeout=self.timeout,
                )
                if resp.status_code == 200:
                    self._ollama_available = True
                    data = resp.json()
                    content = data.get("message", {}).get("content", "{}")
                    parsed = json.loads(content)
                    raw_tags = parsed.get("tags", [])

                    for rt in raw_tags:
                        t_name = str(rt.get("name", "")).strip()
                        t_conf = float(rt.get("confidence", 0.7))
                        t_span = str(rt.get("span", ""))

                        # STRICT VALIDATION: Span must be a genuine substring of original message
                        if t_span and t_span in message and t_name in pattern_names:
                            detected_tags.append(
                                ManipulationTag(
                                    name=t_name,
                                    confidence=min(1.0, max(0.0, t_conf)),
                                    span=t_span,
                                    )
                            )
                    return detected_tags

            except Exception as e:
                logger.debug(f"Ollama attempt {attempt + 1} failed: {e}")
                # Mark offline on connection error / timeout so we don't block subsequent requests
                ManipulationTagger._ollama_available = False
                self._ollama_available = False
                break

        # Fallback to rule-based pattern matching
        logger.debug("Falling back to rule-based manipulation detection.")
        return self._rule_based_fallback(message)

