"""
Source Reliability Weighting Engine for TrustNet Stage 8.

Loads hierarchical source type weights from configuration, assigns calibrated reliability
weights to retrieved Evidence passages, flags unknown source categories, and computes
weighted retrieval statistics.

Supports an ablation toggle (enable_weighting=False) that sets all source weights to 1.0.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml

from trustnet.schemas import Evidence, SourceWeight

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("configs/source_reliability.yaml")


class SourceReliabilityAssigner:
    """
    Assigns provenance reliability weights to evidence passages based on source type.
    """

    def __init__(
        self,
        config_path: Union[str, Path] = DEFAULT_CONFIG_PATH,
        enable_weighting: bool = True,
    ):
        self.config_path = Path(config_path)
        self.enable_weighting = enable_weighting
        self.source_rules: Dict[str, Dict[str, Any]] = {}
        self.default_weight: float = 0.20
        self.unknown_rationale: str = "Unverified source type; default baseline weight applied."
        self._load_config()

    def _load_config(self) -> None:
        """Loads source reliability rules from YAML config file."""
        if not self.config_path.exists():
            logger.warning(
                f"Source reliability config '{self.config_path}' not found; using fallback defaults."
            )
            self.source_rules = {
                "government": {"weight": 1.0, "rationale": "Official government and statutory portal."},
                "wire": {"weight": 0.95, "rationale": "Major accredited news agency / syndicate."},
                "academic": {"weight": 0.90, "rationale": "Peer-reviewed scientific publication."},
                "news": {"weight": 0.75, "rationale": "General digital reporting outlet."},
                "corpus_passage": {"weight": 0.60, "rationale": "Pre-indexed knowledge base document."},
                "social_media": {"weight": 0.30, "rationale": "Unverified social media post."},
            }
            return

        with open(self.config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        self.default_weight = float(data.get("default_weight", 0.20))
        self.unknown_rationale = data.get(
            "unknown_rationale",
            "Unverified source type; assigned conservative baseline weight."
        )
        self.source_rules = data.get("sources", {})
        logger.info(
            f"Loaded {len(self.source_rules)} source reliability rules from {self.config_path}"
        )

    def set_weighting(self, enabled: bool) -> None:
        """Ablation switch: toggles whether empirical source weighting is applied."""
        self.enable_weighting = enabled

    def get_weight(self, source_type: Optional[str]) -> Tuple[float, str, bool]:
        """
        Determines weight, rationale, and unknown flag for a given source_type.
        
        Returns:
            Tuple of (weight: float, rationale: str, is_unknown: bool)
        """
        if not self.enable_weighting:
            return 1.0, "Weighting disabled (ablation mode).", False

        st_clean = (source_type or "").strip().lower()
        if not st_clean:
            return self.default_weight, "Missing source_type; assigned default baseline weight.", True

        if st_clean in self.source_rules:
            rule = self.source_rules[st_clean]
            return float(rule.get("weight", self.default_weight)), rule.get("rationale", ""), False

        return self.default_weight, self.unknown_rationale, True

    def assign_weights(
        self,
        evidence_list: List[Evidence],
    ) -> Tuple[List[SourceWeight], Dict[str, Any]]:
        """
        Assigns source weights to a list of Evidence items and computes a summary.
        
        Args:
            evidence_list: List of Evidence objects.
            
        Returns:
            Tuple of:
            - List of SourceWeight objects corresponding to each evidence item.
            - Summary dictionary containing weighted_mean_score, unknown_sources, etc.
        """
        source_weights: List[SourceWeight] = []
        unknown_sources: List[str] = []
        scores: List[float] = []
        weights: List[float] = []

        for ev in evidence_list:
            st = ev.source_type or "unknown"
            weight, rationale, is_unknown = self.get_weight(st)

            # Record on evidence metadata for downstream consumers
            if ev.metadata is None:
                ev.metadata = {}
            ev.metadata["source_weight"] = weight

            sw = SourceWeight(
                source_type=st,
                weight=round(weight, 4),
                rationale=rationale,
            )
            source_weights.append(sw)

            if is_unknown and st not in unknown_sources:
                unknown_sources.append(st)

            scores.append(float(ev.score or 0.0))
            weights.append(weight)

        # Compute weighted summary statistics
        if weights and sum(weights) > 0:
            weighted_mean_score = float(np.average(scores, weights=weights))
            mean_weight = float(np.mean(weights))
        elif scores:
            weighted_mean_score = float(np.mean(scores))
            mean_weight = 1.0
        else:
            weighted_mean_score = 0.0
            mean_weight = 0.0

        summary = {
            "total_evidence": len(evidence_list),
            "weighting_enabled": self.enable_weighting,
            "mean_source_weight": round(mean_weight, 4),
            "average_weight": round(mean_weight, 4),
            "weighted_mean_retrieval_score": round(weighted_mean_score, 4),
            "unknown_sources": unknown_sources,
        }

        return source_weights, summary
