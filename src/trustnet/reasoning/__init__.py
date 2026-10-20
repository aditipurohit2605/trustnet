"""
TrustNet Stage 8 Reasoning Layer:
- Source Reliability Weighting
- Conflict Detection
- Temporal Freshness & Staleness Analysis
- Manipulation & Rhetorical Tactics Tagging
- Evidence Gap Analysis
- Counterfactual Generation
- Actionable Sharing Advice
"""

from trustnet.reasoning.conflict import ConflictDetector
from trustnet.reasoning.counterfactual import CounterfactualGenerator
from trustnet.reasoning.gaps import GapAnalyzer
from trustnet.reasoning.manipulation import ManipulationTagger
from trustnet.reasoning.share_advice import determine_share_advice
from trustnet.reasoning.source_reliability import SourceReliabilityAssigner
from trustnet.reasoning.temporal import TemporalAnalyzer

__all__ = [
    "SourceReliabilityAssigner",
    "ConflictDetector",
    "TemporalAnalyzer",
    "ManipulationTagger",
    "GapAnalyzer",
    "CounterfactualGenerator",
    "determine_share_advice",
]
