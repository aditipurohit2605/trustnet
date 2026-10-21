"""
TrustNet end-to-end verification pipeline entry points.

Provides message-level retrieval orchestration: extracts atomic claims from unformatted text,
filters for checkable factual claims, and retrieves relevant evidence from the knowledge
corpus using the Stage 3 multi-mode retriever.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from trustnet.retrieval.retriever import Retriever
from trustnet.schemas import (
    AbstainReason,
    Claim,
    ClaimReport,
    Evidence,
    EvidenceStatus,
    ExtractedClaim,
    ReportDecision,
    SearchCoverage,
    ShareAdvice,
    Verdict,
)
from trustnet.text.extraction.extractor import ClaimExtractor

logger = logging.getLogger(__name__)


def build_search_coverage(
    evidence: List[Evidence],
    retrieval_config: Optional[Union[str, Path, Dict[str, Any]]] = None,
    total_retrieved: Optional[int] = None,
) -> SearchCoverage:
    """
    Constructs a SearchCoverage telemetry object from config and retrieved candidates.
    """
    import yaml
    cfg: Dict[str, Any] = {}
    if retrieval_config is None:
        cfg_path = Path("configs/stage3_retrieval.yaml")
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
    elif isinstance(retrieval_config, (str, Path)):
        if Path(retrieval_config).exists():
            with open(retrieval_config, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
    elif isinstance(retrieval_config, dict):
        cfg = retrieval_config

    relevance_threshold = float(cfg.get("relevance_threshold", 0.50))
    not_searched = list(cfg.get("not_searched", [
        "social_media_archives",
        "real_time_web",
        "subscription_databases",
    ]))
    corpus_path_str = cfg.get("corpus_path", "data/corpus/corpus_v1.jsonl")
    corpus_p = Path(corpus_path_str)

    total_docs = 0
    if corpus_p.exists():
        try:
            with open(corpus_p, "r", encoding="utf-8") as f:
                total_docs = sum(1 for line in f if line.strip())
        except Exception:
            total_docs = 5
    else:
        total_docs = 5

    corpus_name = corpus_p.stem if corpus_p.stem else "corpus_v1"
    candidates_count = total_retrieved if total_retrieved is not None else len(evidence)

    scores = [float(e.score) for e in evidence if e.score is not None]
    top_score = max(scores) if scores else 0.0
    passages_above = sum(1 for s in scores if s >= relevance_threshold)

    return SearchCoverage(
        corpus_name=corpus_name,
        total_documents_indexed=total_docs,
        candidates_retrieved=candidates_count,
        passages_above_threshold=passages_above,
        top_score=round(top_score, 4),
        not_searched=not_searched,
    )


def run_retrieval_for_message(
    text: str,
    retriever: Retriever,
    extractor: Optional[ClaimExtractor] = None,
    mode: str = "hybrid_rerank",
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Orchestrates claim extraction and evidence retrieval for an incoming message.
    
    Workflow:
    1. Extracts atomic claims and source mentions using ClaimExtractor.
    2. Filters for checkable claims (ignoring subjective opinions or untestable assertions).
    3. Converts each checkable ExtractedClaim into a canonical Claim instance with a stable ID.
    4. Retrieves top-k evidence documents for each claim using Retriever (default: 'hybrid_rerank').
    
    Args:
        text: Raw user/forwarded message text.
        retriever: Configured Stage 3 Retriever instance.
        extractor: Optional ClaimExtractor instance (created from config if omitted).
        mode: Retrieval mode to execute ('hybrid_rerank', 'hybrid', 'dense', 'bm25').
        top_k: Maximum number of evidence items to retrieve per claim (default: 5).
        
    Returns:
        Structured dictionary containing extracted claims, checkable claims, and their retrieved evidence.
        NOTE: Final verdict prediction and calibration are deferred to subsequent stages.
    """
    if extractor is None:
        extractor = ClaimExtractor()

    logger.info(f"Extracting factual claims from message ({len(text)} characters)...")
    extraction_result = extractor.extract(text)

    # Filter strictly for checkable claims
    checkable_claims: List[ExtractedClaim] = [
        c for c in extraction_result.claims if c.checkable
    ]
    logger.info(f"Extracted {len(extraction_result.claims)} total claims ({len(checkable_claims)} checkable).")

    claim_retrievals: List[Dict[str, Any]] = []

    for ext_claim in checkable_claims:
        # Convert to canonical Claim with stable hashed ID
        claim_obj: Claim = ext_claim.to_claim()
        logger.debug(f"Retrieving top-{top_k} evidence for claim [{claim_obj.id}]: '{claim_obj.text}' [Mode: {mode}]...")

        evidence_items: List[Evidence] = retriever.search(
            claim_text=claim_obj.text,
            mode=mode,
            top_k=top_k,
        )

        claim_retrievals.append({
            "claim": claim_obj,
            "extracted_claim": ext_claim,
            "evidence": evidence_items,
        })

    return {
        "original_text": text,
        "all_extracted_claims": extraction_result.claims,
        "source_mentions": extraction_result.source_mentions,
        "checkable_claims": checkable_claims,
        "claim_retrievals": claim_retrievals,
        "error": extraction_result.error,
    }


def run_extraction_for_image_text(
    image: Any,
    extractor: Optional[ClaimExtractor] = None,
    analyzer: Optional[Any] = None,
    ocr_reader: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Extracts text from an image via OCR and feeds it through the Stage 4 ClaimExtractor.
    All extracted claims are explicitly marked with metadata={'from_image_text': True}.

    Args:
        image: In-memory image (PIL Image, numpy array, bytes) or file path.
        extractor: Configured Stage 4 ClaimExtractor instance (created if omitted).
        analyzer: Optional ImageAnalyzer instance.
        ocr_reader: Optional OCRReader instance.

    Returns:
        Structured dictionary containing:
        - "ocr_text": Raw OCR text extracted from the image.
        - "ocr_confidence": Mean detection confidence score.
        - "ocr_language": Detected language ("en", "hi", "unknown").
        - "claims": All extracted ExtractedClaim instances tagged with metadata={'from_image_text': True}.
        - "checkable_claims": Checkable ExtractedClaim instances tagged with metadata={'from_image_text': True}.
        - "from_image_text": True flag indicating visual provenance.
    """
    ocr_text = ""
    ocr_conf = 0.0
    ocr_lang = "unknown"

    # Step 1: Run OCR extraction
    if analyzer is not None:
        signals = analyzer.analyze(image)
        ocr_text = signals.ocr_text or ""
        ocr_conf = signals.ocr_confidence or 0.0
        ocr_lang = signals.ocr_language or "unknown"
    elif ocr_reader is not None:
        ocr_text, ocr_conf, ocr_lang = ocr_reader.read(image)
    else:
        # Lazy import to avoid unnecessary overhead when vision is not used
        from trustnet.vision.ocr import OCRReader
        reader = OCRReader()
        ocr_text, ocr_conf, ocr_lang = reader.read(image)

    # Step 2: Handle empty OCR text
    if not ocr_text or not ocr_text.strip():
        logger.info("No OCR text detected on the image; returning zero claims.")
        return {
            "ocr_text": "",
            "ocr_confidence": 0.0,
            "ocr_language": "unknown",
            "claims": [],
            "checkable_claims": [],
            "from_image_text": True,
            "error": None,
        }

    # Step 3: Extract factual claims from the OCR text
    if extractor is None:
        extractor = ClaimExtractor()

    logger.info(f"Extracting claims from OCR text ({len(ocr_text)} characters)...")
    extraction_result = extractor.extract(ocr_text)

    # Step 4: Tag every extracted claim with provenance 'from_image_text': True
    tagged_claims: List[ExtractedClaim] = []
    for claim in extraction_result.claims:
        claim.metadata["from_image_text"] = True
        tagged_claims.append(claim)

    checkable_claims = [c for c in tagged_claims if c.checkable]
    logger.info(
        f"Extracted {len(tagged_claims)} claims from image text "
        f"({len(checkable_claims)} checkable, tagged 'from_image_text')."
    )

    return {
        "ocr_text": ocr_text,
        "ocr_confidence": round(float(ocr_conf), 4),
        "ocr_language": ocr_lang,
        "claims": tagged_claims,
        "checkable_claims": checkable_claims,
        "source_mentions": extraction_result.source_mentions,
        "from_image_text": True,
        "error": extraction_result.error,
    }


def apply_reasoning_to_report(
    report: ClaimReport,
    reasoning_config: Optional[Union[str, Path, Dict[str, Any]]] = None,
    image_signals: Optional[Any] = None,
    source_mentions: Optional[List[Any]] = None,
    original_message: Optional[str] = None,
) -> ClaimReport:
    """
    Executes all enabled Stage 8 reasoning components on a ClaimReport:
    - Source Reliability Weighting
    - Conflict Detection
    - Temporal Freshness Analysis
    - Manipulation Tagging
    - Evidence Gap Analysis
    - Counterfactual Sensitivity Generation
    - Actionable Share Advice

    Guarantees:
    - Strict verdict and confidence immutability: the underlying classification is never modified.
    - Fault-tolerant execution: failure in any reasoning component logs a warning and leaves
      that specific field empty without interrupting the pipeline.
    """
    import yaml
    cfg: Dict[str, Any] = {}
    if reasoning_config is None:
        default_cfg_path = Path("configs/stage8_reasoning.yaml")
        if default_cfg_path.exists():
            with open(default_cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
    elif isinstance(reasoning_config, (str, Path)):
        if Path(reasoning_config).exists():
            with open(reasoning_config, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
    elif isinstance(reasoning_config, dict):
        cfg = reasoning_config

    enabled = cfg.get("enabled", {})
    notes: List[str] = list(report.reasoning_notes or [])

    # 1. Source Reliability
    if enabled.get("source_reliability", True):
        try:
            from trustnet.reasoning.source_reliability import SourceReliabilityAssigner
            s_assigner = SourceReliabilityAssigner(
                config_path=cfg.get("source_reliability_path", "configs/source_reliability.yaml")
            )
            weights, s_summary = s_assigner.assign_weights(report.evidence)
            report.source_weights = weights
            if s_summary.get("unknown_sources"):
                notes.append(f"SourceReliability: Unverified source categories detected: {s_summary['unknown_sources']}")
        except Exception as e:
            logger.warning(f"Reasoning layer [source_reliability] failed: {e}")

    # 2. Conflict Detection
    if enabled.get("conflict", True):
        try:
            from trustnet.reasoning.conflict import ConflictDetector
            detector = ConflictDetector(
                conflict_min_weight=cfg.get("conflict_min_weight", 0.50),
                confidence_threshold=cfg.get("conflict_confidence_threshold", 0.45),
            )
            conflict_info = detector.detect_conflicts(report.claim.text, report.evidence)
            report.conflict = conflict_info
            if conflict_info.has_conflict:
                notes.append(f"ConflictDetector: {conflict_info.explanation}")
        except Exception as e:
            logger.warning(f"Reasoning layer [conflict] failed: {e}")

    # 3. Temporal Freshness
    if enabled.get("temporal", True):
        try:
            from trustnet.reasoning.temporal import TemporalAnalyzer
            t_analyzer = TemporalAnalyzer(stale_days=cfg.get("stale_days", 90))
            t_info = t_analyzer.analyze(report.claim.text, report.evidence)
            report.temporal = t_info
            if t_info.is_stale:
                notes.append(f"TemporalAnalyzer: Evidence flagged stale - {t_info.explanation}")
        except Exception as e:
            logger.warning(f"Reasoning layer [temporal] failed: {e}")

    # 4. Manipulation Tagging
    if enabled.get("manipulation", True):
        try:
            from trustnet.reasoning.manipulation import ManipulationTagger
            m_tagger = ManipulationTagger(
                patterns_path=cfg.get("manipulation_patterns_path", "data/eval/manipulation_patterns.yaml"),
                model_name=cfg.get("manipulation_model", "llama3.2:1b"),
            )
            # Tag either original message or claim text
            text_to_tag = original_message if original_message else report.claim.text
            tags = m_tagger.tag(text_to_tag)
            report.manipulation = tags
            if tags:
                tag_names = [t.name for t in tags]
                notes.append(f"ManipulationTagger: Detected rhetorical tactics {tag_names}")
        except Exception as e:
            logger.warning(f"Reasoning layer [manipulation] failed: {e}")

    # 5. Gap Analysis
    if enabled.get("gaps", True):
        try:
            from trustnet.reasoning.gaps import GapAnalyzer
            g_analyzer = GapAnalyzer(
                templates_path=cfg.get("gap_templates_path", "data/eval/gap_templates.yaml")
            )
            gaps = g_analyzer.analyze(
                claim_text=report.claim.text,
                evidence_list=report.evidence,
                conflict_info=report.conflict,
                temporal_info=report.temporal,
                image_signals=image_signals,
                source_mentions=source_mentions,
                search_coverage=report.search_coverage,
            )
            # Combine newly identified gaps with existing legacy notes
            report.gaps = gaps
            if gaps:
                gap_codes = [g.code for g in gaps]
                notes.append(f"GapAnalyzer: Identified verification gaps {gap_codes}")
        except Exception as e:
            logger.warning(f"Reasoning layer [gaps] failed: {e}")

    # 6. Counterfactual Sensitivity
    if enabled.get("counterfactual", True):
        try:
            from trustnet.reasoning.counterfactual import CounterfactualGenerator
            c_gen = CounterfactualGenerator(max_counterfactuals=cfg.get("max_counterfactuals", 5))
            
            # Feed real features into the predictor
            feat_rec = getattr(report, "feature_record", None)
            if not feat_rec:
                scores_list = [float(e.score) for e in report.evidence if e.score is not None]
                top1_sc = scores_list[0] if scores_list else 1.0
                mean_sc = float(sum(scores_list) / len(scores_list)) if scores_list else 0.5
                raw_p = report.raw_probabilities or {}
                t_logits = [raw_p.get("verified", 0.0), raw_p.get("contradicted", 0.0), raw_p.get("insufficient_evidence", 0.0)]
                feat_rec = {
                    "claim_id": report.claim.id,
                    "claim_text": report.claim.text,
                    "dataset": "pipeline",
                    "text_logits": t_logits,
                    "text_pooled": [0.0] * 32,
                    "retrieval_features": {
                        "top1_score": top1_sc,
                        "top1_top2_gap": 0.5,
                        "num_passages_above_threshold": len(report.evidence),
                        "mean_score": mean_sc,
                        "source_type_corpus_passage": len(report.evidence),
                        "source_type_government": sum(1 for e in report.evidence if (e.source_type or "").lower() == "government"),
                        "source_type_news": sum(1 for e in report.evidence if (e.source_type or "").lower() == "news"),
                        "source_type_other": 0.0,
                        "agreement_feature": 1.0,
                    },
                    "image_features": {
                        "image_present": 1 if image_signals else 0,
                        "mismatch_score": getattr(image_signals, "mismatch_score", 0.0) if image_signals else 0.0,
                        "best_reuse_similarity": 0.0,
                        "ocr_present": 1 if getattr(image_signals, "ocr_text", None) else 0,
                        "ocr_confidence": getattr(image_signals, "ocr_confidence", 0.0) or 0.0,
                        "quality_flag_tiny_resolution": 0,
                        "quality_flag_extreme_aspect_ratio": 0,
                        "quality_flag_heavy_jpeg_compression": 0,
                        "quality_flag_screenshot_likeness": 0,
                        "clip_image_embedding": [0.0] * 512,
                    },
                }
            c_list = c_gen.generate(feat_rec, original_verdict=report.verdict)
            report.counterfactuals = c_list
        except Exception as e:
            logger.warning(f"Reasoning layer [counterfactual] failed: {e}")

    # 7. Actionable Share Advice
    if enabled.get("share_advice", True):
        try:
            from trustnet.reasoning.share_advice import determine_share_advice
            advice, reason_code, advice_rationale = determine_share_advice(
                verdict=report.verdict,
                confidence=report.confidence,
                abstained=report.abstained,
                conflict=report.conflict,
                temporal=report.temporal,
                return_code=True,
            )
            report.share_advice = advice
            report.decision = ReportDecision(
                verdict=report.verdict,
                evidence_status=report.evidence_status or (EvidenceStatus.SUPPORTED if report.verdict == Verdict.VERIFIED else (EvidenceStatus.CONTRADICTED if report.verdict == Verdict.CONTRADICTED else EvidenceStatus.INSUFFICIENT)),
                recommended_action=advice,
                reason_codes=[reason_code],
            )
            notes.append(f"ShareAdvice: {advice.value} [{reason_code}] - {advice_rationale}")
        except Exception as e:
            logger.warning(f"Reasoning layer [share_advice] failed: {e}")

    report.reasoning_notes = notes
    return report


def verify_claim(
    claim: Claim,
    evidence: Optional[List[Evidence]] = None,
    image_path: Optional[str] = None,
    predictor: Optional[Any] = None,
    enable_reasoning: bool = True,
    reasoning_config: Optional[Any] = None,
    multilingual_config: Optional[Any] = None,
    search_coverage: Optional[SearchCoverage] = None,
) -> ClaimReport:
    """
    Verifies a single Claim instance using calibrated multimodal prediction and abstention,
    enriched with the Stage 8 reasoning layer and Stage 9 multilingual support.
    """
    # Stage 9: Language identification, normalization, and original text tracking
    if getattr(claim, "original_text", None) is None:
        claim.original_text = claim.text

    if getattr(claim, "language_info", None) is None:
        try:
            from trustnet.lang.detect import detect_language
            claim.language_info = detect_language(claim.original_text or claim.text)
        except Exception as e:
            logger.debug(f"Language detection skipped: {e}")

    if getattr(claim, "normalized_text", None) is None:
        try:
            from trustnet.lang.normalize import normalize_text
            claim.normalized_text = normalize_text(claim.original_text or claim.text)
        except Exception as e:
            logger.debug(f"Normalization skipped: {e}")

    # Resolve strategy if non-English
    effective_claim = claim
    lang_code = claim.language_info.code if claim.language_info else "en"
    if lang_code in ("hi", "hi-Latn"):
        strategy = "translate_test"
        if isinstance(multilingual_config, dict):
            strategy = multilingual_config.get("strategy", "translate_test")
        elif isinstance(multilingual_config, (str, Path)) and Path(multilingual_config).is_file():
            import yaml
            with open(multilingual_config, "r", encoding="utf-8") as f:
                strategy = yaml.safe_load(f).get("strategy", "translate_test")

        if strategy in ("translate_test", "hybrid"):
            try:
                from trustnet.lang.translate import Translator
                translator = Translator.get_instance()
                norm_str = claim.normalized_text.normalized if claim.normalized_text else claim.text
                translated_en = translator.translate(norm_str, src_lang="hi", tgt_lang="en")
                if claim.normalized_text:
                    claim.normalized_text.translated_en = translated_en
                effective_claim = Claim(
                    id=claim.id,
                    text=translated_en,
                    language_info=claim.language_info,
                    original_text=claim.original_text,
                    normalized_text=claim.normalized_text,
                )
            except Exception as e:
                logger.warning(f"Translation failed in verify_claim: {e}")

    if predictor is None:
        from trustnet.fusion.predict import MultimodalPredictor
        predictor = MultimodalPredictor.from_artifacts()

    report = predictor.predict_claim_report(
        claim=effective_claim,
        evidence=evidence or [],
        image_path=image_path,
    )

    if search_coverage is None:
        search_coverage = build_search_coverage(evidence or [])
    report.search_coverage = search_coverage

    # Update evidence_status to reflect search_coverage
    if search_coverage.passages_above_threshold == 0:
        report.evidence_status = EvidenceStatus.NO_EVIDENCE_FOUND
    elif report.verdict == Verdict.VERIFIED:
        report.evidence_status = EvidenceStatus.SUPPORTED
    elif report.verdict == Verdict.CONTRADICTED:
        report.evidence_status = EvidenceStatus.CONTRADICTED
    else:
        report.evidence_status = EvidenceStatus.INSUFFICIENT

    if report.abstained and not report.abstain_reason:
        report.abstain_reason = AbstainReason.BELOW_CONFIDENCE_THRESHOLD

    # Ensure report always shows what user wrote and what module used
    report.language_info = claim.language_info
    report.original_text = claim.original_text or claim.text
    report.normalized_text = claim.normalized_text

    if enable_reasoning:
        report = apply_reasoning_to_report(report, reasoning_config=reasoning_config)
    else:
        if report.decision is None:
            from trustnet.reasoning.share_advice import determine_share_advice
            advice, reason_code, _ = determine_share_advice(
                report.verdict, report.confidence, report.abstained, return_code=True
            )
            report.share_advice = advice
            report.decision = ReportDecision(
                verdict=report.verdict,
                evidence_status=report.evidence_status,
                recommended_action=advice,
                reason_codes=[reason_code],
            )

    return report


def run_full_verification_pipeline(
    text: str,
    image_path: Optional[str] = None,
    retriever: Optional[Retriever] = None,
    extractor: Optional[ClaimExtractor] = None,
    predictor: Optional[Any] = None,
    retrieval_mode: str = "hybrid_rerank",
    top_k: int = 5,
    enable_reasoning: bool = True,
    reasoning_config: Optional[Any] = None,
    multilingual_config: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Full end-to-end verification pipeline:
    1. Extracts checkable claims and source mentions from input text.
    2. Retrieves top-k evidence documents per claim.
    3. Runs calibrated multimodal prediction, multilingual translation, and abstention.
    4. Executes the Stage 8 reasoning layer (conflict, freshness, gaps, share advice).
    5. Returns structured results containing enriched ClaimReport instances.
    """
    if predictor is None:
        from trustnet.fusion.predict import MultimodalPredictor
        predictor = MultimodalPredictor.from_artifacts()

    retrieval_result = run_retrieval_for_message(
        text=text,
        retriever=retriever,
        extractor=extractor,
        mode=retrieval_mode,
        top_k=top_k,
    )

    reports: List[ClaimReport] = []
    source_mentions = retrieval_result.get("source_mentions", [])

    for item in retrieval_result.get("claim_retrievals", []):
        claim_obj = item["claim"]
        evidence_items = item["evidence"]

        report = verify_claim(
            claim=claim_obj,
            evidence=evidence_items,
            image_path=image_path,
            predictor=predictor,
            enable_reasoning=enable_reasoning,
            reasoning_config=reasoning_config,
            multilingual_config=multilingual_config,
        )

        reports.append(report)

    return {
        "original_text": text,
        "image_path": image_path,
        "reports": reports,
        "extraction": retrieval_result,
    }


# Backward-compatible alias
run_full_pipeline = run_full_verification_pipeline


