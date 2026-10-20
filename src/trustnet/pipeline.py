"""
TrustNet end-to-end verification pipeline entry points.

Provides message-level retrieval orchestration: extracts atomic claims from unformatted text,
filters for checkable factual claims, and retrieves relevant evidence from the knowledge
corpus using the Stage 3 multi-mode retriever.
"""

import logging
from typing import Any, Dict, List, Optional

from trustnet.retrieval.retriever import Retriever
from trustnet.schemas import Claim, Evidence, ExtractedClaim
from trustnet.text.extraction.extractor import ClaimExtractor

logger = logging.getLogger(__name__)


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

