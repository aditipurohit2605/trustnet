"""
Inference and Prediction Pipeline with Calibration and Abstention for TrustNet Stage 7.

Orchestrates:
1. Multimodal fusion prediction from Stage 6 models (MLP or LightGBM).
2. Probability calibration via fitted TemperatureScaler.
3. Selective abstention decision via fitted Abstainer.
4. Construction of full ClaimReport with raw and calibrated probabilities.
"""

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.special import softmax

from trustnet.calibration.abstain import Abstainer
from trustnet.calibration.temperature import TemperatureScaler
from trustnet.schemas import Claim, ClaimReport, Evidence, Verdict

logger = logging.getLogger(__name__)

CLASS_NAMES = ["verified", "contradicted", "insufficient_evidence"]
VERDICT_MAP = {
    "verified": Verdict.VERIFIED,
    "contradicted": Verdict.CONTRADICTED,
    "insufficient_evidence": Verdict.INSUFFICIENT_EVIDENCE,
}


class MultimodalPredictor:
    """
    End-to-end predictor incorporating multimodal fusion, temperature scaling,
    and confidence-based abstention.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        model_type: str = "mlp",
        temperature_scaler: Optional[TemperatureScaler] = None,
        abstainer: Optional[Abstainer] = None,
        class_names: Optional[List[str]] = None,
    ):
        self.model = model
        self.model_type = model_type.lower()
        self.scaler = temperature_scaler or TemperatureScaler()
        self.abstainer = abstainer or Abstainer()
        self.class_names = class_names or CLASS_NAMES

    @classmethod
    def from_artifacts(
        cls,
        model_dir: Union[str, Path] = "outputs/stage6/models/mlp_full",
        temperature_path: Union[str, Path] = "outputs/stage7/temperature.json",
        abstainer_path: Union[str, Path] = "outputs/stage7/abstainer.json",
        model_type: str = "mlp",
    ) -> "MultimodalPredictor":
        """
        Loads trained fusion model, fitted temperature scaler, and abstainer from disk.
        Falls back gracefully to default unscaled/default abstainer if artifact files are missing.
        """
        model_dir = Path(model_dir)
        loaded_model = None

        if model_type == "mlp" and (model_dir / "mlp_model.pt").exists():
            import torch
            from trustnet.fusion.mlp import MLPFusionModel, MLPFusionTrainer
            cfg = {}
            if (model_dir / "mlp_config.json").exists():
                with open(model_dir / "mlp_config.json", "r", encoding="utf-8") as f:
                    cfg = json.load(f)
            weights = torch.load(model_dir / "mlp_model.pt", map_location="cpu", weights_only=True)
            text_pooled_dim = weights["text_projector.0.weight"].shape[1] if "text_projector.0.weight" in weights else 768
            clip_dim = weights["clip_projector.0.weight"].shape[1] if "clip_projector.0.weight" in weights else 512
            trainer = MLPFusionTrainer(
                hidden_sizes=cfg.get("hidden_sizes", [128, 64]),
                dropout=cfg.get("dropout", 0.2),
            )
            trainer.model = MLPFusionModel(
                compact_dim=cfg.get("compact_dim", 25),
                clip_dim=clip_dim,
                text_pooled_dim=text_pooled_dim,
                hidden_sizes=cfg.get("hidden_sizes", [128, 64]),
                dropout=cfg.get("dropout", 0.2),
            )
            trainer.model.load_state_dict(weights)
            trainer.model.eval()
            trainer.feature_names = cfg.get("feature_names", [])
            loaded_model = trainer
        elif model_type == "gbm" and (model_dir / "lightgbm_model.txt").exists():
            import lightgbm as lgb
            from trustnet.fusion.gbm import LightGBMFusionClassifier
            clf = LightGBMFusionClassifier()
            clf.booster = lgb.Booster(model_file=str(model_dir / "lightgbm_model.txt"))
            if (model_dir / "feature_names.json").exists():
                with open(model_dir / "feature_names.json", "r", encoding="utf-8") as f:
                    clf.feature_names = json.load(f)
            loaded_model = clf

        # Load temperature scaler
        t_path = Path(temperature_path)
        scaler = TemperatureScaler.load(t_path) if t_path.exists() else TemperatureScaler()

        # Load abstainer
        a_path = Path(abstainer_path)
        abstainer = Abstainer.load(a_path) if a_path.exists() else Abstainer()

        return cls(
            model=loaded_model,
            model_type=model_type,
            temperature_scaler=scaler,
            abstainer=abstainer,
        )

    def predict_record(
        self,
        record: Dict[str, Any],
        active_groups: Optional[List[str]] = None,
        dataset: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Produces raw logits, calibrated probabilities, and abstention verdict for a cached record.
        """
        # 1. Compute raw logits
        raw_logits: np.ndarray
        if self.model is not None:
            if self.model_type == "mlp":
                from trustnet.calibration.ensemble import extract_mlp_logits
                raw_logits = extract_mlp_logits(self.model, [record], active_groups=active_groups)
            else:
                from trustnet.calibration.ensemble import extract_gbm_logits
                raw_logits = extract_gbm_logits(self.model, [record], active_groups=active_groups)
        else:
            # Fallback to text_logits if full model is unavailable
            text_logits = record.get("text_logits", [0.0, 0.0, 0.0])
            raw_logits = np.array([text_logits], dtype=np.float64)

        if raw_logits.ndim == 1:
            raw_logits = raw_logits.reshape(1, -1)

        # 2. Raw Softmax probabilities
        raw_probs = softmax(raw_logits, axis=1)[0]

        # 3. Calibrated Probabilities via Temperature Scaling
        ds_name = dataset or record.get("dataset")
        cal_probs = self.scaler.predict_proba(raw_logits, dataset=ds_name)[0]

        # 4. Abstention Decision
        decision = self.abstainer.decide(cal_probs)[0]
        abstained = (decision == "abstain")

        argmax_idx = int(np.argmax(cal_probs))
        predicted_verdict_str = self.class_names[argmax_idx]
        calibrated_conf = float(cal_probs[argmax_idx])
        if math.isnan(calibrated_conf):
            calibrated_conf = 0.3333

        raw_prob_dict = {
            self.class_names[i]: round(float(raw_probs[i]), 4) for i in range(len(self.class_names))
        }
        cal_prob_dict = {
            self.class_names[i]: round(float(cal_probs[i]), 4) for i in range(len(self.class_names))
        }

        abstention_reason = None
        if abstained:
            abstention_reason = (
                f"Calibrated confidence {calibrated_conf:.4f} fell below the required threshold "
                f"of {self.abstainer.threshold:.4f} for target error rate {self.abstainer.target_error_rate:.2%}."
            )

        return {
            "predicted_verdict": predicted_verdict_str,
            "decision": decision,
            "abstained": abstained,
            "confidence": calibrated_conf,
            "raw_probabilities": raw_prob_dict,
            "calibrated_probabilities": cal_prob_dict,
            "abstention_reason": abstention_reason,
        }

    def predict_claim_report(
        self,
        claim: Claim,
        evidence: Optional[List[Evidence]] = None,
        image_path: Optional[str] = None,
        cached_record: Optional[Dict[str, Any]] = None,
        dataset: Optional[str] = None,
    ) -> ClaimReport:
        """
        Creates a complete ClaimReport preserving original claim text,
        with calibrated confidence and abstention state.
        """
        evidence = evidence or []
        if cached_record is not None:
            pred_info = self.predict_record(cached_record, dataset=dataset)
        else:
            # Query model for expected dimension sizes
            text_dim = 32
            clip_dim = 512
            if self.model is not None and hasattr(self.model, "model"):
                if hasattr(self.model.model, "text_projector"):
                    text_dim = self.model.model.text_projector[0].in_features
                if hasattr(self.model.model, "clip_projector"):
                    clip_dim = self.model.model.clip_projector[0].in_features

            scores_list = [float(e.score) for e in evidence if e.score is not None and not math.isnan(float(e.score))]
            top1_sc = scores_list[0] if scores_list else 1.0
            mean_sc = float(np.mean(scores_list)) if scores_list else 0.5

            rec = {
                "claim_id": claim.id,
                "claim_text": claim.text,
                "dataset": dataset or "unknown",
                "text_logits": [0.0, 0.0, 0.0],
                "text_pooled": [0.0] * text_dim,
                "retrieval_features": {
                    "top1_score": top1_sc,
                    "top1_top2_gap": 0.5,
                    "num_passages_above_threshold": len(evidence),
                    "mean_score": mean_sc,
                    "source_type_corpus_passage": len(evidence),
                    "source_type_government": 0.0,
                    "source_type_news": 0.0,
                    "source_type_other": 0.0,
                    "agreement_feature": 1.0,
                },
                "image_features": {
                    "image_present": 1 if image_path else 0,
                    "mismatch_score": 0.0,
                    "best_reuse_similarity": 0.0,
                    "ocr_present": 0,
                    "ocr_confidence": 0.0,
                    "quality_flag_tiny_resolution": 0,
                    "quality_flag_extreme_aspect_ratio": 0,
                    "quality_flag_heavy_jpeg_compression": 0,
                    "quality_flag_screenshot_likeness": 0,
                    "clip_image_embedding": [0.0] * clip_dim,
                },
            }
            pred_info = self.predict_record(rec, dataset=dataset)

        # Verdict assignment: if abstained, set to INSUFFICIENT_EVIDENCE
        if pred_info["abstained"]:
            final_verdict = Verdict.INSUFFICIENT_EVIDENCE
        else:
            final_verdict = VERDICT_MAP.get(pred_info["predicted_verdict"], Verdict.INSUFFICIENT_EVIDENCE)

        return ClaimReport(
            claim=claim,
            evidence=evidence,
            verdict=final_verdict,
            confidence=pred_info["confidence"],
            abstained=pred_info["abstained"],
            raw_probabilities=pred_info["raw_probabilities"],
            calibrated_probabilities=pred_info["calibrated_probabilities"],
            abstention_reason=pred_info["abstention_reason"],
            gaps=["Confidence below threshold; abstained from verdict."] if pred_info["abstained"] else [],
            language_info=getattr(claim, "language_info", None),
            original_text=getattr(claim, "original_text", None) or claim.text,
            normalized_text=getattr(claim, "normalized_text", None),
        )
