"""
MedVerify AI - Stage 9 + 11: Unified Verification Orchestrator Service

Wires the real AI pipeline end-to-end:
    Stage 11 (Medical Safety Guardrail) ->
    Stage 6 (Disease Classifier) ->
    Stage 7 (Hybrid Retrieval & Evidence Ranking) ->
    Stage 8 (Consensus Verification & NLI Stance Detection) ->
    Stage 10 (Explanation Generation & Faithfulness) ->
    Verdict + Credibility Score + Evidence Breakdown + Safety Disclaimers

This replaces the Stage 3 mock pipeline with real AI inference.
"""

import os
import uuid
import logging
import torch
from typing import Dict, List, Optional
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from app.services.retrieval_engine import HybridRetrievalEngine
from app.services.consensus_engine import (
    VerificationPipeline,
    FaithfulnessVerifier,
    get_certainty_level,
    map_to_spec_label,
)
from app.services.population.applicability import analyze_population_applicability
from app.services.explanation_generator import ExplanationGenerator
from app.services.safety_guardrail import (
    MedicalSafetyGuardrail,
    get_safety_guardrail,
    SafetyAction,
    SafetyResult,
    VulnerabilityFlag,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PATHS (relative to project root)
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VECTOR_STORE_DIR = os.path.join(PROJECT_ROOT, "..", "vector_store")
CLASSIFIER_MODEL_DIR = os.path.join(PROJECT_ROOT, "..", "models", "biobert_disease_classifier")

# Normalize paths
VECTOR_STORE_DIR = os.path.normpath(VECTOR_STORE_DIR)
CLASSIFIER_MODEL_DIR = os.path.normpath(CLASSIFIER_MODEL_DIR)

# Disease category labels (matches 22-category fine-tuned BioBERT)
LABEL_MAP = {
    0: "COVID-19",
    1: "General Cancer",
    2: "Influenza",
    3: "Reproductive Health / Abortion",
    4: "Breast Cancer",
    5: "HIV/AIDS",
    6: "Ebola Virus",
    7: "Alzheimers Disease",
    8: "Prostate Cancer",
    9: "Diabetes",
    10: "Obesity & Weight Management",
    11: "Opioids & Pain Management",
    12: "Measles / MMR",
    13: "Depression",
    14: "Heart Disease",
    15: "Smoking & Tobacco",
    16: "Pregnancy & Maternal Health",
    17: "Heart Attack",
    18: "Stroke",
    19: "Lung Cancer",
    20: "Autism Spectrum Disorder",
    21: "Vaccination"
}


class MedVerifyOrchestrator:
    """
    Singleton-style orchestrator that lazily initializes all AI models
    on first call, then reuses them for subsequent verifications.
    """

    def __init__(self):
        self._classifier_model = None
        self._classifier_tokenizer = None
        self._retrieval_engine = None
        self._verification_pipeline = None
        self._faithfulness_verifier = None
        self._explanation_generator = None
        self._safety_guardrail = None
        self._initialized = False

    def _lazy_init(self):
        """Load all AI models on first use (avoids slow startup)."""
        if self._initialized:
            return

        logger.info("[Orchestrator] Lazy-initializing AI pipeline models...")

        # 1. Disease Classifier (Stage 6)
        if os.path.exists(CLASSIFIER_MODEL_DIR):
            logger.info(f"  Loading disease classifier from: {CLASSIFIER_MODEL_DIR}")
            self._classifier_tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_MODEL_DIR)
            self._classifier_model = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_MODEL_DIR)
            self._classifier_model.eval()
            logger.info("  [OK] Disease classifier loaded.")
        else:
            logger.warning(f"  [WARN] Classifier model not found at {CLASSIFIER_MODEL_DIR}. Using keyword fallback.")

        # 2. Hybrid Retrieval Engine (Stage 7)
        if os.path.exists(VECTOR_STORE_DIR):
            logger.info(f"  Loading FAISS vector store from: {VECTOR_STORE_DIR}")
            self._retrieval_engine = HybridRetrievalEngine(VECTOR_STORE_DIR)
            logger.info("  [OK] Hybrid Retrieval Engine loaded.")
        else:
            logger.warning(f"  [WARN] Vector store not found at {VECTOR_STORE_DIR}. Retrieval disabled.")

        # 3. Consensus Verification Pipeline (Stage 8)
        logger.info("  Loading NLI model for consensus verification...")
        self._verification_pipeline = VerificationPipeline(
            nli_model="cross-encoder/nli-deberta-v3-small"
        )
        self._faithfulness_verifier = FaithfulnessVerifier()
        self._explanation_generator = ExplanationGenerator(
            stance_detector=self._verification_pipeline.stance_detector
        )
        logger.info("  [OK] Consensus Verification Pipeline & Explanation Generator loaded.")

        self._initialized = True

        # 4. Safety Guardrail (Stage 11) — lightweight, no model loading
        self._safety_guardrail = get_safety_guardrail()
        logger.info("  [OK] Medical Safety Guardrail loaded.")

        logger.info("[Orchestrator] All AI models initialized successfully.")

    # ------------------------------------------------------------------
    # SAFETY CHECK
    # ------------------------------------------------------------------

    def check_safety_refusal(self, raw_text: str) -> SafetyResult:
        """
        Run the full Stage 11 safety analysis.
        Returns a SafetyResult with action, flags, disclaimers, and messaging.
        """
        self._lazy_init()
        return self._safety_guardrail.analyze(raw_text)

    # ------------------------------------------------------------------
    # DISEASE CLASSIFICATION (Stage 6)
    # ------------------------------------------------------------------

    def classify_disease(self, claim_text: str) -> str:
        """
        Classify the medical claim into one of 3 Phase 1 disease categories.
        Falls back to keyword matching if model is not available.
        """
        if self._classifier_model and self._classifier_tokenizer:
            inputs = self._classifier_tokenizer(
                claim_text, truncation=True, padding="max_length",
                max_length=128, return_tensors="pt"
            )
            with torch.no_grad():
                outputs = self._classifier_model(**inputs)
                pred = torch.argmax(outputs.logits, dim=1).item()
            return LABEL_MAP.get(pred, "Diabetes")
        else:
            # Keyword fallback
            lower = claim_text.lower()
            if any(kw in lower for kw in ["vaccine", "mmr", "immunization", "autism", "vaccination"]):
                return "Vaccination"
            elif any(kw in lower for kw in ["statin", "heart", "cardiovascular", "cholesterol", "cardiac", "aspirin"]):
                return "Cardiovascular Disease"
            else:
                return "Diabetes"

    # ------------------------------------------------------------------
    # FULL VERIFICATION PIPELINE
    # ------------------------------------------------------------------

    def verify(self, raw_text: str, disease_category: Optional[str] = None) -> Dict:
        """
        Run the full end-to-end verification pipeline.

        Args:
            raw_text: The user's medical claim text
            disease_category: Optional override; auto-classified if None

        Returns:
            Complete verification result dict with verdict, credibility,
            evidence breakdown, consensus, and metadata.
        """
        self._lazy_init()

        start_time = datetime.utcnow()

        # Step 1: Safety Check (Stage 11)
        safety_result = self.check_safety_refusal(raw_text)

        if safety_result.action == SafetyAction.EMERGENCY_REDIRECT:
            return {
                "status": "REFUSED_SAFETY",
                "verdict": None,
                "credibility_score": None,
                "reason": safety_result.emergency_message,
                "safety_action": safety_result.action.value,
                "disclaimer": safety_result.disclaimer,
                "disclaimer_severity": safety_result.disclaimer_severity.value,
                "evidence": [],
                "elapsed_seconds": 0.0,
            }

        if safety_result.action == SafetyAction.REFUSE_PERSONAL:
            return {
                "status": "REFUSED_SAFETY",
                "verdict": None,
                "credibility_score": None,
                "reason": safety_result.refusal_reason,
                "safety_action": safety_result.action.value,
                "disclaimer": safety_result.disclaimer,
                "disclaimer_severity": safety_result.disclaimer_severity.value,
                "evidence": [],
                "elapsed_seconds": 0.0,
            }

        # Step 2: Disease Classification (Stage 6)
        if disease_category:
            disease = disease_category
        else:
            disease = self.classify_disease(raw_text)

        # Step 3: Hybrid Retrieval & Evidence Ranking (Stage 7)
        ranked_evidence = []
        if self._retrieval_engine:
            ranked_evidence = self._retrieval_engine.retrieve_and_rank(
                claim_text=raw_text,
                disease_category=disease,
                faiss_top_k=5,
                pubmed_max=3,
            )

        # Step 3.5: Population Applicability Analysis
        for ev in ranked_evidence:
            try:
                app_res = analyze_population_applicability(raw_text, ev)
                ev["applicability_score"] = app_res.score
                ev["population_match_type"] = app_res.match_result.match_type.value if app_res.match_result else "UNKNOWN"
            except Exception as e:
                logger.warning(f"Population analysis error for evidence: {e}")
                ev["applicability_score"] = 1.0
                ev["population_match_type"] = "UNKNOWN"

        # Step 4: NLI Stance Detection & Consensus (Stage 8)
        if ranked_evidence and self._verification_pipeline:
            verification_result = self._verification_pipeline.verify_claim(
                claim_text=raw_text,
                ranked_evidence=ranked_evidence,
            )
        else:
            # No evidence available
            verification_result = {
                "verdict": "INCONCLUSIVE",
                "credibility_score": 50.0,
                "weighted_consensus": 0.0,
                "total_evidence": 0,
                "raw_counts": {"supporting": 0, "contradicting": 0, "neutral": 0},
                "total_reliability_weight": 0.0,
                "status": "INSUFFICIENT_EVIDENCE",
                "evidence_breakdown": [],
            }

        elapsed = (datetime.utcnow() - start_time).total_seconds()

        # Step 5: Build evidence citation objects with full §11 trace fields
        evidence_citations = []
        for i, ev in enumerate(verification_result.get("evidence_breakdown", [])):
            ev_id = f"ev-{uuid.uuid4().hex[:6]}"
            r_i = ev.get("reliability_score", 0.5)
            p_i = ev.get("applicability_score", 1.0)
            w_i = round(r_i * p_i, 4)
            evidence_citations.append({
                # §11 required fields
                "id": ev_id,
                "title": ev.get("title", ""),
                "source": ev.get("source", "FAISS_STATIC"),
                "source_tier": ev.get("source_tier", "PubMed Article"),
                "evidence_type": ev.get("source_tier", "PubMed Article"),  # evidence type alias
                "pub_year": ev.get("pub_year"),
                "doi": ev.get("doi", ""),
                "url": ev.get("url", ""),
                "citation": f"{ev.get('title', '')} ({ev.get('pub_year', 'n.d.')})",
                "retrieval_score": ev.get("cosine_similarity", 0.0),
                "recency": ev.get("pub_year"),
                # Scoring trace
                "reliability_score": r_i,
                "applicability_score": p_i,
                "population_match_type": ev.get("population_match_type", "UNKNOWN"),
                "W_i": w_i,
                # Stance
                "stance": ev.get("stance", "neutral"),
                "stance_value": ev.get("stance_value", 0),
                "nli_confidence": ev.get("nli_confidence", 0.0),
                "entailment_prob": ev.get("entailment_prob", 0.0),
                "contradiction_prob": ev.get("contradiction_prob", 0.0),
                "cosine_similarity": ev.get("cosine_similarity", 0.0),
                "abstract_chunk": ev.get("chunk_text", ""),
                "source_channel": ev.get("source", "FAISS_STATIC"),
            })

        # Initial verdict display label for explanation generation
        verdict_str_internal = verification_result.get("verdict", "INCONCLUSIVE")
        _verdict_map_early = {
            "SUPPORTED": "Supported", "LIKELY_SUPPORTED": "Supported",
            "INCONCLUSIVE": "Insufficient Evidence",
            "LIKELY_REFUTED": "Contradicted", "REFUTED": "Contradicted",
            "MIXTURE": "Mixed",
        }
        verdict_str = _verdict_map_early.get(verdict_str_internal, "Insufficient Evidence")
        credibility_score = verification_result.get("credibility_score", 50.0)

        # Step 6: Generate Grounded Explanation & Dual-Pass Sentence Faithfulness Verification (Stage 10)
        explanation_sentences = []
        faithfulness_confidence = 100.0
        if self._explanation_generator:
            explanation_sentences, faithfulness_confidence = self._explanation_generator.generate_and_verify_explanation(
                claim_text=raw_text,
                disease_category=disease,
                verdict=verdict_str,
                credibility_score=credibility_score,
                evidence_citations=evidence_citations,
            )

        # Step 7: Build credibility breakdown with real faithfulnessConfidence
        raw_counts = verification_result.get("raw_counts", {})
        total_ev = verification_result.get("total_evidence", 0)

        # Evidence confidence = average NLI confidence across supporting evidence
        nli_confs = [ev.get("nli_confidence", 0.0) for ev in verification_result.get("evidence_breakdown", [])]
        avg_nli = sum(nli_confs) / len(nli_confs) if nli_confs else 0.0

        credibility_breakdown = {
            "overall": credibility_score,
            "evidenceConfidence": round(avg_nli * 100, 1),
            "consensusConfidence": round(abs(verification_result.get("weighted_consensus", 0.0)) * 100, 1),
            "sourceQuality": round(
                (verification_result.get("total_reliability_weight", 0.0) / max(total_ev, 1)) * 100, 1
            ),
            "faithfulnessConfidence": faithfulness_confidence,
        }

        consensus_summary = {
            "weightedConsensusScore": verification_result.get("weighted_consensus", 0.0),
            "totalReliabilityWeight": verification_result.get("total_reliability_weight", 0.0),
            "rawCounts": raw_counts,
        }

        # Map verdict to display label for UI and spec label for research
        verdict_map = {
            "SUPPORTED": "Supported",
            "LIKELY_SUPPORTED": "Supported",
            "INCONCLUSIVE": "Insufficient Evidence",
            "LIKELY_REFUTED": "Contradicted",
            "REFUTED": "Contradicted",
            "MIXTURE": "Mixed",
        }
        verdict_display = verdict_map.get(
            verification_result.get("verdict", "INCONCLUSIVE"), "Insufficient Evidence"
        )
        # Spec label (§13: TRUE/FALSE/MIXTURE/UNPROVEN)
        spec_label = verification_result.get(
            "spec_label",
            map_to_spec_label(verification_result.get("verdict", "INCONCLUSIVE")),
        )

        # Step 7: Build version metadata
        version_metadata = {
            "modelVersion": "qwen3-8b-instruct-v1.2",
            "diseaseClassifierVersion": "biobert-base-cased-v1.2-finetuned",
            "embeddingVersion": "all-MiniLM-L6-v2",
            "knowledgeBaseVersion": "kb-phase1-387chunks-v1.0",
            "rankingFormulaVersion": "rwrav-v1.0-calibrated",
            "promptVersion": "grounded-explanation-v3",
            "faithfulnessModelVersion": "nli-deberta-v3-small-bioscope",
        }

        # Step 8: Generate verdict-specific disclaimer (Stage 11)
        verdict_disclaimer, verdict_disclaimer_severity = self._safety_guardrail.get_verdict_disclaimer(
            verdict=verdict_display,
            vulnerability_flags=safety_result.vulnerability_flags if safety_result.vulnerability_flags else None,
        )

        return {
            "status": "COMPLETED",
            "disease_category": disease,
            "verdict": verdict_display,
            "spec_label": spec_label,
            "credibility_score": credibility_score,
            "credibility_breakdown": credibility_breakdown,
            "consensus_summary": consensus_summary,
            "explanation_sentences": explanation_sentences,
            "evidence_citations": evidence_citations,
            "version_metadata": version_metadata,
            "safety_analysis": {
                "action": safety_result.action.value,
                "vulnerability_flags": [f.value for f in safety_result.vulnerability_flags],
                "matched_patterns": safety_result.matched_patterns,
            },
            "disclaimer": verdict_disclaimer,
            "disclaimer_severity": verdict_disclaimer_severity.value,
            "elapsed_seconds": round(elapsed, 2),
        }


# ---------------------------------------------------------------------------
# MODULE-LEVEL SINGLETON
# ---------------------------------------------------------------------------

_orchestrator_instance: Optional[MedVerifyOrchestrator] = None


def get_orchestrator() -> MedVerifyOrchestrator:
    """Get or create the singleton orchestrator instance."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = MedVerifyOrchestrator()
    return _orchestrator_instance
