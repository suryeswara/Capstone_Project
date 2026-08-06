"""
MedVerify AI - Stage 9: Unified Verification Orchestrator Service

Wires the real AI pipeline end-to-end:
    Stage 6 (Disease Classifier) ->
    Stage 7 (Hybrid Retrieval & Evidence Ranking) ->
    Stage 8 (Consensus Verification & NLI Stance Detection) ->
    Verdict + Credibility Score + Evidence Breakdown

This replaces the Stage 3 mock pipeline with real AI inference.
"""

import os
import uuid
import logging
import torch
from typing import Dict, Optional
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from app.services.retrieval_engine import HybridRetrievalEngine
from app.services.consensus_engine import (
    VerificationPipeline,
    FaithfulnessVerifier,
    get_certainty_level,
)
from app.services.explanation_generator import ExplanationGenerator

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

# Disease category labels (must match Stage 6 training label map)
LABEL_MAP = {0: "Diabetes", 1: "Cardiovascular Disease", 2: "Vaccination"}

# Safety refusal trigger keywords (personal medical advice detection)
SAFETY_REFUSAL_PATTERNS = [
    ("i have", "should i take"),
    ("i have", "how much"),
    ("i have", "chest pain"),
    ("i am", "should i stop"),
    ("my doctor", "should i"),
    ("i feel", "what medication"),
    ("diagnose", "me"),
]


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
        logger.info("[Orchestrator] All AI models initialized successfully.")

    # ------------------------------------------------------------------
    # SAFETY CHECK
    # ------------------------------------------------------------------

    def check_safety_refusal(self, raw_text: str) -> bool:
        """
        Returns True if the input text seeks personal medical advice
        (diagnosis/treatment), which should trigger safety refusal.
        """
        lower = raw_text.lower()
        for pattern_parts in SAFETY_REFUSAL_PATTERNS:
            if all(part in lower for part in pattern_parts):
                return True
        return False

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

        # Step 1: Safety Check
        if self.check_safety_refusal(raw_text):
            return {
                "status": "REFUSED_SAFETY",
                "verdict": None,
                "credibility_score": None,
                "reason": "Input seeks personal diagnostic or treatment advice. "
                          "MedVerify AI verifies general medical claims only.",
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

        # Step 5: Build evidence citation objects
        evidence_citations = []
        for i, ev in enumerate(verification_result.get("evidence_breakdown", [])):
            ev_id = f"ev-{uuid.uuid4().hex[:6]}"
            evidence_citations.append({
                "id": ev_id,
                "title": ev.get("title", ""),
                "source_type": ev.get("source_tier", "PubMed Article"),
                "pub_year": ev.get("pub_year"),
                "doi": ev.get("doi", ""),
                "reliability_score": ev.get("reliability_score", 0.5),
                "stance": ev.get("stance", "neutral"),
                "abstract_chunk": ev.get("chunk_text", ""),
                "url": ev.get("url", ""),
                "source_channel": ev.get("source", "FAISS_STATIC"),
                "nli_confidence": ev.get("nli_confidence", 0.0),
                "entailment_prob": ev.get("entailment_prob", 0.0),
                "contradiction_prob": ev.get("contradiction_prob", 0.0),
                "cosine_similarity": ev.get("cosine_similarity", 0.0),
            })

        # Map verdict to DB enum
        verdict_map = {
            "SUPPORTED": "Supported",
            "LIKELY_SUPPORTED": "Supported",
            "INCONCLUSIVE": "Insufficient Evidence",
            "LIKELY_REFUTED": "Contradicted",
            "REFUTED": "Contradicted",
        }
        verdict_str = verdict_map.get(
            verification_result.get("verdict", "INCONCLUSIVE"), "Insufficient Evidence"
        )
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

        # Map verdict to DB enum
        verdict_map = {
            "SUPPORTED": "Supported",
            "LIKELY_SUPPORTED": "Supported",
            "INCONCLUSIVE": "Insufficient Evidence",
            "LIKELY_REFUTED": "Contradicted",
            "REFUTED": "Contradicted",
        }
        verdict_str = verdict_map.get(
            verification_result.get("verdict", "INCONCLUSIVE"), "Insufficient Evidence"
        )

        # Step 7: Build version metadata
        version_metadata = {
            "modelVersion": "qwen3-8b-instruct-v1.2",
            "diseaseClassifierVersion": "distilbert-base-uncased-finetuned-v1.0",
            "embeddingVersion": "all-MiniLM-L6-v2",
            "knowledgeBaseVersion": "kb-phase1-387chunks-v1.0",
            "rankingFormulaVersion": "rwrav-v1.0-calibrated",
            "promptVersion": "grounded-explanation-v3",
            "faithfulnessModelVersion": "nli-deberta-v3-small-bioscope",
        }

        return {
            "status": "COMPLETED",
            "disease_category": disease,
            "verdict": verdict_str,
            "credibility_score": credibility_score,
            "credibility_breakdown": credibility_breakdown,
            "consensus_summary": consensus_summary,
            "explanation_sentences": explanation_sentences,
            "evidence_citations": evidence_citations,
            "version_metadata": version_metadata,
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
