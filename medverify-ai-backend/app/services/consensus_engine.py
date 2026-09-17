"""
MedVerify AI - Stage 8: Consensus Verification Engine

This module implements:
1. NLI-based Stance Detection (entailment / contradiction / neutral)
2. Weighted Consensus Scoring (R_i-weighted stance aggregation)
3. Dual-Pass Faithfulness Verification (NLI + BioScope Certainty Inflation Guard)
4. Credibility Score Computation & Verdict Mapping
"""

import math
import re
import logging
from typing import List, Dict, Optional, Tuple
from transformers import pipeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BIOSCOPE HEDGE CUE DICTIONARIES (Validated in Stage 0)
# ---------------------------------------------------------------------------

HEDGE_CUES = {
    # Level 1: Weak / Speculative / Association
    "weak": [
        "may", "might", "could", "suggests", "associated with", "correlated with",
        "possible", "potential", "preliminary", "appears to", "trend towards",
        "observed in vitro", "hypothesized", "tentative",
    ],
    # Level 2: Moderate / Indicative
    "moderate": [
        "indicates", "shows", "demonstrates", "supports", "likely",
        "reduces risk of", "increases risk of", "consistent with",
        "evidence suggests", "data supports", "improves",
    ],
    # Level 3: Strong / Causation / Definitive
    "strong": [
        "causes", "cures", "prevents", "proves", "proven to", "definitively",
        "eradicates", "guarantees", "eliminates", "direct cause",
        "has been established", "conclusively",
    ],
}

# NLI label-to-stance mapping for different model label conventions
NLI_LABEL_MAP = {
    "entailment": "supporting",
    "ENTAILMENT": "supporting",
    "contradiction": "contradicting",
    "CONTRADICTION": "contradicting",
    "neutral": "neutral",
    "NEUTRAL": "neutral",
}

STANCE_VALUE_MAP = {
    "supporting": +1,
    "contradicting": -1,
    "neutral": 0,
}

# Spec-mandated label mapping (§13): internal → TRUE/FALSE/MIXTURE/UNPROVEN
INTERNAL_TO_SPEC_LABEL = {
    "SUPPORTED": "TRUE",
    "LIKELY_SUPPORTED": "TRUE",
    "INCONCLUSIVE": "UNPROVEN",
    "LIKELY_REFUTED": "FALSE",
    "REFUTED": "FALSE",
    "MIXTURE": "MIXTURE",
}


def map_to_spec_label(internal_verdict: str) -> str:
    """Map an internal consensus verdict to a spec-mandated label (§13).

    Returns one of: TRUE, FALSE, MIXTURE, UNPROVEN
    """
    return INTERNAL_TO_SPEC_LABEL.get(internal_verdict, "UNPROVEN")


# ---------------------------------------------------------------------------
# CERTAINTY LEVEL EXTRACTION (BioScope Hedge Cue Analysis)
# ---------------------------------------------------------------------------

def get_certainty_level(text: str) -> int:
    """
    Extracts the highest certainty level from a text passage.
    Returns: 1 = weak/speculative, 2 = moderate/indicative, 3 = strong/definitive

    Uses regex word-boundary matching to prevent partial word matches
    (e.g., 'improves' should NOT match the strong cue 'proves').
    """
    text_lower = text.lower()

    def _has_cue(cue: str) -> bool:
        """Check if cue appears as a whole word/phrase in text."""
        pattern = r'\b' + re.escape(cue) + r'\b'
        return bool(re.search(pattern, text_lower))

    for cue in HEDGE_CUES["strong"]:
        if _has_cue(cue):
            return 3
    for cue in HEDGE_CUES["moderate"]:
        if _has_cue(cue):
            return 2
    for cue in HEDGE_CUES["weak"]:
        if _has_cue(cue):
            return 1

    return 2  # Default neutral assumption


# ---------------------------------------------------------------------------
# NLI STANCE DETECTOR
# ---------------------------------------------------------------------------

class NLIStanceDetector:
    """
    Uses a pre-trained Natural Language Inference model to classify
    the relationship between a claim and a piece of evidence as:
    entailment (supporting), contradiction (contradicting), or neutral.

    Uses proper text-classification NLI with premise/hypothesis pair
    formatting for accurate contradiction detection.
    """

    # Map model output labels to our stance labels
    MODEL_LABEL_MAP = {
        "ENTAILMENT": "supporting",
        "CONTRADICTION": "contradicting",
        "NEUTRAL": "neutral",
        "entailment": "supporting",
        "contradiction": "contradicting",
        "neutral": "neutral",
    }

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small"):
        """
        Initialize the NLI pipeline using text-classification.
        The cross-encoder NLI models expect [premise, hypothesis] pairs.
        """
        logger.info(f"Loading NLI model: {model_name}...")
        self.classifier = pipeline(
            "text-classification",
            model=model_name,
            top_k=None,  # Return all label scores
            device=-1,   # CPU
        )
        self.model_name = model_name
        logger.info(f"NLI model loaded: {model_name}")

    def detect_stance(self, claim: str, evidence_text: str) -> Dict:
        """
        Determine the stance of evidence toward a claim.

        Formats input as premise (evidence) + hypothesis (claim) for
        proper NLI inference.

        Returns:
            dict with keys: stance, confidence, entailment_prob,
                            contradiction_prob, neutral_prob
        """
        # Cross-encoder NLI: premise = evidence, hypothesis = claim
        # The tokenizer handles [SEP] automatically with text pairs
        result = self.classifier(
            {"text": evidence_text, "text_pair": claim}
        )

        # Parse label scores from model output
        label_scores = {}
        for item in result:
            label_key = item["label"].upper()
            label_scores[label_key] = item["score"]

        entailment_prob = label_scores.get("ENTAILMENT", 0.0)
        contradiction_prob = label_scores.get("CONTRADICTION", 0.0)
        neutral_prob = label_scores.get("NEUTRAL", 0.0)

        # Determine winning stance by highest probability
        best_label = max(label_scores, key=label_scores.get)
        stance = self.MODEL_LABEL_MAP.get(best_label, "neutral")
        confidence = label_scores[best_label]

        return {
            "stance": stance,
            "stance_value": STANCE_VALUE_MAP.get(stance, 0),
            "confidence": round(confidence, 4),
            "entailment_prob": round(entailment_prob, 4),
            "contradiction_prob": round(contradiction_prob, 4),
            "neutral_prob": round(neutral_prob, 4),
        }


# ---------------------------------------------------------------------------
# DUAL-PASS FAITHFULNESS VERIFIER
# ---------------------------------------------------------------------------

class FaithfulnessVerifier:
    """
    Dual-Pass Faithfulness Verification:
    Pass 1: NLI entailment probability check (threshold >= 0.70)
    Pass 2: BioScope certainty inflation guard (explanation must not
             exceed evidence certainty level)
    """

    NLI_THRESHOLD = 0.70

    def verify(
        self,
        evidence_text: str,
        explanation_text: str,
        nli_entailment_prob: float,
    ) -> Dict:
        """
        Check if an AI-generated explanation is faithful to the evidence.
        """
        evidence_certainty = get_certainty_level(evidence_text)
        explanation_certainty = get_certainty_level(explanation_text)

        certainty_inflated = explanation_certainty > evidence_certainty
        nli_failed = nli_entailment_prob < self.NLI_THRESHOLD

        is_faithful = (not nli_failed) and (not certainty_inflated)

        failure_reasons = []
        if nli_failed:
            failure_reasons.append("NLI_LOW_ENTAILMENT")
        if certainty_inflated:
            failure_reasons.append(
                f"CERTAINTY_INFLATION (Evidence L{evidence_certainty} -> Explanation L{explanation_certainty})"
            )

        return {
            "is_faithful": is_faithful,
            "nli_entailment_prob": round(nli_entailment_prob, 4),
            "evidence_certainty_level": evidence_certainty,
            "explanation_certainty_level": explanation_certainty,
            "certainty_inflated": certainty_inflated,
            "failure_reasons": failure_reasons,
        }


# ---------------------------------------------------------------------------
# WEIGHTED CONSENSUS ENGINE
# ---------------------------------------------------------------------------

class ConsensusEngine:
    """
    Aggregates evidence stances weighted by reliability scores (R_i)
    to compute a final credibility score and verdict.
    """

    # Verdict thresholds on consensus score [-1.0, +1.0]
    VERDICT_THRESHOLDS = {
        "SUPPORTED": 0.30,       # consensus >= 0.30
        "LIKELY_SUPPORTED": 0.10, # 0.10 <= consensus < 0.30
        "INCONCLUSIVE": -0.10,    # -0.10 <= consensus < 0.10
        "LIKELY_REFUTED": -0.30,  # -0.30 <= consensus < -0.10
        "REFUTED": -1.01,         # consensus < -0.30
    }

    def calculate_weighted_consensus(self, evidence_list: List[Dict]) -> Dict:
        """
        Consensus = sum(R_i * stance_i) / sum(R_i)
        stance_i in {+1, -1, 0}

        Includes Neutral Mass Guardrail:
        If evidence is predominantly neutral (>= 70% neutral mass) and directional
        evidence is sparse (<= 1 study per side), the verdict is calibrated to
        INCONCLUSIVE rather than falsely leaning Supported or Refuted.
        """
        total_weight = 0.0
        weighted_stance_sum = 0.0
        directional_weight = 0.0
        neutral_weight = 0.0
        raw_counts = {"supporting": 0, "contradicting": 0, "neutral": 0}

        for item in evidence_list:
            r = item.get("reliability_score", 0.5)
            p = item.get("applicability_score", 1.0)  # P_i ∈ [0, 1]
            s = item.get("stance_value", 0)
            
            w = r * p
            total_weight += w
            weighted_stance_sum += (w * s)

            if s > 0:
                raw_counts["supporting"] += 1
                directional_weight += w
            elif s < 0:
                raw_counts["contradicting"] += 1
                directional_weight += w
            else:
                raw_counts["neutral"] += 1
                neutral_weight += w

        if total_weight == 0 or directional_weight == 0:
            return {
                "weighted_consensus": 0.0,
                "credibility_score": 50.0,
                "verdict": "INCONCLUSIVE",
                "raw_counts": raw_counts,
                "total_evidence": sum(raw_counts.values()),
                "total_reliability_weight": 0.0,
                "status": "INSUFFICIENT_EVIDENCE",
                "spec_label": "UNPROVEN",
            }

        # Consensus score in [-1.0, +1.0]
        consensus_score = weighted_stance_sum / total_weight

        # Map consensus to credibility score [0, 100]
        # consensus -1.0 -> 0, consensus 0.0 -> 50, consensus +1.0 -> 100
        credibility_score = round((consensus_score + 1.0) * 50.0, 2)
        credibility_score = max(0.0, min(100.0, credibility_score))

        # Determine verdict
        verdict = self._map_verdict(consensus_score)

        # -------------------------------------------------------------------
        # Neutral Mass Guardrail (Stage 8 Enhancement)
        # Avoid false polarization when evidence is ambiguous:
        # 1. Neutral dominates (>= 70%) AND directional signals are split (both supp and contra >= 1)
        # 2. Or directional coverage is negligible (< 15%) and consensus is near-zero (|score| < 0.15)
        # -------------------------------------------------------------------
        neutral_ratio = neutral_weight / total_weight if total_weight > 0 else 1.0
        directional_coverage = directional_weight / total_weight if total_weight > 0 else 0.0

        if neutral_ratio >= 0.70 and (raw_counts["supporting"] >= 1 and raw_counts["contradicting"] >= 1):
            logger.info(
                f"[ConsensusEngine] Neutral Mass Guard triggered (split signals): neutral_ratio={neutral_ratio:.2f}, "
                f"supp={raw_counts['supporting']}, contra={raw_counts['contradicting']}. Overriding '{verdict}' -> 'INCONCLUSIVE'."
            )
            verdict = "INCONCLUSIVE"
        elif directional_coverage < 0.15 and abs(consensus_score) < 0.15:
            verdict = "INCONCLUSIVE"

        # -------------------------------------------------------------------
        # MIXTURE Detection (§13)
        # When evidence is highly polarized (both supporting AND contradicting
        # with significant weights), the verdict is MIXTURE rather than
        # reflecting which side barely won.
        # -------------------------------------------------------------------
        supporting_weight = sum(
            (item.get("reliability_score", 0.5) * item.get("applicability_score", 1.0))
            for item in evidence_list if item.get("stance_value", 0) > 0
        )
        contradicting_weight = sum(
            (item.get("reliability_score", 0.5) * item.get("applicability_score", 1.0))
            for item in evidence_list if item.get("stance_value", 0) < 0
        )
        if total_weight > 0:
            support_frac = supporting_weight / total_weight
            contra_frac = contradicting_weight / total_weight
            if (
                support_frac >= 0.20
                and contra_frac >= 0.20
                and raw_counts["supporting"] >= 2
                and raw_counts["contradicting"] >= 2
            ):
                logger.info(
                    f"[ConsensusEngine] MIXTURE detected: support_frac={support_frac:.2f}, "
                    f"contra_frac={contra_frac:.2f}. Overriding verdict '{verdict}' -> 'MIXTURE'."
                )
                verdict = "MIXTURE"

        # Map to spec label (§13: TRUE/FALSE/MIXTURE/UNPROVEN)
        spec_label = map_to_spec_label(verdict)

        return {
            "weighted_consensus": round(consensus_score, 4),
            "credibility_score": credibility_score,
            "verdict": verdict,
            "spec_label": spec_label,
            "raw_counts": raw_counts,
            "total_evidence": sum(raw_counts.values()),
            "total_reliability_weight": round(total_weight, 4),
            "status": "VERIFIED" if verdict != "INCONCLUSIVE" else "INSUFFICIENT_EVIDENCE",
        }

    def _map_verdict(self, consensus: float) -> str:
        """Map consensus score to verdict label."""
        if consensus >= self.VERDICT_THRESHOLDS["SUPPORTED"]:
            return "SUPPORTED"
        elif consensus >= self.VERDICT_THRESHOLDS["LIKELY_SUPPORTED"]:
            return "LIKELY_SUPPORTED"
        elif consensus >= self.VERDICT_THRESHOLDS["INCONCLUSIVE"]:
            return "INCONCLUSIVE"
        elif consensus >= self.VERDICT_THRESHOLDS["LIKELY_REFUTED"]:
            return "LIKELY_REFUTED"
        else:
            return "REFUTED"


# ---------------------------------------------------------------------------
# FULL VERIFICATION PIPELINE (Orchestrator)
# ---------------------------------------------------------------------------

class VerificationPipeline:
    """
    End-to-end verification pipeline that:
    1. Takes ranked evidence from Stage 7
    2. Runs NLI stance detection on each evidence item
    3. Computes weighted consensus score
    4. Returns verdict + credibility score + evidence breakdown
    """

    def __init__(self, nli_model: str = "cross-encoder/nli-deberta-v3-small"):
        self.stance_detector = NLIStanceDetector(model_name=nli_model)
        self.consensus_engine = ConsensusEngine()
        self.faithfulness_verifier = FaithfulnessVerifier()

    def verify_claim(
        self,
        claim_text: str,
        ranked_evidence: List[Dict],
    ) -> Dict:
        """
        Run full verification pipeline on a claim with its ranked evidence.

        Args:
            claim_text: The medical claim to verify
            ranked_evidence: List of evidence dicts from Stage 7 HybridRetrievalEngine

        Returns:
            Full verification result with verdict, credibility score,
            evidence breakdown, and faithfulness checks.
        """
        evidence_with_stances = []

        for i, evidence in enumerate(ranked_evidence):
            evidence_text = evidence.get("chunk_text", evidence.get("title", ""))

            # Run NLI stance detection
            stance_result = self.stance_detector.detect_stance(
                claim=claim_text,
                evidence_text=evidence_text,
            )

            # Merge stance into evidence dict
            enriched = {
                **evidence,
                "stance": stance_result["stance"],
                "stance_value": stance_result["stance_value"],
                "nli_confidence": stance_result["confidence"],
                "entailment_prob": stance_result["entailment_prob"],
                "contradiction_prob": stance_result["contradiction_prob"],
                "neutral_prob": stance_result["neutral_prob"],
                "rank": i + 1,
            }
            evidence_with_stances.append(enriched)

        # Calculate weighted consensus
        consensus_result = self.consensus_engine.calculate_weighted_consensus(
            evidence_with_stances
        )

        return {
            "claim_text": claim_text,
            "verdict": consensus_result["verdict"],
            "credibility_score": consensus_result["credibility_score"],
            "weighted_consensus": consensus_result["weighted_consensus"],
            "total_evidence": consensus_result["total_evidence"],
            "raw_counts": consensus_result["raw_counts"],
            "total_reliability_weight": consensus_result["total_reliability_weight"],
            "status": consensus_result["status"],
            "evidence_breakdown": evidence_with_stances,
        }
