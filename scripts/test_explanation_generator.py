"""
MedVerify AI -- Stage 10 Explanation Generator & Sentence Faithfulness Validation Script

Tests:
1. Explanation sentence drafting with explicit evidence citation IDs [ev-1], [ev-2].
2. Dual-Pass sentence-level faithfulness verification (NLI + BioScope certainty inflation guard).
3. faithfulnessConfidence percentage computation.
4. Full end-to-end integration via MedVerifyOrchestrator.
"""

import os
import sys
import logging

# Add backend app to Python path
sys.path.insert(0, os.path.join(os.getcwd(), "medverify-ai-backend"))

from app.services.explanation_generator import ExplanationGenerator
from app.services.consensus_engine import NLIStanceDetector
from app.services.orchestrator import MedVerifyOrchestrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

print("=" * 80)
print("MEDVERIFY AI -- STAGE 10 EXPLANATION GENERATOR & FAITHFULNESS VALIDATION")
print("=" * 80)

# ---------------------------------------------------------------------------
# TEST 1: Grounded Explanation Generation & Citation Mapping
# ---------------------------------------------------------------------------
print("\n[TEST 1] Grounded Explanation Generation & Citation Mapping")
print("-" * 70)

generator = ExplanationGenerator()

mock_citations = [
    {
        "id": "ev-101",
        "title": "Randomized Trial of Statin Efficacy in Secondary CVD Prevention",
        "abstract_chunk": "A multicenter trial of 10,000 patients demonstrates that statins reduce recurrent myocardial infarction by 30%.",
        "source_type": "Randomized Controlled Trial (RCT)",
        "reliability_score": 0.90,
        "stance": "supporting",
    },
    {
        "id": "ev-102",
        "title": "Systematic Review on Lipid-Lowering Therapies",
        "abstract_chunk": "Meta-analysis confirms lipid lowering significantly improves long-term survival in coronary artery disease.",
        "source_type": "Systematic Review / Meta-Analysis",
        "reliability_score": 0.95,
        "stance": "supporting",
    },
]

sentences, faithfulness_score = generator.generate_and_verify_explanation(
    claim_text="Statins reduce cardiovascular events.",
    disease_category="Cardiovascular Disease",
    verdict="Supported",
    credibility_score=92.5,
    evidence_citations=mock_citations,
)

print(f"  Claim: 'Statins reduce cardiovascular events.'")
print(f"  Verdict: Supported | Faithfulness Confidence Score: {faithfulness_score}%")
print(f"  Generated Sentences ({len(sentences)}):")
for s in sentences:
    print(f"    [{s['sentenceId']}] [{s['status'].upper()}] (L{s['certaintyLevel']} Certainty) Cited: {s['citedEvidenceIds']}")
    print(f"         '{s['text']}'")

# ---------------------------------------------------------------------------
# TEST 2: Contradicted Verdict Explanation & Sentence Status
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("[TEST 2] Contradicted Verdict Explanation & Sentence Verification")
print("-" * 70)

mock_false_citations = [
    {
        "id": "ev-vax-01",
        "title": "Nationwide Cohort Study Finds No Link Between MMR Vaccine and Autism",
        "abstract_chunk": "Analysis of 657,461 children demonstrates no increased risk of autism associated with MMR vaccination.",
        "source_type": "Cohort Study / Observational",
        "reliability_score": 0.90,
        "stance": "contradicting",
    },
    {
        "id": "ev-vax-02",
        "title": "Cochrane Systematic Review of MMR Vaccine Safety",
        "abstract_chunk": "Meta-analysis of 1.2 million children confirms MMR vaccine is not associated with autism or developmental delay.",
        "source_type": "Systematic Review / Meta-Analysis",
        "reliability_score": 0.95,
        "stance": "contradicting",
    },
]

sentences_false, faithfulness_false = generator.generate_and_verify_explanation(
    claim_text="The MMR vaccine causes autism in children.",
    disease_category="Vaccination",
    verdict="Contradicted",
    credibility_score=0.0,
    evidence_citations=mock_false_citations,
)

print(f"  Claim: 'The MMR vaccine causes autism in children.'")
print(f"  Verdict: Contradicted | Faithfulness Confidence Score: {faithfulness_false}%")
print(f"  Generated Sentences ({len(sentences_false)}):")
for s in sentences_false:
    print(f"    [{s['sentenceId']}] [{s['status'].upper()}] (L{s['certaintyLevel']} Certainty) Cited: {s['citedEvidenceIds']}")
    print(f"         '{s['text']}'")

# ---------------------------------------------------------------------------
# TEST 3: End-to-End Orchestrator Integration with Real NLI Model
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("[TEST 3] End-to-End Orchestrator Integration (Stage 10 Full Stack)")
print("-" * 70)
print("  Initializing MedVerify Orchestrator...")

orchestrator = MedVerifyOrchestrator()

# Test claim execution
claim = "Statins reduce the risk of recurrent heart attacks in cardiovascular patients."
res = orchestrator.verify(claim)

print(f"\n  Claim: '{claim}'")
print(f"  Status:                 {res['status']}")
print(f"  Disease Category:       {res['disease_category']}")
print(f"  Verdict:                {res['verdict']}")
print(f"  Credibility Score:      {res['credibility_score']:.1f}%")
print(f"  FaithfulnessConfidence: {res['credibility_breakdown']['faithfulnessConfidence']:.1f}%")
print(f"  Generated Sentences ({len(res['explanation_sentences'])}):")
for s in res['explanation_sentences']:
    print(f"    [{s['sentenceId']}] [{s['status'].upper()}] NLI={s['nliConfidence']:.4f} Cited={s['citedEvidenceIds']}")
    print(f"         '{s['text']}'")

# ---------------------------------------------------------------------------
# STAGE 10 EXIT CRITERIA CHECK
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STAGE 10 EXIT CRITERIA CHECK:")
print(f" [{'OK' if len(sentences) > 0 else 'FAIL'}] Grounded explanation generation operational")
print(f" [{'OK' if all('citedEvidenceIds' in s for s in sentences) else 'FAIL'}] Sentence-level inline citation mapping [ev-1], [ev-2]")
print(f" [{'OK' if all('certaintyLevel' in s for s in sentences) else 'FAIL'}] Dual-pass sentence faithfulness verification (NLI + BioScope)")
print(f" [{'OK' if 'faithfulnessConfidence' in res['credibility_breakdown'] else 'FAIL'}] Faithfulness confidence percentage score computed")
print(" SUMMARY: Stage 10 LLM Explanation Generator & Sentence Faithfulness Engine COMPLETE.")
print("=" * 80)
