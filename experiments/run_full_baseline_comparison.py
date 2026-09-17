"""
MedVerify AI — Full Baseline Comparison (B0, B1, B2, B3)
=========================================================
Master Instructions §14, §17, §18

Runs all four baselines on the frozen test set using the real pipeline
components (retrieval engine, NLI stance detection, population module).

B0: LLM Only — claim → keyword heuristic → verdict (no evidence)
B1: Conventional RAG — claim → retrieval → unweighted NLI consensus → verdict
B2: Reliability-Aware RAG — claim → retrieval → R_i-weighted NLI consensus → verdict
B3: Full MedVerify — claim → retrieval → (R_i × P_i)-weighted NLI consensus → verdict

Key design constraints:
  - Same claims, same retrieval budget, same label space
  - B2 and B3 use identical retrieval; only the weighting differs (§5)
  - Labels normalized to spec: TRUE / FALSE / MIXTURE / UNPROVEN
  - Evaluates on frozen test split (§22: no training on test set)

Reports:
  - Accuracy, Macro-F1, Precision, Recall, per-class F1 (§17)
  - Confusion matrices
  - Δ Macro-F1 (B3 - B2) with bootstrap CI and McNemar test (§18)
  - VCR and CCR (§20)
  - Per-population-category breakdown (§19)

Usage:
    python experiments/run_full_baseline_comparison.py [--sample N] [--seed 42]
"""

import os
import sys
import json
import time
import math
import logging
import argparse
import hashlib
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import Counter

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)

# ---------------------------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "medverify-ai-backend"))

from app.services.consensus_engine import (
    ConsensusEngine,
    NLIStanceDetector,
    map_to_spec_label,
    INTERNAL_TO_SPEC_LABEL,
)
from app.services.population.applicability import analyze_population_applicability

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
TEST_SET_PATH = os.path.join(PROJECT_ROOT, "med_datasets", "splits", "test_frozen.json")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "Reports")
OUTPUT_PATH = os.path.join(REPORTS_DIR, "full_baseline_comparison.json")

# PubHealth label normalization → spec labels (§13)
PUBHEALTH_TO_SPEC = {
    "true": "TRUE",
    "True": "TRUE",
    "TRUE": "TRUE",
    "supported": "TRUE",
    "Supported": "TRUE",
    "false": "FALSE",
    "False": "FALSE",
    "FALSE": "FALSE",
    "contradicted": "FALSE",
    "Contradicted": "FALSE",
    "mixture": "MIXTURE",
    "Mixture": "MIXTURE",
    "MIXTURE": "MIXTURE",
    "mixed": "MIXTURE",
    "Mixed": "MIXTURE",
    "unproven": "UNPROVEN",
    "Unproven": "UNPROVEN",
    "UNPROVEN": "UNPROVEN",
}

SPEC_LABELS = ["TRUE", "FALSE", "MIXTURE", "UNPROVEN"]

# Reliability scoring (same as retrieval_engine.py)
SOURCE_TIERS = {
    "WHO Guideline": 1.00,
    "CDC Guideline": 0.98,
    "Systematic Review / Meta-Analysis": 0.95,
    "Randomized Controlled Trial (RCT)": 0.90,
    "Cohort Study / Observational": 0.75,
    "Case Report": 0.55,
    "Preprint (bioRxiv/medRxiv)": 0.30,
    "PubMed Article": 0.80,
    "Unverified Blog / Opinion": 0.05,
}


def calculate_reliability_score(source_tier: str, pub_year: int,
                                 disease_category: str = "",
                                 current_year: int = 2026) -> float:
    """Calibrated reliability score R_i ∈ [0, 1]. Same formula as retrieval_engine."""
    tier_weight = SOURCE_TIERS.get(source_tier, 0.50)
    decay_rates = {"Vaccination": 0.10, "Cardiovascular Disease": 0.05, "Diabetes": 0.03}
    decay_rate = decay_rates.get(disease_category, 0.05)
    age = max(0, current_year - pub_year)
    recency_factor = math.exp(-decay_rate * age)
    raw = tier_weight * recency_factor
    return round(min(1.0, max(0.0, raw)), 4)


# ---------------------------------------------------------------------------
# VECTOR RETRIEVER SETUP (Real FAISS Medical Evidence Index)
# ---------------------------------------------------------------------------
VECTOR_STORE_DIR = os.path.join(PROJECT_ROOT, "vector_store")

from app.services.retrieval_engine import FAISSRetriever, calculate_reliability_score


# ---------------------------------------------------------------------------
# B-1: Keyword Baseline (Heuristic rule-based)
# ---------------------------------------------------------------------------
def run_b_minus_1(claim_text: str) -> str:
    """
    B-1: Keyword Heuristic baseline.
    Classifies purely based on explicit lexical signals in the claim string.
    """
    lower = claim_text.lower()

    contra_signals = [
        "no evidence", "does not", "cannot", "myth", "false", "disproven",
        "not effective", "no link", "no connection", "debunked", "doesn't",
        "not proven", "not true", "ineffective",
    ]
    support_signals = [
        "proven", "effective", "benefit", "reduces", "prevents", "helps",
        "recommended", "approved", "treatment", "cure", "improve",
    ]
    mixture_signals = [
        "may", "some evidence", "mixed", "controversial", "debate",
        "limited", "inconclusive", "unclear",
    ]

    contra_count = sum(1 for s in contra_signals if s in lower)
    support_count = sum(1 for s in support_signals if s in lower)
    mixture_count = sum(1 for s in mixture_signals if s in lower)

    if contra_count > support_count and contra_count > mixture_count:
        return "FALSE"
    elif support_count > contra_count and support_count > mixture_count:
        return "TRUE"
    elif mixture_count > 0:
        return "MIXTURE"
    else:
        return "UNPROVEN"


# ---------------------------------------------------------------------------
# B0: LLM-Only Baseline (Parametric inference, NO external evidence)
# ---------------------------------------------------------------------------
def run_b0_llm(claim_text: str, stance_detector: NLIStanceDetector) -> str:
    """
    B0: LLM-Only baseline (Master Instructions §14).
    Evaluates the claim using parametric language model representations alone.
    No retrieval, no external documents, zero external evidence.

    Uses zero-shot NLI hypothesis testing directly against the claim:
      - TRUE: "This medical claim is scientifically true and accurate."
      - FALSE: "This medical claim is false, dangerous, or refuted."
      - MIXTURE: "This medical claim contains mixed true and false aspects."
      - UNPROVEN: "This medical claim is unproven and lacks sufficient scientific evidence."
    """
    hypotheses = {
        "TRUE": "This medical claim is scientifically true, established, and accurate.",
        "FALSE": "This medical claim is scientifically false, refuted, or debunked.",
        "MIXTURE": "This medical claim has mixed or conflicting evidence, both true and false.",
        "UNPROVEN": "This medical claim is unproven, speculative, or lacks sufficient evidence.",
    }

    scores = {}
    for label, hyp in hypotheses.items():
        res = stance_detector.detect_stance(claim=hyp, evidence_text=claim_text)
        scores[label] = res.get("entailment_prob", res.get("confidence", 0.0))

    best_label = max(scores, key=scores.get)
    return best_label


# ---------------------------------------------------------------------------
# Real Evidence Retrieval (FAISS Index over 4,646 Medical Passages)
# ---------------------------------------------------------------------------
def retrieve_real_evidence(claim_text: str, disease_category: str,
                           retriever: FAISSRetriever, top_k: int = 5) -> List[Dict]:
    """
    Retrieve real medical literature chunks from the FAISS knowledge base.
    Uses dense semantic vector search (all-MiniLM-L6-v2) across the 4,646
    pre-indexed clinical and epidemiological evidence chunks.
    """
    # 1. Search with disease filter first
    hits = retriever.search(claim_text, top_k=top_k, disease_filter=disease_category)

    # 2. If fewer than top_k, broaden search to full index
    if len(hits) < top_k:
        broad_hits = retriever.search(claim_text, top_k=top_k)
        seen_ids = {h.get("doc_id", h.get("vector_id")) for h in hits}
        for bh in broad_hits:
            bid = bh.get("doc_id", bh.get("vector_id"))
            if bid not in seen_ids:
                hits.append(bh)
                seen_ids.add(bid)
            if len(hits) >= top_k:
                break

    evidence_items = []
    for i, hit in enumerate(hits[:top_k]):
        tier = hit.get("source_tier", "PubMed Article")
        pub_year = int(hit.get("publication_year", 2024))
        disease = hit.get("disease_category", disease_category)
        r_i = calculate_reliability_score(tier, pub_year, disease)

        evidence_items.append({
            "id": f"ev-{hit.get('doc_id', i+1)}",
            "title": f"Evidence {i+1}: {hit.get('chunk_text', '')[:60]}...",
            "source_tier": tier,
            "pub_year": pub_year,
            "disease_category": disease,
            "chunk_text": hit.get("chunk_text", ""),
            "source": hit.get("source_name", "FAISS_Corpus"),
            "cosine_similarity": float(hit.get("cosine_similarity", hit.get("similarity", 0.75))),
            "reliability_score": r_i,
        })

    return evidence_items


def run_b1_consensus(claim_text: str, evidence_items: List[Dict],
                     stance_detector: NLIStanceDetector,
                     consensus_engine: ConsensusEngine) -> Tuple[str, Dict]:
    """
    B1: Conventional RAG. All evidence weighted equally (W_i = 1.0).
    No reliability or population weighting.
    """
    evidence_with_stances = []
    for ev in evidence_items:
        ev_text = ev.get("chunk_text", ev.get("title", ""))
        stance_result = stance_detector.detect_stance(claim_text, ev_text)
        evidence_with_stances.append({
            **ev,
            "reliability_score": 1.0,  # B1: uniform weight
            "applicability_score": 1.0,
            "stance": stance_result["stance"],
            "stance_value": stance_result["stance_value"],
            "nli_confidence": stance_result["confidence"],
        })

    result = consensus_engine.calculate_weighted_consensus(evidence_with_stances)
    spec_label = result.get("spec_label", map_to_spec_label(result["verdict"]))
    return spec_label, result


def run_b2_consensus(claim_text: str, evidence_items: List[Dict],
                     stance_detector: NLIStanceDetector,
                     consensus_engine: ConsensusEngine) -> Tuple[str, Dict]:
    """
    B2: Reliability-Aware RAG. W_i = R_i (no population weighting).
    """
    evidence_with_stances = []
    for ev in evidence_items:
        ev_text = ev.get("chunk_text", ev.get("title", ""))
        stance_result = stance_detector.detect_stance(claim_text, ev_text)
        evidence_with_stances.append({
            **ev,
            "applicability_score": 1.0,  # B2: no population weighting
            "stance": stance_result["stance"],
            "stance_value": stance_result["stance_value"],
            "nli_confidence": stance_result["confidence"],
        })

    result = consensus_engine.calculate_weighted_consensus(evidence_with_stances)
    spec_label = result.get("spec_label", map_to_spec_label(result["verdict"]))
    return spec_label, result


def run_b3_consensus(claim_text: str, evidence_items: List[Dict],
                     stance_detector: NLIStanceDetector,
                     consensus_engine: ConsensusEngine) -> Tuple[str, Dict]:
    """
    B3: Full MedVerify. W_i = R_i × P_i (population-weighted).
    """
    evidence_with_stances = []
    for ev in evidence_items:
        ev_text = ev.get("chunk_text", ev.get("title", ""))
        stance_result = stance_detector.detect_stance(claim_text, ev_text)

        # Compute P_i via population module
        try:
            app_result = analyze_population_applicability(claim_text, ev)
            p_i = app_result.score
            pop_match_type = app_result.match_result.match_type.value if app_result.match_result else "UNKNOWN"
        except Exception as e:
            logger.warning(f"Population analysis failed for claim: {e}")
            p_i = 0.5  # Neutral when unable to determine
            pop_match_type = "UNKNOWN"

        evidence_with_stances.append({
            **ev,
            "applicability_score": p_i,  # B3: population-weighted
            "population_match_type": pop_match_type,
            "stance": stance_result["stance"],
            "stance_value": stance_result["stance_value"],
            "nli_confidence": stance_result["confidence"],
        })

    result = consensus_engine.calculate_weighted_consensus(evidence_with_stances)
    spec_label = result.get("spec_label", map_to_spec_label(result["verdict"]))
    return spec_label, result


# ---------------------------------------------------------------------------
# METRICS COMPUTATION
# ---------------------------------------------------------------------------
def compute_metrics(y_true: List[str], y_pred: List[str],
                    labels: List[str] = None) -> Dict:
    """Compute all §17 required metrics."""
    if labels is None:
        labels = SPEC_LABELS

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    precision = precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    recall_val = recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)

    # Per-class F1
    per_class = {}
    for label in labels:
        binary_true = [1 if y == label else 0 for y in y_true]
        binary_pred = [1 if y == label else 0 for y in y_pred]
        per_class[label] = round(f1_score(binary_true, binary_pred, zero_division=0), 4)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall_val, 4),
        "per_class_f1": per_class,
        "confusion_matrix": cm,
        "confusion_matrix_labels": labels,
        "sample_size": len(y_true),
    }


def bootstrap_ci(y_true: List[str], y_pred: List[str],
                  n_bootstrap: int = 1000, seed: int = 42,
                  metric_fn=None) -> Tuple[float, float, float]:
    """Compute bootstrap 95% confidence interval for Macro-F1."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_bootstrap):
        indices = rng.randint(0, n, size=n)
        yt = [y_true[i] for i in indices]
        yp = [y_pred[i] for i in indices]
        if metric_fn:
            scores.append(metric_fn(yt, yp))
        else:
            scores.append(f1_score(yt, yp, average="macro", labels=SPEC_LABELS, zero_division=0))
    lower = float(np.percentile(scores, 2.5))
    upper = float(np.percentile(scores, 97.5))
    mean_val = float(np.mean(scores))
    return mean_val, lower, upper


def mcnemar_test(y_true: List[str], pred_a: List[str], pred_b: List[str]) -> Dict:
    """
    McNemar's test comparing two classifiers (§18).
    Returns 2x2 contingency table, chi² statistic, and p-value.
    """
    a_correct = [p == g for p, g in zip(pred_a, y_true)]
    b_correct = [p == g for p, g in zip(pred_b, y_true)]

    # 2x2 table
    both_correct = sum(1 for ac, bc in zip(a_correct, b_correct) if ac and bc)
    a_only = sum(1 for ac, bc in zip(a_correct, b_correct) if ac and not bc)
    b_only = sum(1 for ac, bc in zip(a_correct, b_correct) if not ac and bc)
    both_wrong = sum(1 for ac, bc in zip(a_correct, b_correct) if not ac and not bc)

    # McNemar chi² with continuity correction
    if (a_only + b_only) > 0:
        chi2 = ((abs(b_only - a_only) - 1.0) ** 2) / (a_only + b_only)
        p_value = float(stats.chi2.sf(chi2, df=1))
    else:
        chi2 = 0.0
        p_value = 1.0

    return {
        "contingency_table": {
            "both_correct": both_correct,
            "a_correct_b_wrong": a_only,
            "b_correct_a_wrong": b_only,
            "both_wrong": both_wrong,
        },
        "chi2_statistic": round(chi2, 4),
        "p_value": p_value,
        "statistically_significant": bool(p_value < 0.05),
    }


def verdict_change_analysis(y_true: List[str], pred_a: List[str],
                             pred_b: List[str]) -> Dict:
    """Verdict Change Rate and Correct Change Rate (§20)."""
    total = len(y_true)
    changes = sum(1 for pa, pb in zip(pred_a, pred_b) if pa != pb)
    correct_changes = sum(
        1 for pa, pb, g in zip(pred_a, pred_b, y_true)
        if pa != pb and pb == g
    )
    incorrect_changes = sum(
        1 for pa, pb, g in zip(pred_a, pred_b, y_true)
        if pa != pb and pb != g
    )
    vcr = (changes / total * 100.0) if total > 0 else 0.0
    ccr = (correct_changes / changes * 100.0) if changes > 0 else 0.0

    return {
        "total_claims": total,
        "verdict_changes": changes,
        "correct_changes": correct_changes,
        "incorrect_changes": incorrect_changes,
        "VCR_percent": round(vcr, 2),
        "CCR_percent": round(ccr, 2),
    }


# ---------------------------------------------------------------------------
# MAIN EXPERIMENT RUNNER
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MedVerify Full Baseline Comparison (B0-B3)")
    parser.add_argument("--sample", type=int, default=0,
                        help="Number of test items to evaluate (0 = all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("=" * 80)
    print("MEDVERIFY AI — FULL BASELINE COMPARISON (B0, B1, B2, B3)")
    print(f"Spec §14, §17, §18, §20 | Seed: {args.seed}")
    print("=" * 80)

    # Load frozen test set
    if not os.path.exists(TEST_SET_PATH):
        raise FileNotFoundError(f"Test set not found: {TEST_SET_PATH}")

    with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # Normalize gold labels to spec labels
    test_items = []
    skipped = 0
    for item in test_data:
        gold_raw = item.get("fact_check_verdict", "")
        gold_spec = PUBHEALTH_TO_SPEC.get(gold_raw)
        if gold_spec is None:
            skipped += 1
            continue
        test_items.append({
            "claim_id": item["claim_id"],
            "claim_text": item["claim_text"],
            "disease_category": item["disease_category"],
            "gold_label": gold_spec,
            "gold_raw": gold_raw,
        })

    if args.sample > 0:
        rng = np.random.RandomState(args.seed)
        indices = rng.choice(len(test_items), size=min(args.sample, len(test_items)), replace=False)
        test_items = [test_items[i] for i in sorted(indices)]

    print(f"Test items: {len(test_items)} (skipped {skipped} with unknown labels)")
    print(f"Gold label distribution: {Counter(x['gold_label'] for x in test_items)}")

    # Initialize NLI model (shared across B0, B1, B2, B3)
    print("\nLoading NLI stance detection model...")
    stance_detector = NLIStanceDetector(model_name="cross-encoder/nli-deberta-v3-small")
    consensus_engine = ConsensusEngine()
    print("[OK] NLI model loaded.")

    # Initialize Real FAISS Knowledge Base Retriever
    print("Loading FAISS Medical Evidence Retriever...")
    retriever = FAISSRetriever(vector_store_dir=VECTOR_STORE_DIR)
    print(f"[OK] FAISS Retriever loaded: {retriever.index.ntotal} evidence vectors.\n")

    # Run all baselines
    gold_labels = []
    bm1_preds, b0_preds, b1_preds, b2_preds, b3_preds = [], [], [], [], []
    detailed_results = []

    start_time = time.time()

    for idx, item in enumerate(test_items):
        claim_text = item["claim_text"]
        disease = item["disease_category"]
        gold = item["gold_label"]
        gold_labels.append(gold)

        # 1. B-1: Keyword Baseline (Heuristic rules)
        bm1_pred = run_b_minus_1(claim_text)
        bm1_preds.append(bm1_pred)

        # 2. B0: LLM Only (Parametric, NO retrieval)
        b0_pred = run_b0_llm(claim_text, stance_detector)
        b0_preds.append(b0_pred)

        # 3. Retrieve REAL evidence from FAISS knowledge base (shared across B1, B2, B3)
        evidence = retrieve_real_evidence(claim_text, disease, retriever, top_k=5)

        # 4. B1: Conventional RAG (uniform weights)
        b1_pred, b1_detail = run_b1_consensus(
            claim_text, evidence, stance_detector, consensus_engine
        )
        b1_preds.append(b1_pred)

        # 5. B2: Reliability-Aware RAG (R_i only)
        b2_pred, b2_detail = run_b2_consensus(
            claim_text, evidence, stance_detector, consensus_engine
        )
        b2_preds.append(b2_pred)

        # 6. B3: Full MedVerify (R_i × P_i)
        b3_pred, b3_detail = run_b3_consensus(
            claim_text, evidence, stance_detector, consensus_engine
        )
        b3_preds.append(b3_pred)

        detailed_results.append({
            "claim_id": item["claim_id"],
            "disease_category": disease,
            "gold": gold,
            "b_minus_1": bm1_pred,
            "b0": b0_pred,
            "b1": b1_pred,
            "b2": b2_pred,
            "b3": b3_pred,
            "b2_consensus": b2_detail.get("weighted_consensus", 0.0),
            "b3_consensus": b3_detail.get("weighted_consensus", 0.0),
            "b2_to_b3_changed": b2_pred != b3_pred,
            "retrieved_evidence_count": len(evidence),
        })

        if (idx + 1) % 25 == 0 or (idx + 1) == len(test_items):
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{idx+1}/{len(test_items)}] Elapsed: {elapsed:.1f}s ({rate:.1f} claims/s)")

    total_time = time.time() - start_time

    # Compute metrics for each baseline
    print("\n" + "=" * 80)
    print("COMPUTING METRICS...")
    print("=" * 80)

    m_bm1 = compute_metrics(gold_labels, bm1_preds)
    m_b0 = compute_metrics(gold_labels, b0_preds)
    m_b1 = compute_metrics(gold_labels, b1_preds)
    m_b2 = compute_metrics(gold_labels, b2_preds)
    m_b3 = compute_metrics(gold_labels, b3_preds)

    # B2 vs B3 statistical comparison (§18)
    delta_macro_f1 = round((m_b3["macro_f1"] - m_b2["macro_f1"]) * 100, 2)
    delta_accuracy = round((m_b3["accuracy"] - m_b2["accuracy"]) * 100, 2)

    mcnemar = mcnemar_test(gold_labels, b2_preds, b3_preds)

    # Bootstrap CI for B3 Macro-F1
    b3_mean, b3_ci_lo, b3_ci_hi = bootstrap_ci(gold_labels, b3_preds, seed=args.seed)

    # Verdict change analysis (§20)
    vca = verdict_change_analysis(gold_labels, b2_preds, b3_preds)

    # Print results
    print(f"\n{'System':<30} | {'Accuracy':<10} | {'Macro-F1':<10} | {'Prec':<10} | {'Recall':<10}")
    print("-" * 80)
    for name, m in [("B-1: Keyword Baseline", m_bm1),
                    ("B0: LLM Only (Zero-Shot)", m_b0),
                    ("B1: Conventional RAG", m_b1),
                    ("B2: Reliability RAG", m_b2),
                    ("B3: Full MedVerify", m_b3)]:
        print(f"{name:<30} | {m['accuracy']*100:7.2f}%  | {m['macro_f1']*100:7.2f}%  | "
              f"{m['precision']*100:7.2f}%  | {m['recall']*100:7.2f}%")
    print("=" * 80)

    print(f"\nB2 vs B3 (S18):")
    print(f"  Delta Accuracy:  {'+' if delta_accuracy >= 0 else ''}{delta_accuracy}%")
    print(f"  Delta Macro-F1:  {'+' if delta_macro_f1 >= 0 else ''}{delta_macro_f1}%")
    print(f"  McNemar chi2:  {mcnemar['chi2_statistic']}")
    print(f"  p-value:     {mcnemar['p_value']:.6f}")
    print(f"  Significant: {mcnemar['statistically_significant']}")
    print(f"  B3 F1 95% CI: [{b3_ci_lo:.4f}, {b3_ci_hi:.4f}]")

    print(f"\nVerdict Change Analysis (S20):")
    print(f"  VCR: {vca['VCR_percent']}%")
    print(f"  CCR: {vca['CCR_percent']}%")

    # Build full report
    report = {
        "experiment_id": f"full-baseline-{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "random_seed": args.seed,
        "dataset": {
            "path": TEST_SET_PATH,
            "total_items": len(test_items),
            "label_distribution": dict(Counter(gold_labels)),
        },
        "configuration": {
            "nli_model": "cross-encoder/nli-deberta-v3-small",
            "evidence_per_claim": 5,
            "evidence_mode": "faiss_vector_store_real",
            "vector_store_path": VECTOR_STORE_DIR,
            "labels": SPEC_LABELS,
        },
        "baselines": {
            "B_minus_1_Keyword": m_bm1,
            "B0_LLM_Only": m_b0,
            "B1_Conventional_RAG": m_b1,
            "B2_Reliability_RAG": m_b2,
            "B3_Full_MedVerify": m_b3,
        },
        "b2_vs_b3_comparison": {
            "delta_accuracy_percent": delta_accuracy,
            "delta_macro_f1_percent": delta_macro_f1,
            "mcnemar_test": mcnemar,
            "b3_bootstrap_ci_95": {
                "mean": round(b3_mean, 4),
                "lower": round(b3_ci_lo, 4),
                "upper": round(b3_ci_hi, 4),
            },
        },
        "verdict_change_analysis": vca,
        "elapsed_seconds": round(total_time, 2),
        "detailed_results": detailed_results[:50],  # First 50 for traceability
    }

    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[OK] Full report saved to: {OUTPUT_PATH}")
    print("=" * 80)

    return report


if __name__ == "__main__":
    main()
