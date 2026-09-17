"""
MedVerify AI - Primary Contribution Evaluation
================================================
Evaluates the central research contribution:
    W_i = R_i * P_i   (population-aware evidence weighting)

Tasks:
  1. Validate P_i against N=70 ground truth
  2. Audit per-evidence weighting effect (B2 vs B3)
  3. B2 vs B3 on frozen 614-claim test set
  4. Population-mismatch stratified evaluation
  5. Trace every B2->B3 verdict change
  6. Synthesize final honest conclusion

Constraints:
  - Keep current BioBERT classifier
  - Keep current 7,657-chunk Knowledge Base V2
  - Keep frozen 614-claim test set unchanged
  - Do not add new models or tune on the test set
  - Do not change R_i, P_i, or decision thresholds

Usage:
    python experiments/evaluate_primary_contribution.py [--sample N] [--seed 42]
"""

import os
import sys
import json
import math
import time
import logging
import argparse
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, cohen_kappa_score,
    mean_absolute_error,
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
)
from app.services.population.applicability import (
    analyze_population_applicability,
    compute_applicability,
)
from app.services.population.schemas import (
    PopulationProfile, MatchResult, MatchType, ApplicabilityResult,
)
from app.services.population.matcher import match_populations
from app.services.retrieval_engine import FAISSRetriever, calculate_reliability_score

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
TEST_SET_PATH = os.path.join(PROJECT_ROOT, "med_datasets", "splits", "test_frozen.json")
POP_GT_PATH = os.path.join(PROJECT_ROOT, "med_datasets", "evaluation", "population_ground_truth_70.json")
VECTOR_STORE_DIR = os.path.join(PROJECT_ROOT, "vector_store")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

SPEC_LABELS = ["TRUE", "FALSE", "MIXTURE", "UNPROVEN"]

PUBHEALTH_TO_SPEC = {
    "true": "TRUE", "True": "TRUE", "TRUE": "TRUE",
    "supported": "TRUE", "Supported": "TRUE",
    "false": "FALSE", "False": "FALSE", "FALSE": "FALSE",
    "contradicted": "FALSE", "Contradicted": "FALSE",
    "mixture": "MIXTURE", "Mixture": "MIXTURE", "MIXTURE": "MIXTURE",
    "mixed": "MIXTURE", "Mixed": "MIXTURE",
    "unproven": "UNPROVEN", "Unproven": "UNPROVEN", "UNPROVEN": "UNPROVEN",
}

# Scoring mode for P_i - use the frozen 4-dimension formula
PI_SCORING_MODE = "v3_full"


# ============================================================================
# TASK 1: VALIDATE P_i (N=70 Ground Truth)
# ============================================================================

def task1_validate_pi() -> Dict:
    """
    Validate P_i against N=70 population ground truth benchmark.
    Reports: N, Accuracy, Macro-F1, Cohen's Kappa, MAE.
    """
    print("\n" + "=" * 80)
    print("TASK 1: VALIDATE P_i (Population Applicability Score)")
    print("=" * 80)

    with open(POP_GT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("dataset", [])
    n = len(items)
    print(f"  Ground truth benchmark: N={n}")

    # --- (A) Pipeline extraction validation ---
    # Run the actual pipeline to compute P_i from claim + study text
    pipeline_pred_scores = []
    pipeline_pred_labels = []
    gold_scores = []
    gold_labels = []
    ann1_discrete = []
    ann2_discrete = []

    # --- (B) Gold-dimension formula validation ---
    formula_pred_scores = []
    formula_pred_labels = []

    for item in items:
        gold_p = item["gold_P_i"]
        gold_lbl = item["compatibility_label"]
        ann1_p = item.get("annotator_1_P_i", gold_p)
        ann2_p = item.get("annotator_2_P_i", gold_p)

        gold_scores.append(gold_p)
        gold_labels.append(gold_lbl)

        # Discretize annotator scores for Cohen's Kappa
        ann1_discrete.append(
            "COMPATIBLE" if ann1_p >= 0.75
            else ("PARTIAL" if ann1_p >= 0.50 else "MISMATCH")
        )
        ann2_discrete.append(
            "COMPATIBLE" if ann2_p >= 0.75
            else ("PARTIAL" if ann2_p >= 0.50 else "MISMATCH")
        )

        # (A) Pipeline extraction: build profiles from text descriptions
        c_dict = item["claim_population"]
        s_dict = item["study_population"]

        c_prof = PopulationProfile(
            raw_text_snippet=str(c_dict),
            sex=c_dict.get("sex") if c_dict.get("sex") not in ("unspecified", None) else None,
            region=c_dict.get("region") if c_dict.get("region") not in ("unspecified", None) else None,
        )
        s_prof = PopulationProfile(
            raw_text_snippet=str(s_dict),
            sex=s_dict.get("sex") if s_dict.get("sex") not in ("unspecified", None) else None,
            region=s_dict.get("region") if s_dict.get("region") not in ("unspecified", None) else None,
        )

        match_res = match_populations(c_prof, s_prof)
        app_res = compute_applicability(c_prof, s_prof, match_res, scoring_mode=PI_SCORING_MODE)
        pipeline_pred_scores.append(app_res.score)
        pipeline_lbl = (
            "COMPATIBLE" if app_res.score >= 0.75
            else ("PARTIAL" if app_res.score >= 0.50 else "MISMATCH")
        )
        pipeline_pred_labels.append(pipeline_lbl)

        # (B) Gold-dimension formula validation
        a_i = item.get("gold_A_i", 0.5)
        s_i = item.get("gold_S_i", 0.5)
        c_i = item.get("gold_C_i", 0.5)
        g_i = item.get("gold_G_i", 0.5)
        # v3_full frozen formula: P_i = (A_i + S_i + C_i + G_i) / 4.0
        formula_p = (a_i + s_i + c_i + g_i) / 4.0
        formula_pred_scores.append(formula_p)
        formula_lbl = (
            "COMPATIBLE" if formula_p >= 0.75
            else ("PARTIAL" if formula_p >= 0.50 else "MISMATCH")
        )
        formula_pred_labels.append(formula_lbl)

    # Compute metrics
    class_names = ["MISMATCH", "PARTIAL", "COMPATIBLE"]

    # Pipeline metrics
    pipe_acc = accuracy_score(gold_labels, pipeline_pred_labels)
    pipe_f1 = f1_score(gold_labels, pipeline_pred_labels, average="macro",
                       labels=class_names, zero_division=0)
    pipe_mae = mean_absolute_error(gold_scores, pipeline_pred_scores)

    # Formula metrics (gold dimensions -> frozen formula)
    form_acc = accuracy_score(gold_labels, formula_pred_labels)
    form_f1 = f1_score(gold_labels, formula_pred_labels, average="macro",
                       labels=class_names, zero_division=0)
    form_mae = mean_absolute_error(gold_scores, formula_pred_scores)

    # Cohen's Kappa (inter-annotator agreement)
    kappa = cohen_kappa_score(ann1_discrete, ann2_discrete)

    # Confusion matrix (formula-based, as that's the primary validation)
    cm = confusion_matrix(gold_labels, formula_pred_labels, labels=class_names).tolist()

    result = {
        "task": "Task 1: P_i Validation",
        "ground_truth_N": n,
        "scoring_mode": PI_SCORING_MODE,
        "label_distribution": dict(Counter(gold_labels)),
        "formula_validation": {
            "description": "P_i = (A_i + S_i + C_i + G_i) / 4.0 using gold dimension scores",
            "accuracy": round(form_acc, 4),
            "macro_f1": round(form_f1, 4),
            "mae": round(form_mae, 4),
            "confusion_matrix": cm,
            "confusion_matrix_labels": class_names,
        },
        "pipeline_validation": {
            "description": "P_i computed via pipeline extraction (regex-based) from text",
            "accuracy": round(pipe_acc, 4),
            "macro_f1": round(pipe_f1, 4),
            "mae": round(pipe_mae, 4),
        },
        "inter_annotator_agreement": {
            "cohens_kappa": round(kappa, 4),
            "interpretation": (
                "almost perfect" if kappa >= 0.81
                else "substantial" if kappa >= 0.61
                else "moderate" if kappa >= 0.41
                else "fair" if kappa >= 0.21
                else "slight" if kappa >= 0.0
                else "poor"
            ),
        },
    }

    print(f"\n  Formula Validation (gold dimensions):")
    print(f"    Accuracy:  {form_acc * 100:.2f}%")
    print(f"    Macro-F1:  {form_f1 * 100:.2f}%")
    print(f"    MAE:       {form_mae:.4f}")
    print(f"  Pipeline Validation (text extraction):")
    print(f"    Accuracy:  {pipe_acc * 100:.2f}%")
    print(f"    Macro-F1:  {pipe_f1 * 100:.2f}%")
    print(f"    MAE:       {pipe_mae:.4f}")
    print(f"  Cohen's Kappa: {kappa:.4f} ({result['inter_annotator_agreement']['interpretation']})")

    return result


# ============================================================================
# EVIDENCE RETRIEVAL & STANCE DETECTION (shared by Tasks 2-5)
# ============================================================================

def retrieve_evidence(claim_text: str, disease_category: str,
                      retriever: FAISSRetriever, top_k: int = 5) -> List[Dict]:
    """Retrieve real evidence from FAISS knowledge base."""
    hits = retriever.search(claim_text, top_k=top_k, disease_filter=disease_category)

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
            "title": hit.get("chunk_text", "")[:80],
            "source_tier": tier,
            "pub_year": pub_year,
            "disease_category": disease,
            "chunk_text": hit.get("chunk_text", ""),
            "source": hit.get("source_name", "FAISS_Corpus"),
            "cosine_similarity": float(hit.get("cosine_similarity", hit.get("similarity", 0.75))),
            "reliability_score": r_i,
        })

    return evidence_items


def run_b2_with_audit(claim_text: str, evidence_items: List[Dict],
                      stance_detector: NLIStanceDetector,
                      consensus_engine: ConsensusEngine) -> Tuple[str, Dict, List[Dict]]:
    """B2: W_i = R_i. Returns (spec_label, consensus_result, evidence_with_stances)."""
    evidence_with_stances = []
    for ev in evidence_items:
        ev_text = ev.get("chunk_text", ev.get("title", ""))
        stance_result = stance_detector.detect_stance(claim_text, ev_text)
        evidence_with_stances.append({
            **ev,
            "applicability_score": 1.0,  # B2: no population weighting
            "population_match_type": "N/A",
            "p_age": None, "p_sex": None, "p_condition": None, "p_region": None,
            "stance": stance_result["stance"],
            "stance_value": stance_result["stance_value"],
            "nli_confidence": stance_result["confidence"],
        })

    result = consensus_engine.calculate_weighted_consensus(evidence_with_stances)
    spec_label = result.get("spec_label", map_to_spec_label(result["verdict"]))
    return spec_label, result, evidence_with_stances


def run_b3_with_audit(claim_text: str, evidence_items: List[Dict],
                      stance_detector: NLIStanceDetector,
                      consensus_engine: ConsensusEngine) -> Tuple[str, Dict, List[Dict]]:
    """B3: W_i = R_i * P_i. Returns (spec_label, consensus_result, evidence_with_stances)."""
    evidence_with_stances = []
    for ev in evidence_items:
        ev_text = ev.get("chunk_text", ev.get("title", ""))
        stance_result = stance_detector.detect_stance(claim_text, ev_text)

        # Compute P_i via population module
        try:
            app_result = analyze_population_applicability(
                claim_text, ev, scoring_mode=PI_SCORING_MODE
            )
            p_i = app_result.score
            pop_match_type = (
                app_result.match_result.match_type.value
                if app_result.match_result else "UNKNOWN"
            )
            p_age = app_result.p_age
            p_sex = app_result.p_sex
            p_condition = app_result.p_condition
            p_region = app_result.p_region
        except Exception as e:
            logger.warning(f"Population analysis failed: {e}")
            p_i = 0.5
            pop_match_type = "UNKNOWN"
            p_age = p_sex = p_condition = p_region = 0.5

        evidence_with_stances.append({
            **ev,
            "applicability_score": p_i,
            "population_match_type": pop_match_type,
            "p_age": p_age, "p_sex": p_sex,
            "p_condition": p_condition, "p_region": p_region,
            "stance": stance_result["stance"],
            "stance_value": stance_result["stance_value"],
            "nli_confidence": stance_result["confidence"],
        })

    result = consensus_engine.calculate_weighted_consensus(evidence_with_stances)
    spec_label = result.get("spec_label", map_to_spec_label(result["verdict"]))
    return spec_label, result, evidence_with_stances


# ============================================================================
# METRICS HELPERS
# ============================================================================

def compute_metrics(y_true: List[str], y_pred: List[str],
                    labels: List[str] = None) -> Dict:
    """Compute full classification metrics."""
    if labels is None:
        labels = SPEC_LABELS

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    precision = precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    recall_val = recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)

    per_class = {}
    for label in labels:
        bt = [1 if y == label else 0 for y in y_true]
        bp = [1 if y == label else 0 for y in y_pred]
        per_class[label] = round(f1_score(bt, bp, zero_division=0), 4)

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
                  n_bootstrap: int = 2000, seed: int = 42) -> Dict:
    """95% bootstrap confidence interval for Macro-F1."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        yt = [y_true[i] for i in idx]
        yp = [y_pred[i] for i in idx]
        scores.append(f1_score(yt, yp, average="macro", labels=SPEC_LABELS, zero_division=0))
    return {
        "mean": round(float(np.mean(scores)), 4),
        "ci_lower": round(float(np.percentile(scores, 2.5)), 4),
        "ci_upper": round(float(np.percentile(scores, 97.5)), 4),
        "std": round(float(np.std(scores)), 4),
    }


def bootstrap_paired_delta(y_true, pred_a, pred_b,
                            n_bootstrap=2000, seed=42) -> Dict:
    """Bootstrap CI for Delta Macro-F1 = F1(B3) - F1(B2)."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    deltas = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        yt = [y_true[i] for i in idx]
        pa = [pred_a[i] for i in idx]
        pb = [pred_b[i] for i in idx]
        f1_a = f1_score(yt, pa, average="macro", labels=SPEC_LABELS, zero_division=0)
        f1_b = f1_score(yt, pb, average="macro", labels=SPEC_LABELS, zero_division=0)
        deltas.append(f1_b - f1_a)
    return {
        "mean_delta": round(float(np.mean(deltas)), 4),
        "ci_lower": round(float(np.percentile(deltas, 2.5)), 4),
        "ci_upper": round(float(np.percentile(deltas, 97.5)), 4),
        "std": round(float(np.std(deltas)), 4),
    }


def mcnemar_test(y_true: List[str], pred_a: List[str], pred_b: List[str]) -> Dict:
    """McNemar's test comparing B2 vs B3."""
    a_correct = [p == g for p, g in zip(pred_a, y_true)]
    b_correct = [p == g for p, g in zip(pred_b, y_true)]

    both_correct = sum(1 for ac, bc in zip(a_correct, b_correct) if ac and bc)
    a_only = sum(1 for ac, bc in zip(a_correct, b_correct) if ac and not bc)
    b_only = sum(1 for ac, bc in zip(a_correct, b_correct) if not ac and bc)
    both_wrong = sum(1 for ac, bc in zip(a_correct, b_correct) if not ac and not bc)

    # McNemar chi-squared with continuity correction
    if (a_only + b_only) > 0:
        chi2 = ((abs(b_only - a_only) - 1.0) ** 2) / (a_only + b_only)
        p_value = float(stats.chi2.sf(chi2, df=1))
    else:
        chi2 = 0.0
        p_value = 1.0

    # Exact McNemar (binomial) for small discordant counts
    if (a_only + b_only) <= 25 and (a_only + b_only) > 0:
        exact_p = float(stats.binom_test(
            min(a_only, b_only), a_only + b_only, 0.5
        )) if hasattr(stats, 'binom_test') else None
        if exact_p is None:
            # scipy >= 1.7 uses binomtest
            try:
                exact_result = stats.binomtest(
                    min(a_only, b_only), a_only + b_only, 0.5
                )
                exact_p = float(exact_result.pvalue)
            except Exception:
                exact_p = None
    else:
        exact_p = None

    return {
        "contingency_table": {
            "both_correct": both_correct,
            "b2_correct_b3_wrong": a_only,
            "b3_correct_b2_wrong": b_only,
            "both_wrong": both_wrong,
        },
        "discordant_pairs": a_only + b_only,
        "chi2_statistic": round(chi2, 4),
        "p_value_chi2": p_value,
        "p_value_exact_binomial": exact_p,
        "statistically_significant_005": bool(p_value < 0.05),
    }


# ============================================================================
# MAIN EVALUATION PIPELINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MedVerify Primary Contribution Evaluation"
    )
    parser.add_argument("--sample", type=int, default=0,
                        help="Number of test items to evaluate (0 = all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("=" * 80)
    print("MEDVERIFY AI - PRIMARY CONTRIBUTION EVALUATION")
    print("  W_i = R_i * P_i   (Population-Aware Evidence Weighting)")
    print(f"  Scoring mode: {PI_SCORING_MODE}")
    print(f"  Seed: {args.seed}")
    print("=" * 80)

    os.makedirs(REPORTS_DIR, exist_ok=True)

    # ===================================================================
    # TASK 1: P_i Validation
    # ===================================================================
    task1_result = task1_validate_pi()

    # ===================================================================
    # Load test set & initialize models
    # ===================================================================
    print("\n" + "=" * 80)
    print("LOADING MODELS & FROZEN TEST SET")
    print("=" * 80)

    with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)

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
        indices = rng.choice(len(test_items),
                             size=min(args.sample, len(test_items)),
                             replace=False)
        test_items = [test_items[i] for i in sorted(indices)]

    n_claims = len(test_items)
    print(f"  Test items: {n_claims} (skipped {skipped} with unknown labels)")
    print(f"  Label distribution: {dict(Counter(x['gold_label'] for x in test_items))}")

    print("\n  Loading NLI model...")
    stance_detector = NLIStanceDetector(model_name="cross-encoder/nli-deberta-v3-small")
    consensus_engine = ConsensusEngine()
    print("  [OK] NLI model loaded.")

    print("  Loading FAISS retriever...")
    retriever = FAISSRetriever(vector_store_dir=VECTOR_STORE_DIR)
    print(f"  [OK] FAISS loaded: {retriever.index.ntotal} vectors.\n")

    # ===================================================================
    # RUN B2 AND B3 ON ALL CLAIMS (Tasks 2-5 data collection)
    # ===================================================================
    print("=" * 80)
    print("RUNNING B2 & B3 ON ALL CLAIMS (full audit mode)")
    print("=" * 80)

    gold_labels = []
    b2_preds = []
    b3_preds = []
    all_detailed = []        # Complete per-claim audit
    all_evidence_audit = []  # Per-evidence weight audit

    start_time = time.time()

    for idx, item in enumerate(test_items):
        claim_text = item["claim_text"]
        disease = item["disease_category"]
        gold = item["gold_label"]
        gold_labels.append(gold)

        # Retrieve evidence (shared by B2 and B3)
        evidence = retrieve_evidence(claim_text, disease, retriever, top_k=5)

        # B2: W_i = R_i
        b2_pred, b2_detail, b2_evidence = run_b2_with_audit(
            claim_text, evidence, stance_detector, consensus_engine
        )
        b2_preds.append(b2_pred)

        # B3: W_i = R_i * P_i
        b3_pred, b3_detail, b3_evidence = run_b3_with_audit(
            claim_text, evidence, stance_detector, consensus_engine
        )
        b3_preds.append(b3_pred)

        # Per-evidence weight audit
        claim_evidence_audit = []
        for b2_ev, b3_ev in zip(b2_evidence, b3_evidence):
            r_i = b2_ev["reliability_score"]
            p_i = b3_ev["applicability_score"]
            w_b2 = r_i * 1.0  # B2: W = R_i
            w_b3 = r_i * p_i  # B3: W = R_i * P_i
            delta_w = w_b2 - w_b3

            audit_entry = {
                "evidence_id": b2_ev["id"],
                "source_tier": b2_ev["source_tier"],
                "R_i": round(r_i, 4),
                "P_i": round(p_i, 4),
                "W_b2": round(w_b2, 4),
                "W_b3": round(w_b3, 4),
                "delta_W": round(delta_w, 4),
                "population_match_type": b3_ev["population_match_type"],
                "p_age": b3_ev["p_age"],
                "p_sex": b3_ev["p_sex"],
                "p_condition": b3_ev["p_condition"],
                "p_region": b3_ev["p_region"],
                "stance": b2_ev["stance"],
                "stance_value": b2_ev["stance_value"],
            }
            claim_evidence_audit.append(audit_entry)

        all_evidence_audit.append({
            "claim_id": item["claim_id"],
            "evidence_items": claim_evidence_audit,
        })

        # Determine population mismatch category for this claim
        p_i_values = [e["applicability_score"] for e in b3_evidence]
        match_types = [e["population_match_type"] for e in b3_evidence]
        avg_pi = np.mean(p_i_values) if p_i_values else 0.5

        if any(mt == "MISMATCHED" for mt in match_types) or any(p < 0.40 for p in p_i_values):
            pop_category = "clear_mismatch"
        elif avg_pi < 0.75 or any(mt == "PARTIAL" for mt in match_types):
            pop_category = "partial_mismatch"
        else:
            pop_category = "population_compatible"

        all_detailed.append({
            "claim_id": item["claim_id"],
            "claim_text": claim_text,
            "disease_category": disease,
            "gold": gold,
            "b2_verdict": b2_pred,
            "b3_verdict": b3_pred,
            "b2_consensus": round(b2_detail.get("weighted_consensus", 0.0), 4),
            "b3_consensus": round(b3_detail.get("weighted_consensus", 0.0), 4),
            "b2_credibility": b2_detail.get("credibility_score", 50.0),
            "b3_credibility": b3_detail.get("credibility_score", 50.0),
            "verdict_changed": b2_pred != b3_pred,
            "population_category": pop_category,
            "avg_P_i": round(avg_pi, 4),
            "evidence_count": len(evidence),
            "evidence_details": [
                {
                    "R_i": round(e["reliability_score"], 4),
                    "P_i": round(e["applicability_score"], 4),
                    "stance": e["stance"],
                    "pop_match": e["population_match_type"],
                }
                for e in b3_evidence
            ],
        })

        if (idx + 1) % 25 == 0 or (idx + 1) == n_claims:
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{idx+1}/{n_claims}] {elapsed:.1f}s ({rate:.2f} claims/s) "
                  f"| B2:{b2_pred} B3:{b3_pred} Gold:{gold}")

    total_time = time.time() - start_time
    print(f"\n  Total evaluation time: {total_time:.1f}s")

    # ===================================================================
    # TASK 2: WEIGHT AUDIT
    # ===================================================================
    print("\n" + "=" * 80)
    print("TASK 2: WEIGHTING MECHANISM AUDIT")
    print("=" * 80)

    total_evidence_items = sum(len(x["evidence_items"]) for x in all_evidence_audit)
    affected_items = sum(
        1 for x in all_evidence_audit
        for e in x["evidence_items"]
        if abs(e["delta_W"]) > 1e-6
    )
    all_deltas = [
        e["delta_W"] for x in all_evidence_audit
        for e in x["evidence_items"]
    ]
    all_p_i = [
        e["P_i"] for x in all_evidence_audit
        for e in x["evidence_items"]
    ]

    nonzero_deltas = [d for d in all_deltas if abs(d) > 1e-6]

    # Claims whose consensus score changed
    consensus_changed = sum(
        1 for d in all_detailed
        if abs(d["b2_consensus"] - d["b3_consensus"]) > 1e-6
    )
    # Claims whose final verdict changed
    verdict_changed_count = sum(1 for d in all_detailed if d["verdict_changed"])

    task2_result = {
        "task": "Task 2: Weighting Mechanism Audit",
        "total_evidence_items": total_evidence_items,
        "affected_evidence_items": affected_items,
        "affected_percent": round(affected_items / total_evidence_items * 100, 2) if total_evidence_items > 0 else 0,
        "weight_difference_stats": {
            "mean_delta_W": round(float(np.mean(all_deltas)), 6) if all_deltas else 0,
            "median_delta_W": round(float(np.median(all_deltas)), 6) if all_deltas else 0,
            "std_delta_W": round(float(np.std(all_deltas)), 6) if all_deltas else 0,
            "mean_nonzero_delta": round(float(np.mean(nonzero_deltas)), 6) if nonzero_deltas else 0,
            "max_delta_W": round(float(max(all_deltas, key=abs)), 6) if all_deltas else 0,
        },
        "P_i_distribution": {
            "mean": round(float(np.mean(all_p_i)), 4),
            "median": round(float(np.median(all_p_i)), 4),
            "std": round(float(np.std(all_p_i)), 4),
            "min": round(float(np.min(all_p_i)), 4),
            "max": round(float(np.max(all_p_i)), 4),
            "pct_equal_1": round(sum(1 for p in all_p_i if abs(p - 1.0) < 1e-6) / len(all_p_i) * 100, 2),
            "pct_below_0.5": round(sum(1 for p in all_p_i if p < 0.5) / len(all_p_i) * 100, 2),
        },
        "consensus_changes": consensus_changed,
        "verdict_changes": verdict_changed_count,
    }

    print(f"  Total evidence items:  {total_evidence_items}")
    print(f"  Affected items (dW!=0): {affected_items} ({task2_result['affected_percent']}%)")
    print(f"  Mean dW:               {task2_result['weight_difference_stats']['mean_delta_W']}")
    print(f"  Mean P_i:              {task2_result['P_i_distribution']['mean']}")
    print(f"  P_i = 1.0:             {task2_result['P_i_distribution']['pct_equal_1']}%")
    print(f"  P_i < 0.5:             {task2_result['P_i_distribution']['pct_below_0.5']}%")
    print(f"  Consensus changes:     {consensus_changed}/{n_claims}")
    print(f"  Verdict changes:       {verdict_changed_count}/{n_claims}")

    # ===================================================================
    # TASK 3: B2 vs B3 EVALUATION
    # ===================================================================
    print("\n" + "=" * 80)
    print("TASK 3: B2 vs B3 EVALUATION (Frozen Test Set)")
    print("=" * 80)

    m_b2 = compute_metrics(gold_labels, b2_preds)
    m_b3 = compute_metrics(gold_labels, b3_preds)

    delta_acc = round((m_b3["accuracy"] - m_b2["accuracy"]) * 100, 4)
    delta_f1 = round((m_b3["macro_f1"] - m_b2["macro_f1"]) * 100, 4)

    # McNemar test
    mcnemar = mcnemar_test(gold_labels, b2_preds, b3_preds)

    # Bootstrap CIs
    b2_ci = bootstrap_ci(gold_labels, b2_preds, seed=args.seed)
    b3_ci = bootstrap_ci(gold_labels, b3_preds, seed=args.seed)
    delta_ci = bootstrap_paired_delta(gold_labels, b2_preds, b3_preds, seed=args.seed)

    # VCR / CCR
    changes = sum(1 for pa, pb in zip(b2_preds, b3_preds) if pa != pb)
    correct_changes = sum(
        1 for pa, pb, g in zip(b2_preds, b3_preds, gold_labels)
        if pa != pb and pb == g
    )
    incorrect_changes = sum(
        1 for pa, pb, g in zip(b2_preds, b3_preds, gold_labels)
        if pa != pb and pb != g and pa != g
    )
    lateral_changes = sum(
        1 for pa, pb, g in zip(b2_preds, b3_preds, gold_labels)
        if pa != pb and pa != g and pb != g
    )
    degraded_changes = sum(
        1 for pa, pb, g in zip(b2_preds, b3_preds, gold_labels)
        if pa != pb and pa == g
    )

    vcr = (changes / n_claims * 100) if n_claims > 0 else 0
    ccr = (correct_changes / changes * 100) if changes > 0 else 0

    task3_result = {
        "task": "Task 3: B2 vs B3 Evaluation",
        "test_set_size": n_claims,
        "B2_metrics": m_b2,
        "B3_metrics": m_b3,
        "delta_accuracy_percent": delta_acc,
        "delta_macro_f1_percent": delta_f1,
        "mcnemar_test": mcnemar,
        "bootstrap_B2_F1_95CI": b2_ci,
        "bootstrap_B3_F1_95CI": b3_ci,
        "bootstrap_delta_F1_95CI": delta_ci,
        "verdict_change_analysis": {
            "total_claims": n_claims,
            "verdict_changes": changes,
            "VCR_percent": round(vcr, 2),
            "correct_changes": correct_changes,
            "degraded_changes": degraded_changes,
            "lateral_changes": lateral_changes,
            "CCR_percent": round(ccr, 2),
        },
    }

    print(f"\n  {'Metric':<25} | {'B2':>10} | {'B3':>10} | {'Delta':>10}")
    print(f"  {'-'*60}")
    print(f"  {'Accuracy':<25} | {m_b2['accuracy']*100:9.2f}% | {m_b3['accuracy']*100:9.2f}% | {delta_acc:+9.2f}%")
    print(f"  {'Macro-F1':<25} | {m_b2['macro_f1']*100:9.2f}% | {m_b3['macro_f1']*100:9.2f}% | {delta_f1:+9.2f}%")
    print(f"  {'Precision':<25} | {m_b2['precision']*100:9.2f}% | {m_b3['precision']*100:9.2f}%")
    print(f"  {'Recall':<25} | {m_b2['recall']*100:9.2f}% | {m_b3['recall']*100:9.2f}%")
    print(f"\n  Per-class F1:")
    for label in SPEC_LABELS:
        print(f"    {label:<10}: B2={m_b2['per_class_f1'].get(label,0):.4f}  "
              f"B3={m_b3['per_class_f1'].get(label,0):.4f}")
    print(f"\n  McNemar test:")
    print(f"    Discordant pairs: {mcnemar['discordant_pairs']}")
    print(f"    chi2: {mcnemar['chi2_statistic']:.4f}")
    print(f"    p-value (chi2): {mcnemar['p_value_chi2']:.6f}")
    if mcnemar['p_value_exact_binomial'] is not None:
        print(f"    p-value (exact): {mcnemar['p_value_exact_binomial']:.6f}")
    print(f"    Significant (a=0.05): {mcnemar['statistically_significant_005']}")
    print(f"\n  Bootstrap 95% CI for Delta Macro-F1: "
          f"[{delta_ci['ci_lower']:.4f}, {delta_ci['ci_upper']:.4f}]")
    print(f"  VCR: {vcr:.2f}% ({changes}/{n_claims})")
    print(f"  CCR: {ccr:.2f}% ({correct_changes}/{changes if changes > 0 else 'N/A'})")

    # ===================================================================
    # TASK 4: POPULATION MISMATCH STRATIFICATION
    # ===================================================================
    print("\n" + "=" * 80)
    print("TASK 4: POPULATION MISMATCH STRATIFICATION")
    print("=" * 80)

    strata = defaultdict(lambda: {"gold": [], "b2": [], "b3": []})
    for d in all_detailed:
        cat = d["population_category"]
        strata[cat]["gold"].append(d["gold"])
        strata[cat]["b2"].append(d["b2_verdict"])
        strata[cat]["b3"].append(d["b3_verdict"])

    task4_result = {
        "task": "Task 4: Population Mismatch Stratification",
        "categories": {},
    }

    for cat in ["population_compatible", "partial_mismatch", "clear_mismatch"]:
        s = strata[cat]
        n_cat = len(s["gold"])
        if n_cat == 0:
            task4_result["categories"][cat] = {"count": 0, "note": "No claims in this category"}
            continue

        m_b2_cat = compute_metrics(s["gold"], s["b2"])
        m_b3_cat = compute_metrics(s["gold"], s["b3"])

        changes_cat = sum(1 for a, b in zip(s["b2"], s["b3"]) if a != b)
        correct_cat = sum(
            1 for a, b, g in zip(s["b2"], s["b3"], s["gold"])
            if a != b and b == g
        )

        cat_result = {
            "count": n_cat,
            "B2_accuracy": m_b2_cat["accuracy"],
            "B3_accuracy": m_b3_cat["accuracy"],
            "delta_accuracy": round((m_b3_cat["accuracy"] - m_b2_cat["accuracy"]) * 100, 2),
            "B2_macro_f1": m_b2_cat["macro_f1"],
            "B3_macro_f1": m_b3_cat["macro_f1"],
            "delta_macro_f1": round((m_b3_cat["macro_f1"] - m_b2_cat["macro_f1"]) * 100, 2),
            "verdict_changes": changes_cat,
            "correct_changes": correct_cat,
        }
        task4_result["categories"][cat] = cat_result

        print(f"\n  [{cat}] N={n_cat}")
        print(f"    B2 Acc: {m_b2_cat['accuracy']*100:.2f}%  |  B3 Acc: {m_b3_cat['accuracy']*100:.2f}%  |  d: {cat_result['delta_accuracy']:+.2f}%")
        print(f"    B2 F1:  {m_b2_cat['macro_f1']*100:.2f}%  |  B3 F1:  {m_b3_cat['macro_f1']*100:.2f}%  |  d: {cat_result['delta_macro_f1']:+.2f}%")
        print(f"    Verdict changes: {changes_cat}  |  Correct changes: {correct_cat}")

    # ===================================================================
    # TASK 5: VERDICT CHANGE TRACE
    # ===================================================================
    print("\n" + "=" * 80)
    print("TASK 5: B2 -> B3 VERDICT CHANGE TRACE")
    print("=" * 80)

    verdict_changes = []
    for d in all_detailed:
        if d["verdict_changed"]:
            change_type = (
                "IMPROVED" if d["b3_verdict"] == d["gold"] and d["b2_verdict"] != d["gold"]
                else "DEGRADED" if d["b2_verdict"] == d["gold"] and d["b3_verdict"] != d["gold"]
                else "LATERAL"
            )
            verdict_changes.append({
                "claim_id": d["claim_id"],
                "claim_text": d["claim_text"],
                "ground_truth": d["gold"],
                "b2_verdict": d["b2_verdict"],
                "b3_verdict": d["b3_verdict"],
                "b2_consensus": d["b2_consensus"],
                "b3_consensus": d["b3_consensus"],
                "population_mismatch_type": d["population_category"],
                "avg_P_i": d["avg_P_i"],
                "change_type": change_type,
                "is_correct_change": d["b3_verdict"] == d["gold"],
                "evidence_details": d["evidence_details"],
            })

    improved = sum(1 for v in verdict_changes if v["change_type"] == "IMPROVED")
    degraded = sum(1 for v in verdict_changes if v["change_type"] == "DEGRADED")
    lateral = sum(1 for v in verdict_changes if v["change_type"] == "LATERAL")

    task5_result = {
        "task": "Task 5: Verdict Change Trace",
        "total_changes": len(verdict_changes),
        "improved": improved,
        "degraded": degraded,
        "lateral": lateral,
    }

    print(f"  Total verdict changes: {len(verdict_changes)}")
    print(f"    IMPROVED (B3 correct, B2 wrong): {improved}")
    print(f"    DEGRADED (B2 correct, B3 wrong): {degraded}")
    print(f"    LATERAL  (both wrong, different): {lateral}")

    if verdict_changes:
        print(f"\n  Sample changes:")
        for vc in verdict_changes[:5]:
            print(f"    {vc['claim_id']}: B2={vc['b2_verdict']} -> B3={vc['b3_verdict']} "
                  f"(gold={vc['ground_truth']}) [{vc['change_type']}] "
                  f"avg_P_i={vc['avg_P_i']:.3f}")

    # ===================================================================
    # TASK 6: FINAL CONCLUSION
    # ===================================================================
    print("\n" + "=" * 80)
    print("TASK 6: FINAL CONCLUSION")
    print("=" * 80)

    # Determination 1: Is P_i valid?
    pi_valid = (
        task1_result["formula_validation"]["accuracy"] >= 0.85
        and task1_result["formula_validation"]["mae"] <= 0.15
    )

    # Determination 2: Does P_i affect weighting?
    pi_affects = affected_items > 0 and verdict_changed_count > 0

    # Determination 3: Does R_i * P_i improve verification?
    sig = mcnemar["statistically_significant_005"]
    delta_positive = delta_f1 > 0
    ci_excludes_zero = delta_ci["ci_lower"] > 0

    if sig and delta_positive:
        improvement_conclusion = "YES - statistically significant improvement"
    elif delta_positive and not sig:
        improvement_conclusion = "NO - marginal improvement that is NOT statistically significant"
    elif not delta_positive:
        improvement_conclusion = "NO - no improvement observed"
    else:
        improvement_conclusion = "INCONCLUSIVE"

    task6_result = {
        "task": "Task 6: Final Conclusion",
        "determination_1_pi_valid": {
            "answer": pi_valid,
            "accuracy": task1_result["formula_validation"]["accuracy"],
            "mae": task1_result["formula_validation"]["mae"],
            "kappa": task1_result["inter_annotator_agreement"]["cohens_kappa"],
        },
        "determination_2_pi_affects_weighting": {
            "answer": pi_affects,
            "affected_evidence_items": affected_items,
            "verdict_changes": verdict_changed_count,
            "mean_P_i": task2_result["P_i_distribution"]["mean"],
            "pct_P_i_equals_1": task2_result["P_i_distribution"]["pct_equal_1"],
        },
        "determination_3_improves_verification": {
            "answer": improvement_conclusion,
            "delta_macro_f1_percent": delta_f1,
            "delta_accuracy_percent": delta_acc,
            "mcnemar_p_value": mcnemar["p_value_chi2"],
            "statistically_significant": sig,
            "bootstrap_delta_ci": [delta_ci["ci_lower"], delta_ci["ci_upper"]],
            "ci_excludes_zero": ci_excludes_zero,
            "VCR": round(vcr, 2),
            "CCR": round(ccr, 2),
        },
    }

    print(f"\n  1. Is P_i valid?")
    print(f"     {'YES' if pi_valid else 'NO'} - Accuracy: {task1_result['formula_validation']['accuracy']*100:.1f}%, "
          f"MAE: {task1_result['formula_validation']['mae']:.4f}, "
          f"Kappa: {task1_result['inter_annotator_agreement']['cohens_kappa']:.4f}")

    print(f"\n  2. Does P_i affect evidence weighting?")
    print(f"     {'YES' if pi_affects else 'NO'} - {affected_items}/{total_evidence_items} evidence items affected, "
          f"{verdict_changed_count} verdict changes")

    print(f"\n  3. Does R_i * P_i improve verification?")
    print(f"     {improvement_conclusion}")
    print(f"     Delta Macro-F1: {delta_f1:+.4f}%")
    print(f"     McNemar p: {mcnemar['p_value_chi2']:.6f}")
    print(f"     95% CI for Delta F1: [{delta_ci['ci_lower']:.4f}, {delta_ci['ci_upper']:.4f}]")

    # ===================================================================
    # SAVE ALL REPORTS
    # ===================================================================
    print("\n" + "=" * 80)
    print("SAVING REPORTS")
    print("=" * 80)

    # 1. Primary evaluation report
    primary_report = {
        "report": "MedVerify Primary Contribution Evaluation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "scoring_mode": PI_SCORING_MODE,
            "nli_model": "cross-encoder/nli-deberta-v3-small",
            "evidence_per_claim": 5,
            "test_set": TEST_SET_PATH,
            "population_gt": POP_GT_PATH,
            "random_seed": args.seed,
        },
        "task_1_pi_validation": task1_result,
        "task_2_weight_audit": task2_result,
        "task_3_b2_vs_b3": task3_result,
        "task_4_population_stratification": task4_result,
        "task_5_verdict_changes_summary": task5_result,
        "task_6_conclusion": task6_result,
        "elapsed_seconds": round(total_time, 2),
    }

    path1 = os.path.join(REPORTS_DIR, "primary_contribution_evaluation.json")
    with open(path1, "w", encoding="utf-8") as f:
        json.dump(primary_report, f, indent=2, default=str)
    print(f"  [OK] {path1}")

    # 2. Weight audit (per-evidence)
    weight_audit_report = {
        "report": "B2 vs B3 Per-Evidence Weight Audit",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": task2_result,
        "claims": all_evidence_audit,
    }

    path2 = os.path.join(REPORTS_DIR, "b2_b3_weight_audit.json")
    with open(path2, "w", encoding="utf-8") as f:
        json.dump(weight_audit_report, f, indent=2, default=str)
    print(f"  [OK] {path2}")

    # 3. Verdict changes
    verdict_changes_report = {
        "report": "B2 -> B3 Verdict Changes",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": task5_result,
        "changes": verdict_changes,
    }

    path3 = os.path.join(REPORTS_DIR, "b2_b3_verdict_changes.json")
    with open(path3, "w", encoding="utf-8") as f:
        json.dump(verdict_changes_report, f, indent=2, default=str)
    print(f"  [OK] {path3}")

    # 4. Population-stratified results
    pop_strat_report = {
        "report": "Population-Stratified B2 vs B3 Results",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall_B2": m_b2,
        "overall_B3": m_b3,
        "stratified": task4_result["categories"],
        "category_counts": {cat: len(s["gold"]) for cat, s in strata.items()},
    }

    path4 = os.path.join(REPORTS_DIR, "population_stratified_results.json")
    with open(path4, "w", encoding="utf-8") as f:
        json.dump(pop_strat_report, f, indent=2, default=str)
    print(f"  [OK] {path4}")

    print("\n" + "=" * 80)
    print("PRIMARY CONTRIBUTION EVALUATION COMPLETE")
    print("=" * 80)

    return primary_report


if __name__ == "__main__":
    main()
