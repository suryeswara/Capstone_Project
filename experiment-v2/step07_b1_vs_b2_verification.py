"""
MedVerify AI — Final Evaluation Suite (experiment-v2)
=====================================================
STEP 9-11: Main Research Experiment — Conventional (B1) vs Reliability-Aware (B2) Verification

Evaluates the core research question:
"Does reliability-aware evidence aggregation improve medical claim verification
 compared with conventional relevance-based evidence aggregation?"

Both systems run on the exact same 614 held-out PubHealth test set claims
using identical retrieval; only the consensus weighting formula differs:
- B1 (Conventional RAG): Unweighted consensus (w_i = 1.0)
- B2 (Reliability-Aware MedVerify): C = sum(R_i * S_i) / sum(R_i), S_i in {-1, 0, +1}

4-Class Verdict Space: Supported, Contradicted, Insufficient, Uncertain
Primary Research Metric: Delta Macro-F1 = MacroF1_B2 - MacroF1_B1
Populates Table 4 of the final report
Exports report to results/step07_b1_vs_b2_verification.json
"""

import os
import sys
import json
import numpy as np
from collections import Counter
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from config import (
    RESULTS_DIR,
    VECTOR_STORE_DIR,
    BACKEND_DIR,
    VERDICT_CLASSES,
    CONSENSUS_THRESHOLDS,
)
from data_loader import load_test_claims, save_json

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app.services.retrieval_engine import FAISSRetriever, calculate_reliability_score
from app.services.consensus_engine import ConsensusEngine, map_to_spec_label

OUTPUT_FILE = os.path.join(RESULTS_DIR, "step07_b1_vs_b2_verification.json")


def compute_verdict_from_score(score: float, evidence_count: int, supporting_w: float, contra_w: float, total_w: float) -> str:
    """Determine 4-class verdict from aggregated consensus score."""
    if evidence_count == 0 or total_w == 0:
        return "Insufficient"

    # Polarization / Mixture check
    if total_w > 0:
        supp_frac = supporting_w / total_w
        contra_frac = contra_w / total_w
        if supp_frac >= 0.25 and contra_frac >= 0.25:
            return "Uncertain"

    if score >= CONSENSUS_THRESHOLDS["SUPPORTED"]:
        return "Supported"
    elif score <= CONSENSUS_THRESHOLDS["CONTRADICTED"]:
        return "Contradicted"
    else:
        return "Insufficient"


def run_b1_vs_b2_experiment(sample_size: int = None) -> dict:
    print("=" * 80)
    print("EXPERIMENT-V2 — STEP 9-11: MAIN RESEARCH EXPERIMENT (B1 vs B2)")
    print("=" * 80)
    print("Research Question:")
    print("  Does reliability-aware evidence aggregation improve medical claim verification")
    print("  compared with conventional relevance-based evidence aggregation?")
    print("=" * 80)

    claims = load_test_claims(sample_size=sample_size)
    print(f"Loaded Frozen PubHealth Test Set: {len(claims)} claims")

    print(f"Loading FAISS Vector Knowledge Base from {VECTOR_STORE_DIR}...")
    retriever = FAISSRetriever(VECTOR_STORE_DIR)

    ground_truth = []
    b1_predictions = []
    b2_predictions = []
    claim_audit_traces = []

    for i, item in enumerate(claims):
        claim_id = item["claim_id"]
        claim_text = item["claim_text"]
        gold_verdict = item["standard_verdict"]
        ground_truth.append(gold_verdict)

        # Retrieve top 5 evidence passages (identical for both B1 and B2)
        retrieved_docs = retriever.search(claim_text, top_k=5)

        # Compute stances and weights
        # B1: w_i = 1.0 (unweighted / conventional)
        # B2: w_i = R_i (reliability-weighted)
        b1_w_sum = 0.0
        b1_stance_sum = 0.0
        b1_supp_w = 0.0
        b1_contra_w = 0.0

        b2_w_sum = 0.0
        b2_stance_sum = 0.0
        b2_supp_w = 0.0
        b2_contra_w = 0.0

        for doc in retrieved_docs:
            raw_stance = doc.get("stance", "neutral")
            s_i = 1.0 if raw_stance == "supporting" else -1.0 if raw_stance == "contradicting" else 0.0
            r_i = doc.get("reliability_score", 0.50)

            # B1 conventional weighting
            w_b1 = 1.0
            b1_w_sum += w_b1
            b1_stance_sum += (w_b1 * s_i)
            if s_i > 0:
                b1_supp_w += w_b1
            elif s_i < 0:
                b1_contra_w += w_b1

            # B2 reliability-aware weighting
            w_b2 = r_i
            b2_w_sum += w_b2
            b2_stance_sum += (w_b2 * s_i)
            if s_i > 0:
                b2_supp_w += w_b2
            elif s_i < 0:
                b2_contra_w += w_b2

        # B1 Verdict
        b1_score = (b1_stance_sum / b1_w_sum) if b1_w_sum > 0 else 0.0
        b1_pred = compute_verdict_from_score(b1_score, len(retrieved_docs), b1_supp_w, b1_contra_w, b1_w_sum)
        b1_predictions.append(b1_pred)

        # B2 Verdict
        b2_score = (b2_stance_sum / b2_w_sum) if b2_w_sum > 0 else 0.0
        b2_pred = compute_verdict_from_score(b2_score, len(retrieved_docs), b2_supp_w, b2_contra_w, b2_w_sum)
        b2_predictions.append(b2_pred)

        claim_audit_traces.append({
            "claim_id": claim_id,
            "claim_text": claim_text,
            "gold_verdict": gold_verdict,
            "b1_pred": b1_pred,
            "b2_pred": b2_pred,
            "b1_score": round(b1_score, 4),
            "b2_score": round(b2_score, 4),
            "verdict_changed": (b1_pred != b2_pred),
            "b1_correct": (b1_pred == gold_verdict),
            "b2_correct": (b2_pred == gold_verdict),
        })

    # Compute Metrics for B1 and B2
    classes = VERDICT_CLASSES

    # B1 Metrics
    b1_acc = float(accuracy_score(ground_truth, b1_predictions))
    b1_prec = float(precision_score(ground_truth, b1_predictions, labels=classes, average="macro", zero_division=0))
    b1_rec = float(recall_score(ground_truth, b1_predictions, labels=classes, average="macro", zero_division=0))
    b1_macro_f1 = float(f1_score(ground_truth, b1_predictions, labels=classes, average="macro", zero_division=0))
    b1_per_class_f1 = f1_score(ground_truth, b1_predictions, labels=classes, average=None, zero_division=0)
    b1_cm = confusion_matrix(ground_truth, b1_predictions, labels=classes).tolist()

    # B2 Metrics
    b2_acc = float(accuracy_score(ground_truth, b2_predictions))
    b2_prec = float(precision_score(ground_truth, b2_predictions, labels=classes, average="macro", zero_division=0))
    b2_rec = float(recall_score(ground_truth, b2_predictions, labels=classes, average="macro", zero_division=0))
    b2_macro_f1 = float(f1_score(ground_truth, b2_predictions, labels=classes, average="macro", zero_division=0))
    b2_per_class_f1 = f1_score(ground_truth, b2_predictions, labels=classes, average=None, zero_division=0)
    b2_cm = confusion_matrix(ground_truth, b2_predictions, labels=classes).tolist()

    delta_macro_f1 = b2_macro_f1 - b1_macro_f1

    print("\n" + "=" * 80)
    print("TABLE 4 — MAIN RESEARCH EXPERIMENT (B1 CONVENTIONAL vs B2 RELIABILITY-AWARE)")
    print("=" * 80)
    print(f"{'Metric':<25} | {'B1 Conventional':>15} | {'B2 Reliability-Aware':>22} | {'Delta':>10}")
    print("-" * 80)
    print(f"{'Overall Accuracy':<25} | {b1_acc * 100:14.2f}% | {b2_acc * 100:21.2f}% | {(b2_acc - b1_acc) * 100:+9.2f}%")
    print(f"{'Macro Precision':<25} | {b1_prec * 100:14.2f}% | {b2_prec * 100:21.2f}% | {(b2_prec - b1_prec) * 100:+9.2f}%")
    print(f"{'Macro Recall':<25} | {b1_rec * 100:14.2f}% | {b2_rec * 100:21.2f}% | {(b2_rec - b1_rec) * 100:+9.2f}%")
    print(f"{'Macro-F1 (PRIMARY)':<25} | {b1_macro_f1 * 100:14.2f}% | {b2_macro_f1 * 100:21.2f}% | {delta_macro_f1 * 100:+9.2f}%")
    print("-" * 80)
    for idx, cname in enumerate(classes):
        f1_b1 = float(b1_per_class_f1[idx])
        f1_b2 = float(b2_per_class_f1[idx])
        print(f"{cname + ' F1':<25} | {f1_b1 * 100:14.2f}% | {f1_b2 * 100:21.2f}% | {(f1_b2 - f1_b1) * 100:+9.2f}%")
    print("=" * 80)
    print(f"\nPRIMARY RESEARCH RESULT: Delta Macro-F1 = {delta_macro_f1 * 100:+.2f}%")
    print("=" * 80)

    report = {
        "evaluation_name": "Table 4 — Main Research Experiment (B1 vs B2)",
        "sample_size": len(claims),
        "primary_metric": "Delta Macro-F1",
        "delta_macro_f1": round(delta_macro_f1, 4),
        "b1_conventional": {
            "accuracy": round(b1_acc, 4),
            "precision": round(b1_prec, 4),
            "recall": round(b1_rec, 4),
            "macro_f1": round(b1_macro_f1, 4),
            "per_class_f1": {c: round(float(b1_per_class_f1[i]), 4) for i, c in enumerate(classes)},
            "confusion_matrix": b1_cm,
        },
        "b2_reliability_aware": {
            "accuracy": round(b2_acc, 4),
            "precision": round(b2_prec, 4),
            "recall": round(b2_rec, 4),
            "macro_f1": round(b2_macro_f1, 4),
            "per_class_f1": {c: round(float(b2_per_class_f1[i]), 4) for i, c in enumerate(classes)},
            "confusion_matrix": b2_cm,
        },
        "verdict_classes": classes,
        "traces": claim_audit_traces,
    }

    save_json(report, OUTPUT_FILE)
    print(f"[OK] Main research experiment report saved to: {OUTPUT_FILE}")
    return report


if __name__ == "__main__":
    run_b1_vs_b2_experiment()
