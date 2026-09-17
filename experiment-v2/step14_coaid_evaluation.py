"""
MedVerify AI — Final Evaluation Suite (experiment-v2)
=====================================================
STEP 16: CoAID Secondary External Misinformation Evaluation

Evaluates the verification pipeline on the external CoAID dataset:
- Dataset: COVID-19 and Vaccination Misinformation Claims
- Evaluates:
  * Overall: Accuracy, Macro Precision, Macro Recall, Macro-F1
  * Dedicated Vaccination Subset Breakdown
- Populates Table 6 (External Evaluation) of the final report
- Exports report to results/step14_coaid_eval.json
"""

import os
import sys
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from config import RESULTS_DIR, VECTOR_STORE_DIR, BACKEND_DIR
from data_loader import load_coaid_claims, save_json

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app.services.retrieval_engine import FAISSRetriever

OUTPUT_FILE = os.path.join(RESULTS_DIR, "step14_coaid_eval.json")


def evaluate_coaid(sample_size: int = None) -> dict:
    print("=" * 80)
    print("EXPERIMENT-V2 — STEP 16: COAID SECONDARY EXTERNAL EVALUATION")
    print("=" * 80)
    print("Evaluates pipeline accuracy on external COVID-19 & vaccination misinformation")

    claims = load_coaid_claims()
    if sample_size and sample_size < len(claims):
        claims = claims[:sample_size]

    print(f"Loaded Real CoAID Claims: {len(claims)}")

    retriever = FAISSRetriever(VECTOR_STORE_DIR)

    all_ground_truth = []
    all_predictions = []
    vax_ground_truth = []
    vax_predictions = []

    classes = ["Supported", "Contradicted"]

    for item in claims:
        claim_text = item["claim_text"]
        gold = item["standard_verdict"]
        cat = item["disease_category"]

        # Retrieve evidence
        docs = retriever.search(claim_text, top_k=5)

        # Reliability-weighted consensus
        w_sum = 0.0
        s_sum = 0.0
        for d in docs:
            r = d.get("reliability_score", 0.50)
            st = d.get("stance", "neutral")
            s = 1.0 if st == "supporting" else -1.0 if st == "contradicting" else 0.0
            w_sum += r
            s_sum += (r * s)

        score = (s_sum / w_sum) if w_sum > 0 else 0.0
        pred = "Supported" if score >= 0.0 else "Contradicted"

        all_ground_truth.append(gold)
        all_predictions.append(pred)

        if cat == "Vaccination":
            vax_ground_truth.append(gold)
            vax_predictions.append(pred)

    # Overall CoAID Metrics
    acc = float(accuracy_score(all_ground_truth, all_predictions))
    prec = float(precision_score(all_ground_truth, all_predictions, labels=classes, average="macro", zero_division=0))
    rec = float(recall_score(all_ground_truth, all_predictions, labels=classes, average="macro", zero_division=0))
    macro_f1 = float(f1_score(all_ground_truth, all_predictions, labels=classes, average="macro", zero_division=0))

    # Vaccination Subset Metrics
    vax_acc = float(accuracy_score(vax_ground_truth, vax_predictions)) if vax_ground_truth else 0.0
    vax_f1 = float(f1_score(vax_ground_truth, vax_predictions, labels=classes, average="macro", zero_division=0)) if vax_ground_truth else 0.0

    print("\n" + "=" * 80)
    print("COAID EXTERNAL BENCHMARK EVALUATION PERFORMANCE:")
    print("=" * 80)
    print(f"{'Evaluation Slice':<35} | {'Accuracy':>15} | {'Macro-F1':>15}")
    print("-" * 75)
    print(f"{'Overall CoAID Misinformation':<35} | {acc * 100:14.2f}% | {macro_f1 * 100:14.2f}%")
    print(f"{'Vaccination / Viral Subset':<35} | {vax_acc * 100:14.2f}% | {vax_f1 * 100:14.2f}%")
    print("=" * 75)

    report = {
        "evaluation_name": "Table 6 (Partial) — CoAID External Evaluation",
        "sample_size": len(claims),
        "overall": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "macro_f1": round(macro_f1, 4),
        },
        "vaccination_subset": {
            "sample_size": len(vax_ground_truth),
            "accuracy": round(vax_acc, 4),
            "macro_f1": round(vax_f1, 4),
        }
    }

    save_json(report, OUTPUT_FILE)
    print(f"\n[OK] CoAID evaluation report saved to: {OUTPUT_FILE}")
    return report


if __name__ == "__main__":
    evaluate_coaid()
