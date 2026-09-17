"""
MedVerify AI — Final Evaluation Suite (experiment-v2)
=====================================================
STEP 12: Evidence-Quality Verdict Change Ablation

Performs qualitative and quantitative ablation tracing every verdict change (B1 -> B2):
- Evaluates:
  * Total verdict changes
  * Correct changes (B2 matches ground truth and B1 did not)
  * Incorrect changes (B1 matched ground truth and B2 did not)
  * Unchanged count
  * Correct Change Rate (CCR) = Correct Changes / Total Changes
- Analyzes why reliability matters (downweighting low-quality preprints/case reports
  in favor of systematic reviews and RCTs)
- Exports report to results/step08_verdict_change_ablation.json
"""

import os
import sys
import json
from typing import Dict, List

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from config import RESULTS_DIR
from data_loader import load_json, save_json

OUTPUT_FILE = os.path.join(RESULTS_DIR, "step08_verdict_change_ablation.json")
STEP07_RESULTS = os.path.join(RESULTS_DIR, "step07_b1_vs_b2_verification.json")


def run_verdict_ablation() -> Dict:
    print("=" * 80)
    print("EXPERIMENT-V2 — STEP 12: EVIDENCE-QUALITY VERDICT CHANGE ABLATION")
    print("=" * 80)
    print("Research Objective: Demonstrate WHY reliability weighting matters by tracing")
    print("                   cases where the B1 verdict changes under B2.")

    if not os.path.exists(STEP07_RESULTS):
        print("Running Step 7 first to obtain verification traces...")
        from step07_b1_vs_b2_verification import run_b1_vs_b2_experiment
        step07_data = run_b1_vs_b2_experiment()
    else:
        step07_data = load_json(STEP07_RESULTS)

    traces = step07_data.get("traces", [])
    total_claims = len(traces)

    total_changes = 0
    correct_changes = 0
    incorrect_changes = 0
    unchanged = 0

    change_examples = []

    for item in traces:
        b1_pred = item["b1_pred"]
        b2_pred = item["b2_pred"]
        gold = item["gold_verdict"]

        if b1_pred != b2_pred:
            total_changes += 1
            if b2_pred == gold and b1_pred != gold:
                correct_changes += 1
                change_type = "CORRECT_CHANGE"
            elif b1_pred == gold and b2_pred != gold:
                incorrect_changes += 1
                change_type = "INCORRECT_CHANGE"
            else:
                change_type = "NEUTRAL_CHANGE"

            change_examples.append({
                "claim_id": item["claim_id"],
                "claim_text": item["claim_text"],
                "gold_verdict": gold,
                "b1_pred": b1_pred,
                "b2_pred": b2_pred,
                "change_type": change_type,
            })
        else:
            unchanged += 1

    # Correct Change Rate
    ccr = (correct_changes / total_changes) if total_changes > 0 else 0.0

    print("\nVERDICT CHANGE ABLATION SUMMARY TABLE:")
    print("-" * 55)
    print(f"{'Metric':<35} | {'Count / Rate':>15}")
    print("-" * 55)
    print(f"{'Total Evaluated Claims':<35} | {total_claims:>15}")
    print(f"{'Unchanged Verdicts':<35} | {unchanged:>15}")
    print(f"{'Total Verdict Changes (B1 -> B2)':<35} | {total_changes:>15}")
    print(f"{'  * Correct Changes':<35} | {correct_changes:>15}")
    print(f"{'  * Incorrect Changes':<35} | {incorrect_changes:>15}")
    print("-" * 55)
    print(f"{'Correct Change Rate (CCR)':<35} | {ccr * 100:>14.2f}%")
    print("=" * 55)

    report = {
        "evaluation_name": "Evidence-Quality Verdict Change Ablation",
        "total_claims": total_claims,
        "unchanged": unchanged,
        "total_verdict_changes": total_changes,
        "correct_changes": correct_changes,
        "incorrect_changes": incorrect_changes,
        "correct_change_rate": round(ccr, 4),
        "sample_change_cases": change_examples[:10],
    }

    save_json(report, OUTPUT_FILE)
    print(f"[OK] Verdict change ablation report saved to: {OUTPUT_FILE}")
    return report


if __name__ == "__main__":
    run_verdict_ablation()
