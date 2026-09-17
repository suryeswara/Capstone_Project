"""
MedVerify AI — Final Evaluation Suite (experiment-v2)
=====================================================
STEP 15: System Confidence Indicator Calibration (ECE & Brier Score)

Evaluates the calibration of the verification engine's System Confidence Indicator:
- Clarification: Evaluated as System Confidence Indicator (not probability claim is true)
- Metrics:
  * Expected Calibration Error (ECE) across M=10 equal-width bins
  * Brier Score (mean squared difference between confidence and empirical correctness)
  * 10-Bin Reliability Diagram Coordinates (mean confidence vs observed accuracy)
- Populates Table 5 (Trustworthiness) of the final report
- Exports report to results/step12_calibration.json
"""

import os
import sys
import json
import numpy as np
from typing import Dict, List

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from config import RESULTS_DIR
from data_loader import load_json, save_json

OUTPUT_FILE = os.path.join(RESULTS_DIR, "step12_calibration.json")
STEP07_RESULTS = os.path.join(RESULTS_DIR, "step07_b1_vs_b2_verification.json")


def compute_ece_and_diagram(confidences: np.ndarray, accuracies: np.ndarray, num_bins: int = 10) -> tuple:
    """Computes Expected Calibration Error and 10-bin reliability diagram points."""
    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    total_samples = len(confidences)
    diagram_points = []

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == 0:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

        bin_count = int(np.sum(in_bin))

        if bin_count > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            gap = abs(bin_acc - bin_conf)
            ece += (bin_count / total_samples) * gap

            diagram_points.append({
                "bin_index": i + 1,
                "bin_range": [round(float(bin_lower), 2), round(float(bin_upper), 2)],
                "sample_count": bin_count,
                "mean_confidence": round(bin_conf, 4),
                "observed_accuracy": round(bin_acc, 4),
                "calibration_gap": round(gap, 4),
            })
        else:
            diagram_points.append({
                "bin_index": i + 1,
                "bin_range": [round(float(bin_lower), 2), round(float(bin_upper), 2)],
                "sample_count": 0,
                "mean_confidence": round(float((bin_lower + bin_upper) / 2.0), 4),
                "observed_accuracy": 0.0,
                "calibration_gap": 0.0,
            })

    return round(float(ece), 4), diagram_points


def evaluate_calibration() -> Dict:
    print("=" * 80)
    print("EXPERIMENT-V2 — STEP 15: CONFIDENCE CALIBRATION (ECE & BRIER SCORE)")
    print("=" * 80)
    print("System Metric: System Confidence Indicator Calibration")

    if not os.path.exists(STEP07_RESULTS):
        print("Running Step 7 first to obtain predictions...")
        from step07_b1_vs_b2_verification import run_b1_vs_b2_experiment
        step07_data = run_b1_vs_b2_experiment()
    else:
        step07_data = load_json(STEP07_RESULTS)

    traces = step07_data.get("traces", [])
    
    confidences = []
    accuracies = []

    for item in traces:
        # Convert consensus score [-1.0, 1.0] to Confidence Indicator [0.5, 1.0]
        score = abs(item.get("b2_score", 0.0))
        conf = 0.50 + 0.50 * score
        conf = max(0.0, min(1.0, conf))

        is_correct = 1.0 if item.get("b2_correct", False) else 0.0

        confidences.append(conf)
        accuracies.append(is_correct)

    conf_arr = np.array(confidences)
    acc_arr = np.array(accuracies)

    # Compute ECE
    ece, diagram_points = compute_ece_and_diagram(conf_arr, acc_arr, num_bins=10)

    # Compute Brier Score
    brier_score = float(np.mean((conf_arr - acc_arr) ** 2))

    print("\nCONFIDENCE CALIBRATION EVALUATION SUMMARY:")
    print("-" * 55)
    print(f"{'Metric':<35} | {'Score':>15}")
    print("-" * 55)
    print(f"{'Expected Calibration Error (ECE)':<35} | {ece:15.4f}")
    print(f"{'Brier Score':<35} | {brier_score:15.4f}")
    print("=" * 55)

    print("\n10-BIN RELIABILITY DIAGRAM DATA POINTS:")
    print(f"{'Bin':<6} | {'Confidence Range':<18} | {'N':>5} | {'Observed Acc':>14} | {'Gap':>10}")
    print("-" * 65)
    for p in diagram_points:
        if p["sample_count"] > 0:
            crange = f"[{p['bin_range'][0]:.2f}, {p['bin_range'][1]:.2f}]"
            print(f"{p['bin_index']:<6} | {crange:<18} | {p['sample_count']:>5} | {p['observed_accuracy'] * 100:>13.2f}% | {p['calibration_gap']:>10.4f}")
    print("=" * 65)

    report = {
        "evaluation_name": "Confidence Calibration Evaluation",
        "sample_size": len(confidences),
        "metrics": {
            "ece": round(ece, 4),
            "brier_score": round(brier_score, 4),
        },
        "reliability_diagram": diagram_points,
    }

    save_json(report, OUTPUT_FILE)
    print(f"\n[OK] Confidence calibration report saved to: {OUTPUT_FILE}")
    return report


if __name__ == "__main__":
    evaluate_calibration()
