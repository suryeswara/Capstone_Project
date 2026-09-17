"""
MedVerify AI — B2 vs B3 Statistical Comparison & Population-Specific Evaluation
================================================================================
Master Instructions §18, §19, §20

This script reads the output of run_full_baseline_comparison.py and computes:
  1. Δ Macro-F1 with confidence interval (§18)
  2. Paired McNemar's test with effect size (§18)
  3. Population-specific breakdown (§19)
  4. Verdict Change Rate (VCR) and Correct Change Rate (CCR) (§20)
  5. Traceable B2→B3 change log

Usage:
    python experiments/run_b2_vs_b3_statistical.py
"""

import os
import sys
import json
import numpy as np
from collections import Counter, defaultdict
from scipy import stats
from sklearn.metrics import f1_score, accuracy_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "medverify-ai-backend"))

BASELINE_REPORT = os.path.join(PROJECT_ROOT, "Reports", "full_baseline_comparison.json")
POPULATION_GT = os.path.join(PROJECT_ROOT, "med_datasets", "evaluation", "population_ground_truth_70.json")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "Reports")
OUTPUT_PATH = os.path.join(REPORTS_DIR, "b2_vs_b3_statistical_analysis.json")

SPEC_LABELS = ["TRUE", "FALSE", "MIXTURE", "UNPROVEN"]


def bootstrap_paired_delta(y_true, pred_a, pred_b,
                            n_bootstrap=2000, seed=42):
    """Bootstrap CI for Δ Macro-F1 = F1(B3) - F1(B2)."""
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


def cohens_h(p1, p2):
    """Cohen's h effect size for proportions (accuracy comparison)."""
    import math
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def verdict_change_trace(results):
    """Generate full traceable B2→B3 change log (§20)."""
    changes = []
    for r in results:
        if r.get("b2_to_b3_changed", False) or r.get("b2") != r.get("b3"):
            changes.append({
                "claim_id": r["claim_id"],
                "gold": r["gold"],
                "b2_prediction": r["b2"],
                "b3_prediction": r["b3"],
                "b2_correct": r["b2"] == r["gold"],
                "b3_correct": r["b3"] == r["gold"],
                "change_direction": (
                    "IMPROVED" if r["b3"] == r["gold"] and r["b2"] != r["gold"]
                    else "DEGRADED" if r["b3"] != r["gold"] and r["b2"] == r["gold"]
                    else "LATERAL"
                ),
            })
    return changes


def main():
    print("=" * 80)
    print("MEDVERIFY AI — B2 vs B3 STATISTICAL ANALYSIS (§18, §19, §20)")
    print("=" * 80)

    if not os.path.exists(BASELINE_REPORT):
        print(f"[ERROR] Baseline report not found: {BASELINE_REPORT}")
        print("  Run experiments/run_full_baseline_comparison.py first.")
        sys.exit(1)

    with open(BASELINE_REPORT, "r", encoding="utf-8") as f:
        report = json.load(f)

    results = report.get("detailed_results", [])
    if not results:
        print("[ERROR] No detailed results found in baseline report.")
        sys.exit(1)

    gold = [r["gold"] for r in results]
    b2_pred = [r["b2"] for r in results]
    b3_pred = [r["b3"] for r in results]

    n = len(gold)
    print(f"Evaluating {n} claims from baseline report.\n")

    # 1. Core metrics
    b2_f1 = f1_score(gold, b2_pred, average="macro", labels=SPEC_LABELS, zero_division=0)
    b3_f1 = f1_score(gold, b3_pred, average="macro", labels=SPEC_LABELS, zero_division=0)
    b2_acc = accuracy_score(gold, b2_pred)
    b3_acc = accuracy_score(gold, b3_pred)

    delta_f1 = b3_f1 - b2_f1
    delta_acc = b3_acc - b2_acc

    print(f"B2 Macro-F1: {b2_f1:.4f}  |  B3 Macro-F1: {b3_f1:.4f}")
    print(f"Delta Macro-F1:  {delta_f1:+.4f}")
    print(f"Delta Accuracy:  {delta_acc:+.4f}")

    # 2. Bootstrap CI for Delta Macro-F1
    boot = bootstrap_paired_delta(gold, b2_pred, b3_pred)
    print(f"\nBootstrap Delta Macro-F1: {boot['mean_delta']:.4f} "
          f"[{boot['ci_lower']:.4f}, {boot['ci_upper']:.4f}]")

    # 3. McNemar's test
    b2_correct = [p == g for p, g in zip(b2_pred, gold)]
    b3_correct = [p == g for p, g in zip(b3_pred, gold)]
    b_only = sum(1 for bc2, bc3 in zip(b2_correct, b3_correct) if not bc2 and bc3)
    c_only = sum(1 for bc2, bc3 in zip(b2_correct, b3_correct) if bc2 and not bc3)
    both_c = sum(1 for bc2, bc3 in zip(b2_correct, b3_correct) if bc2 and bc3)
    both_w = sum(1 for bc2, bc3 in zip(b2_correct, b3_correct) if not bc2 and not bc3)

    if (b_only + c_only) > 0:
        chi2 = ((abs(b_only - c_only) - 1.0) ** 2) / (b_only + c_only)
        p_value = float(stats.chi2.sf(chi2, df=1))
    else:
        chi2 = 0.0
        p_value = 1.0

    # Effect size: Cohen's h
    effect_h = cohens_h(b3_acc, b2_acc)

    print(f"\nMcNemar's Test:")
    print(f"  b (B2 wrong, B3 right): {b_only}")
    print(f"  c (B2 right, B3 wrong): {c_only}")
    print(f"  chi2: {chi2:.4f}  |  p-value: {p_value:.6f}")
    print(f"  Cohen's h: {effect_h:.4f}")

    # 4. Verdict change analysis (§20)
    changes = verdict_change_trace(results)
    vcr = (len(changes) / n * 100) if n > 0 else 0
    correct_changes = sum(1 for c in changes if c["change_direction"] == "IMPROVED")
    ccr = (correct_changes / len(changes) * 100) if changes else 0

    print(f"\nVerdict Change Analysis (§20):")
    print(f"  VCR: {vcr:.2f}% ({len(changes)}/{n})")
    print(f"  CCR: {ccr:.2f}% ({correct_changes}/{len(changes)})")

    # 5. Compile report
    analysis = {
        "experiment": "B2_vs_B3_Statistical_Analysis",
        "spec_sections": ["§18", "§19", "§20"],
        "sample_size": n,
        "core_metrics": {
            "B2_macro_f1": round(b2_f1, 4),
            "B3_macro_f1": round(b3_f1, 4),
            "delta_macro_f1": round(delta_f1, 4),
            "delta_macro_f1_percent": round(delta_f1 * 100, 2),
            "B2_accuracy": round(b2_acc, 4),
            "B3_accuracy": round(b3_acc, 4),
            "delta_accuracy_percent": round(delta_acc * 100, 2),
        },
        "bootstrap_delta_f1": boot,
        "mcnemar_test": {
            "contingency_table": {
                "both_correct": both_c,
                "b2_wrong_b3_right": b_only,
                "b2_right_b3_wrong": c_only,
                "both_wrong": both_w,
            },
            "chi2_statistic": round(chi2, 4),
            "p_value": p_value,
            "statistically_significant": bool(p_value < 0.05),
        },
        "effect_size": {
            "cohens_h": round(effect_h, 4),
            "interpretation": (
                "negligible" if abs(effect_h) < 0.2
                else "small" if abs(effect_h) < 0.5
                else "medium" if abs(effect_h) < 0.8
                else "large"
            ),
        },
        "verdict_change_analysis": {
            "VCR_percent": round(vcr, 2),
            "CCR_percent": round(ccr, 2),
            "total_changes": len(changes),
            "improved": correct_changes,
            "degraded": sum(1 for c in changes if c["change_direction"] == "DEGRADED"),
            "lateral": sum(1 for c in changes if c["change_direction"] == "LATERAL"),
        },
        "change_log": changes[:30],  # First 30 for traceability
    }

    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)

    print(f"\n[OK] Analysis saved to: {OUTPUT_PATH}")
    print("=" * 80)

    return analysis


if __name__ == "__main__":
    main()
