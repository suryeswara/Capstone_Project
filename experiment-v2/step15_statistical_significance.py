"""
MedVerify AI — Final Evaluation Suite (experiment-v2)
=====================================================
STEP 18: Statistical Significance Analysis (Bootstrap CI & McNemar's Test)

Evaluates the formal statistical significance of the central research result:
  B1 Conventional vs B2 Reliability-Aware Verification
- Primary Difference: Delta Macro-F1 = MacroF1_B2 - MacroF1_B1
- Non-Parametric Bootstrap 95% Confidence Interval (B=1000 resamples)
- McNemar's Paired Test for categorical paired verdict correctness
  (tests whether B2 significantly outperforms B1 on the exact same claims)
- Exports report to results/step15_statistical_significance.json
"""

import os
import sys
import json
import numpy as np
from scipy import stats
from sklearn.metrics import f1_score
from typing import Dict, List, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from config import RESULTS_DIR, VERDICT_CLASSES
from data_loader import load_json, save_json

OUTPUT_FILE = os.path.join(RESULTS_DIR, "step15_statistical_significance.json")
STEP07_RESULTS = os.path.join(RESULTS_DIR, "step07_b1_vs_b2_verification.json")


def compute_bootstrap_delta_f1_ci(
    ground_truth: List[str],
    b1_preds: List[str],
    b2_preds: List[str],
    classes: List[str],
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: int = 42
) -> Tuple[float, float, float]:
    """Computes non-parametric bootstrap 95% confidence interval for Delta Macro-F1."""
    np.random.seed(seed)
    n = len(ground_truth)
    delta_scores = []

    for _ in range(n_bootstraps):
        indices = np.random.choice(n, size=n, replace=True)
        y_true_s = [ground_truth[i] for i in indices]
        y_b1_s = [b1_preds[i] for i in indices]
        y_b2_s = [b2_preds[i] for i in indices]

        f1_b1 = f1_score(y_true_s, y_b1_s, labels=classes, average="macro", zero_division=0)
        f1_b2 = f1_score(y_true_s, y_b2_s, labels=classes, average="macro", zero_division=0)
        delta_scores.append(f1_b2 - f1_b1)

    lower = float(np.percentile(delta_scores, (alpha / 2.0) * 100))
    upper = float(np.percentile(delta_scores, (1.0 - alpha / 2.0) * 100))
    mean_delta = float(np.mean(delta_scores))
    return mean_delta, lower, upper


def compute_mcnemar_test(
    b1_correct_flags: List[bool],
    b2_correct_flags: List[bool]
) -> Dict:
    """
    Computes McNemar's paired test with Edwards continuity correction:
    Contingency table:
      a: Both B1 and B2 correct
      b: B1 correct, B2 incorrect
      c: B1 incorrect, B2 correct
      d: Both B1 and B2 incorrect
    chi2 = (|b - c| - 1)^2 / (b + c)
    """
    a = sum(1 for b1, b2 in zip(b1_correct_flags, b2_correct_flags) if b1 and b2)
    b = sum(1 for b1, b2 in zip(b1_correct_flags, b2_correct_flags) if b1 and not b2)
    c = sum(1 for b1, b2 in zip(b1_correct_flags, b2_correct_flags) if not b1 and b2)
    d = sum(1 for b1, b2 in zip(b1_correct_flags, b2_correct_flags) if not b1 and not b2)

    discordant = b + c
    if discordant == 0:
        return {
            "table": {"both_correct": a, "b1_only": b, "b2_only": c, "both_incorrect": d},
            "statistic": 0.0,
            "p_value": 1.0,
            "is_significant": False,
        }

    # Edwards continuity correction
    chi2_stat = ((abs(b - c) - 1.0) ** 2) / float(discordant)
    p_val = float(stats.chi2.sf(chi2_stat, df=1))

    return {
        "table": {
            "both_correct (a)": a,
            "b1_only (b)": b,
            "b2_only (c)": c,
            "both_incorrect (d)": d,
            "discordant_total": discordant,
        },
        "statistic": round(float(chi2_stat), 4),
        "p_value": float(p_val),
        "is_significant": bool(p_val < 0.05),
    }


def evaluate_significance() -> Dict:
    print("=" * 80)
    print("EXPERIMENT-V2 — STEP 18: STATISTICAL SIGNIFICANCE (B1 vs B2)")
    print("=" * 80)
    print("Primary Research Hypothesis:")
    print("  H1: Reliability-Aware Aggregation (B2) produces statistically significant")
    print("      Macro-F1 gains over Conventional Aggregation (B1).")

    if not os.path.exists(STEP07_RESULTS):
        print("Running Step 7 first to obtain predictions...")
        from step07_b1_vs_b2_verification import run_b1_vs_b2_experiment
        step07_data = run_b1_vs_b2_experiment()
    else:
        step07_data = load_json(STEP07_RESULTS)

    traces = step07_data.get("traces", [])
    classes = step07_data.get("verdict_classes", VERDICT_CLASSES)

    ground_truth = [t["gold_verdict"] for t in traces]
    b1_preds = [t["b1_pred"] for t in traces]
    b2_preds = [t["b2_pred"] for t in traces]

    b1_correct = [t["b1_correct"] for t in traces]
    b2_correct = [t["b2_correct"] for t in traces]

    # 1. Delta Macro-F1 & 95% Bootstrap CI
    print("\n[1/2] Computing Non-Parametric Bootstrap 95% Confidence Interval (B=1000)...")
    mean_delta, ci_lower, ci_upper = compute_bootstrap_delta_f1_ci(
        ground_truth, b1_preds, b2_preds, classes, n_bootstraps=1000, alpha=0.05, seed=42
    )
    obs_delta = step07_data.get("delta_macro_f1", mean_delta)

    # 2. McNemar's Paired Test
    print("[2/2] Computing McNemar's Paired Categorical Test...")
    mcnemar_res = compute_mcnemar_test(b1_correct, b2_correct)

    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE ANALYSIS SUMMARY:")
    print("=" * 80)
    print(f"{'Metric':<35} | {'Statistical Result':>35}")
    print("-" * 75)
    print(f"{'Observed Delta Macro-F1':<35} | {obs_delta * 100:>34.2f}%")
    print(f"{'Bootstrap 95% Confidence Interval':<35} | [{ci_lower * 100:+.2f}%, {ci_upper * 100:+.2f}%]")
    print(f"{'McNemar Chi-Square (df=1)':<35} | {mcnemar_res['statistic']:>35.4f}")
    print(f"{'McNemar p-value':<35} | {mcnemar_res['p_value']:>35.4e}")
    sig_str = "YES (p < 0.05)" if mcnemar_res['is_significant'] else "NO (p >= 0.05)"
    print(f"{'Statistically Significant?':<35} | {sig_str:>35}")
    print("=" * 80)

    report = {
        "evaluation_name": "Statistical Significance Analysis (B1 vs B2)",
        "sample_size": len(traces),
        "delta_macro_f1": round(obs_delta, 4),
        "bootstrap_ci_95": {
            "lower": round(ci_lower, 4),
            "upper": round(ci_upper, 4),
            "mean": round(mean_delta, 4),
            "n_bootstraps": 1000,
        },
        "mcnemar_test": mcnemar_res,
    }

    save_json(report, OUTPUT_FILE)
    print(f"[OK] Statistical significance report saved to: {OUTPUT_FILE}")
    return report


if __name__ == "__main__":
    evaluate_significance()
