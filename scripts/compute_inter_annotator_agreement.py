"""
MedVerify AI — Inter-Annotator Agreement Computation
=====================================================
Master Instructions §16

Computes inter-annotator agreement for the population annotation dataset.

Metrics:
  - Cohen's Kappa (for 2 annotators)
  - Percentage agreement
  - Per-category agreement breakdown

Usage:
    python scripts/compute_inter_annotator_agreement.py
"""

import os
import sys
import json
from collections import Counter
from datetime import datetime

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GT_70_PATH = os.path.join(PROJECT_ROOT, "med_datasets", "evaluation", "population_ground_truth_70.json")
CHALLENGE_PATH = os.path.join(PROJECT_ROOT, "med_datasets", "evaluation", "population_challenge_set.json")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "Reports", "inter_annotator_agreement.json")


def cohens_kappa(ann1, ann2):
    """Compute Cohen's Kappa for two annotators."""
    n = len(ann1)
    if n == 0:
        return 0.0

    # Build confusion matrix
    labels = sorted(set(ann1 + ann2))
    label_to_idx = {l: i for i, l in enumerate(labels)}
    k = len(labels)
    cm = np.zeros((k, k), dtype=int)
    for a, b in zip(ann1, ann2):
        cm[label_to_idx[a], label_to_idx[b]] += 1

    # Observed agreement
    p_o = np.trace(cm) / n

    # Expected agreement
    row_sums = cm.sum(axis=1) / n
    col_sums = cm.sum(axis=0) / n
    p_e = np.sum(row_sums * col_sums)

    if p_e == 1.0:
        return 1.0

    kappa = (p_o - p_e) / (1.0 - p_e)
    return round(float(kappa), 4)


def percentage_agreement(ann1, ann2):
    """Simple percentage agreement."""
    if not ann1:
        return 0.0
    agreements = sum(1 for a, b in zip(ann1, ann2) if a == b)
    return round(agreements / len(ann1) * 100, 2)


def main():
    print("=" * 80)
    print("MEDVERIFY AI — INTER-ANNOTATOR AGREEMENT (§16)")
    print("=" * 80)

    results = {}

    # --- 1. Existing 70-item ground truth ---
    if os.path.exists(GT_70_PATH):
        with open(GT_70_PATH, "r", encoding="utf-8") as f:
            gt_data = json.load(f)

        items = gt_data.get("dataset", [])
        ann1_scores = []
        ann2_scores = []

        for item in items:
            a1 = item.get("annotator_1_P_i")
            a2 = item.get("annotator_2_P_i")
            if a1 is not None and a2 is not None:
                # Categorize into bands for kappa computation
                def categorize_pi(pi):
                    if pi >= 0.75:
                        return "HIGH"
                    elif pi >= 0.50:
                        return "PARTIAL"
                    elif pi >= 0.25:
                        return "LOW"
                    else:
                        return "NOT_APPLICABLE"

                ann1_scores.append(categorize_pi(a1))
                ann2_scores.append(categorize_pi(a2))

        if ann1_scores:
            kappa = cohens_kappa(ann1_scores, ann2_scores)
            pct = percentage_agreement(ann1_scores, ann2_scores)
            print(f"\n[GT-70] Population Ground Truth (N={len(ann1_scores)}):")
            print(f"  Cohen's kappa: {kappa}")
            print(f"  Agreement: {pct}%")

            results["population_gt_70"] = {
                "sample_size": len(ann1_scores),
                "cohens_kappa": kappa,
                "percentage_agreement": pct,
                "annotation_label_distribution": dict(Counter(ann1_scores)),
                "note": (
                    "Both annotator columns contain identical values, indicating "
                    "single-annotator labeling. κ=1.0 is expected. A second independent "
                    "annotator is recommended to validate population labels."
                ) if kappa == 1.0 else "",
            }
        else:
            print("[GT-70] No dual-annotator data found.")
            results["population_gt_70"] = {"note": "No dual-annotator data available"}

    # --- 2. Population challenge set (if available) ---
    if os.path.exists(CHALLENGE_PATH):
        with open(CHALLENGE_PATH, "r", encoding="utf-8") as f:
            challenge_data = json.load(f)

        items = challenge_data.get("dataset", [])
        ann1_labels = []
        ann2_labels = []
        dual_count = 0

        for item in items:
            a1 = item.get("annotator_1")
            a2 = item.get("annotator_2")
            if a1 and a2:
                ann1_labels.append(a1)
                ann2_labels.append(a2)
                dual_count += 1

        if dual_count > 0:
            kappa = cohens_kappa(ann1_labels, ann2_labels)
            pct = percentage_agreement(ann1_labels, ann2_labels)
            print(f"\n[Challenge] Population Challenge Set (N={dual_count}):")
            print(f"  Cohen's kappa: {kappa}")
            print(f"  Agreement: {pct}%")

            results["population_challenge"] = {
                "sample_size": dual_count,
                "cohens_kappa": kappa,
                "percentage_agreement": pct,
            }
        else:
            print(f"\n[Challenge] {len(items)} items found but no dual annotation yet.")
            results["population_challenge"] = {
                "sample_size": len(items),
                "status": "SINGLE_ANNOTATOR",
                "note": "Second annotator labels not yet provided. "
                        "κ cannot be computed with a single annotator.",
            }

    # --- 3. Summary ---
    report = {
        "experiment": "Inter_Annotator_Agreement",
        "spec_section": "§16",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "agreement_statistics": results,
        "methodology": {
            "annotation_labels": ["HIGH", "PARTIAL", "LOW", "NOT_APPLICABLE"],
            "agreement_metric": "Cohen's Kappa",
            "kappa_interpretation": {
                "0.81-1.00": "almost perfect",
                "0.61-0.80": "substantial",
                "0.41-0.60": "moderate",
                "0.21-0.40": "fair",
                "0.00-0.20": "slight",
                "< 0.00": "poor",
            },
        },
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[OK] Agreement report saved to: {OUTPUT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
