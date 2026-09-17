"""
MedVerify AI — Final Evaluation Suite (experiment-v2)
=====================================================
STEP 7: Evidence Reliability (R_i) Evaluation

Evaluates the rule-based clinical evidence hierarchy reliability scoring engine:
- Compares computed R_i with expert hierarchy ground truth across evidence tiers:
  * Systematic review / Meta-analysis (High)
  * Randomized Controlled Trial (High)
  * Cohort study (Moderate)
  * Case report (Low)
  * Preprint (Low/uncertain)
- Metrics:
  * Reliability Ordering Accuracy (%)
  * Spearman Rank Correlation (rho, p-value) with human quality ranking
  * Reliability Tier Agreement Rate (%)
- Exports report to results/step05_evidence_reliability.json
"""

import os
import sys
import json
import numpy as np
from scipy import stats
from typing import Dict, List

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from config import RESULTS_DIR, EXP_DATASETS_DIR, BACKEND_DIR
from data_loader import load_json, save_json

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app.services.retrieval_engine import calculate_reliability_score

OUTPUT_FILE = os.path.join(RESULTS_DIR, "step05_evidence_reliability.json")
BENCHMARK_PATH = os.path.join(EXP_DATASETS_DIR, "evidence_reliability_hierarchy.json")


def evaluate_evidence_reliability() -> Dict:
    print("=" * 80)
    print("EXPERIMENT-V2 — STEP 7: EVIDENCE RELIABILITY (R_i) HIERARCHY EVALUATION")
    print("=" * 80)
    print("Research Question: Does the system consistently give stronger weight to evidence")
    print("                   judged clinically more reliable?")

    benchmark = load_json(BENCHMARK_PATH)
    tiers = benchmark["evidence_tiers"]

    human_ranks = []
    system_ri_scores = []
    comparison_pairs = []
    tier_agreements = []

    eval_table_rows = []

    all_studies = []
    for tier_info in tiers:
        tier_name = tier_info["tier"]
        expected_rank = tier_info["expected_rank_level"]
        gold_quality = tier_info["gold_quality_label"]
        gold_score = tier_info["gold_quality_score"]

        for ex in tier_info["examples"]:
            ri = calculate_reliability_score(
                source_tier=ex["source_tier"],
                pub_year=ex["publication_year"],
                disease_category=ex["disease_category"],
                is_peer_reviewed=ex["is_peer_reviewed"],
                current_year=2026,
            )

            all_studies.append({
                "study_id": ex["study_id"],
                "title": ex["title"],
                "tier": tier_name,
                "human_rank_level": expected_rank,
                "gold_score": gold_score,
                "system_ri": ri,
            })
            human_ranks.append(expected_rank)
            system_ri_scores.append(ri)

            # Check tier agreement (Tier binning)
            if expected_rank >= 4 and ri >= 0.85:
                tier_agreements.append(1)
            elif expected_rank == 3 and (0.65 <= ri < 0.85):
                tier_agreements.append(1)
            elif expected_rank <= 2 and ri < 0.65:
                tier_agreements.append(1)
            else:
                tier_agreements.append(0)

        # Average R_i for this tier
        tier_studies = [s for s in all_studies if s["tier"] == tier_name]
        avg_ri = float(np.mean([s["system_ri"] for s in tier_studies]))
        eval_table_rows.append({
            "tier": tier_name,
            "human_quality": gold_quality,
            "avg_system_ri": round(avg_ri, 4),
        })

    # 1. Pairwise Ordering Accuracy
    # For every pair where Human(A) > Human(B), check if System(A) > System(B)
    correct_orderings = 0
    total_pairs = 0
    for i in range(len(all_studies)):
        for j in range(len(all_studies)):
            if all_studies[i]["human_rank_level"] > all_studies[j]["human_rank_level"]:
                total_pairs += 1
                if all_studies[i]["system_ri"] > all_studies[j]["system_ri"]:
                    correct_orderings += 1

    ordering_accuracy = (correct_orderings / total_pairs) if total_pairs > 0 else 0.0

    # 2. Spearman Rank Correlation
    spearman_corr, spearman_p = stats.spearmanr(human_ranks, system_ri_scores)

    # 3. Tier Agreement Rate
    agreement_rate = float(np.mean(tier_agreements))

    print("\nEVIDENCE RELIABILITY HIERARCHY EVALUATION TABLE:")
    print("-" * 75)
    print(f"{'Evidence Tier':<40} | {'Human Quality':<15} | {'System (R_i)':>12}")
    print("-" * 75)
    for row in eval_table_rows:
        print(f"{row['tier']:<40} | {row['human_quality']:<15} | {row['avg_system_ri']:12.4f}")
    print("=" * 75)

    print("\nRELIABILITY METRICS SUMMARY:")
    print(f"  * Reliability Ordering Accuracy: {ordering_accuracy * 100:.2f}% ({correct_orderings}/{total_pairs} valid pairs)")
    print(f"  * Spearman Rank Correlation (rho): {spearman_corr:.4f} (p = {spearman_p:.4e})")
    print(f"  * Reliability-Tier Agreement:    {agreement_rate * 100:.2f}%")
    print("=" * 75)

    report = {
        "evaluation_name": "Evidence Reliability (R_i) Hierarchy Evaluation",
        "sample_size": len(all_studies),
        "metrics": {
            "ordering_accuracy": round(ordering_accuracy, 4),
            "spearman_correlation": round(float(spearman_corr), 4),
            "spearman_p_value": float(spearman_p),
            "tier_agreement_rate": round(agreement_rate, 4),
        },
        "tier_averages": eval_table_rows,
    }

    save_json(report, OUTPUT_FILE)
    print(f"[OK] Evidence reliability evaluation report saved to: {OUTPUT_FILE}")
    return report


if __name__ == "__main__":
    evaluate_evidence_reliability()
