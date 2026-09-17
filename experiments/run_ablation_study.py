"""
MedVerify AI — Ablation Study
==============================
Master Instructions §23 (Stage 23)

Ablates the population applicability components to isolate their contributions:
  1. Full B3 (R_i × P_i with all 4 dimensions)
  2. No age matching (P_i without A_i)
  3. No sex matching (P_i without S_i)
  4. No condition matching (P_i without C_i)
  5. No region matching (P_i without G_i)
  6. Multiplicative vs Additive weighting

Reports Δ Macro-F1 for each ablation relative to full B3.

Usage:
    python experiments/run_ablation_study.py [--sample N]
"""

import os
import sys
import json
import time
import math
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Tuple
from collections import Counter

import numpy as np
from sklearn.metrics import f1_score, accuracy_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "medverify-ai-backend"))

from app.services.consensus_engine import (
    ConsensusEngine, NLIStanceDetector, map_to_spec_label,
)
from app.services.population.claim_population import extract_claim_population
from app.services.population.study_population import extract_study_population
from app.services.population.matcher import match_populations
from app.services.population.applicability import compute_applicability
from app.services.population.schemas import MatchResult, MatchType

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

TEST_SET_PATH = os.path.join(PROJECT_ROOT, "med_datasets", "splits", "test_frozen.json")
POP_GT_PATH = os.path.join(PROJECT_ROOT, "med_datasets", "evaluation", "population_ground_truth_70.json")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "Reports")
OUTPUT_PATH = os.path.join(REPORTS_DIR, "ablation_study_results.json")

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

SOURCE_TIERS = {
    "Systematic Review / Meta-Analysis": 0.95,
    "Randomized Controlled Trial (RCT)": 0.90,
    "PubMed Article": 0.80,
    "Cohort Study / Observational": 0.75,
}


def build_evidence(claim_text, disease_category):
    """Build simulated evidence items with R_i scores."""
    items = []
    configs = [
        ("Systematic Review / Meta-Analysis", 2024),
        ("Randomized Controlled Trial (RCT)", 2023),
        ("PubMed Article", 2022),
        ("Cohort Study / Observational", 2021),
        ("PubMed Article", 2020),
    ]
    for i, (tier, year) in enumerate(configs):
        r_i = SOURCE_TIERS.get(tier, 0.50) * math.exp(-0.05 * max(0, 2026 - year))
        items.append({
            "id": f"ev-abl-{i+1}",
            "title": f"Evidence {i+1}",
            "source_tier": tier,
            "pub_year": year,
            "chunk_text": f"Study related to {claim_text[:60]}.",
            "reliability_score": round(r_i, 4),
            "disease_category": disease_category,
            "source": "SIMULATED",
        })
    return items


def compute_ablated_pi(claim_text, ev_dict, ablation_mode="full"):
    """
    Compute P_i with one dimension ablated.

    ablation_mode:
      "full" — all 4 dimensions
      "no_age" — A_i forced to 0.5 (neutral)
      "no_sex" — S_i forced to 0.5
      "no_condition" — C_i forced to 0.5
      "no_region" — G_i forced to 0.5
    """
    claim_pop = extract_claim_population(claim_text)
    study_pop = extract_study_population(ev_dict)
    match_result = match_populations(claim_pop, study_pop)

    # Apply ablation by forcing the dimension to neutral (0.5)
    if ablation_mode == "no_age":
        match_result.p_age = 0.5
    elif ablation_mode == "no_sex":
        match_result.p_sex = 0.5
    elif ablation_mode == "no_condition":
        match_result.p_condition = 0.5
    elif ablation_mode == "no_region":
        match_result.p_region = 0.5

    result = compute_applicability(
        claim_pop, study_pop, match_result,
        scoring_mode="v3_frozen_four_dimension",
    )
    return result.score


def run_ablation_variant(test_items, stance_detector, consensus_engine,
                          ablation_mode="full", weighting="multiplicative"):
    """Run a single ablation variant across all test items."""
    gold_labels = []
    predictions = []

    for item in test_items:
        claim_text = item["claim_text"]
        gold = item["gold_label"]
        gold_labels.append(gold)

        evidence = build_evidence(claim_text, item["disease_category"])

        evidence_with_stances = []
        for ev in evidence:
            stance = stance_detector.detect_stance(claim_text, ev["chunk_text"])

            # Compute P_i with ablation
            try:
                p_i = compute_ablated_pi(claim_text, ev, ablation_mode)
            except Exception:
                p_i = 0.5

            r_i = ev["reliability_score"]

            if weighting == "multiplicative":
                # W_i = R_i × P_i (§4)
                app_score = p_i
            elif weighting == "additive":
                # W_i = α × R_i + (1-α) × P_i  → for consensus, we set app=combined
                alpha = 0.65
                combined = alpha * r_i + (1 - alpha) * p_i
                app_score = combined / max(r_i, 0.001)  # normalize so r_i*app ≈ combined
            else:
                app_score = p_i

            evidence_with_stances.append({
                **ev,
                "applicability_score": app_score,
                "stance": stance["stance"],
                "stance_value": stance["stance_value"],
                "nli_confidence": stance["confidence"],
            })

        result = consensus_engine.calculate_weighted_consensus(evidence_with_stances)
        pred = result.get("spec_label", map_to_spec_label(result["verdict"]))
        predictions.append(pred)

    macro_f1 = f1_score(gold_labels, predictions, average="macro",
                        labels=SPEC_LABELS, zero_division=0)
    accuracy = accuracy_score(gold_labels, predictions)

    return {
        "macro_f1": round(macro_f1, 4),
        "accuracy": round(accuracy, 4),
        "sample_size": len(gold_labels),
    }


def main():
    parser = argparse.ArgumentParser(description="MedVerify Ablation Study")
    parser.add_argument("--sample", type=int, default=50,
                        help="Number of test items (0=all, default=50 for speed)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 80)
    print("MEDVERIFY AI — ABLATION STUDY (§23)")
    print("=" * 80)

    # Load test data
    with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    test_items = []
    for item in raw:
        gold_raw = item.get("fact_check_verdict", "")
        gold_spec = PUBHEALTH_TO_SPEC.get(gold_raw)
        if gold_spec:
            test_items.append({
                "claim_id": item["claim_id"],
                "claim_text": item["claim_text"],
                "disease_category": item["disease_category"],
                "gold_label": gold_spec,
            })

    if args.sample > 0:
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(len(test_items), size=min(args.sample, len(test_items)), replace=False)
        test_items = [test_items[i] for i in sorted(idx)]

    print(f"Evaluating {len(test_items)} claims.\n")

    # Load NLI model
    print("Loading NLI model...")
    stance_detector = NLIStanceDetector()
    consensus_engine = ConsensusEngine()
    print("[OK] Model loaded.\n")

    ablation_configs = [
        ("full", "multiplicative", "Full B3 (R_i x P_i, all dimensions)"),
        ("no_age", "multiplicative", "Ablate Age (A_i=0.5)"),
        ("no_sex", "multiplicative", "Ablate Sex (S_i=0.5)"),
        ("no_condition", "multiplicative", "Ablate Condition (C_i=0.5)"),
        ("no_region", "multiplicative", "Ablate Region (G_i=0.5)"),
        ("full", "additive", "Additive Weighting (alpha=0.65)"),
    ]

    results = {}
    full_f1 = None

    for ablation_mode, weighting, description in ablation_configs:
        print(f"Running: {description}...")
        t0 = time.time()
        metrics = run_ablation_variant(
            test_items, stance_detector, consensus_engine,
            ablation_mode=ablation_mode,
            weighting=weighting,
        )
        elapsed = time.time() - t0

        key = f"{ablation_mode}_{weighting}"
        if ablation_mode == "full" and weighting == "multiplicative":
            full_f1 = metrics["macro_f1"]
            delta = 0.0
        else:
            delta = round(metrics["macro_f1"] - (full_f1 or 0), 4)

        results[key] = {
            "description": description,
            "macro_f1": metrics["macro_f1"],
            "accuracy": metrics["accuracy"],
            "delta_f1_vs_full": delta,
            "elapsed_seconds": round(elapsed, 2),
        }

        print(f"  F1={metrics['macro_f1']:.4f}  Acc={metrics['accuracy']:.4f}  "
              f"dF1={delta:+.4f}  ({elapsed:.1f}s)")

    # Summary table
    print("\n" + "=" * 80)
    print(f"{'Variant':<45} | {'F1':<8} | {'dF1':<10} | {'Acc':<8}")
    print("-" * 80)
    for key, r in results.items():
        print(f"{r['description']:<45} | {r['macro_f1']:.4f} | {r['delta_f1_vs_full']:+.4f}   | {r['accuracy']:.4f}")
    print("=" * 80)

    report = {
        "experiment": "Ablation_Study",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "sample_size": len(test_items),
        "seed": args.seed,
        "ablation_results": results,
    }

    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[OK] Report saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
