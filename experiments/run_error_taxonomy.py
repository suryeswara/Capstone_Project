"""
MedVerify AI — Error Taxonomy Analysis
========================================
Master Instructions §21

Classifies every prediction error into the taxonomy:
  E1:  Retrieval failure
  E2:  Reliability failure
  E3:  Population extraction failure
  E4:  Population matching failure
  E5:  Claim extraction failure
  E6:  Claim decomposition failure
  E7:  Evidence interpretation failure
  E8:  Claim/evidence strength mismatch
  E9:  Conflicting evidence
  E10: Explanation hallucination
  E11: Ambiguity

Reads the detailed results from run_full_baseline_comparison.py
and applies heuristic classification rules to categorize errors.

Usage:
    python experiments/run_error_taxonomy.py
"""

import os
import sys
import json
from collections import Counter, defaultdict
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE_REPORT = os.path.join(PROJECT_ROOT, "Reports", "full_baseline_comparison.json")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "Reports")
OUTPUT_PATH = os.path.join(REPORTS_DIR, "error_taxonomy_analysis.json")

ERROR_CODES = {
    "E1": "Retrieval failure",
    "E2": "Reliability failure",
    "E3": "Population extraction failure",
    "E4": "Population matching failure",
    "E5": "Claim extraction failure",
    "E6": "Claim decomposition failure",
    "E7": "Evidence interpretation failure",
    "E8": "Claim/evidence strength mismatch",
    "E9": "Conflicting evidence",
    "E10": "Explanation hallucination",
    "E11": "Ambiguity",
}


def classify_error(item: dict) -> str:
    """
    Heuristically classify a B3 prediction error into the taxonomy.

    This uses the available metadata (consensus scores, gold labels, predictions)
    to infer the most likely error source. In a production setting, this would
    be supplemented with manual annotation.
    """
    gold = item["gold"]
    b3 = item["b3"]
    b2 = item.get("b2", "")
    b3_consensus = item.get("b3_consensus", 0.0)

    # If B3 predicted UNPROVEN but gold was TRUE/FALSE → likely retrieval failure
    # The pipeline didn't find or couldn't use enough directional evidence
    if b3 == "UNPROVEN" and gold in ("TRUE", "FALSE"):
        return "E1"  # Retrieval failure — insufficient evidence retrieved

    # If B2 was correct but B3 broke it → population module introduced error
    if b2 == gold and b3 != gold:
        # Population weighting made it worse
        return "E4"  # Population matching failure

    # If gold is MIXTURE but we predicted TRUE/FALSE → missed conflicting evidence
    if gold == "MIXTURE" and b3 in ("TRUE", "FALSE"):
        return "E9"  # Conflicting evidence (should have detected polarization)

    # If gold is TRUE but we predicted FALSE (or vice versa) → evidence interpretation
    if (gold == "TRUE" and b3 == "FALSE") or (gold == "FALSE" and b3 == "TRUE"):
        # Complete directional reversal → NLI likely misclassified stance
        return "E7"  # Evidence interpretation failure

    # If gold is TRUE/FALSE but we predicted MIXTURE → strength mismatch
    if gold in ("TRUE", "FALSE") and b3 == "MIXTURE":
        return "E8"  # Claim/evidence strength mismatch

    # If consensus was near zero → ambiguity
    if abs(b3_consensus) < 0.05:
        return "E11"  # Ambiguity

    # Default: claim/evidence strength mismatch
    return "E8"


def main():
    print("=" * 80)
    print("MEDVERIFY AI — ERROR TAXONOMY ANALYSIS (§21)")
    print("=" * 80)

    if not os.path.exists(BASELINE_REPORT):
        print(f"[ERROR] Baseline report not found: {BASELINE_REPORT}")
        print("  Run experiments/run_full_baseline_comparison.py first.")
        sys.exit(1)

    with open(BASELINE_REPORT, "r", encoding="utf-8") as f:
        report = json.load(f)

    results = report.get("detailed_results", [])
    total = len(results)

    # Identify B3 errors
    b3_errors = [r for r in results if r["b3"] != r["gold"]]
    b3_correct = total - len(b3_errors)

    print(f"Total claims: {total}")
    print(f"B3 correct: {b3_correct} ({b3_correct/total*100:.1f}%)")
    print(f"B3 errors: {len(b3_errors)} ({len(b3_errors)/total*100:.1f}%)")

    # Classify each error
    error_details = []
    error_counts = Counter()

    for item in b3_errors:
        code = classify_error(item)
        error_counts[code] += 1
        error_details.append({
            "claim_id": item["claim_id"],
            "gold": item["gold"],
            "b3_prediction": item["b3"],
            "b2_prediction": item.get("b2", ""),
            "error_code": code,
            "error_description": ERROR_CODES[code],
            "b3_consensus": item.get("b3_consensus", 0.0),
        })

    # Summary
    print(f"\nError Distribution:")
    print(f"{'Code':<6} | {'Count':<6} | {'%':<8} | {'Description'}")
    print("-" * 60)
    for code in sorted(ERROR_CODES.keys()):
        count = error_counts.get(code, 0)
        pct = (count / max(len(b3_errors), 1)) * 100
        if count > 0:
            print(f"{code:<6} | {count:<6} | {pct:5.1f}%  | {ERROR_CODES[code]}")

    # Build report
    report_out = {
        "experiment": "Error_Taxonomy_Analysis",
        "spec_section": "§21",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "total_claims": total,
        "b3_correct": b3_correct,
        "b3_errors": len(b3_errors),
        "b3_accuracy": round(b3_correct / total, 4) if total > 0 else 0,
        "error_distribution": {
            code: {
                "count": error_counts.get(code, 0),
                "percent": round(error_counts.get(code, 0) / max(len(b3_errors), 1) * 100, 2),
                "description": desc,
            }
            for code, desc in ERROR_CODES.items()
        },
        "error_details": error_details[:50],  # First 50 for traceability
    }

    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report_out, f, indent=2)

    print(f"\n[OK] Error taxonomy saved to: {OUTPUT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
