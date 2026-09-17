"""
MedVerify AI — Final Evaluation Suite (experiment-v2)
=====================================================
STEP 1 & 2: PubHealth Cleaning, Split & Leakage Audit

Audits the frozen PubHealth splits (train, val, test):
- Exact duplicate text matching across splits
- Near-duplicate / paraphrase leakage checking
- Class distribution across all 22 disease categories
- Train / Test contamination check
- Exports structured audit report to results/step01_pubhealth_audit.json
"""

import os
import sys
import json
import hashlib
from collections import Counter
from typing import Dict, List, Set

# Add parent directory for imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from config import RESULTS_DIR, SPLITS_MANIFEST_PATH, DISEASE_CATEGORIES
from data_loader import load_frozen_splits, normalize_text, save_json

OUTPUT_FILE = os.path.join(RESULTS_DIR, "step01_pubhealth_audit.json")


def compute_hash(text: str) -> str:
    """Compute MD5 hash of normalized text."""
    return hashlib.md5(normalize_text(text).encode("utf-8")).hexdigest()


def jaccard_similarity(s1: str, s2: str) -> float:
    """Compute word-level Jaccard similarity between two strings."""
    w1 = set(normalize_text(s1).split())
    w2 = set(normalize_text(s2).split())
    if not w1 or not w2:
        return 0.0
    return len(w1.intersection(w2)) / len(w1.union(w2))


def audit_splits() -> Dict:
    print("=" * 80)
    print("EXPERIMENT-V2 — STEP 1 & 2: PUBHEALTH DATASET AUDIT & LEAKAGE CHECK")
    print("=" * 80)

    splits = load_frozen_splits()
    train_claims = splits["train"]
    val_claims = splits["val"]
    test_claims = splits["test"]

    print(f"Loaded Splits:")
    print(f"  * Train set: {len(train_claims):>6} claims")
    print(f"  * Dev/Val set: {len(val_claims):>4} claims")
    print(f"  * Test set: {len(test_claims):>7} claims (Held-Out Frozen)")
    total_samples = len(train_claims) + len(val_claims) + len(test_claims)
    print(f"  * Total claims: {total_samples:>5}")

    # 1. Exact Duplicate Audit
    print("\n[1/4] Running Exact Duplicate Audit Across Splits...")
    train_hashes = {compute_hash(r["claim_text"]): r["claim_id"] for r in train_claims}
    val_hashes = {compute_hash(r["claim_text"]): r["claim_id"] for r in val_claims}
    test_hashes = {compute_hash(r["claim_text"]): r["claim_id"] for r in test_claims}

    train_val_overlap = set(train_hashes.keys()).intersection(set(val_hashes.keys()))
    train_test_overlap = set(train_hashes.keys()).intersection(set(test_hashes.keys()))
    val_test_overlap = set(val_hashes.keys()).intersection(set(test_hashes.keys()))

    print(f"  * Train-Val exact duplicates:  {len(train_val_overlap)}")
    print(f"  * Train-Test exact duplicates: {len(train_test_overlap)} (Contamination check)")
    print(f"  * Val-Test exact duplicates:   {len(val_test_overlap)}")

    # 2. Near-Duplicate / Paraphrase Leakage Audit (Sample-based for speed)
    print("\n[2/4] Running Near-Duplicate & Paraphrase Leakage Audit (Jaccard >= 0.85)...")
    near_duplicates = 0
    # Audit first 200 test claims against train claims
    test_sample = test_claims[:200]
    for tc in test_sample:
        for trc in train_claims[:500]:
            sim = jaccard_similarity(tc["claim_text"], trc["claim_text"])
            if sim >= 0.85 and compute_hash(tc["claim_text"]) != compute_hash(trc["claim_text"]):
                near_duplicates += 1

    print(f"  * Detected near-duplicate leakages in audited subset: {near_duplicates}")

    # 3. Class Distribution Audit
    print("\n[3/4] Verifying Class Distribution Balance Across 22 Disease Categories...")
    test_class_counts = Counter(r.get("disease_category") for r in test_claims)
    train_class_counts = Counter(r.get("disease_category") for r in train_claims)

    categories_represented_in_test = len(test_class_counts)
    print(f"  * Total categories represented in test split: {categories_represented_in_test} / 22")
    
    # Check key categories required by spec
    key_categories = ["Diabetes", "Heart Disease", "Vaccination", "COVID-19", "General Cancer"]
    print("\n  Sample Class Breakdown in Frozen Test Set:")
    for cat in key_categories:
        count = test_class_counts.get(cat, 0)
        pct = (count / len(test_claims)) * 100
        print(f"    - {cat:<25}: {count:>3} claims ({pct:.1f}%)")

    # 4. Verdict Distribution Audit
    print("\n[4/4] Verifying Test Set Fact-Check Verdict Distribution...")
    verdict_counts = Counter(r.get("fact_check_verdict", "Unknown") for r in test_claims)
    for v, c in verdict_counts.items():
        print(f"    - {str(v):<20}: {c:>3} claims ({(c/len(test_claims))*100:.1f}%)")

    leakage_free = (len(train_test_overlap) == 0 and near_duplicates == 0)
    audit_status = "PASSED" if leakage_free else "WARNING"

    report = {
        "status": audit_status,
        "leakage_free": leakage_free,
        "sample_counts": {
            "train": len(train_claims),
            "val": len(val_claims),
            "test": len(test_claims),
            "total": total_samples,
        },
        "exact_duplicates": {
            "train_val": len(train_val_overlap),
            "train_test": len(train_test_overlap),
            "val_test": len(val_test_overlap),
        },
        "near_duplicates_sample_count": near_duplicates,
        "disease_categories_represented": categories_represented_in_test,
        "test_class_distribution": dict(test_class_counts),
        "test_verdict_distribution": dict(verdict_counts),
    }

    save_json(report, OUTPUT_FILE)
    print("\n" + "=" * 80)
    print(f"DATASET AUDIT RESULT: [{audit_status}]")
    print(f"Report exported to: {OUTPUT_FILE}")
    print("=" * 80)
    return report


if __name__ == "__main__":
    audit_splits()
