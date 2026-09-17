"""
MedVerify AI — Stage 1 / Problems 1 & 2: Dataset Governance & Frozen Splits (All Disease Categories)

Incorporate ALL fine-grained disease categories from the initial med_datasets:
- PubHealth medical corpus (pubhealth_medical.csv.xls) with N >= 40 per category
- Phase 1 manifest (CoAID + PubMed seed claims)
Covers 22 comprehensive clinical disease categories:
  1. COVID-19
  2. General Cancer
  3. Influenza
  4. Reproductive Health / Abortion
  5. Breast Cancer
  6. HIV/AIDS
  7. Ebola Virus
  8. Alzheimers Disease
  9. Prostate Cancer
 10. Diabetes
 11. Obesity & Weight Management
 12. Opioids & Pain Management
 13. Measles / MMR
 14. Depression
 15. Heart Disease
 16. Smoking & Tobacco
 17. Pregnancy & Maternal Health
 18. Heart Attack
 19. Stroke
 20. Lung Cancer
 21. Autism Spectrum Disorder
 22. Vaccination

Performs:
1. Sourcing and canonicalizing across all 22 disease categories.
2. Text normalization and strict exact duplicate removal.
3. Near-duplicate and cross-repost removal using character 4-gram Jaccard similarity.
4. Full reporting of class distribution before and after cleaning.
5. Stratified partitioning into Train (70%), Validation (15%), and Held-Out Test (15%).
6. Verification of 0.0% leakage between train, val, and test splits.
7. Cryptographic SHA-256 freezing into med_datasets/splits/ with complete audit trail.
"""

import os
import re
import json
import hashlib
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(PROJECT_ROOT, "med_datasets", "pubhealth_medical.csv.xls")
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "med_datasets", "processed", "phase1_disease_claims_manifest.json")
SPLITS_DIR = os.path.join(PROJECT_ROOT, "med_datasets", "splits")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

os.makedirs(SPLITS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# 22 Canonical Disease Categories
DISEASE_CATEGORIES = [
    "COVID-19",
    "General Cancer",
    "Influenza",
    "Reproductive Health / Abortion",
    "Breast Cancer",
    "HIV/AIDS",
    "Ebola Virus",
    "Alzheimers Disease",
    "Prostate Cancer",
    "Diabetes",
    "Obesity & Weight Management",
    "Opioids & Pain Management",
    "Measles / MMR",
    "Depression",
    "Heart Disease",
    "Smoking & Tobacco",
    "Pregnancy & Maternal Health",
    "Heart Attack",
    "Stroke",
    "Lung Cancer",
    "Autism Spectrum Disorder",
    "Vaccination"
]

LABEL_MAP = {cat: idx for idx, cat in enumerate(DISEASE_CATEGORIES)}
REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

TOPIC_ALIAS_MAP = {
    "covid-19": "COVID-19",
    "covid": "COVID-19",
    "coronavirus": "COVID-19",
    "aids": "HIV/AIDS",
    "hiv": "HIV/AIDS",
    "flu": "Influenza",
    "influenza": "Influenza",
    "heart attack": "Heart Attack",
    "myocardial infarction": "Heart Attack",
    "heart disease": "Heart Disease",
    "coronary artery disease": "Heart Disease",
    "breast cancer": "Breast Cancer",
    "prostate cancer": "Prostate Cancer",
    "lung cancer": "Lung Cancer",
    "cancer": "General Cancer",
    "alzheimer": "Alzheimers Disease",
    "alzheimer's": "Alzheimers Disease",
    "diabetes": "Diabetes",
    "measles": "Measles / MMR",
    "mmr": "Measles / MMR",
    "ebola": "Ebola Virus",
    "opioid": "Opioids & Pain Management",
    "obesity": "Obesity & Weight Management",
    "depression": "Depression",
    "smoking": "Smoking & Tobacco",
    "tobacco": "Smoking & Tobacco",
    "stroke": "Stroke",
    "pregnancy": "Pregnancy & Maternal Health",
    "maternal": "Pregnancy & Maternal Health",
    "abortion": "Reproductive Health / Abortion",
    "autism": "Autism Spectrum Disorder",
    "vaccination": "Vaccination",
    "vaccine": "Vaccination"
}

def normalize_text(text: str) -> str:
    """Normalize text for rigorous duplicate and near-duplicate matching."""
    text = str(text).strip()
    text = re.sub(r'^["\'\u201c\u201d\u2018\u2019]+|["\'\u201c\u201d\u2018\u2019]+$', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_char_ngrams(text: str, n: int = 4) -> set:
    """Extract character n-grams for fast fuzzy Jaccard similarity."""
    cleaned = re.sub(r'[^\w\s]', '', text.lower())
    if len(cleaned) < n:
        return {cleaned}
    return {cleaned[i:i+n] for i in range(len(cleaned) - n + 1)}

def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Calculate Jaccard similarity coefficient."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union if union > 0 else 0.0

def compute_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def canonicalize_topic(topic_str: str):
    if pd.isna(topic_str):
        return None
    key = str(topic_str).strip().lower()
    return TOPIC_ALIAS_MAP.get(key, None)

def audit_and_freeze():
    print("=" * 80)
    print("MEDVERIFY AI — PHASE 1: DATASET LEAKAGE AUDIT & FROZEN SPLITS (ALL 22 CATEGORIES)")
    print("=" * 80)

    raw_candidates = []

    # 1. Ingest PubHealth CSV (initial raw medical corpus)
    if os.path.exists(CSV_PATH):
        print(f"Reading PubHealth CSV: {os.path.basename(CSV_PATH)}...")
        df_csv = pd.read_csv(CSV_PATH)
        ph_matched = 0
        for _, row in df_csv.iterrows():
            claim_text = normalize_text(row.get("claim", ""))
            if len(claim_text) < 15 or claim_text == "nan":
                continue
            cat = canonicalize_topic(row.get("disease_topic"))
            if cat in LABEL_MAP:
                raw_candidates.append({
                    "claim_id": f"pubhealth-{row.get('claim_id')}",
                    "claim_text": claim_text,
                    "disease_category": cat,
                    "label": LABEL_MAP[cat],
                    "dataset_source": "PubHealth_Initial",
                    "fact_check_verdict": str(row.get("label", "Unknown")).title()
                })
                ph_matched += 1
        print(f"  [OK] Ingested {ph_matched} domain-matched claims across 22 categories from PubHealth.")

    # 2. Ingest Phase 1 Manifest (CoAID Vaccination + PubMed seeds)
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            p1_items = json.load(f)
        p1_matched = 0
        for idx, item in enumerate(p1_items):
            claim_text = normalize_text(item.get("claim_text", ""))
            if len(claim_text) < 15:
                continue
            cat = item.get("disease_category")
            if cat == "Cardiovascular Disease":
                cat = "Heart Disease"
            if cat in LABEL_MAP:
                raw_candidates.append({
                    "claim_id": item.get("claim_id", f"p1-{idx}"),
                    "claim_text": claim_text,
                    "disease_category": cat,
                    "label": LABEL_MAP[cat],
                    "dataset_source": item.get("dataset_source", "Phase1_Manifest"),
                    "fact_check_verdict": item.get("fact_check_verdict", "Unknown")
                })
                p1_matched += 1
        print(f"  [OK] Ingested {p1_matched} additional claims from Phase 1 manifest.")

    print(f"\nTotal raw pool size: {len(raw_candidates)} claims across {len(DISEASE_CATEGORIES)} categories.")

    # 3. Report Raw Class Distribution
    raw_dist = Counter([r["disease_category"] for r in raw_candidates])
    print("\n" + "-" * 75)
    print(f"{'#':<3} | {'Disease Category':<35} | {'Raw Count':>10} | {'Percentage':>10}")
    print("-" * 75)
    for idx, cat in enumerate(DISEASE_CATEGORIES, 1):
        cnt = raw_dist.get(cat, 0)
        pct = (cnt / len(raw_candidates) * 100) if raw_candidates else 0
        print(f"{idx:<3} | {cat:<35} | {cnt:>10d} | {pct:>9.2f}%")
    print("-" * 75)

    # 4. Exact Duplicate Removal
    exact_seen = set()
    exact_unique = []
    exact_duplicates_purged = 0

    for r in raw_candidates:
        key = (r["claim_text"].lower(), r["disease_category"])
        if key in exact_seen:
            exact_duplicates_purged += 1
        else:
            exact_seen.add(key)
            exact_unique.append(r)

    print(f"\n[Exact Duplicate Audit]: Purged {exact_duplicates_purged} duplicates (Retained {len(exact_unique)} claims).")

    # 5. Near-Duplicate & Repost Removal (Jaccard threshold >= 0.80)
    print("\n[Near-Duplicate & Repost Purging (Jaccard >= 0.80)]...")
    for r in exact_unique:
        r["ngrams"] = get_char_ngrams(r["claim_text"], n=4)

    clean_records = []
    near_duplicates_purged = 0
    near_dupe_threshold = 0.80

    for candidate in exact_unique:
        cand_ngrams = candidate["ngrams"]
        is_near_dupe = False
        for kept in clean_records:
            if candidate["disease_category"] == kept["disease_category"]:
                sim = jaccard_similarity(cand_ngrams, kept["ngrams"])
                if sim >= near_dupe_threshold:
                    is_near_dupe = True
                    near_duplicates_purged += 1
                    break
        if not is_near_dupe:
            clean_records.append(candidate)

    print(f"  * Near-duplicates/reposts purged: {near_duplicates_purged}")
    print(f"  * Clean Unique Claims Retained:   {len(clean_records)}")

    # 6. Report Cleaned Class Distribution (Problem 2)
    clean_dist = Counter([r["disease_category"] for r in clean_records])
    print("\n" + "=" * 75)
    print("PROBLEM 2: FINAL CLEANED DATASET CLASS DISTRIBUTION (ALL 22 CATEGORIES)")
    print("=" * 75)
    print(f"{'#':<3} | {'Disease Category':<35} | {'Clean Count':>12} | {'Percentage':>10}")
    print("-" * 75)
    for idx, cat in enumerate(DISEASE_CATEGORIES, 1):
        cnt = clean_dist.get(cat, 0)
        pct = (cnt / len(clean_records) * 100) if clean_records else 0
        print(f"{idx:<3} | {cat:<35} | {cnt:>12d} | {pct:>9.2f}%")
    print("-" * 75)
    print(f"{'TOTAL':<41} | {len(clean_records):>12d} | 100.00%")
    print("=" * 75)

    # 7. Stratified Partitioning: Train (70%), Val (15%), Held-Out Test (15%)
    # STRATIFICATION EXECUTED BEFORE ANY AUGMENTATION
    labels = [r["label"] for r in clean_records]

    train_idx, temp_idx = train_test_split(
        range(len(clean_records)),
        test_size=0.30,
        random_state=42,
        stratify=labels
    )
    temp_labels = [labels[i] for i in temp_idx]
    val_idx_sub, test_idx_sub = train_test_split(
        range(len(temp_idx)),
        test_size=0.50,
        random_state=42,
        stratify=temp_labels
    )
    val_idx = [temp_idx[i] for i in val_idx_sub]
    test_idx = [temp_idx[i] for i in test_idx_sub]

    def serialize_records(indices, split_name):
        res = []
        for i in indices:
            r = dict(clean_records[i])
            r.pop("ngrams", None)
            r["split"] = split_name
            res.append(r)
        return res

    train_data = serialize_records(train_idx, "train")
    val_data = serialize_records(val_idx, "val")
    test_data = serialize_records(test_idx, "test")

    # 8. Leakage Verification Across Splits
    train_texts = {r["claim_text"].lower() for r in train_data}
    val_texts = {r["claim_text"].lower() for r in val_data}
    test_texts = {r["claim_text"].lower() for r in test_data}

    overlap_train_val = train_texts.intersection(val_texts)
    overlap_train_test = train_texts.intersection(test_texts)
    overlap_val_test = val_texts.intersection(test_texts)

    assert len(overlap_train_val) == 0, f"LEAKAGE ERROR: {len(overlap_train_val)} train-val overlap!"
    assert len(overlap_train_test) == 0, f"LEAKAGE ERROR: {len(overlap_train_test)} train-test overlap!"
    assert len(overlap_val_test) == 0, f"LEAKAGE ERROR: {len(overlap_val_test)} val-test overlap!"

    print("\n[Problem 1: Leakage Audit Check Across Splits]:")
    print(f"  * Train / Validation Overlap: {len(overlap_train_val)} claims [0.0% LEAKAGE — PASS]")
    print(f"  * Train / Test Overlap:       {len(overlap_train_test)} claims [0.0% LEAKAGE — PASS]")
    print(f"  * Val / Test Overlap:         {len(overlap_val_test)} claims [0.0% LEAKAGE — PASS]")

    # 9. Freezing Splits & Writing Cryptographic Manifest
    train_path = os.path.join(SPLITS_DIR, "train_frozen.json")
    val_path = os.path.join(SPLITS_DIR, "val_frozen.json")
    test_path = os.path.join(SPLITS_DIR, "test_frozen.json")

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=2)
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2)

    train_sha = compute_sha256(train_path)
    val_sha = compute_sha256(val_path)
    test_sha = compute_sha256(test_path)

    manifest_info = {
        "dataset_name": "MedVerify AI 22-Disease Categories Frozen Splits",
        "clean_sample_count": len(clean_records),
        "disease_categories_count": len(DISEASE_CATEGORIES),
        "disease_categories": DISEASE_CATEGORIES,
        "protocol": "Pre-registered Leakage-Free Stratified 22-Category Protocol",
        "splits": {
            "train": {
                "file": "train_frozen.json",
                "sample_count": len(train_data),
                "sha256": train_sha,
                "class_distribution": dict(Counter([r["disease_category"] for r in train_data]))
            },
            "val": {
                "file": "val_frozen.json",
                "sample_count": len(val_data),
                "sha256": val_sha,
                "class_distribution": dict(Counter([r["disease_category"] for r in val_data]))
            },
            "test": {
                "file": "test_frozen.json",
                "sample_count": len(test_data),
                "sha256": test_sha,
                "class_distribution": dict(Counter([r["disease_category"] for r in test_data]))
            }
        },
        "audit_summary": {
            "raw_candidates": len(raw_candidates),
            "exact_duplicates_purged": exact_duplicates_purged,
            "near_duplicates_purged": near_duplicates_purged,
            "final_clean_samples": len(clean_records),
            "train_val_overlap": len(overlap_train_val),
            "train_test_overlap": len(overlap_train_test),
            "val_test_overlap": len(overlap_val_test)
        }
    }

    manifest_path = os.path.join(SPLITS_DIR, "splits_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_info, f, indent=2)

    report_path = os.path.join(REPORTS_DIR, "phase1_dataset_leakage_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(manifest_info, f, indent=2)

    print("\n" + "=" * 80)
    print("ALL 22 DISEASE CATEGORIES FROZEN AND REGISTERED:")
    print(f"  * Train Set (70%): {len(train_data):4d} claims | SHA-256: {train_sha[:16]}... ({train_path})")
    print(f"  * Val Set   (15%): {len(val_data):4d} claims | SHA-256: {val_sha[:16]}... ({val_path})")
    print(f"  * Test Set  (15%): {len(test_data):4d} claims | SHA-256: {test_sha[:16]}... ({test_path})")
    print(f"\n  Full Audit Report saved to: {report_path}")
    print("=" * 80)
    print("PHASE 1 (ALL 22 DISEASE CATEGORIES) COMPLETE [PASS]")
    print("=" * 80)

if __name__ == "__main__":
    audit_and_freeze()
