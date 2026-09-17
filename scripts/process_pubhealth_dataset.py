"""
MedVerify AI — PubHealth Dataset Processor & Population Benchmark Generator

Parses the user's PubHealth dataset files from 'med_datasets':
1. Reads 'pubhealth_medical_extracted_hybrid(1).jsonl' (structured claim + population annotations)
2. Reads 'pubhealth_medical.csv.xls' (gold fact-checking labels & explanations)
3. Merges claims with gold labels & population profiles
4. Outputs:
   - med_datasets/processed/pubhealth_verified_claims.json
   - med_datasets/evaluation/population_mismatch_benchmark.json
"""

import os
import sys
import json
import csv
import pandas as pd

# Add backend path
sys.path.insert(0, os.path.join(os.getcwd(), "medverify-ai-backend"))

PROJECT_ROOT = os.getcwd()
PUBHEALTH_DIR = os.path.join(PROJECT_ROOT, "med_datasets")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "med_datasets", "processed")
EVAL_DIR = os.path.join(PROJECT_ROOT, "med_datasets", "evaluation")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

print("=" * 80)
print("MEDVERIFY AI — PROCESSING PUBHEALTH DATASET FOR VERIFICATION & EVALUATION")
print("=" * 80)

# -----------------------------------------------------------------------------
# 1. READ GOLD LABELS FROM PUBHEALTH CSV
# -----------------------------------------------------------------------------
csv_path = os.path.join(PUBHEALTH_DIR, "pubhealth_medical.csv.xls")

gold_claims_dict = {}

if os.path.exists(csv_path):
    print(f"Reading gold labels CSV: {os.path.basename(csv_path)}...")
    try:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            cid = str(row.get("claim_id", "")).strip()
            claim_text = str(row.get("claim", "")).strip()
            label = str(row.get("label", "")).strip().lower()
            explanation = str(row.get("explanation", "")).strip()
            main_text = str(row.get("main_text", "")).strip()
            sources = str(row.get("sources", "")).strip()

            if cid and claim_text and claim_text != "nan":
                # Standardize verdict
                verdict_map = {
                    "true": "Supported",
                    "false": "Contradicted",
                    "mixture": "Mixed",
                    "unproven": "Insufficient Evidence",
                    "unverified": "Insufficient Evidence",
                }
                standard_verdict = verdict_map.get(label, "Insufficient Evidence")

                gold_claims_dict[cid] = {
                    "claim_id": cid,
                    "claim_text": claim_text,
                    "gold_label": label,
                    "standard_verdict": standard_verdict,
                    "explanation": explanation if explanation != "nan" else "",
                    "main_text": main_text if main_text != "nan" else "",
                    "sources": sources.split(",") if sources and sources != "nan" else [],
                }
    except Exception as e:
        print(f"  [!] Warning reading {csv_path}: {e}")
else:
    print(f"  [!] Gold labels CSV not found: {csv_path}")

print(f"[OK] Total PubHealth Gold Records Indexed: {len(gold_claims_dict)}")

# -----------------------------------------------------------------------------
# 2. READ EXTRACTED POPULATION ANNOTATIONS (JSONL)
# -----------------------------------------------------------------------------
jsonl_path = os.path.join(PUBHEALTH_DIR, "pubhealth_medical_extracted_hybrid(1).jsonl")
extracted_records = []

if os.path.exists(jsonl_path):
    print(f"\nReading extracted population JSONL: {os.path.basename(jsonl_path)}...")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                extracted_records.append(rec)
            except Exception:
                pass
    print(f"[OK] Total Extracted Population Records: {len(extracted_records)}")

# -----------------------------------------------------------------------------
# 3. MERGE POPULATION ANNOTATIONS WITH GOLD BENCHMARK
# -----------------------------------------------------------------------------
merged_dataset = []
benchmark_pairs = []

for rec in extracted_records:
    cid = str(rec.get("claim_id", ""))
    claim_text = rec.get("claim", "")
    if not claim_text:
        continue

    gold_info = gold_claims_dict.get(cid, {})

    pop_data = rec.get("population") or {}
    age_data = rec.get("age") or {}
    sex_data = rec.get("sex_gender") or {}
    region_data = rec.get("region") or {}

    pop_val = pop_data.get("value") if isinstance(pop_data, dict) else None
    age_val = age_data.get("value") if isinstance(age_data, dict) else None
    sex_val = sex_data.get("value") if isinstance(sex_data, dict) else None
    region_val = region_data.get("value") if isinstance(region_data, dict) else None

    item = {
        "claim_id": f"pubhealth-{cid}",
        "claim_text": claim_text,
        "gold_label": gold_info.get("gold_label", "unknown"),
        "standard_verdict": gold_info.get("standard_verdict", "Insufficient Evidence"),
        "explanation": gold_info.get("explanation", ""),
        "sources": gold_info.get("sources", []),
        "population_annotation": {
            "population": pop_val,
            "age": age_val,
            "sex": sex_val,
            "region": region_val,
        },
        "intervention": rec.get("intervention", {}).get("value") if isinstance(rec.get("intervention"), dict) else None,
        "outcome": rec.get("outcome", {}).get("value") if isinstance(rec.get("outcome"), dict) else None,
    }
    merged_dataset.append(item)

    # Format for population applicability benchmark if population info exists
    if age_val or sex_val or region_val:
        benchmark_pairs.append({
            "claim_id": item["claim_id"],
            "claim": claim_text,
            "gold_verdict": item["standard_verdict"],
            "gold_claim_population": {
                "age_raw": age_val,
                "sex_raw": sex_val,
                "region_raw": region_val,
            },
            "explanation": item["explanation"],
            "sources": item["sources"],
        })

print(f"\n[OK] Successfully merged {len(merged_dataset)} PubHealth records.")
print(f"[OK] Extracted {len(benchmark_pairs)} population-annotated benchmark items.")

# -----------------------------------------------------------------------------
# 4. SAVE PROCESSED BENCHMARK FILES
# -----------------------------------------------------------------------------
out_dataset_path = os.path.join(PROCESSED_DIR, "pubhealth_verified_claims.json")
with open(out_dataset_path, "w", encoding="utf-8") as f:
    json.dump(merged_dataset, f, indent=2)

out_bench_path = os.path.join(EVAL_DIR, "population_mismatch_benchmark.json")
with open(out_bench_path, "w", encoding="utf-8") as f:
    json.dump(benchmark_pairs, f, indent=2)

print("\n" + "=" * 80)
print("PUBHEALTH DATASET PROCESSING COMPLETE [SUCCESS]")
print(f" Saved Processed Claims Dataset : {out_dataset_path}")
print(f" Saved Population Benchmark    : {out_bench_path}")
print("=" * 80)
