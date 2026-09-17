"""
Extract positive (Supported) and negative (Contradicted) public health claims
from the user's PubHealth dataset.
"""

import os
import json

PROJECT_ROOT = os.getcwd()
DATASET_PATH = os.path.join(PROJECT_ROOT, "med_datasets", "processed", "pubhealth_verified_claims.json")

if not os.path.exists(DATASET_PATH):
    print("Dataset not found!")
    exit(1)

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    claims = json.load(f)

supported = [c for c in claims if c.get("standard_verdict") == "Supported" or c.get("gold_label") == "true"]
contradicted = [c for c in claims if c.get("standard_verdict") == "Contradicted" or c.get("gold_label") == "false"]

print("=" * 80)
print(f"PUBHEALTH DATASET CLAIM SAMPLE (Total Claims: {len(claims)})")
print("=" * 80)

print("\n--- POSITIVE / SUPPORTED CLAIMS (TRUE) ---")
for idx, c in enumerate(supported[:6], 1):
    pop = c.get("population_annotation", {})
    age = pop.get("age") or "Not specified"
    sex = pop.get("sex") or "Not specified"
    print(f"{idx}. {c['claim_text']}")
    print(f"   Demographics: Age: {age} | Sex: {sex}")
    print(f"   Explanation: {c['explanation'][:150]}...")
    print()

print("\n--- NEGATIVE / CONTRADICTED CLAIMS (FALSE) ---")
for idx, c in enumerate(contradicted[:6], 1):
    pop = c.get("population_annotation", {})
    age = pop.get("age") or "Not specified"
    sex = pop.get("sex") or "Not specified"
    print(f"{idx}. {c['claim_text']}")
    print(f"   Demographics: Age: {age} | Sex: {sex}")
    print(f"   Explanation: {c['explanation'][:150]}...")
    print()
