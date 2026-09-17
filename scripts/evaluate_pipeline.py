"""
MedVerify AI - Stage 13: End-to-End Pipeline Evaluation

Evaluates the full verification pipeline against PubHealth gold labels.
1. Loads gold-labeled claims from med_datasets/processed/pubhealth_verified_claims.json
2. Samples 200 claims (stratified by verdict)
3. Runs each through the orchestrator pipeline
4. Compares predicted verdict vs. gold label
5. Reports: Accuracy, Precision, Recall, F1 (per-verdict + macro)
"""

import os
import sys
import json
import random
import time
import logging
from collections import Counter, defaultdict

# Add backend to path
sys.path.insert(0, os.path.join(os.getcwd(), "medverify-ai-backend"))

from sklearn.metrics import classification_report, accuracy_score, f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.getcwd()
DATASET_PATH = os.path.join(PROJECT_ROOT, "med_datasets", "processed", "pubhealth_verified_claims.json")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "med_datasets", "evaluation", "pipeline_evaluation_results.json")

SAMPLE_SIZE = 200
RANDOM_SEED = 42

# Gold label to standard verdict mapping
GOLD_TO_VERDICT = {
    "true": "Supported",
    "false": "Contradicted",
    "mixture": "Mixed",
    "unproven": "Insufficient Evidence",
    "unknown": "Insufficient Evidence",
}

# Pipeline verdict normalization
PIPELINE_VERDICT_NORMALIZE = {
    "Supported": "Supported",
    "Contradicted": "Contradicted",
    "Mixed": "Mixed",
    "Insufficient Evidence": "Insufficient Evidence",
}

print("=" * 80)
print("MEDVERIFY AI — STAGE 13: END-TO-END PIPELINE EVALUATION")
print("=" * 80)

# 1. Load dataset
print(f"\n[1/5] Loading PubHealth gold-labeled claims from: {DATASET_PATH}")
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    all_claims = json.load(f)

# Filter claims with valid gold labels
valid_claims = [
    c for c in all_claims
    if c.get("gold_label") in GOLD_TO_VERDICT and c.get("claim_text", "").strip()
]
print(f"  Total claims with valid gold labels: {len(valid_claims)}")

# 2. Stratified sampling
print(f"\n[2/5] Stratified sampling of {SAMPLE_SIZE} claims...")
random.seed(RANDOM_SEED)

by_label = defaultdict(list)
for c in valid_claims:
    by_label[c["gold_label"]].append(c)

sample = []
per_label = SAMPLE_SIZE // len(by_label)
for label, claims in by_label.items():
    n = min(per_label, len(claims))
    sample.extend(random.sample(claims, n))

# Fill remaining if needed
remaining = SAMPLE_SIZE - len(sample)
if remaining > 0:
    pool = [c for c in valid_claims if c not in sample]
    sample.extend(random.sample(pool, min(remaining, len(pool))))

print(f"  Sample size: {len(sample)}")
print(f"  Distribution: {dict(Counter(c['gold_label'] for c in sample))}")

# 3. Run pipeline
print("\n[3/5] Running verification pipeline on sampled claims...")
print("  (This may take several minutes depending on model loading and PubMed API calls)")

# Lazy import orchestrator to avoid loading models until needed
from app.services.orchestrator import get_orchestrator
orchestrator = get_orchestrator()

results = []
gold_verdicts = []
pred_verdicts = []
errors = 0

for i, claim in enumerate(sample):
    claim_text = claim["claim_text"]
    gold_label = claim["gold_label"]
    expected_verdict = GOLD_TO_VERDICT[gold_label]

    try:
        result = orchestrator.verify(claim_text)
        predicted_verdict = result.get("verdict", "Insufficient Evidence")

        if result.get("status") == "REFUSED_SAFETY":
            predicted_verdict = "Insufficient Evidence"  # Safety refusals count as inconclusive

        # Normalize
        predicted_verdict = PIPELINE_VERDICT_NORMALIZE.get(predicted_verdict, "Insufficient Evidence")

        gold_verdicts.append(expected_verdict)
        pred_verdicts.append(predicted_verdict)

        results.append({
            "claim_id": claim.get("claim_id", f"sample-{i}"),
            "claim_text": claim_text[:200],
            "gold_label": gold_label,
            "expected_verdict": expected_verdict,
            "predicted_verdict": predicted_verdict,
            "correct": expected_verdict == predicted_verdict,
            "credibility_score": result.get("credibility_score"),
            "disease_category": result.get("disease_category"),
            "elapsed_seconds": result.get("elapsed_seconds", 0),
        })

        if (i + 1) % 20 == 0:
            correct_so_far = sum(1 for r in results if r["correct"])
            print(f"  Progress: {i+1}/{len(sample)} | Running accuracy: {correct_so_far/(i+1)*100:.1f}%")

    except Exception as e:
        errors += 1
        logger.warning(f"  Error on claim {i}: {e}")
        gold_verdicts.append(expected_verdict)
        pred_verdicts.append("Insufficient Evidence")
        results.append({
            "claim_id": claim.get("claim_id", f"sample-{i}"),
            "claim_text": claim_text[:200],
            "gold_label": gold_label,
            "expected_verdict": expected_verdict,
            "predicted_verdict": "ERROR",
            "correct": False,
            "error": str(e),
        })

# 4. Compute metrics
print("\n[4/5] Computing evaluation metrics...")

verdict_labels = ["Supported", "Contradicted", "Mixed", "Insufficient Evidence"]
accuracy = accuracy_score(gold_verdicts, pred_verdicts)
macro_f1 = f1_score(gold_verdicts, pred_verdicts, labels=verdict_labels, average="macro", zero_division=0)

report_str = classification_report(
    gold_verdicts, pred_verdicts,
    labels=verdict_labels,
    zero_division=0,
)

print(f"\n{'='*70}")
print("  PIPELINE EVALUATION RESULTS")
print(f"{'='*70}")
print(f"  Total Evaluated: {len(sample)}")
print(f"  Errors:          {errors}")
print(f"  Accuracy:        {accuracy*100:.2f}%")
print(f"  Macro-F1:        {macro_f1*100:.2f}%")
print(f"\n{report_str}")

# 5. Save results
print(f"\n[5/5] Saving evaluation results to: {OUTPUT_PATH}")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

eval_output = {
    "evaluation_type": "end_to_end_pipeline",
    "sample_size": len(sample),
    "errors": errors,
    "accuracy": round(accuracy, 4),
    "macro_f1": round(macro_f1, 4),
    "classification_report": report_str,
    "per_claim_results": results,
}

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(eval_output, f, indent=2)

print(f"\n{'='*80}")
print("STAGE 13 PIPELINE EVALUATION COMPLETE")
print(f"{'='*80}")
