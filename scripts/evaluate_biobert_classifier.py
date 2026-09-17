"""
MedVerify AI — Phase 2: BioBERT Disease Classifier Test Evaluation (All 22 Categories)

Evaluates the fine-tuned 22-category BioBERT classifier strictly on:
- med_datasets/splits/test_frozen.json (614 claims across 22 disease categories)
- Computes:
  * Accuracy
  * Precision (Macro)
  * Recall (Macro)
  * Macro-F1 (Primary Metric)
  * Per-Class F1 for all 22 disease categories
  * Full Confusion Matrix (22 x 22)
- Exports report to reports/biobert_test_evaluation.json
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS_DIR = os.path.join(PROJECT_ROOT, "med_datasets", "splits")
TEST_FILE = os.path.join(SPLITS_DIR, "test_frozen.json")
MANIFEST_FILE = os.path.join(SPLITS_DIR, "splits_manifest.json")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "biobert_disease_classifier")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
OUTPUT_REPORT = os.path.join(REPORTS_DIR, "biobert_test_evaluation.json")

def evaluate_classifier():
    print("=" * 80)
    print("MEDVERIFY AI — PHASE 2: BIOBERT HELD-OUT TEST EVALUATION (ALL 22 CATEGORIES)")
    print("=" * 80)

    if not os.path.exists(TEST_FILE):
        raise FileNotFoundError(f"Test split not found at: {TEST_FILE}")

    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    class_names = manifest.get("disease_categories", [])
    num_classes = len(class_names)
    print(f"Total Disease Categories Evaluated: {num_classes}")

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_records = json.load(f)

    print(f"Loaded Frozen Held-Out Test Set: {len(test_records)} claims")
    test_df = pd.DataFrame(test_records)

    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(f"Fine-tuned BioBERT model directory not found at: {MODEL_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_preds = []
    test_labels = list(test_df["label"])

    print("\nExecuting Inference over 614 Held-Out Test Claims across 22 categories...")
    with torch.no_grad():
        for text in test_df["claim_text"]:
            inputs = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors="pt").to(device)
            outputs = model(**inputs)
            pred = int(torch.argmax(outputs.logits, dim=1).item())
            test_preds.append(pred)

    # Compute Core Metrics
    acc = float(accuracy_score(test_labels, test_preds))
    prec = float(precision_score(test_labels, test_preds, average="macro", zero_division=0))
    rec = float(recall_score(test_labels, test_preds, average="macro", zero_division=0))
    macro_f1 = float(f1_score(test_labels, test_preds, average="macro", zero_division=0))
    per_class_f1 = f1_score(test_labels, test_preds, average=None, zero_division=0)
    cm = confusion_matrix(test_labels, test_preds, labels=list(range(num_classes))).tolist()

    print("\n" + "=" * 80)
    print("BIOBERT HELD-OUT TEST EVALUATION SUMMARY TABLE:")
    print("=" * 80)
    print(f"{'Overall Metric':<40} | {'Result':>15}")
    print("-" * 80)
    print(f"{'Overall Accuracy':<40} | {acc * 100:14.2f}%")
    print(f"{'Macro Precision':<40} | {prec * 100:14.2f}%")
    print(f"{'Macro Recall':<40} | {rec * 100:14.2f}%")
    print(f"{'Macro-F1 (Primary Metric across 22 Classes)':<40} | {macro_f1 * 100:14.2f}%")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("PER-CLASS F1 BREAKDOWN (ALL 22 DISEASE CATEGORIES):")
    print("=" * 80)
    print(f"{'#':<3} | {'Disease Category':<35} | {'Test N':>8} | {'Per-Class F1':>15}")
    print("-" * 80)
    per_class_dict = {}
    for idx, cname in enumerate(class_names):
        c_support = sum(test_df["label"] == idx)
        f1_val = float(per_class_f1[idx])
        per_class_dict[cname] = round(f1_val, 4)
        print(f"{idx+1:<3} | {cname:<35} | {c_support:>8d} | {f1_val * 100:14.2f}%")
    print("=" * 80)

    # Save artifact
    evaluation_payload = {
        "evaluation_dataset": "med_datasets/splits/test_frozen.json",
        "sample_size": len(test_records),
        "disease_categories_count": num_classes,
        "disease_categories": class_names,
        "metrics": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "macro_f1": round(macro_f1, 4),
            "per_class_f1": per_class_dict
        },
        "confusion_matrix": cm
    }

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(evaluation_payload, f, indent=2)

    print(f"\n[OK] BioBERT 22-Category Test Evaluation Report saved to: {OUTPUT_REPORT}")
    print("=" * 80)
    gate_status = "PASS" if macro_f1 >= 0.85 else "INVESTIGATE"
    print(f"BIOBERT 22-CATEGORY EVALUATION STATUS: [{gate_status}] (Macro-F1: {macro_f1 * 100:.2f}%)")
    print("=" * 80)

    return evaluation_payload

if __name__ == "__main__":
    evaluate_classifier()
