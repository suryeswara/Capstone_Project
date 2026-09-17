"""
MedVerify AI — Final Evaluation Suite (experiment-v2)
=====================================================
STEP 4: BioBERT Medical / Disease Classifier Evaluation

Evaluates the fine-tuned BioBERT classifier on the frozen held-out test split
(med_datasets/splits/test_frozen.json) across all 22 disease categories:
- Primary Metric: Macro-F1
- Secondary Metrics: Accuracy, Precision (Macro), Recall (Macro)
- Target Disease Category Breakdown: Diabetes F1, Cardiovascular F1, Vaccination F1
- Full 22x22 Confusion Matrix
- Exports report to results/step02_biobert_eval.json
"""

import os
import sys
import json
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from config import RESULTS_DIR, BIOBERT_MODEL_DIR, TEST_FROZEN_PATH, DISEASE_CATEGORIES
from data_loader import load_json, save_json

OUTPUT_FILE = os.path.join(RESULTS_DIR, "step02_biobert_eval.json")


def evaluate_biobert(sample_size: int = None) -> dict:
    print("=" * 80)
    print("EXPERIMENT-V2 — STEP 4: BIOBERT MEDICAL CLAIM CLASSIFIER EVALUATION")
    print("=" * 80)
    print("Protocol Statement: BioBERT was evaluated as the implemented medical claim classifier.")

    if not os.path.exists(BIOBERT_MODEL_DIR):
        raise FileNotFoundError(f"BioBERT model directory not found: {BIOBERT_MODEL_DIR}")

    test_records = load_json(TEST_FROZEN_PATH)
    if sample_size:
        test_records = test_records[:sample_size]
    print(f"Loaded Frozen Test Split: {len(test_records)} claims")

    # Load BioBERT Tokenizer & Model
    print(f"Loading BioBERT from: {BIOBERT_MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(BIOBERT_MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(BIOBERT_MODEL_DIR)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Executing inference on device: {device}...")

    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for i, item in enumerate(test_records):
            text = item.get("claim_text", "")
            gold_label = item.get("label", 0)
            true_labels.append(gold_label)

            inputs = tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt"
            ).to(device)
            outputs = model(**inputs)
            pred = int(torch.argmax(outputs.logits, dim=1).item())
            pred_labels.append(pred)

    num_classes = len(DISEASE_CATEGORIES)
    acc = float(accuracy_score(true_labels, pred_labels))
    prec = float(precision_score(true_labels, pred_labels, average="macro", zero_division=0))
    rec = float(recall_score(true_labels, pred_labels, average="macro", zero_division=0))
    macro_f1 = float(f1_score(true_labels, pred_labels, average="macro", zero_division=0))
    per_class_f1_list = f1_score(true_labels, pred_labels, average=None, zero_division=0)
    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(num_classes))).tolist()

    per_class_f1_map = {}
    for idx, cname in enumerate(DISEASE_CATEGORIES):
        val = float(per_class_f1_list[idx]) if idx < len(per_class_f1_list) else 0.0
        per_class_f1_map[cname] = round(val, 4)

    # Key categories requested in prompt:
    # Diabetes, Cardiovascular / Heart Disease, Vaccination
    diabetes_f1 = per_class_f1_map.get("Diabetes", 0.0)
    cardio_f1 = per_class_f1_map.get("Heart Disease", per_class_f1_map.get("Cardiovascular Disease", 0.0))
    vaccination_f1 = per_class_f1_map.get("Vaccination", 0.0)

    print("\n" + "=" * 80)
    print("TABLE 1 — BIOBERT MEDICAL CLAIM CLASSIFIER PERFORMANCE")
    print("=" * 80)
    print(f"{'Metric':<30} | {'BioBERT Score':>15}")
    print("-" * 50)
    print(f"{'Accuracy':<30} | {acc * 100:14.2f}%")
    print(f"{'Macro Precision':<30} | {prec * 100:14.2f}%")
    print(f"{'Macro Recall':<30} | {rec * 100:14.2f}%")
    print(f"{'Macro-F1 (Primary)':<30} | {macro_f1 * 100:14.2f}%")
    print(f"{'Diabetes F1':<30} | {diabetes_f1 * 100:14.2f}%")
    print(f"{'Cardiovascular F1':<30} | {cardio_f1 * 100:14.2f}%")
    print(f"{'Vaccination F1':<30} | {vaccination_f1 * 100:14.2f}%")
    print("=" * 80)

    results = {
        "evaluation_name": "Table 1 — BioBERT Evaluation",
        "model": "BioBERT (fine-tuned 22-disease category classifier)",
        "sample_size": len(test_records),
        "primary_metric": "Macro-F1",
        "metrics": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "macro_f1": round(macro_f1, 4),
            "diabetes_f1": round(diabetes_f1, 4),
            "cardiovascular_f1": round(cardio_f1, 4),
            "vaccination_f1": round(vaccination_f1, 4),
        },
        "per_class_f1": per_class_f1_map,
        "confusion_matrix": cm,
    }

    save_json(results, OUTPUT_FILE)
    print(f"\n[OK] BioBERT evaluation report saved to: {OUTPUT_FILE}")
    return results


if __name__ == "__main__":
    evaluate_biobert()
