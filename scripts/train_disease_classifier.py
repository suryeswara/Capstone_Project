"""
MedVerify AI — Phase 2: BioBERT Disease Classifier Fine-Tuning (All 22 Categories)

Trains BioBERT (dmis-lab/biobert-base-cased-v1.2) strictly on:
- med_datasets/splits/train_frozen.json (2,861 claims across 22 disease categories)
- Validates on med_datasets/splits/val_frozen.json (613 claims)
- Saves best checkpoint to models/biobert_disease_classifier/
- Handles class imbalance via inverse-frequency class-weighted CrossEntropyLoss.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS_DIR = os.path.join(PROJECT_ROOT, "med_datasets", "splits")
MANIFEST_FILE = os.path.join(SPLITS_DIR, "splits_manifest.json")
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "models", "biobert_disease_classifier")

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

class MedicalClaimDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }

def train_classifier():
    print("=" * 80)
    print("MEDVERIFY AI — PHASE 2: BIOBERT 22-CATEGORY DISEASE CLASSIFIER FINE-TUNING")
    print("=" * 80)

    # Load categories from manifest
    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    disease_categories = manifest.get("disease_categories", [])
    num_classes = len(disease_categories)
    print(f"Total Target Disease Categories: {num_classes}")

    label_map = {cat: idx for idx, cat in enumerate(disease_categories)}

    train_file = os.path.join(SPLITS_DIR, "train_frozen.json")
    val_file = os.path.join(SPLITS_DIR, "val_frozen.json")

    with open(train_file, "r", encoding="utf-8") as f:
        train_records = json.load(f)
    with open(val_file, "r", encoding="utf-8") as f:
        val_records = json.load(f)

    print(f"Loaded Frozen Train: {len(train_records)} claims")
    print(f"Loaded Frozen Val:   {len(val_records)} claims")

    train_df = pd.DataFrame(train_records)
    val_df = pd.DataFrame(val_records)

    # Compute inverse class frequencies for weighted loss
    counts_map = train_df["label"].value_counts().to_dict()
    total_samples = len(train_df)
    class_weights = []
    for c_id in range(num_classes):
        cnt = counts_map.get(c_id, 1)
        w = total_samples / (num_classes * cnt)
        class_weights.append(w)

    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

    MODEL_NAME = "dmis-lab/biobert-base-cased-v1.2"
    print(f"\nLoading BioBERT backbone: '{MODEL_NAME}' with num_labels={num_classes}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )

    train_dataset = MedicalClaimDataset(train_df["claim_text"], train_df["label"], tokenizer, max_len=64)
    val_dataset = MedicalClaimDataset(val_df["claim_text"], val_df["label"], tokenizer, max_len=64)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Hardware: {device}")
    model.to(device)
    class_weights_tensor = class_weights_tensor.to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)

    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    epochs = 4
    best_val_macro_f1 = 0.0

    print("\nStarting Fine-Tuning Loop across 22 categories...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(batch["label"].numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        val_macro_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val Macro-F1: {val_macro_f1*100:.2f}%")

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            model.save_pretrained(MODEL_SAVE_DIR)
            tokenizer.save_pretrained(MODEL_SAVE_DIR)
            print(f"  --> Saved new best checkpoint to {MODEL_SAVE_DIR} (Val Macro-F1: {val_macro_f1:.4f})")

    print("\n" + "=" * 80)
    print(f"22-CATEGORY BIOBERT FINE-TUNING FINISHED. Best Validation Macro-F1: {best_val_macro_f1:.4f}")
    print(f"Model saved to: {MODEL_SAVE_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    train_classifier()
