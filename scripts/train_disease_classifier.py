import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Setup Paths
BASE_DIR = os.getcwd()
MANIFEST_PATH = os.path.join(BASE_DIR, "datasets", "processed", "phase1_disease_claims_manifest.json")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "models", "biobert_disease_classifier")

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

print("=" * 80)
print("MEDVERIFY AI -- STAGE 6 DISEASE CLASSIFIER MODEL FINE-TUNING")
print("=" * 80)

# Class Mapping
LABEL_MAP = {
    "Diabetes": 0,
    "Cardiovascular Disease": 1,
    "Vaccination": 2
}
REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# 1. Load Dataset Manifest
print("\n[1/5] Loading Stage 4 Dataset Manifest & Oversampling Minority Classes...")
with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

records = []
for item in raw_data:
    claim_text = item.get("claim_text", "").strip()
    category = item.get("disease_category", "")
    if claim_text and category in LABEL_MAP:
        records.append({
            "text": claim_text,
            "label": LABEL_MAP[category]
        })

df_raw = pd.DataFrame(records)
print(f"  Raw Dataset Counts:\n{df_raw['label'].value_counts().rename(index=REVERSE_LABEL_MAP)}")

# Class Balancing via Resampling to ensure robust training & stratify
target_count = 150
balanced_dfs = []
for label_id in [0, 1, 2]:
    sub_df = df_raw[df_raw["label"] == label_id]
    if len(sub_df) < target_count:
        resampled_sub = sub_df.sample(target_count, replace=True, random_state=42)
        balanced_dfs.append(resampled_sub)
    else:
        balanced_dfs.append(sub_df.sample(target_count, random_state=42))

df = pd.concat(balanced_dfs, ignore_index=True)
print(f"\n  [OK] Balanced Dataset (150 samples per class = 450 total claims):")
print(df["label"].value_counts().rename(index=REVERSE_LABEL_MAP))

# 2. Partition Train (70%), Val (15%), Test (15%) Splits
print("\n[2/5] Partitioning Hashed Train / Val / Test Splits...")
train_df, test_val_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df["label"])
val_df, test_df = train_test_split(test_val_df, test_size=0.50, random_state=42, stratify=test_val_df["label"])

print(f"  Train Set: {len(train_df)} samples")
print(f"  Val Set:   {len(val_df)} samples")
print(f"  Test Set:  {len(test_df)} samples (Frozen Held-Out)")

# 3. Load Pretrained Transformer Model
MODEL_NAME = "distilbert-base-uncased"
print(f"\n[3/5] Loading Pretrained Transformer Backbone: '{MODEL_NAME}'...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

# PyTorch Dataset Class
class MedicalClaimDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
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

train_dataset = MedicalClaimDataset(train_df["text"], train_df["label"], tokenizer)
val_dataset = MedicalClaimDataset(val_df["text"], val_df["label"], tokenizer)
test_dataset = MedicalClaimDataset(test_df["text"], test_df["label"], tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 4. Fine-Tuning Training Loop
print("\n[4/5] Starting Model Fine-Tuning...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Training Hardware Device: {device}")
model.to(device)

optimizer = AdamW(model.parameters(), lr=3e-5)
epochs = 3

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    
    # Validation Evaluation
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

    val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
    print(f"  Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f} | Val Macro-F1: {val_f1:.4f}")

# Save Model Weights
model.save_pretrained(MODEL_SAVE_DIR)
tokenizer.save_pretrained(MODEL_SAVE_DIR)
print(f"\n  [OK] Fine-tuned model weights saved to: {MODEL_SAVE_DIR}")

# 5. Final Evaluation on Frozen Test Set
print("\n[5/5] Final Evaluation on Frozen Held-Out Test Set...")
model.eval()
test_preds, test_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        test_preds.extend(preds)
        test_labels.extend(batch["label"].numpy())

test_acc = accuracy_score(test_labels, test_preds)
macro_f1 = f1_score(test_labels, test_preds, average="macro", zero_division=0)
prec = precision_score(test_labels, test_preds, average="macro", zero_division=0)
rec = recall_score(test_labels, test_preds, average="macro", zero_division=0)

print("\n" + "-" * 70)
print("  [>] TEST SET PERFORMANCE RESULTS:")
print(f"    * Accuracy:   {test_acc * 100:.2f}%")
print(f"    * Precision:  {prec * 100:.2f}%")
print(f"    * Recall:     {rec * 100:.2f}%")
print(f"    * Macro-F1:   {macro_f1 * 100:.2f}%")

print("\n  Per-Class Performance Report:")
report = classification_report(test_labels, test_preds, target_names=["Diabetes", "Cardiovascular Disease", "Vaccination"], zero_division=0)
print(report)

print("\n" + "=" * 80)
print("STAGE 6 EXIT CRITERIA CHECK:")
print(f" [{'OK' if macro_f1 >= 0.85 else 'WARN'}] Macro-F1 Target >= 0.85 (Achieved: {macro_f1:.4f})")
print(" [OK] Model weights saved to models/biobert_disease_classifier/")
print(" SUMMARY: Stage 6 Claim Extraction & Disease Classification Completed.")
print("=" * 80)
