# MedVerify AI — Stage 6 Model Training & Claim Extraction Specification

**Document Version:** 1.0.0  
**Stage:** 6 (Claim Extraction + Disease Classification)  
**Target Output Models:** `Qwen 3 / Llama 3.1` (Claim Extractor) & `BioBERT` (Disease Classifier)  
**Exit Criteria Bar:** Macro-F1 $\ge 0.85$ on frozen held-out test split  

---

## 1. Overview & Objectives

Stage 6 implements the front end of the AI/ML verification pipeline:
1. **Claim Extraction:** Converts raw, informal, or noisy user text into a singular, verifiable atomic claim assertion.
2. **Disease Classification:** Routes the extracted claim to one of the 3 Phase 1 disease domains (**Diabetes**, **Cardiovascular Disease**, **Vaccination**) using a fine-tuned biomedical transformer model (`BioBERT`).

```
 [Raw User Input] ──► [Claim Extraction LLM] ──► [Structured Atomic Claim]
                                                           │
                                                           ▼
 [Downstream RAG] ◄── [Domain Routing] ◄── [BioBERT Disease Classifier]
```

---

## 2. Subsystem A: Claim Extraction Engine

### Model Choice
* **Primary:** `Qwen 3 8B Instruct` or `Llama 3.1 8B Instruct` (Open-weight, Apache 2.0 / Community License, self-hostable).

### Prompt Engineering Schema
```text
SYSTEM PROMPT:
You are an expert medical NLP assistant for MedVerify AI. Your task is to extract the core verifiable medical claim assertion from the user's input text.
Rules:
1. Strip out personal anecdotes, emotional filler, greeting phrases, and slang.
2. Express the core assertion in a clear, declarative, scientific sentence.
3. Preserve original certainty language (e.g., if the user says "might cause", do not upgrade to "causes").

USER INPUT:
"Hey doc, my friend posted on Facebook that drinking green tea every morning cures type 2 diabetes completely without insulin. Is this true?"

EXTRACTED ATOMIC CLAIM:
"Green tea consumption cures type 2 diabetes completely without insulin therapy."
```

---

## 3. Subsystem B: Disease Classification Model (`BioBERT`)

### Model Backbone & Fine-Tuning Strategy
* **Base Model:** `dmis-lab/biobert-v1.1` (BERT pretrained on PubMed abstracts and PMC full-text articles).
* **Classification Head:** 3-class linear classification head with Softmax output.
* **Target Classes:**
  1. `Diabetes` (Label ID: 0)
  2. `Cardiovascular Disease` (Label ID: 1)
  3. `Vaccination` (Label ID: 2)

### Training Hyperparameters
* **Loss Function:** Cross-Entropy Loss (with Focal Loss weighting if class imbalance is detected).
* **Optimizer:** AdamW ($\text{lr} = 2 \times 10^{-5}$, $\text{weight\_decay} = 0.01$).
* **Batch Size:** 16
* **Epochs:** 4–5 epochs with early stopping on Validation Macro-F1.
* **Sequence Length:** 128 tokens.

---

## 4. Dataset Partitioning & Governance

Fine-tuning uses the Stage 4 dataset manifest (`phase1_disease_claims_manifest.json`):

| Split | Percentage | Purpose |
| :--- | :---: | :--- |
| **Train Set** | **70%** (~270 claims) | Fine-tuning `BioBERT` linear classification head. |
| **Validation Set** | **15%** (~58 claims) | Tuning learning rate and monitoring loss for early stopping. |
| **Held-Out Test Set** | **15%** (~59 claims) | Final performance sign-off against exit criteria. |

---

## 5. Evaluation Protocol & Exit Criteria

To prevent the minority-class recall failure seen in prior literature (Paper 1), evaluation reports both aggregate accuracy and per-class recall:

$$\text{Macro-F1} = \frac{\text{F1}_{\text{Diabetes}} + \text{F1}_{\text{Cardiovascular}} + \text{F1}_{\text{Vaccination}}}{3}$$

### Stage 6 Exit Criteria Bar:
- [ ] **Macro-F1 $\ge 0.85$** on the frozen held-out test split.
- [ ] **Per-Class Recall $\ge 0.80$** for Diabetes, Cardiovascular, and Vaccination individually.
- [ ] Model weights saved to `models/biobert_disease_classifier/`.
- [ ] Model version logged in API version metadata object (`diseaseClassifierVersion: "biobert-v2.1-finetuned"`).

---

## 6. Implementation Code Blueprint (`train_disease_classifier.py`)

The training script executes the fine-tuning workflow:
1. Loads dataset manifest from `datasets/processed/phase1_disease_claims_manifest.json`.
2. Tokenizes texts using `AutoTokenizer` from HuggingFace `BioBERT`.
3. Sets up PyTorch `DataLoader` and `Trainer`.
4. Runs fine-tuning loop and evaluates test set metrics.
5. Saves model weights and generates evaluation report.
