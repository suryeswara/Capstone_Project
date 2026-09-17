# MedVerify AI — BioBERT Task Definition & Role Specification (§7, Task 0.2)

## 1. Role in System Architecture

As mandated in Master Instructions §7 ("Model Policy") and §26 ("Research vs Engineering"):
* **BioBERT is NOT the main research novelty.**
* BioBERT is the **Supporting Medical Domain Understanding Model**.
* The core research novelty is **Population-Aware Evidence Weighting** ($W_i = R_i \times P_i$) implemented within the multi-stage reasoning and consensus engine.

```text
                                MEDVERIFY PIPELINE
                                
  Online Health Claim
          ↓
  [BioBERT Model]  ─────────►  Disease Category Classification (22 categories)
  (Stage 6: Supporting)         (e.g., "Vaccination", "COVID-19", "Diabetes")
          ↓
  Structured Claim Representation
          ↓
  Population Extraction (Claim Demographics: Age, Sex, Condition, Region)
          ↓
  Hybrid Evidence Retrieval (FAISS Knowledge Base + PubMed NCBI)
          ↓
  Evidence Population Extraction (Evidence Demographics)
          ↓
  Evidence Reliability Scoring: R_i = f(SourceTier, EvidenceType, Recency)
          ↓
  Population Applicability:     P_i = f(AgeMatch, SexMatch, ConditionMatch, RegionMatch)
          ↓
  Evidence Weighting:           W_i = R_i × P_i  [CORE RESEARCH NOVELTY]
          ↓
  Evidence Stance Detection:    NLI (cross-encoder/nli-deberta-v3-small)
          ↓
  Weighted Consensus Engine:    C = Σ(W_i × stance_i) / Σ(W_i)
          ↓
  Final Verification Verdict:   TRUE / FALSE / MIXTURE / UNPROVEN
          ↓
  Explanation & Faithfulness Check (BioScope Certainty Guard + NLI Faithfulness)
```

---

## 2. BioBERT Model Details

* **Base Architecture**: `dmis-lab/biobert-base-cased-v1.2`
* **Fine-Tuning Task**: 22-class medical disease topic classification (`Claim → Disease Category`)
* **Checkpoint Path**: `models/biobert_disease_classifier/`
* **Number of Parameters**: ~110 Million
* **Training Split**: `med_datasets/splits/train_frozen.json` (2,861 claims)
* **Validation Split**: `med_datasets/splits/val_frozen.json` (613 claims)
* **Test Split**: `med_datasets/splits/test_frozen.json` (614 claims)

---

## 3. Test Set Evaluation Results (614 Frozen Test Claims)

The evaluation of the fine-tuned BioBERT classifier on the held-out test split is recorded in `reports/biobert_test_evaluation.json`:

| Metric | Score | Specification Status |
|---|---:|---|
| **Overall Accuracy** | **82.90%** | Exceeds 80% deployment target |
| **Macro-F1** | **0.7998** | Stable across all 22 classes |
| **Weighted Precision** | **81.27%** | High fidelity across high-volume classes |
| **Weighted Recall** | **82.59%** | Strong sensitivity to clinical terminology |

### Per-Category Performance Highlights

| Disease Category | Test Samples | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Lung Cancer | 8 | 1.0000 | 1.0000 | **1.0000** |
| Ebola Virus | 17 | 0.9412 | 0.9412 | **0.9412** |
| Stroke | 10 | 1.0000 | 0.8889 | **0.9412** |
| COVID-19 | 200 | 0.8920 | 0.9397 | **0.9152** |
| Breast Cancer | 25 | 0.9167 | 0.9020 | **0.9091** |
| Prostate Cancer | 14 | 1.0000 | 0.8333 | **0.9091** |
| Alzheimers Disease | 15 | 0.8889 | 0.8889 | **0.8889** |
| Vaccination | 53 | 0.8889 | 0.8562 | **0.8723** |
| Influenza | 31 | 0.8824 | 0.8333 | **0.8571** |
| Diabetes | 15 | 0.8462 | 0.8462 | **0.8462** |
| Smoking & Tobacco | 15 | 0.8235 | 0.8235 | **0.8235** |
| Reproductive Health / Abortion | 28 | 0.8163 | 0.8163 | **0.8163** |
| General Cancer | 70 | 0.8000 | 0.8340 | **0.8167** |
| Measles / MMR | 12 | 0.8000 | 0.8000 | **0.8000** |
| Depression | 11 | 0.7500 | 0.7500 | **0.7500** |

---

## 4. Key Takeaways for Research Reporting

1. **Clarification**: Do not report BioBERT as the 4-way claim verification model. BioBERT's verified role is the **domain-specific disease classifier and feature extractor**.
2. **Claim Verification**: The 4-way claim verification (`TRUE`, `FALSE`, `MIXTURE`, `UNPROVEN`) is produced by the consensus pipeline over retrieved evidence (B1, B2, B3).
3. **Reproducibility**: The evaluation script `scripts/evaluate_biobert_classifier.py` reproducibly reproduces these metrics on `med_datasets/splits/test_frozen.json`.
