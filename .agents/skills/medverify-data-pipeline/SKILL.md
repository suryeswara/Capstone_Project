---
name: medverify-data-pipeline
description: How MedVerify AI's training data is sourced, processed, and ingested — dataset sources (CoAID, PubHealth), the processing scripts, the manifest format, and how to add new disease categories or data sources.
---

# MedVerify AI — Data Pipeline Reference

## Dataset Sources

| Dataset | Disease Focus | Type | Script |
|---------|--------------|------|--------|
| **CoAID** | Vaccination / COVID-19 | Fact-checked fake/real health claims | `download_datasets.py` |
| **PubHealth** | General / Multi-disease | Public health claim verifications | `process_pubhealth_dataset.py` |
| **PubMed NCBI** | All (live) | Real-time literature abstracts | `download_real_disease_knowledge.py` |

---

## Processing Pipeline

### Step 1: Download Raw Data
```bash
python scripts/download_datasets.py          # CoAID
python scripts/download_real_disease_knowledge.py  # PubMed abstracts
```

### Step 2: Process Into Manifest
```bash
python scripts/process_pubhealth_dataset.py   # PubHealth → manifest entries
python scripts/sample_pubhealth_claims.py     # Sample subset for balance
```

### Step 3: Build Knowledge Base (FAISS Index)
```bash
python scripts/build_knowledge_base.py
# Reads:  datasets/processed/phase1_disease_claims_manifest.json
# Writes: vector_store/faiss_index.bin
#         vector_store/vector_metadata.json
```

### Step 4: Train Disease Classifier
```bash
python scripts/train_disease_classifier.py
# Reads:  datasets/processed/phase1_disease_claims_manifest.json
# Writes: models/biobert_disease_classifier/
```

---

## Manifest Format (`phase1_disease_claims_manifest.json`)

Each record in the manifest:
```json
{
  "claim_id": "coaid-fake_05-0",
  "claim_text": "Spraying chlorine on skin kills viruses in the body.",
  "explanation": "Fact-checked by CoAID dataset as contradicted health claim.",
  "disease_category": "Vaccination",
  "dataset_source": "CoAID",
  "fact_check_verdict": "Contradicted",    // "Supported" | "Contradicted" | "Unverified"
  "evidence_sources": ["WHO", "CDC"],       // Used to assign source_tier
  "pub_year": 2024
}
```

---

## Adding a New Disease Category

1. **Add training data** to the manifest with `"disease_category": "YourDisease"`
2. **Update label map** in `orchestrator.py`:
   ```python
   LABEL_MAP = {0: "Diabetes", 1: "Cardiovascular Disease", 2: "Vaccination", 3: "YourDisease"}
   ```
3. **Add decay rate** in `retrieval_engine.py`:
   ```python
   DISEASE_DECAY_RATES = {
       ...,
       "YourDisease": 0.05,  # tune based on how fast field evolves
   }
   ```
4. **Retrain classifier**: `python scripts/train_disease_classifier.py`
5. **Rebuild FAISS index**: `python scripts/build_knowledge_base.py`
6. **Seed new category** in DB: add to `disease_categories` table
7. Update frontend filters and labels to include the new category

---

## Adding a New Data Source

1. Write a download/process script in `scripts/` that outputs records in the manifest format above
2. Append records to `datasets/processed/phase1_disease_claims_manifest.json`
3. Re-run `build_knowledge_base.py` to rebuild the FAISS index
4. The new records will be automatically assigned `source_tier` based on the `evidence_sources` field

---

## Data Quality Notes

- **Reliability score at build time**: assigned heuristically
  - `Supported` verdict → 0.95
  - `Contradicted` verdict → 0.85
  - Other → 0.50
- **At runtime**: reliability is **recalculated** using the full formula `R_i = TierWeight × exp(-λ × age) × PeerReviewFactor`
- **Disease filter**: FAISS retrieval can be filtered by `disease_category` — only relevant chunks are returned

---

## `pub health all/` Directory

Contains raw PubHealth JSONL files extracted and processed for the hybrid retrieval corpus.
See `pubhealth_medical_extracted_hybrid.jsonl` for the hybrid-filtered records.

---

## Benchmarking

```bash
python scripts/benchmark_live_app.py
# Tests the full end-to-end pipeline against a set of known claims
# Reports: accuracy, retrieval hit rate, consensus score distribution
```
