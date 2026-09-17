# MedVerify AI — Dataset Specification & Hierarchy (§3, Task 0.1)

## 1. Overview & Dataset Hierarchy

This document formalizes and freezes the dataset hierarchy for MedVerify AI, resolving the relationship between the 8,770 raw records and the 4,088 clean frozen records.

```text
RAW PUBHEALTH DATASET
      8,770 records
(Train: 7,028 | Dev: 852 | Test: 890)
            ↓
Filtering & Domain Normalization:
  - Clinical disease category filtering (22 canonical categories)
  - Exclusion of non-medical / non-disease public health claims
  - Text length sanity filter (length >= 15 characters, non-null)
  - Ingestion of high-quality clinical seeds (CoAID + PubMed)
            ↓
Deduplication & Leakage Prevention:
  - Exact text deduplication
  - Near-duplicate / repost purging (char 4-gram Jaccard >= 0.80)
            ↓
CLEAN FROZEN CORPUS
      4,088 records
            ↓
Pre-registered Stratified 70 / 15 / 15 Split:
  ├── Train: 2,861 records (70%)
  ├── Val:     613 records (15%)
  └── Test:    614 records (15%) [FROZEN HELD-OUT TEST BENCHMARK]
```

---

## 2. Clarification on the "890 Test" vs "614 Test" Reference

* **The 890 number**: In the original raw PubHealth publication (Kotonya & Stenetorp, 2020), the uncurated test split contained 890 records. However, this raw split contained:
  * Non-disease and general policy claims (e.g., healthcare legislation, general food hygiene, traffic safety).
  * Records with unmapped or empty disease topics.
  * Near-duplicate claims cross-reposted from online forums.
* **The 614 number**: When the dataset was rigorously filtered to the 22 canonical clinical disease categories, deduplicated against training samples, and partitioned via stratified sampling (seed=42), the resulting held-out test set is exactly **614 records**.
* **Decision**: The **614 records** in `med_datasets/splits/test_frozen.json` constitute the **official, frozen held-out test set** for all MedVerify evaluations. The reference to 890 test claims is formally superseded.

---

## 3. Detailed Filtering & Removal Breakdown

| Stage | Filter / Operation | Records Retained | Records Removed | Rationale |
|---|---|---:|---:|---|
| **Raw Ingestion** | PubHealth Raw Medical Corpus (`pubhealth_medical.csv.xls`) | 8,770 | — | Initial baseline corpus |
| **Sanity Filter** | Text length >= 15 chars, non-empty, non-null | 8,754 | 16 | Malformed, truncated, or empty entries |
| **Domain Scope** | 22 Canonical Disease Categories filter | 3,923 | 4,831 | Removed claims outside medical disease scope (e.g., non-clinical public health, traffic laws, general politics, environmental health) |
| **Clinical Seeds** | Ingestion of Phase 1 Manifest (CoAID COVID-19/Vaccination seeds) | 4,432 | +509 added | Expanded clinical coverage for low-resource disease categories |
| **Exact Deduplication** | Purge identical `(claim_text.lower(), disease_category)` | 4,219 | 213 | Avoid identical repetitions across forums |
| **Near-Duplicate Purge** | 4-gram Jaccard similarity >= 0.80 within category | 4,088 | 131 | Purged cross-reposts, paraphrases, and social media echoes to prevent data leakage |
| **Final Clean Corpus** | **Frozen Dataset** | **4,088** | — | Clean, balanced, leakage-free dataset |

---

## 4. Frozen Split Specification

The 4,088 clean records were partitioned using stratified sampling across the 22 disease categories:

| Split | Count | Percentage | File Path | SHA-256 Checksum |
|---|---:|---:|---|---|
| **Train** | 2,861 | 70.0% | `med_datasets/splits/train_frozen.json` | `54f82fcb54002093838184a67553019283c5d84092fe90d9842e55dbec96b287` |
| **Val** | 613 | 15.0% | `med_datasets/splits/val_frozen.json` | `685eb5232e6da6bde41b4bfbbd28b6d6c90eed6d48f1ed7ff143e11cebbaf61e` |
| **Test** | 614 | 15.0% | `med_datasets/splits/test_frozen.json` | `3a25756f8f0673622daa75a79ab1972ba2581a81f29f790d3d600dda3873cdeb` |
| **Total** | **4,088** | **100.0%** | `med_datasets/splits/splits_manifest.json` | — |

### Leakage Verification
- Train ∩ Val Overlap: **0 records (0.0% leakage — PASS)**
- Train ∩ Test Overlap: **0 records (0.0% leakage — PASS)**
- Val ∩ Test Overlap: **0 records (0.0% leakage — PASS)**

---

## 5. Label Space

The official research verification labels across all splits and baselines are strictly standardized to:

```text
1. TRUE
2. FALSE
3. MIXTURE
4. UNPROVEN
```

Internal mapping from dataset annotations and consensus reasoning states:

| Dataset / Internal Label | Official Research Label | Definition |
|---|---|---|
| `true`, `Supported`, `LIKELY_SUPPORTED` | **TRUE** | Claim is corroborated by high-reliability, population-applicable evidence. |
| `false`, `Contradicted`, `LIKELY_REFUTED`, `REFUTED` | **FALSE** | Claim is directly refuted or disproven by reliable evidence. |
| `mixture`, `Mixed`, `MIXTURE` | **MIXTURE** | Claim contains both true and false elements, or evidence is divided. |
| `unproven`, `INCONCLUSIVE`, `Insufficient Evidence` | **UNPROVEN** | Scientific evidence is insufficient, absent, or inconclusive. |
