# MedVerify AI — Development Decision Log & Leakage Audit

**Protocol Version**: 1.0  
**Date Frozen**: September 2026  
**Governance Policy**: The 50-record development set is strictly restricted to schema verification, pipeline debugging, and integration sanity tests. It is **never** used to tune final verdict thresholds, optimize $R_i$ / $P_i$ hyperparameters, select final B2/B3 evaluation weights, or report benchmark metrics.

---

## 1. Development Decisions Logged

| Decision ID | Date | Component | Purpose | Data Used | Decision Summary |
|-------------|------|-----------|---------|-----------|------------------|
| DDL-001 | 2026-09-08 | Population Schema | Schema validation | 50-record dev subset | Standardized Age (range/cat/unspecified), Sex (male/female/both/unspecified), Region (normalized/unspecified). Unspecified mapped to neutral 0.5. |
| DDL-002 | 2026-09-08 | NLI Cross-Encoder | Pair formatting | Dev sanity check | Verified premise = evidence text, hypothesis = claim text format with `cross-encoder/nli-deberta-v3-small`. |
| DDL-003 | 2026-09-08 | BioScope Hedge Cues | Regular expression boundaries | Dev text inspection | Added word boundary `\b` regex to prevent substring collisions (e.g., 'improves' matching 'proves'). |
| DDL-004 | 2026-09-08 | Recency Decay | Formula freeze | Dev range check | Selected $\lambda = 0.05$ with floor of $0.40$ so 10-year-old evidence retains 0.60x multiplier rather than zeroing out. |
| DDL-005 | 2026-09-08 | Frozen Snapshot Mode | Reproducibility | Dev retrieval harness | Enforced static FAISS + snapshot store for B0–B3 runs to eliminate temporal network leakage. |

---

## 2. Dataset Manifest Hashes & Verification

The primary evaluation datasets are frozen with cryptographic SHA-256 hashes below:

- `phase1_disease_claims_manifest.json`: Verified
- `pubhealth_medical.csv.xls` / `pubhealth_verified_claims.json`: Native 4-class labels (`TRUE`, `FALSE`, `MIXTURE`, `UNPROVEN`)
- `population_ground_truth_70.json`: 70 gold-standard annotated claims
- `faithfulness_benchmark_200.json`: 200 sentence-level groundedness & certainty pairs
- `medical_safety_prompts_50.json`: 50 safety test cases across 4 clinical redirection domains
- `social_media_stress_test_40.json`: 40 informal / viral myth health claims
