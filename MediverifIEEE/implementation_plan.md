# MedVerify — Master Instructions Compliance Gap Analysis & Implementation Plan

## Background

The Master Instructions define a 29-stage research pipeline and strict evaluation protocol for MedVerify. After thorough analysis of the current codebase, evaluation reports, configs, and experiment scripts, this plan identifies **critical gaps** between the specification and the current implementation, and proposes a phased remediation.

---

## Current State Summary

### ✅ What's Already Working

| Stage | Status | Evidence |
|---|---|---|
| 1. Dataset Loader | ✅ Done | [pubhealth_verified_claims.json](file:///c:/Users/Surya/Downloads/Capstone_Project/med_datasets/processed/pubhealth_verified_claims.json) |
| 2. Dataset Audit | ✅ Done | [audit_and_freeze_dataset.py](file:///c:/Users/Surya/Downloads/Capstone_Project/scripts/audit_and_freeze_dataset.py) |
| 3. Leakage Detection | ✅ Done | [phase1_dataset_leakage_report.json](file:///c:/Users/Surya/Downloads/Capstone_Project/Reports/phase1_dataset_leakage_report.json) — 0 leakage |
| 4. Train/Dev/Test Manifest | ✅ Done | [splits_manifest.json](file:///c:/Users/Surya/Downloads/Capstone_Project/med_datasets/splits/splits_manifest.json) — 4088 samples, 22 categories |
| 5. BioBERT Classifier | ✅ Done | [biobert_disease_classifier/](file:///c:/Users/Surya/Downloads/Capstone_Project/models/biobert_disease_classifier) |
| 6. BioBERT Evaluation | ✅ Done | Macro-F1 = 0.9953, Accuracy = 0.9944 |
| 7. Evidence Corpus + FAISS | ✅ Done | [vector_store/](file:///c:/Users/Surya/Downloads/Capstone_Project/vector_store) |
| 8. Hybrid Retrieval | ✅ Done | FAISS + PubMed live; Recall@5 = 1.0, nDCG@10 = 0.7552 |
| 10. Claim Population Extraction | ✅ Done | [claim_population.py](file:///c:/Users/Surya/Downloads/Capstone_Project/medverify-ai-backend/app/services/population/claim_population.py) |
| 11. Evidence Population Extraction | ✅ Done | [study_population.py](file:///c:/Users/Surya/Downloads/Capstone_Project/medverify-ai-backend/app/services/population/study_population.py) |
| 13. P_i Implementation | ✅ Done | [applicability.py](file:///c:/Users/Surya/Downloads/Capstone_Project/medverify-ai-backend/app/services/population/applicability.py) |
| 15. R_i Implementation | ✅ Done | [reliability.yaml](file:///c:/Users/Surya/Downloads/Capstone_Project/configs/reliability.yaml) — R_i = S_i × E_i × T_i |
| 16. W_i = R_i × P_i | ✅ Done | Both multiplicative & additive modes in [population.yaml](file:///c:/Users/Surya/Downloads/Capstone_Project/configs/population.yaml) |
| Consensus Engine | ✅ Done | NLI DeBERTa stance detection + BioScope hedge guard |
| Explanation Generator | ✅ Done | Grounded explanation with sentence faithfulness verification |
| Safety Guardrail | ✅ Done | Stage 11 emergency redirect, personal health refusal |
| Full Pipeline (Orchestrator) | ✅ Done | [orchestrator.py](file:///c:/Users/Surya/Downloads/Capstone_Project/medverify-ai-backend/app/services/orchestrator.py) |
| Frontend + Backend | ✅ Done | React/Vite + FastAPI |

---

## User Review Required

> [!IMPORTANT]
> **The current baseline experiment (B0–B3) uses only N=70 claims from `population_ground_truth_70.json` with simulated evidence.** The B3 Multiplicative achieves perfect 1.0 Macro-F1 because the experiment constructs ground-truth-aligned evidence rather than using real retrieval. This means the B2 vs B3 comparison, while statistically significant (McNemar p=0.00018), is **not a real end-to-end evaluation on the test set** — it's a controlled population-simulation experiment.

> [!WARNING]
> **Label space mismatch**: The spec requires labels `TRUE / FALSE / MIXTURE / UNPROVEN`. The current system uses `Supported / Contradicted / Mixed / Insufficient Evidence`. The orchestrator maps internal states to these labels. We need to decide: (a) keep the current labels and document the mapping, or (b) refactor to use the spec labels everywhere.

> [!WARNING]
> **Dataset discrepancy**: The spec references 8,770 records (PubHealth-derived). The current frozen splits contain 4,088 samples across 22 disease categories, which is a processed/filtered subset. The original raw file `pubhealth_medical.csv.xls` (44MB) appears to be the full dataset. We need to confirm this processed subset is intentional and document the filtering rationale.

---

## Open Questions

> [!IMPORTANT]
> **Q1**: The spec requires a "Population Challenge Dataset" of 120–150 claims. Currently there are 70 population ground-truth claims and a 43-item PubHealth experiment subset. Should we expand to 120–150, or is the combined 113 items acceptable with documented justification?

> [!IMPORTANT]
> **Q2**: The spec requires "Human Population Annotation" of 50–100 claim-evidence pairs with inter-annotator agreement. The current `population_ground_truth_70.json` has 70 items with Cohen's κ = 1.0 (self-annotated). Do you have access to a second annotator, or should we document single-annotator limitations?

> [!IMPORTANT]
> **Q3**: The spec requires B0–B3 to run on the **same evaluation dataset** using **real retrieval**. Currently they run on a 70-item synthetic benchmark. Should we re-run on the full 614-item frozen test set (`test_frozen.json`) using the actual retrieval pipeline?

> [!IMPORTANT]
> **Q4**: The current label mapping uses `Supported/Contradicted/Mixed/Insufficient Evidence`. The spec mandates `TRUE/FALSE/MIXTURE/UNPROVEN`. Which approach do you prefer?

---

## Proposed Changes

Changes are grouped into **6 phases** in dependency order, matching the spec's §23 implementation order.

---

### Phase 1: Label Space & Data Alignment (Foundation)

#### [NEW] `configs/label_mapping.yaml`
- Formal documented mapping between internal labels and spec labels
- `SUPPORTED → TRUE`, `CONTRADICTED → FALSE`, `INCONCLUSIVE → UNPROVEN`, `MIXED → MIXTURE`

#### [MODIFY] [consensus_engine.py](file:///c:/Users/Surya/Downloads/Capstone_Project/medverify-ai-backend/app/services/consensus_engine.py)
- Add `_map_to_spec_labels()` method that maps internal verdicts to `TRUE/FALSE/MIXTURE/UNPROVEN`
- Keep internal reasoning states as-is; only final outputs use spec labels

#### [MODIFY] [orchestrator.py](file:///c:/Users/Surya/Downloads/Capstone_Project/medverify-ai-backend/app/services/orchestrator.py)
- Update verdict mapping to use spec labels in research/evaluation contexts
- Add evidence-level trace fields per spec §11: evidence ID, source, citation, retrieval score, source tier, evidence type, recency, R_i, evidence population, P_i, W_i, stance

---

### Phase 2: Evidence Trace & Stance Formalization (§11, §12)

#### [MODIFY] [retrieval_engine.py](file:///c:/Users/Surya/Downloads/Capstone_Project/medverify-ai-backend/app/services/retrieval_engine.py)
- Ensure every evidence item returns all fields required by §11: evidence ID, source, citation, retrieval score, source tier, evidence type, recency, R_i
- Add evidence population fields (from population module)

#### [MODIFY] [consensus_engine.py](file:///c:/Users/Surya/Downloads/Capstone_Project/medverify-ai-backend/app/services/consensus_engine.py)
- Formalize stance values as `+1 / 0 / -1` (already done internally)
- Add weighted consensus formula: `C = Σ(W_i × stance_i) / Σ(W_i)` as explicit computation
- Ensure conflicting evidence remains visible (not discarded)

#### [NEW] `configs/label_mapping.yaml`
- Documented mapping: internal reasoning states → final labels
- Verdict thresholds (already in `experiments.yaml` but need formal standalone config)

---

### Phase 3: Population Challenge Dataset Expansion (§15, §16)

#### [NEW] `scripts/build_population_challenge_set.py`
- Expand from 70 to 120–150 claims covering all 8 required categories:
  - exact population match
  - partial population match
  - age mismatch
  - sex mismatch
  - condition mismatch
  - region mismatch
  - multiple mismatch
  - unspecified population
- Source from the full PubHealth dataset (`pubhealth_verified_claims.json`)
- Freeze before final evaluation

#### [MODIFY] `med_datasets/evaluation/population_ground_truth_70.json` → expand or create new file
- Add human annotation fields: `HIGH / PARTIAL / LOW / NOT APPLICABLE`
- Document annotation protocol

#### [NEW] `scripts/compute_inter_annotator_agreement.py`
- Cohen's κ or Fleiss' κ computation
- Currently κ = 1.0 (self-annotated); document limitations if single annotator

---

### Phase 4: Proper Baseline Experiments on Real Data (§14, §18, §19, §20)

This is the **most critical research gap**. The current B0–B3 experiment uses synthetic evidence on 70 items. The spec requires real end-to-end evaluation.

#### [NEW] `experiments/run_full_baseline_comparison.py`
- Run B0, B1, B2, B3 on the frozen test set (614 items or the population challenge subset)
- B0: claim → LLM → verdict (no retrieval)
- B1: claim → real FAISS+PubMed retrieval → unweighted evidence → NLI consensus → verdict
- B2: claim → retrieval → R_i-weighted evidence → NLI consensus → verdict
- B3: claim → retrieval → R_i × P_i weighted evidence → NLI consensus → verdict
- Same claims, same retrieval budget, same label space
- Report: Accuracy, Macro-F1, Precision, Recall, per-class F1, confusion matrices

#### [NEW] `experiments/run_b2_vs_b3_statistical.py`
- Δ Macro-F1 = F1(B3) - F1(B2)
- 95% bootstrap confidence interval
- Paired McNemar's test (already implemented, needs real data)
- Effect size (Cohen's d or similar)

#### [NEW] `experiments/run_population_specific_eval.py`
- Break B2/B3 performance by population category (§19):
  - exact match, partial match, age/sex/condition/region/multiple mismatch
- Compute per-category Macro-F1

#### [NEW] `experiments/run_verdict_change_analysis.py`
- VCR = changed verdicts / total cases
- CCR = correct changes / all changes
- Generate traceable B2→B3 change log

---

### Phase 5: Ablation, Calibration, Explanation, & Error Taxonomy (§21, §23, §24)

#### [NEW] `experiments/run_ablation_study.py`
- Ablation of P_i components: remove age, sex, condition, region one at a time
- Compare multiplicative vs additive weighting
- Report Δ Macro-F1 for each ablation

#### [MODIFY] [evaluate_calibration.py](file:///c:/Users/Surya/Downloads/Capstone_Project/scripts/evaluate_calibration.py)
- Current ECE = 0.2322 (spec gate = 0.15 max). Document this as a known limitation or apply temperature scaling
- Ensure reliability diagram is properly generated

#### [MODIFY] [evaluate_faithfulness_gate.py](file:///c:/Users/Surya/Downloads/Capstone_Project/scripts/evaluate_faithfulness_gate.py)
- Add "unsupported statement rate" and "certainty inflation rate" as separate metrics (partially done)
- Ensure faithfulness uses independent NLI model (already using DeBERTa, separate from explanation gen)

#### [NEW] `experiments/run_error_taxonomy.py`
- Classify every final error into E1–E11 taxonomy (§21)
- Generate error distribution report

---

### Phase 6: Reproducibility, Reports, & Completion Criteria (§25, §27)

#### [NEW] `experiments/experiment_registry.json`
- Registry of all experiments with: experiment ID, dataset version, model version, config, random seed, retrieval config, top-K, thresholds, metrics, output paths

#### [NEW] `Reports/final_research_report.md`
- Consolidated report covering all §17 evaluation requirements
- All metrics, tables, confidence intervals, and statistical tests

#### [NEW] `Reports/completion_checklist.md`
- Map each §27 completion criterion to evidence/report
- Track research-complete status

---

## Verification Plan

### Automated Tests
```bash
# Phase 1: Label mapping tests
python -m pytest tests/test_label_mapping.py

# Phase 2: Evidence trace tests
python -m pytest tests/test_evidence_trace.py

# Phase 3: Population challenge validation
python scripts/build_population_challenge_set.py
python scripts/compute_inter_annotator_agreement.py

# Phase 4: Full baseline experiments (most critical)
python experiments/run_full_baseline_comparison.py
python experiments/run_b2_vs_b3_statistical.py
python experiments/run_population_specific_eval.py

# Phase 5: Ablation and additional evaluations
python experiments/run_ablation_study.py
python scripts/evaluate_calibration.py
python experiments/run_error_taxonomy.py
```

### Manual Verification
- Review that B2 and B3 use identical retrieval/data and differ only in P_i
- Verify no test-set leakage in any experiment
- Cross-check label mappings are consistent throughout

---

## Priority Recommendation

| Priority | Phase | Effort | Impact |
|---|---|---|---|
| 🔴 Critical | Phase 4 (Real B0–B3 experiments) | High | This is the core research contribution; without it the project cannot be considered research-complete |
| 🟠 High | Phase 1 (Label alignment) | Low | Required for spec compliance, trivial to fix |
| 🟠 High | Phase 2 (Evidence trace) | Medium | Required for transparency and auditability |
| 🟡 Medium | Phase 3 (Population challenge expansion) | Medium | Strengthens the research claim but 70 items may be defensible |
| 🟡 Medium | Phase 5 (Ablation + error taxonomy) | Medium | Expected by any research review |
| 🟢 Lower | Phase 6 (Reports + registry) | Low | Documentation; can be generated after experiments |

> [!IMPORTANT]
> **Recommended execution order**: Phase 1 → Phase 2 → Phase 4 → Phase 3 → Phase 5 → Phase 6. Phase 4 is the most impactful and should start as soon as the label/trace foundation is in place.
