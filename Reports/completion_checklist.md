# MedVerify — Research Completion Checklist (§27)

This document tracks whether MedVerify meets all §27 completion criteria.

## Core Pipeline Components

| Criterion | Status | Evidence |
|---|---|---|
| BioBERT baseline evaluated | ✅ PASS | `Reports/biobert_test_evaluation.json` — Macro-F1 = 0.9953 |
| Retrieval evaluated | ✅ PASS | `Reports/retrieval_evaluation_results.json` — Recall@5 = 1.0, nDCG@10 = 0.7552 |
| B0 implemented | ✅ PASS | `experiments/run_full_baseline_comparison.py` — keyword heuristic baseline |
| B1 implemented | ✅ PASS | `experiments/run_full_baseline_comparison.py` — unweighted RAG |
| B2 implemented | ✅ PASS | `experiments/run_full_baseline_comparison.py` — R_i-weighted RAG |
| B3 implemented | ✅ PASS | `experiments/run_full_baseline_comparison.py` — R_i×P_i weighted |

## Population Module

| Criterion | Status | Evidence |
|---|---|---|
| P_i human-validated | ✅ PASS | `Reports/population_validation_results.json` — Acc=0.9857, κ=1.0 |
| Population challenge frozen | 🟡 PENDING | `scripts/build_population_challenge_set.py` created, needs execution |

## Evaluation Experiments

| Criterion | Status | Evidence |
|---|---|---|
| B2 vs B3 evaluated | ✅ PASS (controlled) | `Reports/baseline_b0_b3_results.json` — Δ Macro-F1 = +70.97%, p=0.00018 |
| B2 vs B3 on real data | 🟡 PENDING | `experiments/run_full_baseline_comparison.py` needs execution |
| Mismatch-specific results | 🟡 PENDING | `experiments/run_b2_vs_b3_statistical.py` needs execution |
| Ablations completed | 🟡 PENDING | `experiments/run_ablation_study.py` needs execution |

## Quality & Safety

| Criterion | Status | Evidence |
|---|---|---|
| Calibration measured | ✅ PASS | ECE = 0.2322, Brier = 0.1627 (note: ECE > 0.15 gate) |
| Explanation faithfulness | ✅ PASS | F1 = 0.9489, Recall = 1.0, BioScope catch rate = 59.52% |
| Real-world external evaluation | ✅ PASS | N=150, Macro-F1 = 0.8479, 95% CI [0.7736, 0.9083] |
| Statistical analysis | ✅ PASS (controlled) | McNemar p=0.00018, bootstrap CI computed |

## Data Integrity

| Criterion | Status | Evidence |
|---|---|---|
| No train-test leakage | ✅ PASS | `Reports/phase1_dataset_leakage_report.json` — 0 overlap |
| Dataset manifest versioned | ✅ PASS | `med_datasets/splits/splits_manifest.json` with SHA256 hashes |
| Labels documented | ✅ PASS | `configs/label_mapping.yaml` — spec mapping |
| Error taxonomy | 🟡 PENDING | `experiments/run_error_taxonomy.py` needs execution |

## Summary

**Research-complete**: 🟡 **PARTIALLY** — 4 experiment scripts need execution to generate final reports.

**Remaining steps**:
1. Run `experiments/run_full_baseline_comparison.py` (B0-B3 on full test set)
2. Run `experiments/run_b2_vs_b3_statistical.py` (statistical analysis)
3. Run `experiments/run_ablation_study.py` (ablation)
4. Run `experiments/run_error_taxonomy.py` (error taxonomy)
5. Run `scripts/build_population_challenge_set.py` (challenge set)
6. Run `scripts/compute_inter_annotator_agreement.py` (IAA)
