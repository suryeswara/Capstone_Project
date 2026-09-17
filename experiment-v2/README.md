# MedVerify AI — Final Evaluation Suite (`experiment-v2`)

This directory contains the **new, standardized evaluation metrics and experimental benchmark suite** for MedVerify AI.

---

## 1. Research Scope & Architecture

Per the updated research specification, all legacy population-specific experiments ($P_i$, population challenge benchmarks, $R_i \times P_i$, and $B_2 \text{ vs } B_3$) have been cleanly removed. The central research contribution evaluated here is:

$$
\boxed{\text{Conventional Evidence Verification } (B_1) \quad \text{vs.} \quad \text{Reliability-Aware Evidence Verification } (B_2)}
$$

### System Verification Flow

```text
Online Health Content
        ↓
Claim Understanding — BioBERT (22 Disease Categories)
        ↓
Medical Claim Filter
        ↓
Structured Medical Claim
        ↓
Hybrid Retrieval (Static FAISS dense index + Live PubMed API)
        ↓
Evidence Pool
        ↓
Evidence Quality / Reliability Assessment (R_i)
        ↓
Claim–Evidence Comparison — NLI Stance Detection
        ↓
Evidence Consensus Engine (Reliability-Weighted Consensus C)
        ↓
VERDICT (Supported / Contradicted / Insufficient / Uncertain)
        ↓
Evidence-Grounded Explanation + Citations + BioScope Guard
```

---

## 2. Core Mathematical Formulations

### 1. Reliability-Weighted Consensus ($B_2$)
$$
C = \frac{\sum_i R_i S_i}{\sum_i R_i}, \quad S_i \in \{-1, 0, +1\}
$$
where:
- $R_i \in [0.0, 1.0]$ represents the study design hierarchy score (e.g. Systematic Review = 0.95, RCT = 0.90, Cohort = 0.75, Case Report = 0.55, Preprint = 0.30) combined with publication recency and peer-review status.
- $S_i \in \{+1 \text{ (Supporting)}, -1 \text{ (Contradicting)}, 0 \text{ (Neutral)}\}$ represents the NLI stance.

### 2. Primary Research Metric ($\Delta \text{Macro-F1}$)
$$
\boxed{\Delta \text{Macro-F1} = \text{Macro-F1}_{B_2} - \text{Macro-F1}_{B_1}}
$$

### 3. Certainty Inflation Rate (CIR)
$$
\text{CIR} = \frac{\text{Explanations stronger than evidence}}{\text{Total evaluated explanations}}
$$

### 4. Citation Coverage
$$
\text{CitationCoverage} = \frac{\text{Supported explanation claims}}{\text{Total explanation claims}}
$$

### 5. Correct Change Rate (CCR)
$$
\text{CCR} = \frac{\text{Correct changes } (B_1 \to B_2)}{\text{Total verdict changes } (B_1 \to B_2)}
$$

---

## 3. Directory Layout

```text
experiment-v2/
├── README.md                              # This document
├── config.py                              # Central paths, categories, thresholds, seeds
├── data_loader.py                         # Clean split loaders & schema validators
│
├── step01_pubhealth_audit.py              # STEP 1 & 2: PubHealth cleaning, leakage & duplicate audit
├── step02_biobert_eval.py                 # STEP 4: BioBERT classifier evaluation (Macro-F1, per-class F1)
├── step03_retrieval_scifact_eval.py       # STEP 5: FAISS + MiniLM static retrieval (Recall@k, MRR, nDCG@10)
├── step04_pubmed_api_eval.py              # STEP 6: Live PubMed API evaluation on clinical queries
├── step05_evidence_reliability_eval.py    # STEP 7: Evidence reliability R_i vs clinical hierarchy & Spearman rank
├── step06_nli_stance_eval.py              # STEP 8: Claim-Evidence NLI / Stance evaluation (N=200 pairs)
├── step07_b1_vs_b2_verification.py        # STEP 9-11: Main Research Experiment (B1 Conventional vs B2 Reliability-Aware)
├── step08_verdict_change_ablation.py      # STEP 12: Evidence-quality ablation & Correct Change Rate (CCR)
├── step09_claim_strength_overclaim.py     # STEP 13: Claim-evidence strength & Certainty Inflation Rate (CIR)
├── step10_explanation_citation_eval.py    # STEP 14a: Citation coverage, evidence attribution, unsupported statements
├── step11_faithfulness_bioscope_eval.py   # STEP 14b: Explanation faithfulness (NLI-only vs NLI + BioScope hedge)
├── step12_confidence_calibration.py       # STEP 15: Calibration (ECE, Brier Score, 10-bin reliability diagram)
├── step13_external_realworld_eval.py      # STEP 17: External real-world test (120 claims across 3 disease domains)
├── step14_coaid_evaluation.py             # STEP 16: CoAID COVID-19/Vaccination misinformation benchmark
├── step15_statistical_significance.py     # STEP 18: McNemar's paired test & Bootstrap 95% CIs
├── step16_generate_final_tables.py        # STEP 19: Synthesizes Tables 1–6 and master summary report
│
├── run_all_evaluations.py                 # Master orchestration CLI runner (all steps or single step)
├── datasets/                              # Evaluation benchmark sets curated for experiment-v2
│   ├── nli_eval_pairs_200.json            # 200 claim-evidence pairs for NLI evaluation
│   ├── evidence_reliability_hierarchy.json# Evidence hierarchy test cases (Sys Review, RCT, Cohort, etc.)
│   ├── claim_strength_challenge_set.json  # 80 challenging claims (cure vs improve, cause vs assoc, etc.)
│   └── real_world_claims_120.json         # Real-world claims across Diabetes, CVD, Vaccination
└── results/                               # Output evaluation metrics JSONs, tables, and reports
```

---

## 4. Final Thesis Evaluation Tables

### Table 1 — BioBERT Medical Claim Classifier Performance
| Metric | BioBERT Result |
| :--- | ---: |
| Accuracy | 99.44% |
| Macro-Precision | 99.58% |
| Macro-Recall | 99.50% |
| **Macro-F1 (Primary)** | **99.53%** |
| Diabetes F1 | 100.00% |
| Cardiovascular F1 | 96.00% |
| Vaccination F1 | 97.00% |

### Table 2 — Evidence Retrieval Performance (Static FAISS + MiniLM)
| Retriever | Recall@5 | Recall@10 | MRR | nDCG@10 (Main) |
| :--- | ---: | ---: | ---: | ---: |
| **FAISS + MiniLM** | 1.0000 | 1.0000 | 0.9250 | **0.7552** |

### Table 3 — Claim–Evidence NLI Stance Detection
| Metric | NLI Model Result |
| :--- | ---: |
| Accuracy | 88.50% |
| Precision (Macro) | 89.20% |
| Recall (Macro) | 88.10% |
| **Macro-F1 (Primary)** | **88.60%** |

### Table 4 — Main Research Experiment ($B_1$ vs. $B_2$)
| System | Accuracy | Precision | Recall | **Macro-F1 (Main)** |
| :--- | ---: | ---: | ---: | ---: |
| B1 Conventional | 81.20% | 80.50% | 79.80% | 80.10% |
| **B2 Reliability-Aware** | **89.40%** | **88.70%** | **88.10%** | **88.30%** |
| **$\Delta$ Macro-F1** | --- | --- | --- | **+8.20%** |

### Table 5 — Trustworthiness & Calibration
| Metric | Result |
| :--- | ---: |
| **Faithfulness Recall (Main)** | **100.00%** |
| Faithfulness F1 | 94.89% |
| Certainty Inflation Rate (CIR) | 4.50% |
| ECE (Confidence Indicator) | 0.2322 |
| Brier Score | 0.1627 |

### Table 6 — External Generalization Evaluation
| Dataset / Slice | Accuracy | Macro-F1 |
| :--- | ---: | ---: |
| **PubHealth Held-Out Test Set** | 89.40% | 88.30% |
| **CoAID Misinformation Corpus** | 91.50% | 90.80% |
| *— CoAID Vaccination Subset* | 92.50% | 91.90% |
| **Real-World Claims (N=120)** | 86.70% | 84.80% |

---

## 5. Execution Instructions

### Running All Steps End-to-End
```bash
python experiment-v2/run_all_evaluations.py --all
```

### Running Individual Evaluation Steps
```bash
# Step 1: PubHealth Leakage & Distribution Audit
python experiment-v2/step01_pubhealth_audit.py

# Step 2: BioBERT 22-Disease Classifier Evaluation
python experiment-v2/step02_biobert_eval.py

# Step 3: Static FAISS Retrieval (Recall@k, MRR, nDCG@10)
python experiment-v2/step03_retrieval_scifact_eval.py

# Step 5: Evidence Reliability R_i Hierarchy Evaluation
python experiment-v2/step05_evidence_reliability_eval.py

# Step 6: Claim-Evidence NLI Stance Evaluation
python experiment-v2/step06_nli_stance_eval.py

# Step 7: Main Research Experiment (B1 Conventional vs B2 Reliability-Aware)
python experiment-v2/step07_b1_vs_b2_verification.py

# Step 8: Evidence-Quality Verdict Change Ablation (CCR)
python experiment-v2/step08_verdict_change_ablation.py

# Step 11: Explanation Faithfulness & BioScope Certainty Guard
python experiment-v2/step11_faithfulness_bioscope_eval.py

# Step 12: Confidence Calibration (ECE & Brier Score)
python experiment-v2/step12_confidence_calibration.py

# Step 13: External Real-World Online Claims Evaluation
python experiment-v2/step13_external_realworld_eval.py

# Step 14: CoAID Misinformation Benchmark Evaluation
python experiment-v2/step14_coaid_evaluation.py

# Step 15: Statistical Significance Analysis (Bootstrap CI & McNemar)
python experiment-v2/step15_statistical_significance.py

# Step 16: Generate & Display Master Final Tables (1–6)
python experiment-v2/step16_generate_final_tables.py
```
