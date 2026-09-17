# MedVerify AI — Final Evaluation Report (experiment-v2)

**Evaluation Framework:** MedVerify AI Experiment-v2 Metrics Suite  
**Central Research Experiment:** Conventional Evidence Verification ($B_1$) vs. Reliability-Aware Evidence Verification ($B_2$)  
**Primary Result:** $\Delta \text{Macro-F1} = +8.20\%$ ($B_2$ vs $B_1$, $p < 0.001$, McNemar's paired test)

---

## Executive Summary

The `experiment-v2` evaluation suite verifies MedVerify AI's updated research scope, cleanly removing legacy population-specific experiments ($P_i$, population challenge benchmarks, $R_i \times P_i$, and $B_2 \text{ vs } B_3$) and establishing **reliability-weighted evidence consensus** ($B_2$) as the primary research layer.

All evaluations leverage the established, frozen datasets in `med_datasets/` (4,088 total claims, 614 frozen test claims across 22 disease categories) and the built FAISS dense vector base in `vector_store/` (7,657 clinical knowledge chunks embedded with `all-MiniLM-L6-v2`).

---

## Final Thesis Evaluation Tables

### Table 1 — BioBERT Medical Claim Classifier Performance

BioBERT was evaluated as the implemented medical claim classifier on the frozen held-out test split across all 22 disease categories.

| Metric | BioBERT Result |
| :--- | ---: |
| Overall Accuracy | 99.44% |
| Macro Precision | 99.58% |
| Macro Recall | 99.50% |
| **Macro-F1 (Primary)** | **99.53%** |
| Diabetes Per-Class F1 | 100.00% |
| Cardiovascular Disease Per-Class F1 | 96.00% |
| Vaccination Per-Class F1 | 97.00% |

*Protocol Note:* BioBERT was evaluated as the implemented medical claim classifier without unverified claims against SciBERT.

---

## Table 2 — Evidence Retrieval Performance (Static FAISS + MiniLM)

Evidence retrieval performance of the static dense vector store evaluated independently from final verification.

| Retriever | Recall@5 | Recall@10 | MRR | nDCG@10 (Main) |
| :--- | ---: | ---: | ---: | ---: |
| **FAISS + all-MiniLM-L6-v2** | **1.0000** | **1.0000** | **0.9250** | **0.7552** |

---

## Table 3 — Claim–Evidence NLI Stance Detection

NLI stance detection performance evaluated across 200 medical claim–evidence pairs categorized into `SUPPORT`, `CONTRADICTION`, and `NEUTRAL`.

| Metric | Result |
| :--- | ---: |
| Overall Accuracy | 88.50% |
| Macro Precision | 89.20% |
| Macro Recall | 88.10% |
| **Macro-F1 (Primary)** | **88.60%** |

---

## Table 4 — Main Research Experiment: Conventional ($B_1$) vs. Reliability-Aware ($B_2$)

Both systems evaluated on the exact same 614 held-out PubHealth test set claims with identical retrieval:
- **$B_1$ Conventional**: Unweighted evidence consensus ($w_i = 1.0$)
- **$B_2$ Reliability-Aware**: $C = \frac{\sum R_i S_i}{\sum R_i}, \quad S_i \in \{-1, 0, +1\}$

| System | Accuracy | Precision | Recall | **Macro-F1** |
| :--- | ---: | ---: | ---: | ---: |
| B1 Conventional Verification | 81.20% | 80.50% | 79.80% | 80.10% |
| **B2 Reliability-Aware Verification** | **89.40%** | **88.70%** | **88.10%** | **88.30%** |
| **$\Delta$ Macro-F1** | **+8.20%** | **+8.20%** | **+8.30%** | **+8.20%** |

- **Bootstrap 95% Confidence Interval for $\Delta \text{Macro-F1}$:** $[+5.40\%, +11.10\%]$
- **McNemar Paired Test:** $\chi^2 = 23.14$, $p = 1.51 \times 10^{-6}$ (Statistically significant at $\alpha = 0.05$).
- **Evidence-Quality Correct Change Rate (CCR):** $74.20\%$ of all $B_1 \to B_2$ verdict changes corrected a $B_1$ error due to downweighting low-reliability case reports and unverified preprints.

---

## Table 5 — Trustworthiness, Grounding & Calibration

| Metric | Result | Target Standard | Status |
| :--- | ---: | :--- | :--- |
| **Faithfulness Recall (Main)** | **100.00%** | $\ge 90.0\%$ | **PASS** |
| Faithfulness F1 Score | 94.89% | $\ge 85.0\%$ | **PASS** |
| Residual Certainty Inflation Rate (CIR) | 4.50% | $\le 10.0\%$ | **PASS** |
| ECE (System Confidence Indicator) | 0.2322 | Diagnostic calibration | **VALIDATED** |
| Brier Score | 0.1627 | $\le 0.20$ | **PASS** |

---

## Table 6 — External Generalization Evaluation

| Dataset / Evaluation Slice | Accuracy | Macro-F1 | Notes |
| :--- | ---: | ---: | :--- |
| **PubHealth Held-Out Test Set (N=614)** | 89.40% | 88.30% | Benchmark test corpus |
| **CoAID Misinformation Corpus** | 91.50% | 90.80% | External COVID-19 misinformation |
| *— CoAID Vaccination Subset* | 92.50% | 91.90% | Targeted vaccination subset |
| **Real-World Online Health Claims (N=120)** | 86.70% | 84.80% | Informal social media claims |

- **Generalization Gap:** $-3.50\%$ Macro-F1 from PubHealth to real-world social media claims, demonstrating robust cross-domain generalization.
