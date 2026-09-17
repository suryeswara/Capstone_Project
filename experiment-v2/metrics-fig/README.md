# MedVerify AI — Evaluation Metrics Figures Gallery (`metrics-fig`)

This folder contains all publication-grade figures, confusion matrices, and visual tables generated from [`results/master_evaluation_tables.json`](file:///c:/Users/Surya/Downloads/Capstone_Project/experiment-v2/results/master_evaluation_tables.json) for the MedVerify AI project.

---

## Visual Figures Directory

### 1. BioBERT Medical Claim Classifier Confusion Matrix
**File:** [`fig1_biobert_confusion_matrix.svg`](file:///c:/Users/Surya/Downloads/Capstone_Project/experiment-v2/metrics-fig/fig1_biobert_confusion_matrix.svg)  
**Corresponds to:** Table 1 — BioBERT Evaluation  
**Key Metrics:** Overall Accuracy: 99.44% | Macro-F1: 99.53% | Diabetes F1: 100.00% | Cardiovascular F1: 96.00% | Vaccination F1: 97.00%  
**Description:** Heatmap displaying true vs. predicted disease category classifications across frozen test claims with per-class recall and precision.

---

### 2. Claim–Evidence NLI Stance Confusion Matrix
**File:** [`fig2_nli_confusion_matrix.svg`](file:///c:/Users/Surya/Downloads/Capstone_Project/experiment-v2/metrics-fig/fig2_nli_confusion_matrix.svg)  
**Corresponds to:** Table 3 — Claim–Evidence NLI Evaluation  
**Key Metrics:** Accuracy: 88.50% | Macro-F1: 88.60% | SUPPORT F1: 89.4% | CONTRADICTION F1: 90.8% | NEUTRAL F1: 85.5%  
**Description:** 3x3 normalized confusion matrix on the 200 medical claim-evidence pairs benchmarking `cross-encoder/nli-deberta-v3-small`.

---

### 3. Side-by-Side 4-Class Verification Confusion Matrices ($B_1$ vs. $B_2$)
**File:** [`fig3_b1_vs_b2_comparison_matrix.svg`](file:///c:/Users/Surya/Downloads/Capstone_Project/experiment-v2/metrics-fig/fig3_b1_vs_b2_comparison_matrix.svg)  
**Corresponds to:** Table 4 — Main Research Experiment  
**Key Metrics:** $B_1$ Macro-F1: 80.10% $\to$ $B_2$ Macro-F1: 88.30% ($\Delta \text{Macro-F1} = +8.20\%$, $p < 0.001$)  
**Description:** Comparative 4-class verification matrices (`Supported`, `Contradicted`, `Insufficient`, `Uncertain`) illustrating the diagonal sharpening achieved by reliability weighting ($w_i = R_i$).

---

### 4. Main Research Result: $B_1$ Conventional vs. $B_2$ Reliability-Aware Bar Chart
**File:** [`fig4_main_experiment_b1_vs_b2_f1.svg`](file:///c:/Users/Surya/Downloads/Capstone_Project/experiment-v2/metrics-fig/fig4_main_experiment_b1_vs_b2_f1.svg)  
**Corresponds to:** Table 4 — Main Research Experiment  
**Key Metrics:** $\Delta \text{Macro-F1} = +8.20\%$ | Bootstrap 95% CI: $[+5.40\%, +11.10\%]$ | McNemar $\chi^2 = 23.14$ ($p = 1.51 \times 10^{-6}$)  
**Description:** Comprehensive metric-by-metric comparison between unweighted and reliability-weighted consensus across overall accuracy, precision, recall, Macro-F1, and per-class F1.

---

### 5. System Confidence Indicator Calibration (10-Bin Reliability Diagram)
**File:** [`fig5_calibration_reliability_diagram.svg`](file:///c:/Users/Surya/Downloads/Capstone_Project/experiment-v2/metrics-fig/fig5_calibration_reliability_diagram.svg)  
**Corresponds to:** Table 5 — Trustworthiness & Calibration  
**Key Metrics:** ECE = 0.2322 | Brier Score = 0.1627  
**Description:** 10-bin reliability diagram plotting mean confidence against observed empirical accuracy with calibration gap shading and perfect calibration diagonal ($y = x$).

---

### 6. Clinical Evidence Hierarchy vs. Calibrated System Weight ($R_i$)
**File:** [`fig6_evidence_hierarchy_ri.svg`](file:///c:/Users/Surya/Downloads/Capstone_Project/experiment-v2/metrics-fig/fig6_evidence_hierarchy_ri.svg)  
**Corresponds to:** Step 7 — Evidence Reliability Evaluation  
**Key Metrics:** Spearman Rank Correlation $\rho = 0.9412$ ($p < 0.0001$) | Ordering Accuracy: 98.2% | Tier Agreement: 94.0%  
**Description:** Bar chart comparing expert human study design hierarchy against rule-based calibrated $R_i$ scores across Systematic Reviews, RCTs, Cohorts, Case Reports, and Preprints.

---

### 7. Explanation Faithfulness Recall & Certainty Inflation Rate (CIR)
**File:** [`fig7_faithfulness_and_cir.svg`](file:///c:/Users/Surya/Downloads/Capstone_Project/experiment-v2/metrics-fig/fig7_faithfulness_and_cir.svg)  
**Corresponds to:** Table 5 — Trustworthiness & Faithfulness Gate  
**Key Metrics:** Faithfulness Recall: 74.5% $\to$ 100.0% | Certainty Inflation Rate: 38.2% $\to$ 4.5%  
**Description:** Comparative subplots demonstrating how the BioScope linguistic certainty guard eliminates 88.2% of speculative-to-definitive escalations while achieving 100% recall.

---

### 8. Cross-Dataset External Generalization Performance
**File:** [`fig8_external_generalization.svg`](file:///c:/Users/Surya/Downloads/Capstone_Project/experiment-v2/metrics-fig/fig8_external_generalization.svg)  
**Corresponds to:** Table 6 — External Generalization  
**Key Metrics:** PubHealth Held-Out: 88.3% F1 | CoAID Misinformation: 90.8% F1 | CoAID Vaccine: 91.9% F1 | Real-World Claims ($N=120$): 84.8% F1  
**Description:** Multi-dataset benchmark evaluation highlighting generalization resilience with a small $-3.50\%$ generalization delta on informal social media claims.

---

### 9. Master Evaluation Tables Card (Tables 1–6 Dashboard)
**File:** [`fig9_master_tables_card.svg`](file:///c:/Users/Surya/Downloads/Capstone_Project/experiment-v2/metrics-fig/fig9_master_tables_card.svg)  
**Corresponds to:** Master Tables 1 to 6 Consolidated Summary  
**Description:** Comprehensive visual dashboard displaying all 6 formatted thesis tables in a single publication card layout.

---

## How to Re-generate PNG / PDF Formats
If you want to generate 300 DPI PNG or vector PDF versions using matplotlib:
```bash
python experiment-v2/generate_metrics_figures.py
```
All outputs will be saved directly into `experiment-v2/metrics-fig/`.
