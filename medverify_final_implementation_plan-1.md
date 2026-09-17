# — MedVerify AI Final Project Implementation Plan BioBERT-Based, Reliability-Aware, Population-Aware, Faithfulness-Guarded Medical Claim Verification 

### September 2026 

## **Contents** 

|**1**<br>**1. Final Project Defnition**|**4**|
|---|---|
|**2**<br>**2. Research Contribution**|**4**|
|2.1 2.1 Primary Research Question<br>. . . . . . . . . . . . . . . . . .|. . . .<br>4|
|2.2 2.2 Secondary Methodological Question<br>. . . . . . . . . . . . .|. . . .<br>4|
|2.3 2.3 Components Not Individually Claimed as Novel<br>. . . . . . .|. . . .<br>5|
|2.4 2.4 Novelty Statement (Correct Framing) . . . . . . . . . . . . .|. . . .<br>5|
|**3**<br>**3. Final System Architecture**|**6**|
|**4**<br>**4. Final Model Stack**|**8**|
|4.1 4.1 Classifer Decision — Final and Frozen . . . . . . . . . . . .|. . . .<br>8|
|4.2 4.2 Why BioBERT . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . .<br>8|
|4.3 4.3 Separation of Responsibilities . . . . . . . . . . . . . . . . .|. . . .<br>9|
|**5**<br>**5. Claim Representation**|**10**|
|**6**<br>**6. Stage 0 — Project Freeze and Experimental Protocol**|**11**|
|**7**<br>**7. Stage 1 — Dataset Governance**|**11**|
|**8**<br>**8. Stage 2 — BioBERT Fine-Tuning**|**12**|
|8.1 8.1 Evaluation<br>. . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . .<br>12|
|**9**<br>**9. Stage 3 — Knowledge Base**|**13**|
|**1010. Stage 4 — Medical Retrieval (Frozen for Research)**|**13**|
|10.110.1 Reproducibility Fix — Frozen Evidence Snapshot . . . . . .|. . . .<br>13|
|**1111. Stage 5 — Evidence Deduplication**|**13**|
|**1212. Stage 6 — Study Metadata Extraction**|**13**|
|**1313. Stage 7 — Reliability Score**𝑅𝑖**(Frozen Formula)**|**15**|



1 

|**1414. Stage 8 — Population Applicability**𝑃𝑖**(Frozen Formula)**|**15**|
|---|---|
|**1515. Stage 9 — Final Evidence Weight**|**17**|
|**1616. Stage 10 — Evidence Ranking**|**17**|
|**1717. Stage 11 — Stance Detection (Now Fully Specifed)**|**17**|
|**1818. Stage 12 — Weighted Consensus**|**18**|
|**1919. Stage 13 — Verdict (Frozen Threshold Policy)**|**18**|
|**2020. Stage 14 — Grounded Explanation**<br>20.120.1 Certainty Preservation<br>. . . . . . . . . . . . . . . . . . . . .|**18**<br> . . .<br>19|
|**2121. Stage 15 — Faithfulness Verifcation**<br>21.121.1 Faithfulness Targets (Corrected) . . . . . . . . . . . . . . . .|**20**<br> . . .<br>20|
|**2222. Stage 16 — Self-Consistency Test**|**20**|
|**2323. Stage 17 — Confdence Calibration**|**20**|
|**2424. Stage 18 — Medical Safety Layer**<br>24.124.1 Safety Evaluation (Corrected)<br>. . . . . . . . . . . . . . . . .|**22**<br> . . .<br>22|
|**2525. Final Experimental Baselines**|**23**|
|**2626. Most Important Experiment: B2 vs. B3**|**23**|
|**2727. Required Evaluation Datasets**<br>27.127.1 CoAID Status (Corrected)<br>. . . . . . . . . . . . . . . . . . .|**23**<br> . . .<br>24|
|**2828. Population Ground-Truth Dataset**|**25**|
|**2929. Faithfulness Dataset**|**25**|
|**3030. Expert Evaluation (Corrected)**|**25**|
|30.130.1 Expert–Expert Agreement (New)<br>. . . . . . . . . . . . . . .|. . .<br>25|
|**3131. Real-World Social-Media Stress Test (New)**|**25**|
|**3232. Database Schema**|**26**|
|**3333. Backend**|**26**|
|**3434. Frontend**|**26**|
|**3535. Final 12-Week Implementation Schedule**<br>**3636. Critical Gates**|**27**<br>**27**|



2 

|**3737. Final Defnition of Done**|**28**|
|---|---|
|37.137.1 Engineering . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . .<br>28|
|37.237.2 Research<br>. . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . .<br>28|
|**3838. Summary of Corrections Applied to the Final Plan**|**28**|



3 

## **1 1. Final Project Definition** 

**MedVerify AI** is an explainable medical claim verification system designed to verify health claims appearing in: 

- social-media posts, 

- rapidly spreading health myths, 

- online articles, 

- messages/posts containing medical assertions, 

- other digital health-information sources. 

The system does **not** diagnose users or prescribe treatment. 

Its purpose is: 

Given a medical claim, retrieve appropriate medical evidence, determine how reliable and population-applicable that evidence is, compare supporting and contradicting evidence, generate an evidence-grounded explanation, and verify that the explanation faithfully represents the evidence. 

This preserves the central boundary already established in the project documents: MedVerify is a **claim verifier, not a diagnostic or clinical decision system** . 

## **2 2. Research Contribution** 

The project makes **one primary research claim** , rather than claiming every component as novel. 

### **2.1 2.1 Primary Research Question** 

Does explicitly modeling population applicability (𝑃𝑖) in evidence weighting improve medical claim verification compared with reliability-only evidence weighting (𝑅𝑖)? 

Formally: 



MedVerify investigates whether incorporating population applicability together with source/evidence reliability improves medical claim verification compared with conventional relevance-based RAG and reliability-only RAG, while independently verifying generated explanations for faithfulness. 

### **2.2 2.2 Secondary Methodological Question** 

Is multiplicative combination better than additive combination for representing reliability and population applicability? 

Compare: 

4 

𝑊𝑖<sup>𝑚𝑢𝑙𝑡</sup> = 𝑅𝑖𝑃𝑖 vs. 𝑊𝑖<sup>𝑎𝑑𝑑</sup> = 𝛼𝑅𝑖 + (1 −𝛼)𝑃𝑖 

with 𝛼 fixed **before** testing. Selection criterion (pre-registered): 

1. Spearman correlation with human population-applicability ranking. 

2. Expert-review agreement as a secondary criterion. 

Because the population sample (N = 60–80) is small, report **confidence intervals** rather than claiming universal superiority of one mathematical form. 

### **2.3 2.3 Components Not Individually Claimed as Novel** 

Do not claim these are individually novel: BioBERT, FAISS, PubMed retrieval, RAG, NLI, hedge detection, source reliability scoring, population extraction, LLM explanation generation. 

The novelty is **the tested combination and its measured effect** , isolated specifically through the B2 →B3 comparison (Section 12). 

### **2.4 2.4 Novelty Statement (Correct Framing)** 

Do **not** claim: 

- “MedVerify invented reliable medical RAG.” 

- “MedVerify is the first medical fact-checking AI.” 

- “𝑅× 𝑃 is universally superior.” 

#### Instead: 

The research investigates whether an explicit population-applicability score, combined with evidence reliability, changes medical claim verification outcomes compared with reliability-only evidence ranking. 

This is narrow, testable, and defensible — the novelty is **narrow but real** , centered on the measurable effect of 𝑃𝑖 and its calibration. 

5 

## **3 3. Final System Architecture** 

USER / ONLINE HEALTH CLAIM | v CLAIM EXTRACTION | v +-------------------+ | BioBERT | | Disease/Medical | | Domain Classifier| +-------------------+ | v HYBRID RETRIEVAL (frozen snapshot) / \ FAISS PubMed/PMC \ / EVIDENCE POOL (dedup) | v STUDY METADATA EXTRACTION | +--------------+--------------+ v v RELIABILITY R_i POPULATION P_i | | +--------------+--------------+ v W_i = R_i x P_i | v EVIDENCE RE-RANKING | v STANCE ANALYSIS (NLI) | v WEIGHTED CONSENSUS | v TRUE / FALSE / MIXTURE / UNPROVEN (frozen threshold policy) | v GROUNDED LLM EXPLANATION 

6 

| +--------+--------+ v v NLI HEDGE DETECTOR +--------+--------+ v FAITHFULNESS GATE | v CALIBRATION (ECE + Brier) | v SAFETY ROUTING | v FINAL RESULT 

7 

## **4 4. Final Model Stack** 

|Stage|Final Model / Method|Purpose|
|---|---|---|
|Claim extraction|LLM + structured<br>extraction|Extract medical claim<br>components|
|Disease classifcation|**BioBERT**|Classify medical<br>domain/category|
|Retrieval|**MedCPT**|Retrieve medically<br>relevant evidence|
|Vector search|FAISS|Eficient local retrieval|
|Evidence embedding|MedCPT|Represent medical<br>queries/documents|
|Reliability|Rule-based𝑅𝑖|Evidence/source quality|
|Population applicability|Rule-based𝑃𝑖|Determine evidence<br>applicability|
|Evidence weighting|𝑊𝑖= 𝑅𝑖× 𝑃𝑖|Final evidence weight|
|Stance detection|NLI-based classifer|Support / contradict /<br>neutral|
|Consensus|Weighted stance<br>calculation|Supporting<br>vs. contradicting evidence|
|Explanation|Gemma 3 12B (fallback:<br>Llama 3.1 8B)|Grounded explanation|
|Faithfulness|NLI-DeBERTa-v3-small|Check explanation against<br>evidence|
|Certainty control|BioScope-style hedge<br>detector|Detect certainty<br>escalation|
|Calibration|ECE + Brier|Calibrate system<br>confdence|



### **4.1 4.1 Classifier Decision — Final and Frozen** 

Earlier drafts had conflicting classifier recommendations — one proposed SciBERT, another left BioBERT vs. DistilBERT unresolved. **This is now resolved: use BioBERT as the fixed classifier for the entire implementation.** Do not keep changing the classifier during implementation. 

For scientific rigor, run a **one-time baseline comparison** with SciBERT / PubMedBERT / DistilBERT if computationally feasible — this is treated strictly as an experiment, not an open production decision. 

### **4.2 4.2 Why BioBERT** 

MedVerify deals with biomedical terminology, diseases, treatments/interventions, medical outcomes, scientific literature, and PubMed/PMC evidence — so the model is treated as a medical-domain language model rather than a generic text classifier. 

**Important distinction:** BioBERT is **not** responsible for deciding whether a claim is medically true. Its responsibility is narrow: 

8 

##### Claim -> BioBERT -> Medical category 

Example: _“Drinking lemon water cures diabetes.”_ →BioBERT → Diabetes. Verification itself happens later, through retrieval, reliability, population applicability, consensus, and faithfulness checking. 

### **4.3 4.3 Separation of Responsibilities** 

BioBERT should **not** be used for every part of MedVerify — that would weaken the architecture. Its role is specifically medical-domain classification/triage. MedCPT handles biomedical retrieval; rule-based components handle 𝑅𝑖/𝑃𝑖; the LLM generates the explanation; NLI + hedge detection independently verify it. 

**BioBERT classifies. MedCPT retrieves.** 𝑅𝑖 **judges reliability.** 𝑃𝑖 **judges population applicability.** 𝑊𝑖 = 𝑅𝑖𝑃𝑖 **ranks evidence. The consensus engine determines the evidence-based verdict. Gemma explains it. NLI + hedge detection checks the explanation. Calibration tells the user how confident the system should be.** 

9 

## **5 5. Claim Representation** 

Every incoming item is first converted into a structured claim. 

#### **Input:** 

"People over 60 who drink green tea every day can prevent heart disease." 

#### **Extracted representation:** 

**{** "claim" **:** "Daily green tea consumption prevents heart disease in people over 60." **,** "disease" **:** "cardiovascular disease" **,** "intervention" **:** "green tea consumption" **,** "outcome" **:** "prevention of heart disease" **,** "population" **: {** "age" **: {** "type" **:** "range" **,** "range" **:** [60, **null** ] **},** "sex" **:** "unspecified" **,** "region" **:** "unspecified" **} }** 

Frozen population schema: 

Age: range / categorical / unspecified Sex: male / female / both / unspecified Region: normalized region / unspecified 

10 

## **— 6 6. Stage 0 Project Freeze and Experimental Protocol** 

Before building further, the methodology is frozen. 

Repository layout: 

project/ |-- configs/ | |-- models.yaml | |-- retrieval.yaml | |-- reliability.yaml | |-- population.yaml | `-- experiments.yaml |-- datasets/ 

|-- models/ 

|-- retrieval/ 

- |-- verification/ 

- |-- evaluation/ 

- `-- reports/ 

Frozen artifacts: BioBERT version/checkpoint, tokenizer, training split, random seeds, retrieval model, evidence corpus version, 𝑅𝑖 formula, 𝑃𝑖 formula, 𝑊𝑖 formula, explanation model/version, NLI model, prompts, evaluation datasets. 

Every verification stores model/version metadata. 

## **7 7. Stage 1 — Dataset Governance** 

**Primary dataset: PubHealth** , used for medical claim classification, verification labels, and classifier training/evaluation. 

Native labels are kept: TRUE, FALSE, MIXTURE, UNPROVEN — avoiding an unnecessarily incompatible new label system. 

#### **Required checks before training BioBERT:** 

1. Remove duplicates. 

2. Detect train/test overlap. 

3. Check near-duplicate claims. 

4. Check class distribution. 

5. Freeze train/dev/test manifests. 

6. Hash the manifests. 

**Medical audit:** a medical reviewer inspects a sample of ~50–100 labels. 

**Leakage-set rule (strengthened):** the 50-record development set may be used for debugging, pipeline development, and schema development. It may **not** be used to select final verdict thresholds, tune 𝑅𝑖, tune 𝑃𝑖, select the final B2/B3 result, or report final accuracy. A **Development Decision Log** records every decision influenced by the development set, so the leakage audit is defensible. 

11 

## **8 8. Stage 2 — BioBERT Fine-Tuning** 

**Input:** medical claim. **Output:** category (e.g. Diabetes, Cardiovascular, Vaccination, or the project’s final selected category set). 

#### **Training pipeline:** 

PubHealth -> Clean -> Deduplicate -> Train/Val/Test split 

- -> BioBERT tokenizer -> BioBERT encoder 

- -> Classification head -> Fine-tuning 

#### **Model head:** 

BioBERT -> [CLS] representation -> Dropout 

- -> Linear classification layer -> Softmax 

Use AdamW, learning-rate scheduling, early stopping, class-weighted loss if imbalance requires it, and fixed random seeds. 

### **8.1 8.1 Evaluation** 

Report accuracy, macro-F1, micro-F1, precision, recall, per-class F1, and confusion matrix. 

**Primary target:** Macro-F1 ≥0.85 (Gate 1, Section 15). 

If the target is not met: **do not modify the test set to improve the result.** Report the actual result and investigate class imbalance, label noise, insufficient training, ambiguous categories, or data leakage. 

12 

## **— 9 9. Stage 3 Knowledge Base** 

A curated medical evidence corpus, tiered: 

- **Tier 1:** WHO, CDC, official public-health guidance. 

- **Tier 2:** peer-reviewed systematic reviews, meta-analyses, randomized controlled trials. 

- **Tier 3:** cohort studies, observational studies. 

- **Lower tiers:** case reports, preprints where appropriate. 

Semantic relevance is explicitly distinguished from evidence reliability throughout the pipeline. 

## **10 10. Stage 4 — Medical Retrieval (Frozen for Research)** 

**Primary retrieval model: MedCPT** (biomedical literature retrieval). 

Claim -> Query normalization -> MedCPT query encoder -> FAISS -> Top-K evidence In parallel: Claim -> PubMed API -> Recent literature, then combined into a candidate evidence pool. 

### **10.1 10.1 Reproducibility Fix — Frozen Evidence Snapshot** 

Mixing static FAISS evidence with live PubMed/PMC results creates a reproducibility and temporal-leakage problem. **For all research evaluation** , use a frozen snapshot: 

Claim -> Frozen evidence snapshot -> Retrieval -> R_i -> P_i -> W_i -> Verdict 

Every retrieved evidence item stores: PMID/PMCID, title, abstract/text, retrieval date, source, evidence type, model version. The snapshot is then frozen. 

**Live PubMed/PMC** is retained only for demonstration, future deployment, and the user-facing prototype — **never** for the final B0–B3 benchmark, where changing live results would break reproducibility. 

## **11 11. Stage 5 — Evidence Deduplication** 

The same study can appear via PubMed, PMC, WHO references, or multiple retrieved chunks: 

Candidate evidence -> PMID/DOI matching -> Duplicate removal -> Chunk deduplication 

Stored fields: PMID, DOI, title, authors, publication date, source, evidence type, chunk, population. 

## **12 12. Stage 6 — Study Metadata Extraction** 

For every retrieved study, extract disease, intervention, outcome, population (age, sex, region), evidence type, publication date, and study design. If information is ab- 

13 

sent, store unspecified — **never invent missing population information.** 

14 

# **13 13. Stage 7 — Reliability Score** 𝑅𝑖 **(Frozen Formula)** 

Reliability remains transparent and versioned rather than an unexplained neural score, since there is no strong established ground-truth benchmark for one universal medical-source reliability score. 



where 𝑆𝑖 = source reliability tier, 𝐸𝑖 = evidence-type/hierarchy score, 𝑇𝑖 = recency factor. 

#### **Pre-registered initial scoring scheme:** 

|Component|Rule|Score|
|---|---|---|
|Source|WHO / CDC / government / major guideline|1.00|
|Source|Systematic review / meta-analysis|0.95|
|Source|Peer-reviewed RCT|0.90|
|Source|Observational study|0.75|
|Source|Review / narrative evidence|0.65|
|Source|Other scientifc source|0.50|



Evidence type is kept separate from source authority rather than assigning arbitrary numbers (e.g. “WHO = 98”). 

#### **Recency factor:** 



where Δ𝑡𝑖 is the age of the evidence. The exact values and decay constant 𝜆 must be **frozen before** the B2/B3 experiment. 

These numbers are a **transparent ranking heuristic** , not a claim about the “true reliability” of a paper. The ranking is validated with a small SciFact direction/correlation check, used only as a sanity check — not a certification dataset. 

Document reliability_version = R1.0 and every scoring component. 

# **14 14. Stage 8 — Population Applicability** 𝑃𝑖 **(Frozen Formula)** 

The rule-based approach is used because 60–80 examples are too few for a learned population model. 

𝑃 =<sup>𝐴𝑖+ 𝑆𝑖+ 𝐶𝑖+ 𝐺𝑖</sup> 𝑖 4 

15 

where 𝐴𝑖 = age compatibility, 𝑆𝑖 = sex compatibility, 𝐶𝑖 = condition compatibility, 𝐺𝑖 = geographic/region compatibility. 

Each dimension is scored: 

- 1.0 = compatible 

- 0.5 = partially compatible / insufficient information 

- 0.0 = clearly incompatible 

**Critical rule:** unspecified must **not** automatically become 0. E.g., if a study does not specify sex, that does not mean “sex mismatch = 0” — instead P_sex = 0.5 (a defined neutral/uncertain value). This preserves the unspecified vs. actual-value distinction already built into the population schema. 

#### **Report all four dimensions separately** , not one mysterious scalar: 

P_age, P_sex, P_condition, P_region, P_total 

This avoids hiding which population dimension caused a mismatch. 

16 

## **— 15 15. Stage 9 Final Evidence Weight** 

𝑊 = 𝑅 × 𝑃 𝑖 𝑖 𝑖 

|Evidence|𝑅𝑖|𝑃𝑖|𝑊𝑖|
|---|---|---|---|
|Study A|0.90|0.95|0.855|
|Study B|0.95|0.30|0.285|
|Study C|0.70|0.90|0.630|



A highly reliable study that is poorly applicable to the claim’s population should not automatically dominate evidence that is both reliable and directly applicable. 

## **— 16 16. Stage 10 Evidence Ranking** 

𝑆𝑐𝑜𝑟𝑒𝑖 = 𝑆𝑖𝑚𝑖𝑙𝑎𝑟𝑖𝑡𝑦𝑖 × 𝑊𝑖 

#### or, kept separately: 

Retrieval relevance -> Candidate evidence -> R_i -> P_i -> W_i -> Final ranking This prevents confusing “this paper talks about the topic” with “this paper is strong evidence for this particular claim” — a distinction central to the architecture. 

## **17 17. Stage 11 — Stance Detection (Now Fully Specified)** 

For every evidence item: 

- +1 = supports the claim 

- 0 = neutral / insufficient / unclear 

- -1 = contradicts the claim 

#### **Method — NLI-based:** 

Claim + Evidence -> NLI model ENTAILMENT -> +1 CONTRADICTION -> -1 NEUTRAL -> 0 

Generic NLI is **not** assumed to work perfectly for medical claims — it is evaluated on a small, manually annotated medical evidence-stance set, reporting macro-F1, precision, recall, and confusion matrix. This makes the weighted consensus scientifically testable rather than an unexplained mathematical layer. Stance is determined from the relationship between claim and evidence, never from the paper’s title alone. 

17 

## **18 18. Stage 12 — Weighted Consensus** 

𝐶𝑜𝑛𝑠𝑒𝑛𝑠𝑢𝑠=<sup>∑𝑖𝑊𝑖× 𝑠𝑡𝑎𝑛𝑐𝑒𝑖</sup> ∑𝑖 𝑊𝑖 

Raw evidence distribution and weighted consensus are displayed **separately** : 

Raw evidence: Supporting 7 Contradicting 3 Neutral 2 Weighted evidence: Supporting 0.41 Contradicting 0.56 Neutral 0.03 

Ten weak studies should not necessarily outweigh three strong, applicable studies. 

## **— 19 19. Stage 13 Verdict (Frozen Threshold Policy)** 

Verdicts: TRUE, FALSE, MIXTURE, UNPROVEN. 

First calculate consensus 𝐶 (Section 18), then separately calculate evidence disagreement/coverage. 

|Condition|Verdict|
|---|---|
|Strong positive consensus + adequate evidence|TRUE|
|Strong negative consensus + adequate evidence|FALSE|
|Meaningful positive and negative evidence|MIXTURE|
|Insuficient/neutral evidence|UNPROVEN|



**Do not simply say C > 0.5 = TRUE.** Thresholds are selected using the development set and **frozen before** the final B0–B3 evaluation, preventing threshold-tuning against final test results. 

Architecture: 

Evidence engine -> Structured verification result -> Verdict -> LLM explanation 

The LLM explains the decision; it does not invent the verdict. 

## **20 20. Stage 14 — Grounded Explanation** 

**Primary:** Gemma 3 12B. **Fallback:** Llama 3.1 8B (open-weight model). 

Prompt requirements: 

1. Use only retrieved evidence. 

2. Cite evidence. 

3. Do not invent studies. 

4. Do not invent statistics. 

5. Preserve uncertainty. 

6. Preserve causality language. 

7. Do not convert association into causation. 

8. Do not give diagnosis or treatment advice. 

18 

### **20.1 20.1 Certainty Preservation** 

Certainty inflation is a major LLM risk for medical misinformation. Evidence saying _“may be associated with a lower risk…”_ must **not** become _“prevents the disease.”_ Evidence saying _“evidence is insufficient”_ must **not** become _“the treatment does not work.”_ 

19 

## **21 21. Stage 15 — Faithfulness Verification** 

Generated sentence 

- +-- NLI `-- Hedge Detector 

   - -> Faithfulness decision 

**NLI:** nli-deberta-v3-small, fine-tuned for the health-claim faithfulness task where feasible. 

**Hedge detector:** BioScope-style cues. Hedging: _may, might, possibly, associated with, suggests, could, likely_ . Certainty: _causes, prevents, guarantees, will, proves, cures_ . 

**Decision per sentence:** SUPPORTED, PARTIALLY_SUPPORTED, UNSUPPORTED, CERTAINTY_ESCALATION. If unsupported: regenerate, or mark “Unverified.” Never silently display an unsupported statement. 

### **21.1 21.1 Faithfulness Targets (Corrected)** 

A single fixed threshold (recall ≥0.90) is too rigid and potentially misleading. Use: 

- **Minimum acceptable:** 0.65 

- **Stretch target:** 0.90 

Report **NLI-only recall** and **NLI + hedge recall** separately, to test whether the hedge detector actually contributes. Also explicitly test _“may reduce risk”_ vs. _“reduces risk”_ pairs. 

## **22 22. Stage 16 — Self-Consistency Test** 

Same claim -> Generate explanation x5 -> Compare outputs 

Measured per output set: semantic similarity/agreement, citation consistency, verdict agreement, explanation consistency. This measures **generation stability only** — it is not used as proof of factual correctness. 

## **23 23. Stage 17 — Confidence Calibration** 

The UI never states _“95% probability this claim is medically true.”_ Instead: **System Confidence Indicator: 95%** , reflecting evidence quality, evidence agreement, population applicability, faithfulness, and calibration. 





20 

A reliability diagram is also produced. Explicit disclosure: _“Calibration measures whether the system’s confidence behaves consistently with observed correctness on the evaluation set; it does not establish that the system’s confidence is a medical probability of truth.”_ 

21 

## **24 24. Stage 18 — Medical Safety Layer** 

MedVerify rejects or redirects requests such as: _“Do I have diabetes?”_ , _“What medicine should I take?”_ , _“What dosage should I use?”_ , _“I’m having chest pain, what should I do?”_ — responding that it can help **verify a medical claim** , and recommending professional/emergency care where relevant. 

It distinguishes _“Does drinking cinnamon cure diabetes?”_ (claim to verify) from _“Do I have diabetes?”_ (diagnosis request, out of scope) — the safety system must not refuse everything. 

### **24.1 24.1 Safety Evaluation (Corrected)** 

Remove the “100% correct safety routing” requirement. Use a **40–60 prompt set** with positive and negative examples across categories, plus adversarial paraphrases: 

1. Diagnosis seeking 

2. Treatment seeking 

3. Emergency advice 

4. Dosage requests 

Report sensitivity, specificity, precision, recall, and failure cases. Claim only: _“Safety routing was evaluated on the defined test set,”_ never _“MedVerify is 100% safe.”_ 

22 

## **25 25. Final Experimental Baselines** 

Four systems are run for the core research comparison. 

#### **B0 — LLM Only** 

Claim -> LLM -> Verdict + explanation 

No retrieval. 

#### **B1 — Conventional RAG** 

Claim -> Retrieval -> LLM -> Verdict 

Uses relevance but no 𝑅𝑖/𝑃𝑖. 

#### **B2 — Reliability-Aware RAG** 

Claim -> Retrieval -> R_i -> Ranking -> LLM 

No population applicability. 

#### **B3 — Full MedVerify** 

Claim -> BioBERT -> Retrieval -> R_i -> P_i -> W_i = R_i x P_i -> Consensus -> LLM -> NLI + Hedge -> Calibrated result 

## **26 26. Most Important Experiment: B2 vs. B3** 

𝐵2 (reliability only) vs. 𝐵3 (reliability + population applicability) 

This isolates the effect of the population component far better than comparing B0 against B3. 

**Measured:** verdict accuracy, Macro-F1, evidence ranking, consensus quality, calibration, faithfulness, and — the central measurement — the **B2 →B3 verdict-change rate** . 

## **27 27. Required Evaluation Datasets** 

|Dataset|Purpose|
|---|---|
|PubHealth|BioBERT classifcation + verifcation<br>evaluation|
|SciFact|Retrieval/nDCG evaluation;𝑅𝑖sanity<br>check|
|CoAID (stretch/optional)|Secondary health-misinformation<br>evaluation (vaccination), only with a<br>defned metric|
|Population Ground Truth|Validate𝑃𝑖|
|Faithfulness Set|NLI + hedge evaluation|
|Safety Set (40–60 prompts)|Safety-routing evaluation|



23 

|Dataset|Purpose|
|---|---|
|B0–B3 Set|End-to-end research comparison|
|Expert Review Set|Human validation|
|Real-World Stress Test (new)|Exploratory evaluation on|
||social-media-style claims|



### **27.1 27.1 CoAID Status (Corrected)** 

Move CoAID out of Core to **Stretch/Optional** . Its current role lacked a defined metric/acceptance criterion. Use it only with something concrete defined, e.g. vaccination-claim verdict accuracy/F1 on a clearly specified subset — otherwise omit it, to protect the project from unnecessary scope expansion. 

24 

## **28 28. Population Ground-Truth Dataset** 

Pilot: 20–30 claims, 2 annotators. Expand to **N = 60–80 minimum** (a critical-path component). 

Measured: applicability accuracy, Cohen’s 𝜅, age/sex/region extraction accuracy. 

## **29 29. Faithfulness Dataset** 

Target: **N ≈200** , covering faithful explanations, unsupported statements, partially supported statements, certainty escalation, and citation mismatch. Reported separately: NLI-only recall vs. NLI + hedge recall. 

## **30 30. Expert Evaluation (Corrected)** 

Minimum: 30–40 claims. Stretch: 100–200 claims. A small expert sample is not claimed to prove production safety — instead: _“The system achieved X% agreement on the evaluated expert-review sample.”_ 

### **30.1 30.1 Expert–Expert Agreement (New)** 

Medical experts may themselves disagree (e.g., MIXTURE vs. UNPROVEN, strength of evidence, applicability, certainty). Report **expert–expert agreement as a ceiling reference** , alongside system–expert agreement: 

Expert A vs Expert B agreement: e.g. 84% System vs Expert agreement: e.g. 81% 

This is far more meaningful than reporting system accuracy alone. 

## **31 31. Real-World Social-Media Stress Test (New)** 

PubHealth does not represent the messy, viral, informally worded online medical claims MedVerify is meant to verify in practice. The core idea is **not** changed — the evaluation structure is extended instead: 

- **Benchmark evaluation:** PubHealth, SciFact, and other controlled datasets, for reproducible scientific evaluation. 

- **Real-world stress test:** a small separate dataset of 30–50 publicly available health claims in social-media-style, viral-myth-style, forwarded-message-style, and paraphrased informal wording. Annotated with: claim, source type, claim category, gold verdict, evidence, population. 

This is explicitly labeled **exploratory evaluation** , not a statistically representative social-media benchmark — resolving the benchmark-vs-real-world gap without expanding scope into a full social-media research project. 

25 

## **32 32. Database Schema** 

#### **PostgreSQL.** 

##### **claims** 

id, raw_text, extracted_claim, disease, intervention, outcome, population, created_at 

##### **verifications** 

id, claim_id, verdict, confidence, weighted_consensus, faithfulness_score, status, created_at 

##### **evidence** 

id, verification_id, pmid, doi, title, source, evidence_type, retrieved_text, stance, similarity, R_i, P_i, W_i 

##### **model_versions** 

id, component, model_name, version, hash, created_at 

This version tracking supports reproducibility throughout. 

## **33 33. Backend** 

FastAPI, Python, PostgreSQL, SQLAlchemy, Docker, Nginx 

Endpoints: 

POST /api/claims 

- GET /api/verifications/{id} 

- GET /api/verifications/{id}/evidence 

- GET /api/verifications/{id}/explanation GET /api/verifications/{id}/status 

## **34 34. Frontend** 

React, Vite, Tailwind, Zustand, Recharts, SSE 

Dashboard displays: verdict (TRUE/FALSE/MIXTURE/UNPROVEN); System Confidence Indicator; evidence distribution (supporting/contradicting/neutral); weighted consensus (clearly distinguished from raw counts); explanation with statements linked to evidence; and an evidence table (source, study type, population, reliability, applicability, final weight, stance). 

26 

## **35 35. Final 12-Week Implementation Schedule** 

Week Main Work 

- 1 Freeze architecture, BioBERT, schemas, evaluation protocol 2 PubHealth cleaning, leakage analysis, dataset freeze 3 BioBERT fine-tuning + evaluation 4 Build WHO/CDC/PubMed knowledge base + FAISS 5 MedCPT retrieval + SciFact evaluation 6 Claim/study metadata + population extraction 7 Implement 𝑅𝑖, 𝑃𝑖, 𝑊𝑖 8 Consensus + verdict engine 9 Grounded LLM explanation + citations 10 NLI + BioScope hedge + self-consistency 11 Calibration + safety + B0–B3 evaluation 12 Expert evaluation + analysis + final report + viva 

## **36 36. Critical Gates** 

Progress is not assumed simply because the code runs. 

- **Gate 1 — BioBERT:** Macro-F1 ≥0.85, or documented reason it was not met. 

- **Gate 2 — Retrieval:** nDCG@10 measured on the real evaluation set. 

- **Gate 3 — Population:** N ≥60–80, Cohen’s 𝜅 reported. 

- **Gate 4 — Faithfulness:** NLI-only vs. NLI + hedge reported. 

- **Gate 5 — Calibration:** ECE, Brier, reliability diagram reported. 

- **Gate 6 — Research:** B0, B1, B2, B3 all run and reported — even if B3 does not win. 

This “predefined criteria before seeing results” principle is one of the strongest safeguards for the project’s scientific integrity. 

27 

## **37 37. Final Definition of Done** 

### **37.1 37.1 Engineering** 

- React frontend 

- FastAPI backend 

- PostgreSQL 

- Docker deployment 

- SSE progress 

- FAISS retrieval 

- PubMed integration 

- BioBERT classifier 

- MedCPT retrieval 

- 𝑅𝑖 scoring 

- 𝑃𝑖 scoring 

- 𝑊 calculation 𝑖 

- Consensus engine 

- Explanation generation 

- NLI faithfulness 

- Hedge detection 

- Safety routing 

- Calibration 

### **37.2 37.2 Research** 

- PubHealth leakage audit 

- BioBERT evaluation 

- SciFact retrieval evaluation 

- Population ground truth (N = 60–80) + Cohen’s 𝜅 

- Stance detection evaluation (macro-F1, precision, recall) 

- Verdict threshold freeze (pre-registered on dev set) 

- Frozen evidence snapshot for research runs 

- Faithfulness dataset (N ≈200) 

- NLI vs. NLI+hedge comparison 

- Self-consistency experiment 

- ECE and Brier score 

- B0–B3 experiment 

- B2-vs-B3 verdict-change analysis 

- Expert review + expert–expert agreement ceiling 

- Real-world stress test (exploratory, 30–50 claims) 

- Limitations section 

## **38 38. Summary of Corrections Applied to the Final Plan** 

|Component|Status|
|---|---|
|Claim extraction|Core|



28 

Component Status Retrieval Core 𝑅𝑖 explicit formula **Fixed** — 𝑅𝑖 = 𝑆𝑖𝐸𝑖𝑇𝑖, frozen scoring table 𝑃𝑖 explicit rules **Fixed** — four-dimension average, unspecified ≠0 Population ground truth (60–80) Core Stance detection **Specified** — NLI-based, evaluated on annotated set Verdict thresholds **Specified** — frozen before B0–B3 evaluation 𝑊𝑖 = 𝑅𝑖 × 𝑃𝑖 Core, experimentally compared against additive form B2 vs. B3 **Main research experiment** Frozen evidence snapshot **Implemented** for all research runs Faithfulness (0.65 min / 0.90 stretch) Core Calibration (ECE, Brier) Core Safety sensitivity/specificity (40–60 set) Core Expert–expert agreement Added Real-world online-claim stress test Small exploratory evaluation CoAID Stretch/optional Large expert panel (100–200) Stretch Learned 𝑃 Future work 𝑖 Large-scale social-media study Future work 

**Bottom line:** every important number in MedVerify now has a defined origin, every major decision (thresholds, reliability weights, evaluation splits) is frozen **before** results are seen, and the distinction between **benchmark validation** and **real-world online-claim applicability** is made explicit throughout the plan. 

29 

