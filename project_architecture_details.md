# MedVerify AI: Complete Project Architecture & Model Justification Report

## 1. Project Overview
**MedVerify AI** is an enterprise-grade, evidence-grounded AI system designed to verify medical claims and combat health misinformation. 

Unlike traditional Large Language Models (LLMs) that often hallucinate or weigh facts based on internet "popularity," MedVerify relies exclusively on **Reliability-Weighted Scientific Consensus**. It retrieves real medical evidence and scores it based on the quality of the study (e.g., Meta-Analysis > Case Report) and its demographic applicability to the claim.

---

## 2. Core Architecture Pipeline

The system is built as a multi-stage pipeline, conceptually modeled after an automated **Medical Courtroom**:

1. **The Triage Desk (Domain Classification):** Incoming claims are classified into specific biomedical domains (e.g., Cardiology, Virology, Oncology) to route them appropriately.
2. **The Medical Librarian (Hybrid Retrieval):** The system searches for evidence in two places:
   - A local **FAISS** vector store containing curated medical chunks (fast, static).
   - The **NCBI PubMed E-Utilities API** for live, real-time fetching of clinical trials and systematic reviews (dynamic).
3. **The Demographic Analyst (Population Extraction):** The system extracts patient demographics (Age, Sex, Region, Condition) from both the user's claim and the retrieved evidence to compute an **Applicability Score ($P_i$)**.
4. **The Medical Jury (NLI & Consensus):** An AI model judges whether the evidence *Supports*, *Contradicts*, or is *Neutral* toward the claim. The final verdict is determined by a weighted vote: **Weight = Reliability ($R_i$) × Applicability ($P_i$)**.
5. **The Fraud Inspector (BioScope Guard):** Before presenting the verdict, a linguistic algorithm checks the generated explanation against the BioScope corpus to ensure the AI isn't inflating the certainty of speculative evidence.

---

## 3. AI Models Used & Justification (The "Why")

MedVerify intentionally avoids using a single, monolithic generative LLM (like GPT-4) for the entire process. Instead, it uses specialized, fine-tuned models for distinct tasks to maximize precision, speed, and safety.

### 3.1. The Triage Model (Domain Classifier)
* **Model Used:** Fine-tuned `dmis-lab/biobert-base-cased-v1.2`.
* **Role:** Classifies the medical claim into disease domains (Diabetes, Cardiovascular Disease, Vaccination).
* **Why this model?** BioBERT is pre-trained on **4.5 billion words of biomedical text** (PubMed abstracts + PMC full-text articles), giving it native understanding of medical terminology, abbreviations, and clinical language that general-purpose models like DistilBERT fundamentally lack. For a medical domain classifier, domain-specific pre-training directly translates to higher precision on rare disease terms and clinical phrasing.

### 3.2. The Embedding Model (Semantic Search)
* **Model Used:** `SentenceTransformers` -> `all-MiniLM-L6-v2`.
* **Role:** Converts claim text and medical abstracts into 384-dimensional dense vectors to perform similarity search in the FAISS database.
* **Why this model?** When querying a vector database, latency is paramount. `all-MiniLM-L6-v2` provides an exceptional balance of semantic accuracy and blazing-fast inference speed, allowing us to search thousands of documents in milliseconds without needing massive GPU resources.

### 3.3. The Jury Model (Stance Detection / NLI)
* **Model Used:** Cross-Encoder -> `cross-encoder/nli-deberta-v3-small`.
* **Role:** Natural Language Inference (NLI). It takes a Premise (the medical evidence) and a Hypothesis (the user's claim) and outputs whether they Entail (Support), Contradict, or are Neutral.
* **Why this model?** 
  - **Cross-Encoders vs. Bi-Encoders:** Cross-encoders process both sentences simultaneously, allowing the attention mechanism to compare words between the claim and the evidence directly. This results in far higher accuracy for logical entailment than Bi-Encoders.
  - **DeBERTa-v3:** DeBERTa uses disentangled attention and enhanced mask decoder training, making it currently state-of-the-art for NLI tasks compared to older models like RoBERTa.

### 3.4. The Faithfulness Guard
* **Model Used:** BioScope Linguistic Hedge Cue Regex Engine (Algorithmic / Rules-based).
* **Role:** Scans the generated explanations for certainty inflation (e.g., upgrading "might help" to "cures").
* **Why this model?** Instead of relying on a neural network that could itself hallucinate, we use deterministic linguistic rules derived from the BioScope biomedical corpus. If evidence is rated at Level 1 (Speculative), the system physically blocks the output from using Level 3 (Definitive) vocabulary.

---

## 4. Technical Stack

* **Backend Gateway:** `FastAPI` + `Uvicorn` (Python 3.12). Chosen for its native asynchronous capabilities, allowing for Server-Sent Events (SSE) streaming to the frontend.
* **Vector Database:** `FAISS` (Facebook AI Similarity Search). Chosen because it runs entirely in-memory and handles dense vector similarity (L2 distance / Cosine Similarity) instantly.
* **Relational Database:** `SQLAlchemy 2.0` with SQLite/PostgreSQL for structured persistence (users, verification history, population analytics).
* **Frontend:** `React 18` + `Vite` + `Tailwind CSS`. Chosen for rapid component rendering and managing the complex, real-time state of the 5-gauge credibility dashboard.
* **Infrastructure:** `Docker` + `Docker Compose` + `Nginx`. Ensures the system is platform-agnostic and ready for cloud deployment.

---

## 5. The Advanced Consensus Formula

The secret sauce of MedVerify is the **Reliability-Weighted Relative Agreement Voting (RWRAV)** formula, upgraded with **Population Applicability**:

$$C = \frac{\sum_{i=1}^{N} (R_i \times P_i \times S_i)}{\sum_{i=1}^{N} (R_i \times P_i)}$$

Where:
* $S_i$ = Stance score (+1 Support, -1 Contradict, 0 Neutral) provided by the DeBERTa NLI model.
* $R_i$ = Reliability Score (e.g., 0.95 for Meta-Analysis, 0.40 for Case Report).
* $P_i$ = Population Applicability Score (Demographic match between claim and evidence).

**Why this matters:** If an AI finds a low-quality Case Report ($R_i = 0.40$) on the wrong demographic ($P_i = 0.2$), its voting power drops to $0.08$. This prevents obscure, low-quality studies from overriding high-quality systematic reviews, solving a major flaw in basic RAG applications.
