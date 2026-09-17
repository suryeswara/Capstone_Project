# MedVerify AI: Reliability-Weighted Biomedical Claim Verification & Faithfulness Engine

MedVerify AI is an enterprise-grade, evidence-grounded AI system designed to verify medical claims, detect health misinformation, and compute reliability-weighted scientific consensus from peer-reviewed clinical literature (PubMed & static FAISS vector store).

---

## 🏛️ The Core Analogy: The Automated Medical Courtroom

To understand how MedVerify AI processes a medical assertion, imagine a high-tech **Medical Courtroom**:

```text
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                            USER MEDICAL CLAIM                           │
 └────────────────────────────────────┬────────────────────────────────────┘
                                      │
                                      ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │ 1. THE TRIAGE DESK (Stage 6: BioBERT Disease Classifier)                │
 │    Classifies claim into domain: Diabetes, Cardiovascular, Vaccination  │
 └────────────────────────────────────┬────────────────────────────────────┘
                                      │
                                      ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │ 2. THE MEDICAL LIBRARIAN & INVESTIGATOR (Stage 7: Hybrid Retrieval)     │
 │    Fetches peer-reviewed literature from FAISS Vault + Live PubMed API  │
 └────────────────────────────────────┬────────────────────────────────────┘
                                      │
                                      ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │ 3. DEMOGRAPHIC ANALYST (Stage 12: Population Applicability Scoring)     │
 │    Matches Patient Demographics (Age, Sex, Region) in Claim vs Evidence │
 └────────────────────────────────────┬────────────────────────────────────┘
                                      │
                                      ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │ 4. THE EXPERT MEDICAL JURY (Stage 8 & 12: Weighted Consensus)          │
 │    Cross-Encoder NLI weighs source reliability (Ri) & applicability (Pi)│
 └────────────────────────────────────┬────────────────────────────────────┘
                                      │
                                      ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │ 5. THE FRAUD & CERTAINTY INSPECTOR (Stage 10: BioScope Faithfulness)    │
 │    Guards against certainty inflation (e.g. "suggests" -> "cures")      │
 └────────────────────────────────────┬────────────────────────────────────┘
                                      │
                                      ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │ 6. THE FINAL VERDICT & CREDIBILITY REPORT (5-Gauge Score + Citations)   │
 └─────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technologies & AI Models Used

| Component Layer | Technology / Model Used | Purpose / Role |
| :--- | :--- | :--- |
| **Domain Classifier** | `BioBERT` (`dmis-lab/biobert-base-cased-v1.2`) | Fine-tuned 3-class sequence classifier for medical domain triage (pre-trained on PubMed + PMC) |
| **Dense Vector Embeddings** | `SentenceTransformers` (`all-MiniLM-L6-v2`) | 384-dimensional dense semantic embedding generation |
| **Vector Store Index** | `FAISS` (Facebook AI Similarity Search) | Sub-millisecond dense retrieval across structured medical chunks |
| **Live Evidence Retrieval** | `NCBI PubMed E-Utilities API` | Live fetch of PubMed systematic reviews & RCT abstracts |
| **Stance Detection (NLI)** | `Cross-Encoder` (`cross-encoder/nli-deberta-v3-small`) | Natural Language Inference (Entailment, Contradiction, Neutral) |
| **Hedge & Certainty Scanner** | BioScope Linguistic Hedge Cue Regex Engine | 3-tier certainty level extraction ($L1$ speculative, $L2$ indicative, $L3$ definitive) |
| **Backend API Gateway** | `FastAPI` (Python 3.12) + `Uvicorn` | Asynchronous REST gateway with Server-Sent Events (SSE) streaming |
| **Database ORM** | `SQLAlchemy 2.0` + `PostgreSQL` / `SQLite` | Schema persistence for users, claims, and population metrics |
| **Security & Auth** | `PBKDF2-HMAC-SHA256` + `JWT (HS256)` | Cryptographic password hashing and bearer token user authentication |
| **Frontend UI** | `React 18` + `Vite` + `Tailwind CSS` + `Zustand` | Interactive dashboard with 5-gauge credibility breakdown & live progress bar |
| **Containerization** | `Docker` + `Docker Compose` + `Nginx` | Production-ready multi-container deployment architecture |

---

## 📋 Stage-by-Stage Implementation Journey

Below is the comprehensive analysis of all stages implemented in this project, tracking our evolution from architecture freeze to advanced population scoring and deployment readiness.

---

### Stage 0: Pre-Build Governance & Architectural Rules Freeze
- **Analogy:** *Building the Constitutional Blueprint before constructing the courthouse.*
- **Implementation:**
  - Established strict separation between general medical claim verification and personal diagnostic advice.
  - Defined the non-negotiable consensus formula: Reliability-Weighted Relative Agreement Voting ($RWRAV$).

### Stage 1: REST API Contract & State Machine Freeze
- **Analogy:** *Architecting the standardized legal proceedings and court docket.*
- **Implementation:**
  - Authored OpenAPI-compliant Data Transfer Objects (`dto.py`).
  - Formulated the state machine transitions (CREATED $\rightarrow$ EXTRACTING $\rightarrow$ ... $\rightarrow$ COMPLETED).
  - Defined the 5-tier verdict taxonomy: `Supported`, `Likely Supported`, `Inconclusive`, `Likely Refuted`, `Refuted`.

### Stage 2: Backend Core & Database Schema Setup
- **Analogy:** *Laying the solid steel foundation and vault storage of the courthouse.*
- **Implementation:**
  - Built FastAPI server skeleton with request ID tracking.
  - Created SQLAlchemy 2.0 database models: `UserModel`, `ClaimModel`, `VerificationModel`, `EvidenceCitationModel`.
  - Configured PostgreSQL primary with automatic SQLite local fallback (`medverify_dev.db`).

### Stage 3: Interactive Frontend Web Application & Mock API
- **Analogy:** *Constructing the interactive public gallery and jury display screens.*
- **Implementation:**
  - Created modern React 18 + Vite web app with Tailwind CSS styling.
  - Designed the **5-Gauge Credibility Breakdown UI**.
  - Integrated state-machine timeline progress bar and sentence-level evidence citation badges (`[ev-1]`).

### Stage 4: Dataset Governance & Phase 1 Categorization
- **Analogy:** *Cataloging historical legal precedent and medical evidence files.*
- **Implementation:**
  - Curated biomedical datasets (CoAID, PubMed, SciFact) into 3 Phase 1 disease domains (Diabetes, Cardiovascular, Vaccination).
  - Extended ingestion logic for the real-world **PUBHEALTH dataset**, structuring thousands of complex medical claims into verified JSONL artifacts for robust testing.

### Stage 5: FAISS Vector Indexing & Semantic Search Engine
- **Analogy:** *Creating a sub-millisecond digital search engine for the medical library.*
- **Implementation:**
  - Encoded medical text chunks using `all-MiniLM-L6-v2` into 384-dimensional dense vectors.
  - Built `FAISSRetriever` (`vector_store/index.faiss`) using L2 distance mapped to Cosine Similarity.

### Stage 6: Biomedical Disease Domain Classifier Fine-Tuning
- **Analogy:** *Training the Triage Officer to assign cases to the correct specialist.*
- **Implementation:**
  - Fine-tuned `dmis-lab/biobert-base-cased-v1.2` (pre-trained on 4.5B biomedical words from PubMed abstracts + PMC full-text) as a 3-class sequence classifier for medical domain triage.
  - Hyperparameters: AdamW lr=2e-5, weight_decay=0.01, 4 epochs, batch_size=16, max_length=256.
  - Class-balanced dataset: 150 samples per class (450 total), stratified 70/15/15 train/val/test split.

### Stage 7: Hybrid Evidence Retrieval Engine & Calibrated Reliability Scoring
- **Analogy:** *Dispatching investigators to search both the physical vault and live worldwide medical archives.*
- **Implementation:**
  - Built `HybridRetrievalEngine` combining static vector search (`FAISSRetriever`) with live PubMed NCBI Entrez API search (`PubMedRetriever`).
  - Implemented study design hierarchy reliability weights ($R_i$), scaling from Meta-Analysis ($0.95$) to Case Reports ($0.40$).
  - Applied exponential publication year recency decay.

### Stage 8: Reliability-Weighted Consensus Engine & NLI Stance Detector
- **Analogy:** *The Medical Jury deliberating on the claim based on evidence quality.*
- **Implementation:**
  - Integrated Cross-Encoder NLI (`cross-encoder/nli-deberta-v3-small`) formatted with premise (evidence) and hypothesis (claim) text pairs.
  - Calculated weighted consensus score based solely on evidence reliability ($R_i$) (Upgraded in Stage 12).

### Stage 9: Unified Backend Integration & Real AI Pipeline Wiring
- **Analogy:** *Wiring all courtroom departments into a seamless automated system.*
- **Implementation:**
  - Created `MedVerifyOrchestrator` implementing a lazy-loading singleton pattern for all AI models.
  - Connected FastAPI background tasks to execute the real AI pipeline end-to-end.

### Stage 10: Grounded LLM Explanation Generator & Sentence Faithfulness Verifier
- **Analogy:** *The Court Reporter writing the final verdict, audited by the Fraud Inspector.*
- **Implementation:**
  - Created `ExplanationGenerator` drafting patient-accessible explanations with inline citations.
  - Implemented **Dual-Pass Faithfulness Verification**: Probability check + BioScope Certainty Inflation Guard (verifying sentence certainty level $\le$ evidence certainty level).

### Stage 11: Real-Time SSE Streaming, Safety Refusal & Auth System
- **Analogy:** *Installing live trial broadcasting, emergency triage guards, and secure security badges.*
- **Implementation:**
  - Built Server-Sent Events (SSE) streaming API for real-time progress updates.
  - Implemented fast-path sub-15ms safety refusal for personal medical advice queries.
  - Built Authentication System (PBKDF2-HMAC-SHA256, JWT, User DB).

---

## ✨ NEW IMPROVEMENTS & ADVANCED FEATURES ✨

### Stage 12: Population Applicability Scoring (Precision Demographics)
- **Problem:** A claim about *"drug efficacy in pediatric females"* should not be strongly supported by a trial conducted strictly on *"geriatric males"*.
- **Implementation:** 
  - Integrated a `analyze_population_applicability` step into the pipeline.
  - Added extraction of **Age, Sex, and Region** demographics from both the claim and the retrieved evidence.
  - Computed the **Applicability Score ($P_i$)** dynamically (from $0.1$ to $1.0$) based on demographic overlap.
  - **Consensus Equation Upgrade:** Evolved the weighting formula from purely reliability-based to **$W_i = R_i \times P_i$**. A high-quality Meta-Analysis ($R_i = 0.95$) on the wrong population ($P_i = 0.2$) is now properly heavily discounted ($W_i = 0.19$).

### Stage 13: UI Enhancements for Evidence Explainability
- **Implementation:**
  - Modified the frontend `EvidenceCard` components to surface the new population metrics.
  - Added colored UI badges representing the `populationMatchType` (e.g., EXACT_MATCH, PARTIAL_MATCH, NO_MATCH, MISMATCH).
  - Explicitly display the $P_i$ score and extracted demographic factors directly to the user for maximum transparency.

### Stage 14: Dockerization & Production Deployment Architecture
- **Implementation:**
  - Fully containerized the system for cloud-native deployment.
  - Created `Dockerfile` for the FastAPI backend and a multi-stage `Dockerfile` (Node.js + Nginx) for the Vite React frontend.
  - Wrote a unified `docker-compose.yml` to orchestrate the backend, frontend, and reverse proxy networking seamlessly.
  - Setup Nginx to properly route `/api` traffic to the backend and serve static frontend assets.

---

## 🚀 Running the Project Locally

### Development Mode (Direct Python/Node)
**1. Start the Backend API Server:**
```bash
cd medverify-ai-backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
* API Server: [http://localhost:8000](http://localhost:8000)
* Interactive Swagger Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

**2. Start the Frontend Web Application:**
```bash
cd medverify-ai-frontend
npm run dev
```
* Web Dashboard: [http://localhost:5173](http://localhost:5173)

### Production Mode (Docker Compose)
```bash
docker-compose up --build
```
* The unified application will be available at [http://localhost:80](http://localhost:80)

