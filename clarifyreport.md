# MedVerify AI — Complete Project Report
### Capstone Project | September 2026

> **Author:** Surya | **Report Type:** Comprehensive Development & Architecture Clarification

---

## Table of Contents

1. [Project Vision & Goal](#1-project-vision--goal)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Complete Folder Structure (with Explanations)](#3-complete-folder-structure)
4. [AI Pipeline Stages (What Was Built)](#4-ai-pipeline-stages)
5. [Backend: FastAPI Application](#5-backend-fastapi-application)
6. [Frontend: React + Vite Application](#6-frontend-react--vite-application)
7. [ML Models & Justification](#7-ml-models--justification)
8. [Datasets & Data Pipeline](#8-datasets--data-pipeline)
9. [Vector Database (FAISS)](#9-vector-database-faiss)
10. [Configuration & DevOps](#10-configuration--devops)
11. [Scripts (Automation & Evaluation)](#11-scripts-automation--evaluation)
12. [Tests](#12-tests)
13. [Experiments & Baseline Comparisons](#13-experiments--baseline-comparisons)
14. [Reports & Documentation](#14-reports--documentation)
15. [The Advanced Consensus Formula](#15-the-advanced-consensus-formula)
16. [Technology Stack Summary](#16-technology-stack-summary)
17. [What Was Done — Chronological Summary](#17-what-was-done--chronological-summary)

---

## 1. Project Vision & Goal

**MedVerify AI** is an enterprise-grade, evidence-grounded AI system that **verifies medical claims** and combats health misinformation by computing a **Reliability-Weighted Scientific Consensus** derived from peer-reviewed clinical literature.

### Why This Matters

Traditional approaches to fact-checking medical claims either:
- Rely on a single generative LLM (GPT-4 etc.) which **hallucinates**, or
- Use simple keyword-matching that ignores **evidence quality** and **demographic relevance**.

MedVerify solves both problems by:
1. Fetching **real medical evidence** (FAISS vector store + live PubMed API)
2. Scoring evidence by **source reliability** (Meta-Analysis > RCT > Case Report)
3. Scoring evidence by **demographic applicability** (does this study apply to *this* patient population?)
4. Running NLI (Natural Language Inference) to determine whether evidence supports, contradicts, or is neutral toward the claim
5. Guarding the final explanation with a **BioScope linguistic certainty guard** to prevent hallucination inflation

---

## 2. High-Level Architecture

The system is conceptually modeled as an automated **Medical Courtroom**:

```
User Claim (Text/Image)
        │
        ▼
┌─────────────────────────┐
│  Stage 11: Safety Guard  │  ← Blocks harmful, off-topic, or dangerous prompts
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  Stage 6: Disease        │  ← Fine-tuned BioBERT (dmis-lab/biobert-base-cased-v1.2) classifies domain
│  Classifier (Triage)     │     (Diabetes / Cardiovascular / Vaccination …)
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  Stage 7: Hybrid         │  ← FAISS dense retrieval + live PubMed API fetch
│  Retrieval Engine        │
└─────────────────────────┘
        │
        ▼
┌────────────────────────────────┐
│  Stage 12: Population           │  ← Extracts Age / Sex / Region from claim
│  Demographic Extraction (NEW)  │     and each piece of evidence, computes P_i
└────────────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  Stage 8: Consensus      │  ← Cross-Encoder NLI (DeBERTa-v3) computes stance
│  Engine (Medical Jury)   │     Weighted Vote: W_i = R_i × P_i × Stance_i
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  Stage 10: Explanation   │  ← Generates human-readable verdict + explanation
│  Generator               │
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  BioScope Faithfulness   │  ← Dual-pass: NLI entailment + linguistic hedge
│  Guard                   │     cue scan to block certainty inflation
└─────────────────────────┘
        │
        ▼
Verdict + 5-Gauge Credibility Dashboard (streamed via SSE to React Frontend)
```

---

## 3. Complete Folder Structure

```
Capstone_Project/
├── .agents/                        ← Agent skills (AI assistant customizations)
├── .vscode/                        ← VS Code workspace settings
├── MediverifIEEE/                  ← IEEE paper drafts and academic references
├── Reports/                        ← Project reports, stage specs, PDF plans
├── Resources/                      ← Research resources & reference materials
├── configs/                        ← YAML configuration files for pipeline tuning
├── docs/                           ← Technical documentation (retrieval flow etc.)
├── experiments/                    ← Experiment scripts and results
├── med_datasets/                   ← ALL training, evaluation, and raw datasets
│   ├── raw/                        ← Original downloaded datasets
│   ├── processed/                  ← Cleaned manifests used by the pipeline
│   └── evaluation/                 ← Benchmark datasets for evaluation metrics
├── medverify-ai-backend/           ← FastAPI Python backend (main server)
│   ├── app/
│   │   ├── api/                    ← REST API route handlers
│   │   ├── core/                   ← Security utilities (JWT, password hashing)
│   │   ├── db/                     ← Database models and session management
│   │   ├── middleware/             ← Request middleware
│   │   ├── schemas/                ← Pydantic DTOs for request/response
│   │   └── services/               ← Core AI pipeline services
│   │       └── population/         ← Demographic extraction sub-package
│   └── Dockerfile                  ← Backend container definition
├── medverify-ai-frontend/          ← React + TypeScript + Vite frontend
│   └── src/
│       ├── components/             ← Reusable UI components (8 sub-groups)
│       ├── pages/                  ← 11 full application pages
│       ├── store/                  ← Zustand global state
│       ├── lib/                    ← API client utilities
│       └── types/                  ← TypeScript type definitions
├── models/
│   └── biobert_disease_classifier/ ← Fine-tuned disease classifier weights
├── scripts/                        ← 26 automation, training, and evaluation scripts
├── tests/                          ← Unit/integration tests
├── vector_store/                   ← FAISS index files (pre-built)
├── docker-compose.yml              ← Multi-container orchestration
└── capstone_project_report.md      ← Original short summary report
```

---

### 3.1 `.agents/` — AI Assistant Skills

This directory contains custom skills (instruction sets) for the Antigravity AI coding assistant. These skills teach the AI about every part of the project so it can answer questions accurately without re-reading code every time.

| Skill File | Purpose |
|---|---|
| `medverify-project-overview` | Top-level map of the entire project |
| `medverify-backend-api` | FastAPI endpoints, JWT auth, database sessions |
| `medverify-ai-pipeline` | Full AI inference pipeline details |
| `medverify-frontend` | React page structure, component organization |
| `medverify-data-pipeline` | Dataset sourcing, processing, and ingestion |
| `medverify-vector-db` | FAISS vector store architecture |
| `medverify-ml-testing` | How to test every ML model and stage |
| `medverify-devops` | Docker, environment variables, migrations |

---

### 3.2 `configs/` — YAML Configuration Files

Centralizes all tunable parameters so the pipeline can be adjusted without changing source code.

| File | Purpose |
|---|---|
| `models.yaml` | Model names, paths, and device settings (CPU/GPU) |
| `retrieval.yaml` | Top-K results, similarity thresholds for FAISS and PubMed |
| `reliability.yaml` | Source reliability weight table (Meta-Analysis = 0.95, Case Report = 0.40, etc.) |
| `population.yaml` | Age bucket definitions, sex normalization rules, region mapping |
| `experiments.yaml` | Experiment hyperparameters (batch sizes, test set sizes) |

---

### 3.3 `docs/` — Technical Documentation

| File | Purpose |
|---|---|
| `current-retrieval-flow.md` | Detailed description of how the hybrid retrieval engine fetches, re-ranks, and deduplicates evidence from FAISS + PubMed |

---

### 3.4 `experiments/` — Experiment Runs

| File/Folder | Purpose |
|---|---|
| `population_applicability/run_pubhealth_experiment.py` | Runs the population scoring on the PUBHEALTH dataset and reports precision/recall of demographic matching |
| `run_baseline_comparison.py` | Compares system performance against 3 baselines: B0 (No weighting), B1 (Only reliability weighting), B2 (Only population weighting), B3 (Full RWRAV formula) |

---

### 3.5 `med_datasets/` — All Datasets

#### `med_datasets/raw/`

| File | Purpose |
|---|---|
| `downloaded_raw_claims.json` | 164 KB of raw medical claims downloaded directly (Phase 1 diseases) |
| `scifact_claims.jsonl` | SciFact scientific claim dataset (placeholder) |
| `scifact_corpus.jsonl` | SciFact evidence corpus (placeholder) |

#### `med_datasets/processed/`

| File | Purpose |
|---|---|
| `phase1_disease_claims_manifest.json` | 165 KB structured manifest of Phase 1 medical claims (Diabetes, Cardiovascular, Vaccination) with labels and metadata |
| `pubhealth_verified_claims.json` | 12.4 MB cleaned & verified PUBHEALTH dataset claims used for training and evaluation |

#### `med_datasets/evaluation/`

| File | Purpose |
|---|---|
| `faithfulness_benchmark_200.json` | 200 curated claim-evidence pairs to benchmark the BioScope faithfulness guard |
| `medical_safety_prompts_50.json` | 50 adversarial/harmful prompts to test the safety guardrail |
| `population_ground_truth_70.json` | 70 manually annotated claim-evidence pairs with gold-standard demographic labels |
| `population_mismatch_benchmark.json` | 9.5 MB benchmark specifically for testing population mismatch detection |
| `pubhealth_experiment_results.json` | Stored results from PUBHEALTH pipeline evaluation runs |
| `social_media_stress_test_40.json` | 40 social media style claims (informal language, emojis) for robustness testing |

> **Why multiple evaluation sets?** Each benchmark tests a *different* failure mode — safety, hallucination, demographic error, and informal language robustness.

---

### 3.6 `models/biobert_disease_classifier/`

Contains the fine-tuned disease classification model weights (267 MB `.safetensors` file).

| File | Purpose |
|---|---|
| `config.json` | Model architecture config (BioBERT `bert` architecture, 12 layers) |
| `model.safetensors` | Fine-tuned weights (~267 MB) for 3-class disease classification |
| `tokenizer.json` | Vocabulary and tokenization rules |
| `tokenizer_config.json` | Tokenizer metadata |

This model was fine-tuned from `dmis-lab/biobert-base-cased-v1.2` (pre-trained on 4.5B words of PubMed + PMC biomedical text) to classify medical claims into **Diabetes**, **Cardiovascular Disease**, or **Vaccination** categories.

---

### 3.7 `vector_store/`

The pre-built FAISS index for sub-millisecond semantic similarity search.

| File | Purpose |
|---|---|
| `faiss_index.bin` | 594 KB binary FAISS index (384-dimensional L2 space) containing encoded medical literature chunks |
| `vector_metadata.json` | 217 KB metadata mapping each FAISS vector ID to its source text, source type, reliability score, and study design |

> The FAISS index is built by `scripts/build_knowledge_base.py` and loaded once at startup by the `HybridRetrievalEngine`.

---

### 3.8 `Reports/` — Stage Reports & Academic References

| File | Purpose |
|---|---|
| `MedVerify_Final_Complete_Plan (1).pdf` | Full 14-stage implementation plan PDF |
| `PICOs-RAG_ PICO-supported...pdf` | Academic reference paper on PICO-guided RAG in evidence-based medicine |
| `baseline_b0_b3_results.json` | Results of the baseline comparison experiment |
| `calibration_reliability_diagram.json` | Calibration curve data for the credibility score |
| `development_decision_log.md` | Log of architectural decisions made during development |
| `medverify_project_understanding_report.md` | Deep-dive understanding report (19 KB) |
| `stage0_validation.py` | Stage 0: validation script for BioScope hedge cues and NLI model checks |
| `stage1_api_contract.md` | Stage 1: API contract specification |
| `stage6_model_training_spec.md` | Stage 6: Disease classifier training specification |

---

## 4. AI Pipeline Stages

The entire pipeline was built across 14 numbered stages. Here is what each stage delivered:

| Stage | Name | What Was Built |
|---|---|---|
| Stage 0 | Foundation Validation | Validated BioScope hedge cues, NLI model compatibility, and dataset quality |
| Stage 1 | API Contract | Defined all REST endpoints, request/response schemas, and SSE streaming contract |
| Stage 2 | Database Schema | SQLAlchemy models for Users, Verifications, Evidence, and Population Analytics |
| Stage 3 | Mock Pipeline | Initial mock end-to-end pipeline (placeholder AI) to test API flow |
| Stage 4 | Frontend Scaffold | React + Vite + Tailwind project setup with routing |
| Stage 5 | Auth System | JWT-based login/registration with PBKDF2-HMAC-SHA256 password hashing |
| Stage 6 | Disease Classifier | Fine-tuned BioBERT (`dmis-lab/biobert-base-cased-v1.2`) for 3-class medical domain classification |
| Stage 7 | Hybrid Retrieval | FAISS dense search + live NCBI PubMed API + BM25 re-ranking |
| Stage 8 | Consensus Engine | Cross-Encoder NLI (DeBERTa-v3) stance detection + RWRAV consensus formula |
| Stage 9 | Orchestrator | Wired all AI stages into a single `MedVerifyOrchestrator` singleton |
| Stage 10 | Explanation Generator | Human-readable verdict generation with faithfulness checking |
| Stage 11 | Safety Guardrail | Multi-layer medical safety filter blocking harmful queries |
| Stage 12 | Population Scoring | Demographic extraction (Age/Sex/Region) + Population Applicability Score P_i |
| Stage 13 | Frontend Overhaul | 5-gauge credibility dashboard, SSE streaming timeline, evidence cards |
| Stage 14 | Containerization | Docker + Docker Compose + Nginx for production-ready deployment |

---

## 5. Backend: FastAPI Application

**Path:** [`medverify-ai-backend/`](file:///c:/Users/Surya/Downloads/Capstone_Project/medverify-ai-backend)

### 5.1 [`app/main.py`](file:///c:/Users/Surya/Downloads/Capstone_Project/medverify-ai-backend/app/main.py)
The FastAPI application entry point. Registers all routers (`auth`, `verifications`, `health`), configures CORS, and starts the Uvicorn ASGI server.

### 5.2 [`app/config.py`](file:///c:/Users/Surya/Downloads/Capstone_Project/medverify-ai-backend/app/config.py)
Loads all environment variables (database URL, JWT secret, model paths) using Pydantic's `BaseSettings`.

---

### 5.3 `app/api/` — REST API Routes

| File | Routes | Purpose |
|---|---|---|
| [`auth.py`](file:///c:/Users/Surya/Downloads/Capstone_Project/medverify-ai-backend/app/api/auth.py) | `POST /auth/register`, `POST /auth/login`, `GET /auth/me` | User registration, JWT login, and profile fetch |
| [`verifications.py`](file:///c:/Users/Surya/Downloads/Capstone_Project/medverify-ai-backend/app/api/verifications.py) | `POST /verify`, `GET /verify/stream/{id}` (SSE), `GET /history`, `GET /reports/{id}` | Main claim verification endpoint with SSE streaming, history, and report generation |
| [`health.py`](file:///c:/Users/Surya/Downloads/Capstone_Project/medverify-ai-backend/app/api/health.py) | `GET /health` | System health check (confirms AI models are loaded) |

> **SSE Streaming:** The `/verify/stream/{id}` endpoint uses FastAPI's `StreamingResponse` to emit JSON events every 250ms, allowing the frontend to display a live progress timeline as each pipeline stage completes.

---

### 5.4 `app/core/` — Security

| File | Purpose |
|---|---|
| [`security.py`](file:///c:/Users/Surya/Downloads/Capstone_Project/medverify-ai-backend/app/core/security.py) | Password hashing (PBKDF2-HMAC-SHA256), JWT token creation and validation using `python-jose` |

---

### 5.5 `app/db/` — Database Layer

| File | Purpose |
|---|---|
| [`models.py`](file:///c:/Users/Surya/Downloads/Capstone_Project/medverify-ai-backend/app/db/models.py) | SQLAlchemy ORM models: `User`, `VerificationResult`, `EvidenceItem`, `PopulationAnalytics` |
| [`session.py`](file:///c:/Users/Surya/Downloads/Capstone_Project/medverify-ai-backend/app/db/session.py) | Database session factory; supports both SQLite (development) and PostgreSQL (production) |

---

### 5.6 `app/schemas/` — Pydantic DTOs

| File | Purpose |
|---|---|
| [`dto.py`](file:///c:/Users/Surya/Downloads/Capstone_Project/medverify-ai-backend/app/schemas/dto.py) | All request/response data transfer objects: `ClaimRequest`, `VerificationResponse`, `EvidenceCard`, `CredibilityBreakdown`, `PopulationMatch` |

---

### 5.7 `app/services/` — Core AI Services

This is the heart of the project. All ML inference happens here.

#### [`orchestrator.py`](file:///c:/Users/Surya/Downloads/Capstone_Project/medverify-ai-backend/app/services/orchestrator.py) (Stage 9 + 11) — The Master Coordinator
The `MedVerifyOrchestrator` singleton class wires the entire pipeline together. It:
- Lazily loads all AI models on first call (to avoid startup time penalty)
- Calls the safety guardrail first
- Runs the disease classifier
- Calls the hybrid retrieval engine
- Triggers population demographic extraction
- Passes results to the consensus engine
- Calls the explanation generator
- Returns the complete `VerificationResponse`

#### [`retrieval_engine.py`](file:///c:/Users/Surya/Downloads/Capstone_Project/medverify-ai-backend/app/services/retrieval_engine.py) (Stage 7) — Hybrid Evidence Retrieval
The `HybridRetrievalEngine` class:
- Loads the FAISS index and metadata from `vector_store/`
- Encodes the query using `all-MiniLM-L6-v2` SentenceTransformer
- Performs L2-distance nearest-neighbor search on FAISS
- Simultaneously queries the **NCBI PubMed Entrez API** for live abstracts (systematic reviews + RCTs)
- Merges and deduplicates results
- Attaches `reliability_score` to each piece of evidence based on study design

#### [`consensus_engine.py`](file:///c:/Users/Surya/Downloads/Capstone_Project/medverify-ai-backend/app/services/consensus_engine.py) (Stage 8) — NLI Stance Detection
The `VerificationPipeline` class:
- Loads `cross-encoder/nli-deberta-v3-small` for Natural Language Inference
- For each evidence item, computes **Entailment / Contradiction / Neutral** toward the claim
- Aggregates into a single consensus score using the RWRAV formula: `C = Σ(R_i × P_i × S_i) / Σ(R_i × P_i)`
- Maps the consensus score to a verdict: **VERIFIED / DISPUTED / UNVERIFIED / INSUFFICIENT_EVIDENCE**

The `FaithfulnessVerifier` class:
- Performs dual-pass faithfulness checking:
  - **Pass 1:** NLI entailment check between explanation and evidence
  - **Pass 2:** BioScope hedge cue regex scan (blocks certainty inflation)

#### [`explanation_generator.py`](file:///c:/Users/Surya/Downloads/Capstone_Project/medverify-ai-backend/app/services/explanation_generator.py) (Stage 10) — Human-Readable Output
Generates a structured, human-readable explanation of the verdict including:
- Summary sentence with confidence level
- Evidence breakdown with source citations
- Population applicability notes
- Certainty caveats based on BioScope level

#### [`safety_guardrail.py`](file:///c:/Users/Surya/Downloads/Capstone_Project/medverify-ai-backend/app/services/safety_guardrail.py) (Stage 11) — Medical Safety Filter
`MedicalSafetyGuardrail` is a multi-layer filter that runs **before** any AI processing:
- Detects **self-harm** and **crisis** language → blocks and redirects to emergency services
- Detects **non-medical** off-topic queries → blocks with explanation
- Detects **personally identifiable information (PII)** → warns and redacts
- Detects **solicitation for medical advice** → blocks with disclaimer
- Detects **drug interactions / dosage queries** → blocks with pharmacist referral
- Returns `SafetyResult` with a `SafetyAction` enum: `ALLOW`, `BLOCK`, `WARN`, `REDACT`

#### [`image_claim_extractor.py`](file:///c:/Users/Surya/Downloads/Capstone_Project/medverify-ai-backend/app/services/image_claim_extractor.py) — Image-to-Text Claim Extraction
Handles cases where the user submits a screenshot or image of a medical claim. Extracts the text using OCR-style processing before feeding it to the pipeline.

---

### 5.8 `app/services/population/` — Demographic Extraction Sub-Package

This is the most significant architectural addition — a complete sub-package for precision demographic analysis.

| File | Purpose |
|---|---|
| [`schemas.py`](file:///c:/Users/Surya/Downloads/Capstone_Project/medverify-ai-backend/app/services/population/schemas.py) | Pydantic models: `AgeProfile`, `SexProfile`, `RegionProfile`, `PopulationProfile`, `ApplicabilityScore` |
| [`normalizer.py`](file:///c:/Users/Surya/Downloads/Capstone_Project/medverify-ai-backend/app/services/population/normalizer.py) | Normalizes raw age mentions (e.g., "elderly", "65 years old", "pediatric") into standardized age buckets; normalizes sex mentions |
| [`claim_population.py`](file:///c:/Users/Surya/Downloads/Capstone_Project/medverify-ai-backend/app/services/population/claim_population.py) | Extracts demographic information **from the user's claim text** using regex + NLP |
| [`study_population.py`](file:///c:/Users/Surya/Downloads/Capstone_Project/medverify-ai-backend/app/services/population/study_population.py) | Extracts demographic information **from the retrieved evidence abstracts** using regex + NLP |
| [`clinical_trials.py`](file:///c:/Users/Surya/Downloads/Capstone_Project/medverify-ai-backend/app/services/population/clinical_trials.py) | Specialized parser for clinical trial abstracts (handles CONSORT-style reporting, inclusion/exclusion criteria) |
| [`matcher.py`](file:///c:/Users/Surya/Downloads/Capstone_Project/medverify-ai-backend/app/services/population/matcher.py) | Compares claim demographics vs. evidence demographics; returns a match score for each axis (Age, Sex, Region) |
| [`applicability.py`](file:///c:/Users/Surya/Downloads/Capstone_Project/medverify-ai-backend/app/services/population/applicability.py) | Aggregates individual axis match scores into a single `P_i` score using configurable weights; exposes `analyze_population_applicability()` called by the orchestrator |
| [`__init__.py`](file:///c:/Users/Surya/Downloads/Capstone_Project/medverify-ai-backend/app/services/population/__init__.py) | Public API: re-exports `analyze_population_applicability`, `PopulationProfile`, `ApplicabilityScore` |

**How `P_i` is calculated:**

```
P_i = (0.40 × Age_Match) + (0.30 × Sex_Match) + (0.30 × Region_Match)

Where each axis match is:
  1.00 = Exact match
  0.70 = Partial match (overlapping age ranges)
  0.50 = Broad match (same general category)
  0.10 = Mismatch (explicitly different demographics)
  0.50 = Unknown (no demographic info in either)
```

---

## 6. Frontend: React + Vite Application

**Path:** [`medverify-ai-frontend/`](file:///c:/Users/Surya/Downloads/Capstone_Project/medverify-ai-frontend)

### 6.1 Technology Choices

- **React 18** + **TypeScript** — Type-safe component tree
- **Vite** — Sub-second HMR during development
- **Tailwind CSS** — Utility-first styling
- **Zustand** — Lightweight global state management
- **Recharts** — Credibility gauge charts

### 6.2 Pages (`src/pages/`)

| Page | Purpose |
|---|---|
| [`Landing.tsx`](file:///c:/Users/Surya/Downloads/Capstone_Project/medverify-ai-frontend/src/pages/Landing.tsx) | Marketing landing page (14 KB) — hero section, feature highlights, CTA |
| [`Verify.tsx`](file:///c:/Users/Surya/Downloads/Capstone_Project/medverify-ai-frontend/src/pages/Verify.tsx) | Main claim verification interface — text input, image upload, SSE streaming timeline |
| [`Dashboard.tsx`](file:///c:/Users/Surya/Downloads/Capstone_Project/medverify-ai-frontend/src/pages/Dashboard.tsx) | User dashboard — recent verifications, statistics |
| [`HistoryPage.tsx`](file:///c:/Users/Surya/Downloads/Capstone_Project/medverify-ai-frontend/src/pages/HistoryPage.tsx) | Full verification history with search and filter |
| [`ReportPage.tsx`](file:///c:/Users/Surya/Downloads/Capstone_Project/medverify-ai-frontend/src/pages/ReportPage.tsx) | Detailed report view for a single verification |
| [`Analytics.tsx`](file:///c:/Users/Surya/Downloads/Capstone_Project/medverify-ai-frontend/src/pages/Analytics.tsx) | Aggregated analytics charts (verdict distribution, domain distribution) |
| [`Consensus.tsx`](file:///c:/Users/Surya/Downloads/Capstone_Project/medverify-ai-frontend/src/pages/Consensus.tsx) | Consensus breakdown explorer — shows NLI scores per evidence item |
| [`EvidenceExplorer.tsx`](file:///c:/Users/Surya/Downloads/Capstone_Project/medverify-ai-frontend/src/pages/EvidenceExplorer.tsx) | Browse raw evidence retrieved for a claim |
| [`FaithfulnessLab.tsx`](file:///c:/Users/Surya/Downloads/Capstone_Project/medverify-ai-frontend/src/pages/FaithfulnessLab.tsx) | Developer tool: test BioScope faithfulness checking interactively |
| [`Admin.tsx`](file:///c:/Users/Surya/Downloads/Capstone_Project/medverify-ai-frontend/src/pages/Admin.tsx) | Admin panel — user management, system stats |
| [`NotFound.tsx`](file:///c:/Users/Surya/Downloads/Capstone_Project/medverify-ai-frontend/src/pages/NotFound.tsx) | 404 error page |

### 6.3 Components (`src/components/`)

| Sub-group | Purpose |
|---|---|
| `auth/` | Login and registration forms |
| `consensus/` | Consensus score visualizations |
| `credibility/` | 5-gauge credibility dashboard (Overall, Evidence Confidence, Consensus, Source Quality, Faithfulness) |
| `evidence/` | Evidence cards with demographic badges, reliability badges, stance indicators |
| `faithfulness/` | BioScope certainty level visualization |
| `layout/` | Navbar, sidebar, footer, page wrappers |
| `pipeline/` | SSE streaming progress timeline component |
| `ui/` | Atomic UI primitives: buttons, modals, badges, spinners |

### 6.4 State Management (`src/store/useAppStore.ts`)

10 KB Zustand store managing:
- Auth state (user, JWT token)
- Current verification session (streaming progress, result)
- History list
- Analytics data
- UI state (sidebar open/close, theme)

---

## 7. ML Models & Justification

### 7.1 Disease Classifier — Fine-tuned BioBERT

| Attribute | Value |
|---|---|
| **Base Model** | `dmis-lab/biobert-base-cased-v1.2` |
| **Pre-training corpus** | PubMed abstracts + PMC full-text (4.5B biomedical words) |
| **Fine-tuned on** | Phase 1 claims manifest (Diabetes / Cardiovascular / Vaccination) |
| **Output** | 3-class softmax classification |
| **Optimizer** | AdamW lr=2e-5, weight_decay=0.01 |
| **Epochs** | 4 with early stopping on Val Macro-F1 |
| **Why BioBERT?** | Pre-trained exclusively on biomedical text (PubMed + PMC), giving it native understanding of clinical terminology, disease names, and medical abbreviations. A general-purpose model like DistilBERT lacks this domain knowledge — e.g., it cannot natively distinguish `T2DM` (Type 2 Diabetes Mellitus) or `MI` (Myocardial Infarction) as clinical terms vs. ambiguous general-language words. |

### 7.2 Embedding Model — `all-MiniLM-L6-v2`

| Attribute | Value |
|---|---|
| **Model** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Output** | 384-dimensional dense vectors |
| **Why this model?** | Best-in-class speed vs. accuracy trade-off for semantic similarity. Handles medical text well without domain-specific fine-tuning. 6-layer architecture means inference in <10ms. |

### 7.3 NLI / Stance Detection — `nli-deberta-v3-small`

| Attribute | Value |
|---|---|
| **Model** | `cross-encoder/nli-deberta-v3-small` |
| **Task** | Natural Language Inference (Entailment / Contradiction / Neutral) |
| **Why Cross-Encoder?** | Unlike bi-encoders, cross-encoders process the claim and evidence *together*, enabling attention across both texts. This is dramatically more accurate for logical entailment. |
| **Why DeBERTa-v3?** | Uses disentangled attention (relative positional encoding separate from content encoding) + enhanced mask decoder — current state-of-the-art for NLI benchmarks (SuperGLUE). |

### 7.4 Faithfulness Guard — BioScope Regex Engine

| Attribute | Value |
|---|---|
| **Type** | Deterministic rule-based (no neural network) |
| **Source** | BioScope biomedical corpus hedge cue dictionary |
| **Why deterministic?** | A neural model used for safety checking could itself hallucinate. Deterministic rules derived from the BioScope corpus are guaranteed to fire correctly. |

**BioScope Certainty Levels:**

| Level | Label | Example Cues | Blocked from upgrading to |
|---|---|---|---|
| 1 | Weak / Speculative | may, might, suggests, possible | Level 2 or 3 language |
| 2 | Moderate / Indicative | indicates, demonstrates, likely | Level 3 language |
| 3 | Strong / Definitive | causes, cures, proves, definitively | — (maximum) |

---

## 8. Datasets & Data Pipeline

### 8.1 Dataset Sources

| Dataset | Purpose | Size |
|---|---|---|
| **Phase 1 Custom Claims** | Initial 3-domain training data (Diabetes, Cardio, Vaccination) | 164 KB |
| **PUBHEALTH** | Real-world health claim dataset from news & social media with expert fact-check labels | 44 MB (raw), 12 MB (processed) |
| **PubMed (Live API)** | Real-time evidence retrieval during inference (not stored — fetched on-demand) | Dynamic |

### 8.2 Data Pipeline Flow

```
scripts/download_datasets.py
        │
        ▼ (Raw JSON/CSV files in med_datasets/raw/)
scripts/process_pubhealth_dataset.py
        │
        ▼ (Cleaned JSON in med_datasets/processed/)
scripts/build_knowledge_base.py
        │
        ▼ (FAISS index in vector_store/)
Pipeline Ready
```

### 8.3 Evaluation Benchmark Files

| File | Test Target | Size |
|---|---|---|
| `faithfulness_benchmark_200.json` | BioScope guard precision/recall | 200 items |
| `medical_safety_prompts_50.json` | Safety guardrail precision | 50 prompts |
| `population_ground_truth_70.json` | P_i score accuracy (Cohen's Kappa) | 70 annotated pairs |
| `population_mismatch_benchmark.json` | Large-scale mismatch detection | 9.5 MB |
| `social_media_stress_test_40.json` | Informal language robustness | 40 claims |

---

## 9. Vector Database (FAISS)

**Path:** [`vector_store/`](file:///c:/Users/Surya/Downloads/Capstone_Project/vector_store)

### Architecture

```
Medical Literature Chunks (text)
        │
        ▼ (SentenceTransformer: all-MiniLM-L6-v2)
384-dimensional dense vectors
        │
        ▼
FAISS IndexFlatL2 (faiss_index.bin)
        │
vector_metadata.json ← maps vector ID → {text, source, study_design, reliability_score, pub_year}
```

### Key Facts

| Property | Value |
|---|---|
| **Index type** | `IndexFlatL2` — exact L2 distance (no approximation) |
| **Embedding dimension** | 384 |
| **Index file size** | ~594 KB |
| **Metadata file size** | ~217 KB |
| **Approximate chunks indexed** | ~1,500 medical literature chunks |
| **Default Top-K** | 10 results (configurable in `configs/retrieval.yaml`) |
| **Re-ranking** | BM25 lexical re-ranking applied after FAISS retrieval |

---

## 10. Configuration & DevOps

### 10.1 YAML Config Centralization

```yaml
# configs/reliability.yaml
study_design_weights:
  meta_analysis: 0.95
  systematic_review: 0.90
  rct: 0.80
  cohort_study: 0.65
  case_control: 0.55
  case_report: 0.40
  expert_opinion: 0.30
  in_vitro: 0.20
```

### 10.2 Docker & Deployment Architecture

```
[Browser]
    │
    ▼
[Nginx :80]
    ├── /api/* → [FastAPI Backend :8000]
    └── /*     → [React Static Build (Vite dist/)]
```

| File | Purpose |
|---|---|
| [`docker-compose.yml`](file:///c:/Users/Surya/Downloads/Capstone_Project/docker-compose.yml) | Orchestrates backend + frontend + nginx containers |
| [`medverify-ai-backend/Dockerfile`](file:///c:/Users/Surya/Downloads/Capstone_Project/medverify-ai-backend/Dockerfile) | Python 3.12 container, installs pip requirements, runs uvicorn |
| [`medverify-ai-frontend/Dockerfile`](file:///c:/Users/Surya/Downloads/Capstone_Project/medverify-ai-frontend/Dockerfile) | Node 20 container, builds Vite app, serves via Nginx |
| [`medverify-ai-frontend/nginx.conf`](file:///c:/Users/Surya/Downloads/Capstone_Project/medverify-ai-frontend/nginx.conf) | Nginx reverse proxy: `/api/*` → backend, `/` → React SPA |

### 10.3 Database Configuration

| Environment | Database | Setup |
|---|---|---|
| Development | SQLite (`medverify_dev.db`) | Auto-created at startup |
| Production | PostgreSQL | Set `DATABASE_URL` environment variable |

---

## 11. Scripts (Automation & Evaluation)

**Path:** [`scripts/`](file:///c:/Users/Surya/Downloads/Capstone_Project/scripts) — **26 Python scripts**

### Data & Infrastructure Scripts

| Script | Purpose |
|---|---|
| `download_datasets.py` | Downloads SciFact, CoAID raw datasets |
| `download_real_disease_knowledge.py` | Fetches PubMed abstracts for the knowledge base |
| `process_pubhealth_dataset.py` | Ingests PUBHEALTH CSV → cleaned JSONL |
| `build_knowledge_base.py` | Builds FAISS index from processed medical chunks |
| `train_disease_classifier.py` | Fine-tunes BioBERT (`dmis-lab/biobert-base-cased-v1.2`) on Phase 1 disease claims |
| `sample_pubhealth_claims.py` | Samples N claims from PUBHEALTH for quick testing |
| `verify_dataset_governance.py` | Audits dataset quality and label balance |
| `init_postgres_db.py` | Initializes PostgreSQL database |
| `sync_db_schema.py` | Synchronizes ORM schema to database |
| `seed_user.py` | Creates a test admin user |

### Evaluation Scripts

| Script | What It Measures |
|---|---|
| `evaluate_biobert_classifier.py` | Accuracy / F1 of disease classifier on held-out test set |
| `evaluate_pipeline.py` | End-to-end pipeline accuracy on Phase 1 benchmark |
| `evaluate_faithfulness_gate.py` | Precision/recall of BioScope guard on 200-item benchmark |
| `evaluate_safety_layer.py` | Safety guardrail precision on 50 adversarial prompts |
| `evaluate_calibration.py` | Reliability diagram (how well credibility scores match actual accuracy) |
| `evaluate_population_kappa.py` | Cohen's Kappa inter-annotator agreement for population matching |
| `evaluate_self_consistency.py` | Tests if repeated runs on same claim produce consistent verdicts |
| `benchmark_live_app.py` | End-to-end latency benchmarking against the running FastAPI server |

### Benchmark Generation Scripts

| Script | What It Generates |
|---|---|
| `generate_faithfulness_benchmark_200.py` | 200 annotated claim-evidence pairs for faithfulness testing |
| `generate_population_gt_70.py` | 70 gold-standard demographic annotation examples |

### Test Scripts (within `scripts/`)

| Script | What It Tests |
|---|---|
| `test_consensus_engine.py` | Unit tests for NLI stance detection + RWRAV formula |
| `test_explanation_generator.py` | Tests explanation generation quality |
| `test_pipeline_integration.py` | End-to-end integration tests for the full pipeline |
| `test_retrieval_engine.py` | Tests FAISS retrieval + PubMed API fallback |
| `test_sse_streaming.py` | Tests Server-Sent Events streaming correctness |
| `test_auth_system.py` | Tests JWT registration / login / token validation |

---

## 12. Tests

**Path:** [`tests/`](file:///c:/Users/Surya/Downloads/Capstone_Project/tests)

| File | Purpose |
|---|---|
| [`test_population_extraction.py`](file:///c:/Users/Surya/Downloads/Capstone_Project/tests/test_population_extraction.py) | Validates that the NLP correctly extracts Age/Sex/Region from 50+ diverse claim and abstract texts |
| [`test_population_matching.py`](file:///c:/Users/Surya/Downloads/Capstone_Project/tests/test_population_matching.py) | Validates `P_i` score calculation against manually annotated ground truth |

---

## 13. Experiments & Baseline Comparisons

**Path:** [`experiments/`](file:///c:/Users/Surya/Downloads/Capstone_Project/experiments)

### Baseline Comparison (`run_baseline_comparison.py`)

Compares 4 system configurations to justify the full RWRAV formula:

| Baseline | Description |
|---|---|
| **B0** | No weighting — all evidence treated equally |
| **B1** | Reliability weighting only (R_i) |
| **B2** | Population weighting only (P_i) |
| **B3** | Full RWRAV: R_i × P_i (our system) |

Results stored in [`Reports/baseline_b0_b3_results.json`](file:///c:/Users/Surya/Downloads/Capstone_Project/Reports/baseline_b0_b3_results.json).

### Population Applicability Experiment

Runs the population extraction pipeline on PUBHEALTH and reports:
- Coverage: % of claims with extractable demographics
- Precision / Recall on `population_ground_truth_70.json`
- Distribution of P_i scores across all verifications

---

## 14. Reports & Documentation

**Path:** [`Reports/`](file:///c:/Users/Surya/Downloads/Capstone_Project/Reports) and [`docs/`](file:///c:/Users/Surya/Downloads/Capstone_Project/docs)

| File | Purpose |
|---|---|
| [`medverify_project_understanding_report.md`](file:///c:/Users/Surya/Downloads/Capstone_Project/Reports/medverify_project_understanding_report.md) | 19 KB deep understanding report of the entire system |
| [`development_decision_log.md`](file:///c:/Users/Surya/Downloads/Capstone_Project/Reports/development_decision_log.md) | Chronological log of key architectural decisions |
| [`stage1_api_contract.md`](file:///c:/Users/Surya/Downloads/Capstone_Project/Reports/stage1_api_contract.md) | Full REST API contract specification |
| [`stage6_model_training_spec.md`](file:///c:/Users/Surya/Downloads/Capstone_Project/Reports/stage6_model_training_spec.md) | Training configuration for the disease classifier |
| [`docs/current-retrieval-flow.md`](file:///c:/Users/Surya/Downloads/Capstone_Project/docs/current-retrieval-flow.md) | 17 KB technical description of the complete retrieval pipeline |
| [`project_architecture_details.md`](file:///c:/Users/Surya/Downloads/Capstone_Project/project_architecture_details.md) | Architecture + model justification document |

---

## 15. The Advanced Consensus Formula

The intellectual core of MedVerify is the **Reliability-Weighted Relative Agreement Voting (RWRAV)** formula, extended with Population Applicability:

$$C = \frac{\sum_{i=1}^{N} (R_i \times P_i \times S_i)}{\sum_{i=1}^{N} (R_i \times P_i)}$$

| Symbol | Meaning | Range |
|---|---|---|
| **C** | Final consensus score | [-1.0, +1.0] |
| **S_i** | Stance from DeBERTa NLI (+1 Support, -1 Contradict, 0 Neutral) | {-1, 0, +1} |
| **R_i** | Source reliability weight (by study design hierarchy) | [0.20, 0.95] |
| **P_i** | Population applicability score (demographic match) | [0.10, 1.00] |

**Verdict mapping from C:**

| C Score | Verdict | Display |
|---|---|---|
| C ≥ 0.60 | VERIFIED | ✅ |
| 0.20 ≤ C < 0.60 | LIKELY TRUE | ⚠️ |
| -0.20 < C < 0.20 | DISPUTED | 🔶 |
| -0.60 < C ≤ -0.20 | LIKELY FALSE | ⚠️ |
| C ≤ -0.60 | CONTRADICTED | ❌ |
| < 3 evidence items | INSUFFICIENT EVIDENCE | ℹ️ |

> **Key insight:** A Meta-Analysis (R=0.95) on the wrong demographic (P=0.10) gets W=0.095 — nearly silent. This prevents obscure, mismatched studies from overriding relevant evidence.

---

## 16. Technology Stack Summary

| Layer | Technology | Why |
|---|---|---|
| Disease Classification | BioBERT (`dmis-lab/biobert-base-cased-v1.2`) | Domain pre-trained on 4.5B biomedical words (PubMed + PMC) |
| Semantic Embedding | `all-MiniLM-L6-v2` | Best speed/accuracy for similarity |
| NLI / Stance | `nli-deberta-v3-small` | SOTA NLI with cross-encoder precision |
| Vector Search | FAISS IndexFlatL2 | Sub-millisecond exact search |
| Live Evidence | NCBI PubMed E-Utilities API | Real-time peer-reviewed literature |
| Faithfulness Guard | BioScope Regex Engine | Deterministic, no hallucination risk |
| Backend | FastAPI + Uvicorn (Python 3.12) | Async-native, SSE streaming support |
| ORM | SQLAlchemy 2.0 | Unified SQLite/PostgreSQL support |
| Frontend | React 18 + TypeScript + Vite | Fast HMR, type safety |
| Styling | Tailwind CSS | Utility-first, consistent design |
| State Management | Zustand | Lightweight, no boilerplate |
| Charts | Recharts | Composable chart library for React |
| Auth | JWT + PBKDF2-HMAC-SHA256 | Secure, stateless authentication |
| Containerization | Docker + Docker Compose | Production-ready, platform-agnostic |
| Reverse Proxy | Nginx | API routing + SPA serving |

---

## 17. What Was Done — Chronological Summary

### Phase 1: Foundation & Data
1. **Dataset Design** — Defined Phase 1 domains; downloaded raw claims; created `phase1_disease_claims_manifest.json`
2. **PUBHEALTH Integration** — Wrote `process_pubhealth_dataset.py` to parse 44 MB PUBHEALTH XLS into 12 MB structured JSON
3. **Knowledge Base Construction** — Fetched PubMed abstracts; built FAISS index with `build_knowledge_base.py`

### Phase 2: Core ML Pipeline
4. **Disease Classifier Training** — Fine-tuned BioBERT (`dmis-lab/biobert-base-cased-v1.2`) on Phase 1 claims; saved to `models/biobert_disease_classifier/`
5. **Hybrid Retrieval Engine** — Built `retrieval_engine.py` combining FAISS + live PubMed API + BM25 re-ranking
6. **Consensus Engine** — Built `consensus_engine.py` with DeBERTa-v3 NLI and the full RWRAV formula
7. **BioScope Faithfulness Guard** — Implemented dual-pass faithfulness verification
8. **Explanation Generator** — Built `explanation_generator.py` for human-readable verdict generation

### Phase 3: Population Applicability (Major Feature)
9. **Population Schemas** — Defined `AgeProfile`, `SexProfile`, `RegionProfile` Pydantic models
10. **Demographic Extraction** — Built `claim_population.py` and `study_population.py`
11. **Clinical Trial Parser** — Wrote `clinical_trials.py` for CONSORT-format abstracts
12. **Normalizer** — Built `normalizer.py` to normalize "elderly" → `65+`, "pediatric" → `0-17` etc.
13. **Matcher & Applicability** — Built `matcher.py` and `applicability.py`; integrated into orchestrator
14. **Ground Truth Generation** — Generated `population_ground_truth_70.json` and `population_mismatch_benchmark.json`

### Phase 4: Safety & Security
15. **Medical Safety Guardrail** — Built `safety_guardrail.py` with 6 detection categories
16. **Auth System** — Implemented JWT + PBKDF2 authentication

### Phase 5: Backend API & Streaming
17. **Database Layer** — Built SQLAlchemy models and sessions
18. **REST API** — Built all routes in `verifications.py` and `auth.py`
19. **SSE Streaming** — Implemented Server-Sent Events streaming every 250ms for live progress
20. **Orchestrator** — Wired all services into `MedVerifyOrchestrator` singleton

### Phase 6: Frontend
21. **Project Scaffold** — Created React + TypeScript + Vite + Tailwind project
22. **Landing Page** — Built 14 KB marketing landing page
23. **Verify Page** — Built main claim input UI with SSE streaming timeline
24. **5-Gauge Dashboard** — Built credibility breakdown with 5 gauge charts
25. **Evidence Cards** — Built evidence cards with demographic + reliability badges
26. **All Remaining Pages** — Dashboard, History, Report, Analytics, Consensus, Evidence Explorer, Faithfulness Lab, Admin

### Phase 7: Evaluation & Benchmarking
27. **Evaluation Benchmarks** — Generated all 6 evaluation datasets
28. **Evaluation Scripts** — Wrote 9 evaluation scripts (faithfulness, safety, pipeline, calibration, kappa, self-consistency)
29. **Baseline Comparison** — Ran B0–B3 experiment; confirmed RWRAV outperforms all baselines
30. **Integration Tests** — Wrote auth, pipeline, consensus, retrieval, SSE test scripts

### Phase 8: DevOps & Documentation
31. **Docker** — Created Dockerfiles for backend and frontend
32. **Docker Compose** — Orchestrated multi-container deployment
33. **Nginx Config** — Configured reverse proxy for API routing + SPA serving
34. **Documentation** — Created `.agents` skills (8 skill files), stage reports, architecture details, this report

---

> *Report generated: **15 September 2026***  
> *Project: **MedVerify AI — Capstone Project***  
> *Status: **Active Development — All 14 Stages Complete***
