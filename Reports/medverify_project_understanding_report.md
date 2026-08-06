# MedVerify AI — Technical & AI/ML Architecture Project Understanding Report

**Author:** Technical Lead & AI/ML Architect  
**Project:** MedVerify AI — An Explainable Medical Claim Verification System  
**Source Documents Analyzed:** `MedVerify AI.pdf`, `MedVerify_AI_Implementation_Plan-2.pdf`, `MedVerify_AI_Two_Phase_Plan.pdf`, `Research_Gap_Analysis_Health_Misinformation.pdf`  

---

## Executive Summary

MedVerify AI addresses a critical public health challenge: the rapid spread of health and medical misinformation across social media and digital platforms. Traditional AI detection tools rely on static black-box classifiers (e.g., predicting True/False without evidence) or generic LLMs prone to ungrounded hallucinations. 

MedVerify AI introduces a novel, multi-layered architecture that combines **Medical NLP, Hybrid Retrieval-Augmented Generation (Hybrid RAG), Evidence Reliability Ranking, Medical Consensus Analysis, Grounded LLM Explanation, and a mandatory Faithfulness Verification layer** (NLI + Hedge/Certainty Detection).

---

## 1. Project Summary & Core Objectives

### Problem Statement
Health misinformation leads to delayed diagnoses, ineffective self-medication, vaccine hesitancy, and heightened health risks. Current AI approaches suffer from:
1. **Static Pretrained Knowledge:** Inability to account for daily evolving medical research.
2. **Black-Box Classification:** Providing labels without citations or explanations.
3. **LLM Hallucinations & Certainty Inflation:** LLMs generating fluent but unsupported justifications or upgrading correlation ("associated with") into causation ("cures"/"causes").
4. **Uncalibrated Credibility Scores:** Conflating vector similarity with scientific trustworthiness.

### Primary Objective
To design, implement, and scientifically validate an explainable, evidence-backed medical claim verification platform anchored in peer-reviewed literature and official public health guidelines.

### Boundary Principles & Rules
* **Verification vs. Diagnosis:** MedVerify AI is strictly an **evidence-grounded claim verifier**. It **never** provides clinical diagnosis, medical triage, or treatment recommendations. Input classifier guardrails route diagnosis-seeking queries to a safe refusal/redirect response.
* **Credibility Score Definition:** The Credibility Score is a **System Confidence Indicator** reflecting evidence quality, consensus alignment, and faithfulness — **never** the absolute probability of medical truth.
* **No Guarantees of Truth:** Neither RAG retrieval nor NLI entailment are treated as absolute truth oracles; human-in-the-loop expert audits remain essential.

---

## 2. System Architecture

MedVerify AI is structured as a decoupled, microservice-oriented architecture designed for scalability, version control, and strict component isolation.

```
                                  [ User Input / Client ]
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │  React + Tailwind Web UI  │
                               └─────────────┬─────────────┘
                                             │ REST API (JSON)
                                             ▼
                               ┌───────────────────────────┐
                               │  Spring Boot / FastAPI    │
                               │     API Gateway           │
                               └─────────────┬─────────────┘
                                             │ Async Orchestration
                                             ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                                  AI / ML Microservice                                  │
 │                                                                                        │
 │  [ Claim Extraction ] ──► [ Disease Classifier ] ──► [ Hybrid Retrieval Engine ]       │
 │     (Qwen 3 / Llama 3)         (BioBERT)                (FAISS + PubMed API)           │
 │                                                                   │                    │
 │  ┌────────────────────────────────────────────────────────────────┘                    │
 │  ▼                                                                                     │
 │  [ Evidence Ranking ] ──► [ Consensus Engine ] ──► [ LLM Explanation Generator ]       │
 │    (Source Tier/Recency)    (Weighted Tally)           (Grounded Prompting)            │
 │                                                                   │                    │
 │  ┌────────────────────────────────────────────────────────────────┘                    │
 │  ▼                                                                                     │
 │  [ Faithfulness Verification ] ──► [ Credibility Indicator ]                           │
 │     (NLI + Hedge Detector)            (Calibrated Metric)                              │
 └───────────────────────────────────────────┬────────────────────────────────────────────┘
                                             │
                                             ▼
                              ┌─────────────────────────────┐
                              │    PostgreSQL Relational DB  │
                              │ (Claims, Verifications, DB) │
                              └─────────────────────────────┘
```

### Architectural Layer Responsibilities
1. **Frontend Layer:** React SPA providing claim submission, real-time verification progress via state-machine status polling/WebSocket, structured verdict rendering, evidence breakdown, and reference links.
2. **Backend API Gateway / Orchestrator:** Manages REST endpoints, request ID propagation, rate limiting, authentication (Phase 2), database persistence, and verification state-machine lifecycle execution.
3. **AI/ML Service (FastAPI):** Hosts model pipelines (BioBERT, LLMs, NLI models, FAISS vector indices), handles external PubMed API communications with circuit breakers, and calculates reliability and credibility scores.

---

## 3. End-to-End AI/ML Pipeline

The core execution flow follows a strict 10-step pipeline:

```
[1. User Claim] ──► [2. Claim Extraction] ──► [3. Disease Classification] ──► [4. Evidence Retrieval]
                                                                                        │
[8. Faithfulness Check] ◄── [7. Grounded LLM Expl.] ◄── [6. Consensus Engine] ◄── [5. Evidence Ranking]
          │
          ▼
[9. Credibility Score] ──► [10. Final Verified Result & References]
```

1. **Claim Extraction:** Extracts the core verifiable medical assertion from raw input text using an open-weight LLM (`Qwen 3 8B`/`Llama 3.1 8B`).
2. **Disease Classification:** Routes the claim to its medical category (e.g., Diabetes, Cardiovascular, Vaccination) using a fine-tuned `BioBERT` classifier.
3. **Evidence Retrieval (Hybrid RAG):** Simultaneously queries:
   * *Static Corpus:* FAISS vector store containing pre-indexed WHO, CDC, and PMC guidelines embedded via `BAAI/bge-large-en-v1.5`.
   * *Live Retrieval:* Live PubMed NCBI E-utilities API for recent literature.
4. **Evidence Ranking Engine:** Scores retrieved passages using a transparent, multi-tiered reliability function (Source Tier + Peer-Review Status + Recency Decay).
5. **Medical Consensus Analysis:** Computes a reliability-weighted stance score ($\text{Consensus} = \frac{\sum (R_i \times \text{stance}_i)}{\sum R_i}$) alongside the raw study count.
6. **Grounded LLM Explanation:** Generates a structured narrative requiring explicit chunk citations and strict preservation of original evidence certainty language.
7. **Faithfulness Verification:** Evaluates explanation sentences against cited evidence using a dual-pass checker:
   * *Pass A (NLI Entailment):* `MedNLI`-fine-tuned classifier checking sentence entailment.
   * *Pass B (Hedge/Certainty Detection):* BioScope-style cue matching to detect certainty escalation (e.g., evidence says "may reduce risk", explanation states "prevents").
8. **Credibility Indicator Calculation:** Synthesizes weighted consensus, evidence reliability, recency, and faithfulness pass-rate into a calibrated confidence percentage.
9. **Safety Guardrail & Refusal Check:** Intercepts diagnosis/treatment advice seeking and emits safety disclosures.
10. **Final Evidence-Backed Result Assembly:** Packages verdict, confidence indicator, evidence breakdown, explanation, and clickable DOIs/URLs.

---

## 4. Frontend & Backend Responsibilities

| Subsystem | Key Responsibilities | Tech Stack / Tooling |
| :--- | :--- | :--- |
| **Frontend** | • Input form (Text in Phase 1; Image/Audio in Phase 2)<br>• Real-time verification progress display (State Machine)<br>• Structured Result Dashboard: Verdict badge, Credibility Indicator gauge, Consensus distribution chart, Grounded Explanation with sentence-level citation toggles, Reference list | React, Vanilla CSS / Tailwind CSS |
| **Backend API Gateway** | • REST API endpoints (`/api/claims`, `/api/verifications`)<br>• Request ID propagation and log tracing<br>• Asynchronous verification state machine management<br>• PostgreSQL schema migrations and audit logging<br>• Rate limiting, authorization, and input sanitization | Spring Boot (Java) or FastAPI (Python consolidation) |
| **AI Microservice** | • Model loading and inference pipelines<br>• FAISS vector indexing and similarity search<br>• Live PubMed NCBI E-utilities API fetching with caching & retries<br>• Score calibration and NLI Faithfulness checking | Python 3.10+, FastAPI, HuggingFace Transformers, FAISS, PyTorch |

---

## 5. Database Design

### Relational Store (PostgreSQL)
* `disease_categories`: Supported medical domains, active phase, model versions.
* `claims`: User submitted raw text, extracted claim, assigned disease category, timestamp, request ID.
* `verifications`: Unique verification execution record containing verdict, credibility score, weighted consensus score, faithfulness pass status, model versions, prompt version, KB version.
* `evidence_citations`: Per-verification retrieved chunks, source name, tier, DOI/URL, reliability score, stance (+1/0/-1).
* `users` *(Phase 2)*: Authentication, role-based access.
* `feedback` *(Phase 2)*: User ratings and comments for expert audit queues (never auto-retrained without expert review).

### Vector Store (FAISS)
* **Index Structure:** Flat or IVF index rebuilt during scheduled knowledge updates.
* **Chunk Metadata Fields:** `document_chunk_text`, `source_name`, `source_tier`, `publication_date`, `disease_category`, `evidence_type` (meta-analysis, RCT, cohort, case report), `doi_or_pmid`, `reliability_score`.

---

## 6. Primary Research Contribution

MedVerify AI directly addresses key research gaps identified across recent IEEE health misinformation literature:

1. **Reliability-Weighted RAG (RWRAV):** First system to replace raw vector similarity scoring with a calibrated, source-quality tiering system ($\text{Meta-Analysis} > \text{RCT} > \text{Cohort} > \text{Case Report} > \text{Preprint} > \text{Blog}$).
2. **Faithfulness Verification (Attribution Grounding):** Introduces post-generation NLI entailment and hedge detection, eliminating LLM hallucination and certainty inflation risks.
3. **Calibrated Consensus Engine:** Distinguishes study volume from study quality, displaying both raw stance counts and reliability-weighted consensus.
4. **Reproducible Benchmark Evaluation:** Establishes rigorous evaluation protocols on published datasets (`PubHealth`, `SciFact`, `CoAID`, `MedNLI`) with frozen test set hashes and explicit Baseline A/B ablations.

---

## 7. Phased Development Strategy

### Phase 1 — Core Research MVP (Target: Weeks 1–9)
* **Language & Modality:** English text-only claims.
* **Disease Scope:** 3 core categories with strong benchmark dataset coverage:
  1. **Diabetes**
  2. **Cardiovascular Disease**
  3. **Vaccination**
* **Vector Store:** Local FAISS index.
* **Evidence Corpus:** Static WHO/CDC guidelines + Live PubMed/PMC API.
* **Validation & Gates:** Stage 0 pre-build notebook validation + Two-Pass Review Gate (Engineering Reliability & Research Validity).

### Phase 2 — Expansion & Trust Infrastructure (Target: Weeks 10–17)
* **Multimodal Inputs:** Image OCR (Tesseract / VLM fallback) and Audio Speech-to-Text (Whisper).
* **Multilingual:** Arabic support (AraBERT/CAMeLBERT, BGE-M3 cross-lingual embeddings).
* **Disease Expansion:** 8–10 total categories (adding Cancer, Infectious Diseases, Nutrition, Mental Health, etc.).
* **Trust & Security:** Vector-store poisoning anomaly detection, versioned KB management, user accounts/history, active learning feedback queue.
* **Infrastructure:** Load testing, Qdrant/Weaviate vector DB migration evaluation, Admin/Analytics surveillance dashboard.

---

## 8. Models and Datasets

### Models
| Pipeline Role | Primary Choice | Alternative / Fallback | Notes |
| :--- | :--- | :--- | :--- |
| **Claim Extraction** | Qwen 3 (8B Instruct) | Llama 3.1 (8B) / Gemma 3 | Open-weight, self-hostable |
| **Disease Classifier** | BioBERT | PubMedBERT / ClinicalBERT | Fine-tuned on PubHealth/HealthFact subset |
| **Retrieval Embeddings** | BAAI/bge-large-en-v1.5 | all-MiniLM-L6-v2 | High scientific text recall |
| **LLM Explanation** | Qwen 3 (8B Instruct) | Llama 3.1 (8B) | Single model instance handles extraction & explanation |
| **Faithfulness Check** | MedNLI-fine-tuned DeBERTa/BioBERT + BioScope Hedge Rules | Independent LLM-as-Judge (Phase 2) | NLI entailment + certainty cue matching |

### Datasets & Assigned Roles
* `PubHealth` (~11,800 claims): Fine-tuning & evaluation of verdict classifier; training Faithfulness module.
* `SciFact` (~1,400 claims / 5,000 abstracts): Ground-truth sentence-level rationale evaluation for Evidence Ranking (nDCG@10) and Consensus calibration.
* `CoAID` (~5,000 claims / 296,000 posts): Domain evaluation for Vaccination/Infectious disease and social media text stress testing.
* `HealthFact` (~11,000 claims): Secondary fine-tuning set to expand disease category diversity.
* `MedNLI` (~14,000 sentence pairs): Clinical NLI dataset used specifically to train the Faithfulness entailment checking head.

---

## 9. Key Technical Risks & Mitigations

1. **Relevance $\neq$ Reliability:** Similarity search only finds topically related text.  
   * *Mitigation:* Explicit 3-layer architecture (Retrieval $\rightarrow$ Reliability Scoring $\rightarrow$ Platt Calibration).
2. **Causation vs. Association Trap:** LLM elevating weak correlation to definitive cause.  
   * *Mitigation:* BioScope-style hedge/certainty detector comparing evidence vs. explanation certainty levels.
3. **Data Leakage:** Indexed evidence documents overlapping with benchmark test sets.  
   * *Mitigation:* Freeze, hash, and lock test set manifests in Stage 4 before indexing knowledge base in Stage 5.
4. **Live PubMed API Bottlenecks:** External API latency or rate limiting delaying user response.  
   * *Mitigation:* Redis/Local caching layer, asynchronous timeouts, and automatic fallback to indexed static corpus.
5. **Label Quality & Journalist Bias in Benchmarks:** `PubHealth` labels originate from news fact-checkers, not medical doctors.  
   * *Mitigation:* Conduct a 50–100 item expert medical audit of dataset labels (Stage 4) prior to training.

---

## 10. Document Conflicts Identified

During cross-document analysis, the following inconsistencies between initial design and refined plans were identified and resolved:

1. **Scope Conflict:** `MedVerify AI.pdf` proposed 12 disease categories, OCR, Speech-to-Text, and Arabic support in a single prototype. `MedVerify_AI_Implementation_Plan-2.pdf` correctly de-scopes Phase 1 to 3 categories, text-only, and English.  
   * *Resolution:* Follow the phased roadmap strictly. Phase 1 must validate the 3-category core before Phase 2 expansion.
2. **Reliability Weighting Method:** `MedVerify AI.pdf` asserted static arbitrary weights (WHO=98, Systematic Review=95, RCT=90) without derivation.  
   * *Resolution:* `MedVerify_AI_Two_Phase_Plan.pdf` replaces hardcoded numbers with a versioned, formulaic reliability function calibrated against `SciFact` labels in Stage 0.
3. **Missing Faithfulness Step:** `MedVerify AI.pdf` lacked any mechanism to check if the generated explanation matched retrieved evidence.  
   * *Resolution:* Added mandatory Stage 10 Faithfulness Verification (NLI + Hedge detection).

---

## 11. Recommended Step-by-Step Development Order (Phase 1)

```
[Stage 0: Pre-Build Notebook Validation] (SciFact calibration & NLI+Hedge check)
  │
  ▼
[Stage 1: Architecture & API Contracts Freeze] (DTOs, REST Spec, State Machine Enum)
  │
  ▼
[Stage 2: Backend & Infrastructure Skeleton] (Spring Boot/FastAPI + Postgres + Docker Compose)
  │
  ▼
[Stage 3: Stable API & Mock Frontend Integration] (UI rendering complete via Mock API)
  │
  ▼
[Stage 4: Dataset Governance & Label Audit] (Hashed manifests + 50-item medical label audit)
  │
  ▼
[Stage 5: Knowledge Base Construction] (Chunking, BGE Embeddings, FAISS Index, Leakage Check)
  │
  ▼
[Stage 6: Claim Extraction & Disease Classifier] (BioBERT fine-tuning, Macro-F1 >= 0.85)
  │
  ▼
[Stage 7: Hybrid Retrieval & Evidence Ranking] (FAISS + PubMed API + Calibrated Reliability)
  │
  ▼
[Stage 8: Reliability-Weighted Consensus Engine] (Weighted Stance Score vs Raw Count)
  │
  ▼
[Stage 9: Grounded LLM Explanation Generation] (Qwen 3 / Llama 3.1 with Strict Citations)
  │
  ▼
[Stage 10: Faithfulness Verification Engine] (MedNLI Entailment + BioScope Hedge Detection)
  │
  ▼
[Stage 11: Credibility Indicator & Medical Safety Guardrail] (Safety Routing & UI Confidence)
  │
  ▼
[Stage 12: Security & Reliability Hardening] (Rate limiting, Circuit breakers, Prompt Injection)
  │
  ▼
[Stage 13: Research Evaluation & Two-Pass Review Gate] (Baseline A/B Ablation & Sign-Off)
```

---
*Report compiled and validated against primary source specifications. System context established.*
