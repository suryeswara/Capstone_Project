---
name: medverify-project-overview
description: Top-level map of the MedVerify AI project — its full pipeline stages, all components, where each file lives, and how everything connects end-to-end.
---

# MedVerify AI — Project Overview

## What It Is

**MedVerify AI** is a medical claim fact-verification platform. A user submits a health claim (e.g. "Statins prevent heart attacks") and receives:
- A **verdict** (Supported / Contradicted / Mixed / Insufficient Evidence)
- A **credibility score** (0–100)
- Ranked **evidence citations** from FAISS + PubMed
- A **grounded explanation** with sentence-level citations

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React + TypeScript + Vite + Tailwind CSS |
| Backend | FastAPI (Python 3.11+) |
| AI/ML | SentenceTransformers, HuggingFace Transformers, FAISS |
| Database | PostgreSQL (Docker) / SQLite (dev fallback) |
| Auth | JWT (custom, no OAuth) |
| Containerization | Docker + Docker Compose |
| Reverse Proxy | Nginx (frontend container) |

---

## Pipeline Stages (End-to-End)

```
User Claim (raw text)
      │
      ▼
[Stage 6] Disease Classifier (BioBERT fine-tuned DistilBERT)
      │   → Diabetes | Cardiovascular Disease | Vaccination
      ▼
[Stage 7] Hybrid Retrieval Engine
      │   → FAISS static top-5 + PubMed live top-3
      │   → Merge, deduplicate, rank by R_i score
      ▼
[Stage 8] Consensus Engine (NLI DeBERTa)
      │   → Stance detection: supporting / contradicting / neutral
      │   → Weighted consensus score
      │   → BioScope certainty inflation guard
      ▼
[Stage 10] Explanation Generator
      │   → Grounded sentences with [ev-N] citations
      │   → Dual-pass faithfulness verification (NLI + BioScope)
      ▼
Verdict + Credibility Score + Evidence + Explanation
```

---

## Repository Structure

```
Capstone_Project/
├── medverify-ai-backend/          # FastAPI backend + AI pipeline
│   └── app/
│       ├── api/                   # REST API routes
│       │   ├── auth.py            # JWT auth endpoints
│       │   ├── verifications.py   # Claim submission + SSE streaming
│       │   └── health.py          # /health endpoint
│       ├── services/              # Core AI pipeline
│       │   ├── orchestrator.py    # Unified pipeline wiring (Stage 9)
│       │   ├── retrieval_engine.py # FAISS + PubMed hybrid (Stage 7)
│       │   ├── consensus_engine.py # NLI consensus (Stage 8)
│       │   ├── explanation_generator.py # Grounded LLM explanation (Stage 10)
│       │   └── population/        # Population applicability engine
│       ├── db/
│       │   ├── models.py          # SQLAlchemy ORM models
│       │   └── session.py         # DB session factory
│       ├── schemas/dto.py         # Pydantic DTOs
│       ├── core/security.py       # JWT + bcrypt
│       └── config.py              # Settings via pydantic-settings
├── medverify-ai-frontend/         # React/Vite frontend
│   └── src/
│       ├── pages/                 # 11 pages (Landing, Verify, Dashboard, Report, etc.)
│       ├── components/            # Reusable UI components
│       ├── store/                 # State management
│       └── types/                 # TypeScript types
├── vector_store/                  # Persisted FAISS index
│   ├── faiss_index.bin
│   └── vector_metadata.json
├── models/                        # Fine-tuned ML models
│   └── biobert_disease_classifier/
├── datasets/                      # Raw + processed datasets
│   └── processed/phase1_disease_claims_manifest.json
├── scripts/                       # One-time build/setup scripts
│   ├── build_knowledge_base.py    # Build FAISS index (Stage 5)
│   ├── train_disease_classifier.py
│   ├── download_datasets.py
│   └── ...
└── docker-compose.yml             # Full stack: postgres + backend + frontend
```

---

## Database Schema (SQLAlchemy)

| Table | Key columns |
|-------|------------|
| `users` | id, email, hashed_password, full_name, role |
| `disease_categories` | id, name, active_phase, classifier_model_version |
| `claims` | id, user_id, raw_text, extracted_claim, disease_category |
| `verifications` | id, claim_id, status, verdict, credibility_score, explanation_json, consensus_summary |
| `evidence_citations` | id, verification_id, title, source_type, reliability_score, stance, url |

---

## Status State Machine

```
CREATED → EXTRACTING → CLASSIFYING → RETRIEVING → RANKING
       → CONSENSUS → GENERATING → VERIFYING → COMPLETED
                                             → FAILED
                                             → REFUSED_SAFETY
```

---

## Running Locally (Dev)

```bash
# Start PostgreSQL
docker compose up postgres -d

# Backend
cd medverify-ai-backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Frontend
cd medverify-ai-frontend
npm install && npm run dev
```

## Running Full Stack (Docker)

```bash
docker compose up --build
# Frontend: http://localhost:80
# Backend:  http://localhost:8000
# API Docs: http://localhost:8000/docs
```
