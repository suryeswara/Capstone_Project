---
name: medverify-backend-api
description: Reference for MedVerify AI's FastAPI backend — all REST endpoints, JWT auth flow, database session management, DTO schemas, SSE streaming, and how to add new routes or models.
---

# MedVerify AI — Backend API Reference

## Framework & Entry Point

- **Framework**: FastAPI + Uvicorn
- **Entry**: `medverify-ai-backend/app/main.py`
- **API Docs**: `http://localhost:8000/docs` (Swagger UI auto-generated)
- **Base prefix**: `/api`

---

## Authentication System

**File**: `app/api/auth.py`  
**Method**: JWT (custom Bearer tokens, no OAuth)  
**Security**: `app/core/security.py` — bcrypt password hashing + HS256 JWT

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/auth/register` | Register new user (email, password, full_name) |
| POST | `/api/auth/login` | Authenticate → returns JWT token |
| GET | `/api/auth/me` | Get current user profile + stats |
| GET | `/api/auth/history` | Get user's verification history |

### JWT Flow
```
POST /api/auth/login → { access_token: "Bearer eyJ..." }
All protected routes → Authorization: Bearer <token>
→ get_current_user() dependency extracts & validates token
```

### Key DTOs
- `UserRegisterDTO`: `email`, `password`, `full_name`
- `UserLoginDTO`: `email`, `password`
- `TokenDTO`: `access_token`, `token_type`
- `UserProfileDTO`: `id`, `email`, `full_name`, `role`, `stats`

---

## Verification Endpoints

**File**: `app/api/verifications.py`

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/verifications` | Submit a claim → returns `verification_id` |
| GET | `/api/verifications/{id}` | Poll verification status |
| GET | `/api/verifications/{id}/report` | Get full report when COMPLETED |
| GET | `/api/verifications/{id}/stream` | SSE stream of progress events |

### Submit Claim (`SubmitClaimRequestDTO`)
```json
{
  "raw_text": "Statins prevent heart attacks",
  "disease_category_override": null  // optional: force a category
}
```

### Response (`SubmitClaimResponseDTO`)
```json
{
  "verification_id": "ver-abc123",
  "status": "CREATED"
}
```

### Full Report (`VerificationReportDTO`)
```json
{
  "id": "ver-abc123",
  "verdict": "Supported",
  "credibility_score": 82.4,
  "credibility_breakdown": { ... },
  "consensus": { "score": 0.65, "supporting": 4, "contradicting": 1 },
  "evidence": [ { "title": "...", "reliability_score": 0.90, "stance": "supporting" } ],
  "explanation": [ { "sentence": "...", "status": "verified" } ],
  "population_analysis": { ... },
  "version_metadata": { "pipeline_version": "1.0", "models": [...] }
}
```

---

## SSE Streaming (Real-Time Progress)

**Pattern**: FastAPI `BackgroundTasks` + Server-Sent Events

```python
# Pipeline runs in background
background_tasks.add_task(run_real_verification_pipeline, ver_id, raw_text)

# Frontend polls via:
GET /api/verifications/{id}/stream
# Returns: text/event-stream
# Event format: data: {"status": "RETRIEVING", "progress": 45, "label": "..."}
```

### Stage Labels (shown in UI)
| Status | Progress | Label |
|--------|----------|-------|
| EXTRACTING | 10% | Extracting verifiable atomic claim assertion... |
| CLASSIFYING | 25% | Classifying medical domain using fine-tuned DistilBERT... |
| RETRIEVING | 45% | Retrieving evidence from FAISS vector store & live PubMed API... |
| RANKING | 65% | Ranking evidence quality by source tier and recency... |
| CONSENSUS | 80% | Analyzing reliability-weighted medical consensus via NLI... |
| GENERATING | 90% | Generating grounded explanation with sentence citations... |
| VERIFYING | 95% | Evaluating faithfulness (NLI entailment + BioScope certainty)... |

---

## Database Session

**File**: `app/db/session.py`

```python
# Dev: SQLite fallback (USE_SQLITE_FALLBACK=true)
# Prod: PostgreSQL via DATABASE_URL env var

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

def get_db():  # FastAPI dependency
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

---

## Configuration (`app/config.py`)

All settings via `pydantic-settings` (env vars or `.env` file):

| Variable | Default | Purpose |
|----------|---------|---------|
| `POSTGRES_USER` | `medverify` | DB username |
| `POSTGRES_PASSWORD` | `medverify_secret` | DB password |
| `POSTGRES_SERVER` | `localhost` | DB host |
| `POSTGRES_DB` | `medverify_db` | DB name |
| `JWT_SECRET_KEY` | — | Must be set in `.env` for prod |
| `USE_SQLITE_FALLBACK` | `false` | Use SQLite for dev |
| `CONSENSUS_MODE` | `baseline` | `baseline` or `population` |
| `POPULATION_SCORING_MODE` | `v1_age_only` | `v1_age_only` / `v2_age_sex` / `v3_full` |

---

## Adding a New API Route

1. Create `app/api/my_route.py` with `router = APIRouter(prefix="/api/my_route")`
2. Define endpoint functions with `@router.get/post/...`
3. Register in `app/main.py`: `app.include_router(my_route.router)`
4. Add any new DTOs to `app/schemas/dto.py`
5. Add any new DB tables to `app/db/models.py` and run migration

## Database Migrations (Alembic)

```bash
cd medverify-ai-backend
alembic revision --autogenerate -m "description"
alembic upgrade head
```

Or use the sync script: `python scripts/sync_db_schema.py`
