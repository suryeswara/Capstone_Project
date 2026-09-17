---
name: medverify-devops
description: How to build, run, deploy, and debug the MedVerify AI full stack — Docker Compose setup, environment variables, database migrations, testing scripts, and common troubleshooting patterns.
---

# MedVerify AI — DevOps & Deployment Reference

## Docker Compose Architecture

**File**: `docker-compose.yml` (project root)

```
┌─────────────────────────────────────────────┐
│              Docker Network                  │
│                                             │
│  ┌──────────┐   ┌──────────┐   ┌─────────┐ │
│  │ postgres │   │ backend  │   │frontend │ │
│  │ :5432    │◄──│ :8000    │◄──│ :80     │ │
│  │ postgres │   │ FastAPI  │   │ Nginx   │ │
│  │ _16-alp. │   │ Python   │   │ React   │ │
│  └──────────┘   └──────────┘   └─────────┘ │
└─────────────────────────────────────────────┘
```

### Services

| Service | Image | Port | Data |
|---------|-------|------|------|
| `postgres` | `postgres:16-alpine` | 5432 | `postgres_data` volume |
| `backend` | Custom Dockerfile | 8000 | Mounts `vector_store/` + `models/` |
| `frontend` | Custom Dockerfile (Nginx) | 80 | Serves `dist/` |

---

## Environment Variables

### Backend (`.env` file in `medverify-ai-backend/`)

```env
# Database
POSTGRES_USER=medverify
POSTGRES_PASSWORD=medverify_secret
POSTGRES_SERVER=localhost        # Use 'postgres' in Docker
POSTGRES_PORT=5432
POSTGRES_DB=medverify_db

# Auth
JWT_SECRET_KEY=your-secret-key-here   # CHANGE IN PRODUCTION

# Dev options
USE_SQLITE_FALLBACK=false             # true for local dev without Docker
CONSENSUS_MODE=population             # 'baseline' or 'population'
POPULATION_SCORING_MODE=v1_age_only   # v1_age_only | v2_age_sex | v3_full
```

Copy from `.env.example` to get started:
```bash
cp medverify-ai-backend/.env.example medverify-ai-backend/.env
```

---

## Running the Stack

### Full Docker Stack (Recommended)
```bash
# From project root
docker compose up --build

# Access:
# Frontend:  http://localhost
# Backend:   http://localhost:8000
# API Docs:  http://localhost:8000/docs
```

### Dev Mode (No Docker)
```bash
# Terminal 1: Database only
docker compose up postgres -d

# Terminal 2: Backend
cd medverify-ai-backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Terminal 3: Frontend
cd medverify-ai-frontend
npm install
npm run dev   # http://localhost:5173
```

### SQLite Dev (No Docker at all)
```bash
# Set in .env:
USE_SQLITE_FALLBACK=true
# Then run backend normally
```

---

## Database Migrations

```bash
cd medverify-ai-backend

# Auto-generate migration from model changes
alembic revision --autogenerate -m "add new column"

# Apply migrations
alembic upgrade head

# Or use the sync script (simpler, creates tables directly)
python ../scripts/sync_db_schema.py

# Seed a test user
python ../scripts/seed_user.py
```

---

## One-Time Setup Order

Run these in sequence for a fresh environment:

```bash
# 1. Download datasets
python scripts/download_datasets.py
python scripts/download_real_disease_knowledge.py

# 2. Process & manifest
python scripts/process_pubhealth_dataset.py

# 3. Train disease classifier
python scripts/train_disease_classifier.py

# 4. Build FAISS vector index
python scripts/build_knowledge_base.py

# 5. Start DB & apply schema
docker compose up postgres -d
python scripts/sync_db_schema.py

# 6. Seed test user
python scripts/seed_user.py

# 7. Start full stack
docker compose up --build
```

---

## Testing

### Unit & Integration Tests
```bash
cd scripts

python test_retrieval_engine.py          # FAISS retrieval tests
python test_consensus_engine.py          # NLI consensus tests
python test_explanation_generator.py     # Faithfulness tests
python test_pipeline_integration.py      # End-to-end pipeline
python test_auth_system.py               # JWT auth flow
python test_sse_streaming.py             # SSE progress streaming
```

### Benchmarking
```bash
python scripts/benchmark_live_app.py
```

---

## Common Issues & Fixes

| Problem | Fix |
|---------|-----|
| FAISS index not found | Run `python scripts/build_knowledge_base.py` |
| Classifier model missing | Run `python scripts/train_disease_classifier.py` |
| DB connection refused | Start PostgreSQL: `docker compose up postgres -d` |
| JWT token errors | Check `JWT_SECRET_KEY` is set in `.env` |
| PubMed API timeout | Adjust `TIMEOUT = 8` in `retrieval_engine.py` |
| Frontend can't reach backend | Check Nginx proxy in `nginx.conf`: `/api → backend:8000` |
| SQLite + Docker conflict | Set `USE_SQLITE_FALLBACK=false` in Docker env |
| Model memory error | Reduce `faiss_top_k` and `pubmed_max` in orchestrator |

---

## Logs

```bash
# All services
docker compose logs -f

# Backend only
docker compose logs -f backend

# Backend uvicorn logs include per-request pipeline stage timings
```

---

## Production Checklist

- [ ] Set `JWT_SECRET_KEY` to a strong random value
- [ ] Change default PostgreSQL credentials
- [ ] Mount `vector_store/` and `models/` as Docker volumes (not baked into image)
- [ ] Set `CONSENSUS_MODE=population` for full applicability scoring
- [ ] Enable HTTPS via a reverse proxy (e.g. Traefik or Caddy) in front of Nginx
- [ ] Set CORS origins in `app/main.py` to only allow the frontend domain
