---
name: medverify-ml-testing
description: How to test every ML model and AI pipeline stage in MedVerify AI — test scripts, what each tests, expected outputs, how to run them, and how to interpret pass/fail results.
---

# MedVerify AI — ML Model Testing Reference

## Overview

All test scripts live in two locations:

| Location | Coverage |
|----------|---------|
| `scripts/` | Stage-by-stage integration & validation scripts |
| `tests/` | Unit tests for population extraction & matching |

> [!IMPORTANT]
> All scripts must be run from the **project root** (`Capstone_Project/`), not from `scripts/` or `medverify-ai-backend/`. They add the backend to `sys.path` automatically.

---

## Quick Reference — Run All Tests

```bash
# From project root (Capstone_Project/)

# Stage 7: Hybrid Retrieval (FAISS + PubMed)
python scripts/test_retrieval_engine.py

# Stage 8: Consensus NLI Engine
python scripts/test_consensus_engine.py

# Stage 10: Explanation Generator & Faithfulness
python scripts/test_explanation_generator.py

# Stage 9: Full End-to-End Pipeline Integration
python scripts/test_pipeline_integration.py

# Auth & DB Integration
python scripts/test_auth_system.py

# SSE Streaming Progress
python scripts/test_sse_streaming.py

# Population Extraction Unit Tests
python tests/test_population_extraction.py
python tests/test_population_matching.py

# Live App Benchmark (requires running server at :8000)
python scripts/benchmark_live_app.py
```

---

## Test 1 — Retrieval Engine (`test_retrieval_engine.py`)

**Stage 7 | File**: `scripts/test_retrieval_engine.py`

### What It Tests
| Sub-Test | What it checks |
|----------|---------------|
| TEST 1 | `calculate_reliability_score()` sanity check across all source tiers |
| TEST 2 | FAISS vector similarity search — `FAISSRetriever.search()` with disease filter |
| TEST 3 | Live PubMed NCBI API — `PubMedRetriever.search()` (requires internet) |
| TEST 4 | Full `HybridRetrievalEngine.retrieve_and_rank()` for 3 disease categories |

### Prerequisites
- FAISS index built: `vector_store/faiss_index.bin` must exist
- Internet connection for PubMed API (TEST 3)

### Expected Output
```
[TEST 1] Reliability Scoring: WHO 2025 Vaccination → R_i ≈ 0.90
[TEST 2] FAISS results ≥ 1 for Diabetes query
[TEST 3] PubMed returns ≥ 1 live articles
[TEST 4] Ranked evidence ordered by R_i descending
SUMMARY: Stage 7 Hybrid Retrieval & Evidence Ranking Engine COMPLETE.
```

### Key Assertions
- `cosine_similarity` values are between 0 and 1
- FAISS results have `"source": "FAISS_STATIC"`
- PubMed results have `"source": "PUBMED_LIVE"` and a valid `url`
- Rankings are sorted `R_i` descending

---

## Test 2 — Consensus Engine (`test_consensus_engine.py`)

**Stage 8 | File**: `scripts/test_consensus_engine.py`

### What It Tests
| Sub-Test | What it checks |
|----------|---------------|
| TEST 1 | BioScope certainty level extraction — 6 sentences, 3 levels (L1/L2/L3) |
| TEST 2 | Dual-pass faithfulness: NLI prob + certainty inflation guard |
| TEST 3 | Weighted consensus scoring — 5 scenarios covering all verdict types |
| TEST 4 | End-to-end `VerificationPipeline.verify_claim()` with real NLI DeBERTa model |
| TEST 5 | False claim (MMR-autism myth) → must return `Contradicted` |

> [!NOTE]
> TEST 4 downloads `cross-encoder/nli-deberta-v3-small` on first run (~150 MB). Allow 30–60 seconds.

### BioScope Certainty Levels
| Level | Keywords | Example |
|-------|---------|---------|
| L1 — Weak | may, might, suggests, possible | "may reduce blood sugar" |
| L2 — Moderate | indicates, shows, improves, reduces risk | "shows improved outcomes" |
| L3 — Strong | causes, cures, proves, prevents | "cures type 2 diabetes" |

### Critical Scenario — TEST 3 Scenario B
Tests that **1 Meta-Analysis contradicting beats 4 Case Reports supporting**:
```
R_i weighted:
  Meta-Analysis (R_i=0.92) × -1 = -0.92
  4 × Case Report (R_i=0.25) × +1 = +1.00
  Result: Weighted consensus < 0 → Contradicted ✓
```
This verifies the reliability-weighted system overrides raw vote count.

### Expected Output
```
[TEST 1] BioScope: ALL PASSED (6/6)
[TEST 2] Faithfulness: ALL PASSED (4/4)
[TEST 3] Scenario B → Contradicted (correctly overrides 4:1 raw vote)
[TEST 4] Statins claim → Supported, credibility ≥ 60%
[TEST 5] MMR-autism claim → Contradicted, credibility ≤ 30%
SUMMARY: Stage 8 Consensus Verification Engine COMPLETE.
```

---

## Test 3 — Explanation Generator (`test_explanation_generator.py`)

**Stage 10 | File**: `scripts/test_explanation_generator.py`

### What It Tests
| Sub-Test | What it checks |
|----------|---------------|
| TEST 1 | Grounded sentence drafting with `[ev-N]` inline citation IDs |
| TEST 2 | Contradicted verdict → explanation uses correct negation language |
| TEST 3 | Full orchestrator integration: `MedVerifyOrchestrator.verify()` end-to-end |

### Expected Sentence Output Format
```
[s1] [VERIFIED] (L2 Certainty) Cited: ['ev-101', 'ev-102']
     'Based on current evidence, statins show effectiveness in ...'
```

### Key Assertions
- Every sentence has `sentenceId`, `text`, `status`, `certaintyLevel`, `citedEvidenceIds`
- Status is one of: `"verified"` | `"unsupported"` | `"contradiction"`
- `faithfulnessConfidence` is a float in [0.0, 100.0]
- `credibility_breakdown.faithfulnessConfidence` present in orchestrator result

---

## Test 4 — Full Pipeline Integration (`test_pipeline_integration.py`)

**Stage 9 | File**: `scripts/test_pipeline_integration.py`

### What It Tests
| Sub-Test | Claim | Expected |
|----------|-------|---------|
| TEST 1 | Safety patterns (4 inputs) | Personal advice → `REFUSED`, factual → passes |
| TEST 2 | Disease classification (5 claims) | Correct category label |
| TEST 3 | True claim: Statins | `Supported`, credibility ≥ 60% |
| TEST 4 | False claim: MMR-autism | `Contradicted`, credibility ≤ 30% |
| TEST 5 | Safety refusal: chest pain + aspirin | `REFUSED_SAFETY` |
| TEST 6 | Diabetes claim: Metformin | `Supported` |

### Safety Refusal Patterns Tested
```python
("I have chest pain, should I take aspirin?", True)   # → REFUSED
("I have diabetes, how much insulin?",        True)   # → REFUSED
("Statins reduce cardiovascular risk.",       False)  # → passes
("MMR vaccine causes autism.",                False)  # → passes
```

### Running Time
~2–5 minutes on first run (model loading). Subsequent runs are faster.

---

## Test 5 — Auth System (`test_auth_system.py`)

**File**: `scripts/test_auth_system.py`  
**Method**: Uses FastAPI `TestClient` (no server needed)

### What It Tests
| Test | Endpoint | Check |
|------|----------|-------|
| TEST 1 | POST `/api/auth/register` | New user created, JWT returned |
| TEST 2 | POST `/api/auth/register` (dup) | 400 Bad Request |
| TEST 3 | POST `/api/auth/login` | Correct → 200 + JWT; Wrong → 401 |
| TEST 4 | GET `/api/auth/me` | Profile + DB stats |
| TEST 5 | POST `/api/claims` + GET `/api/auth/history` | Claim linked to user in DB |

### Prerequisites
- DB tables must exist: script calls `Base.metadata.create_all(bind=engine)`
- SQLite or PostgreSQL must be reachable

---

## Test 6 — SSE Streaming (`test_sse_streaming.py`)

**File**: `scripts/test_sse_streaming.py`  
**Method**: Uses FastAPI `TestClient`

### What It Tests
- Claim submission → `verification_id` returned
- SSE stream emits events: `CREATED → EXTRACTING → ... → COMPLETED`
- Each event has `status`, `progress`, `label`, `isTerminal`
- Safety refusal streams `REFUSED_SAFETY` as terminal event

### Expected SSE Event Shape
```json
{
  "status": "RETRIEVING",
  "progress": 45,
  "label": "Retrieving evidence from FAISS vector store & live PubMed API...",
  "isTerminal": false
}
```

---

## Test 7 — Population Tests (`tests/`)

**Files**: `tests/test_population_extraction.py`, `tests/test_population_matching.py`

### What They Test
| File | What |
|------|------|
| `test_population_extraction.py` | Age normalization, sex normalization, region normalization, claim + study population extraction |
| `test_population_matching.py` | Population matching: MATCHED / MISMATCHED / UNKNOWN / PARTIAL |

### Running
```bash
# From project root
python tests/test_population_extraction.py
python tests/test_population_matching.py
```

---

## Benchmark — Live App (`benchmark_live_app.py`)

**File**: `scripts/benchmark_live_app.py`  
**Requires**: Server running at `http://127.0.0.1:8000`

### What It Measures
| Metric | Description |
|--------|-------------|
| Submission latency | POST `/api/claims` response time (ms) |
| SSE stream latency | Full streaming until `isTerminal=true` |
| End-to-end latency | Total verification pipeline time |
| Report accuracy | Verdict + credibility score validation |
| Safety refusal speed | Fast-path timing for personal advice queries |

### Start Server First
```bash
cd medverify-ai-backend
uvicorn app.main:app --port 8000
# Then in another terminal:
python scripts/benchmark_live_app.py
```

---

## Common Test Failures & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `FAISS index not found` | Index not built | `python scripts/build_knowledge_base.py` |
| `NLI model download fails` | No internet | Connect to internet; model is ~150 MB |
| `ModuleNotFoundError: app` | Wrong working directory | Run from project root, not `scripts/` |
| `DB connection error` in auth test | No DB | Set `USE_SQLITE_FALLBACK=true` in `.env` |
| PubMed TEST 3 returns 0 | Network timeout | Check internet; FAISS-only tests still pass |
| `Classifier model not found` | Model not trained | `python scripts/train_disease_classifier.py` |
| TEST 3 Scenario B fails | Weighted consensus bug | Check `ConsensusEngine.calculate_weighted_consensus()` in `consensus_engine.py` |

---

## What "Exit Criteria" Mean

Each test script ends with a `STAGE X EXIT CRITERIA CHECK` block. These are the minimum conditions that must all be `[OK]` before moving to the next stage:

```
STAGE 7 EXIT CRITERIA CHECK:
 [OK] FAISS static vector retrieval operational
 [OK] PubMed live API returned N real-time results
 [OK] Hybrid merge + deduplication working
 [OK] Reliability-ranked output sorted by R_i (descending)
 SUMMARY: Stage 7 COMPLETE.
```

A `[WARN]` means soft failure (e.g. disease classifier gives a different label — acceptable if model not fine-tuned).  
A `[FAIL]` means hard failure — must be fixed before proceeding.
