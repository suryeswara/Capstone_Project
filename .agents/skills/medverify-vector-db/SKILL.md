---
name: medverify-vector-db
description: Architecture reference for MedVerify AI's FAISS vector database — how it is built, stored, loaded, and queried in the hybrid retrieval pipeline.
---

# MedVerify AI — Vector Database Architecture

## Overview

The project uses a **FAISS flat index** (not a hosted/cloud vector DB) as a static knowledge base of fact-checked medical claims, paired with a **live PubMed API** retriever. Together they form the **Hybrid Retrieval Engine** (Stage 7 of the pipeline).

---

## File Locations

| File | Purpose |
|------|---------|
| `vector_store/faiss_index.bin` | FAISS binary index (~580 KB, ~4600 vectors) |
| `vector_store/vector_metadata.json` | JSON array of chunk metadata keyed by vector_id |
| `scripts/build_knowledge_base.py` | One-time script to build & save the index |
| `medverify-ai-backend/app/services/retrieval_engine.py` | Runtime FAISS + PubMed retriever classes |
| `medverify-ai-backend/app/services/orchestrator.py` | Lazy-loads `HybridRetrievalEngine` on first request |

---

## Build Process (`build_knowledge_base.py`)

Run once from the project root:
```bash
python scripts/build_knowledge_base.py
```

### Steps:
1. **Load embedding model**: `all-MiniLM-L6-v2` (SentenceTransformers, 384-D)
2. **Ingest corpus**: Reads `datasets/processed/phase1_disease_claims_manifest.json`
3. **Chunk & format**: Each record → `"Claim: <text>. Explanation: <explanation>"` string
4. **Embed**: `model.encode(texts, normalize_embeddings=True)` → float32 numpy array
5. **Index**: `faiss.IndexFlatIP(384)` (Inner Product = Cosine similarity on L2-normalized vectors)
6. **Persist**: `faiss.write_index(index, "vector_store/faiss_index.bin")` + metadata JSON

### Each metadata entry has:
```json
{
  "vector_id": 0,
  "chunk_text": "Claim: ... Explanation: ...",
  "raw_claim": "...",
  "disease_category": "Vaccination | Diabetes | Cardiovascular Disease",
  "source_name": "CoAID | PubHealth | ...",
  "source_tier": "WHO/CDC Guideline | Systematic Review / Meta-Analysis",
  "reliability_score": 0.85,
  "verdict_stance": "Supported | Contradicted | Unverified",
  "publication_year": 2024,
  "doc_id": "coaid-fake_05-0"
}
```

---

## Runtime Query Flow (`retrieval_engine.py`)

### FAISSRetriever.search(query, top_k, disease_filter)
1. Encode query with same `all-MiniLM-L6-v2` model, normalized
2. `index.search(query_vec, top_k * 3)` — fetches 3x candidates for post-filter
3. Optional `disease_filter` prune (e.g. only "Vaccination" chunks)
4. Returns top_k results with `cosine_similarity` scores

### PubMedRetriever.search(query, disease_category, max_results)
1. Calls NCBI E-utilities `esearch.fcgi` → gets PMIDs
2. Calls `esummary.fcgi` → gets titles, pub types, dates
3. Maps pub_type → source_tier → reliability score

### HybridRetrievalEngine.retrieve_and_rank(claim_text, disease_category)
1. FAISS top-5 + PubMed top-3
2. Merge & deduplicate by `doc_id`
3. Recalculate `reliability_score` for all using formula:
   - `R_i = TierWeight × exp(-λ × age) × PeerReviewFactor`
   - λ (decay rate) per disease: Vaccination=0.10, Cardiovascular=0.05, Diabetes=0.03
4. Sort descending by `reliability_score`
5. Returns ranked evidence list to the Consensus Engine (Stage 8)

---

## Source Tier Weights (SOURCE_TIERS)

| Tier | Weight |
|------|--------|
| WHO Guideline | 1.00 |
| CDC Guideline | 0.98 |
| WHO/CDC Guideline | 0.99 |
| Systematic Review / Meta-Analysis | 0.95 |
| Randomized Controlled Trial (RCT) | 0.90 |
| PubMed Article | 0.80 |
| Cohort Study / Observational | 0.75 |
| Case Report | 0.55 |
| Preprint (bioRxiv/medRxiv) | 0.30 |
| Unverified Blog / Opinion | 0.05 |

---

## Orchestrator Integration (`orchestrator.py`)

- `MedVerifyOrchestrator` uses **lazy initialization** — all models load on first API call
- `VECTOR_STORE_DIR` = `<project_root>/../vector_store` (one level above the `medverify-ai-backend/` folder)
- If index is missing, retrieval is disabled with a warning (graceful degradation)
- The **same** `all-MiniLM-L6-v2` model used at build time is reused at runtime — this is critical for vector consistency; switching models breaks all similarity scores

---

## Key Design Decisions

- **IndexFlatIP** (not HNSW/IVF): Exact search, appropriate for ~4,600 vectors; no approximation needed at this scale
- **Normalized embeddings**: Enables cosine similarity via inner product — no separate normalization step needed at query time
- **Hybrid retrieval**: Static FAISS catches known fact-checked claims; live PubMed catches recent evidence not in the static corpus
- **No dedicated vector DB server** (e.g. Pinecone/Weaviate): Everything is file-based, suitable for capstone/academic deployment
- **Disease-aware decay**: Different diseases have different knowledge half-lives; Vaccination knowledge decays faster (λ=0.10) than Diabetes guidelines (λ=0.03)

---

## Rebuilding the Index

If the dataset changes, rebuild with:
```bash
cd <project_root>
python scripts/build_knowledge_base.py
```
This overwrites both `faiss_index.bin` and `vector_metadata.json`. The backend will pick up the new index on next restart (lazy init re-runs).
