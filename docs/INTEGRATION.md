# MedVerify inference pipeline: integration guide

The pipeline in `medverify-ai-backend/app/pipeline/` replaces the old orchestrator, consensus engine,
retrieval engine and population module. It **only loads** the final v4 artifacts. It never retrains,
rebuilds FAISS, or invents evidence or scores. If an artifact is missing, `/api/claims` answers
`503` with the reason and `/health/pipeline` shows it.

## Flow (matches the architecture blueprint)

1. **Validate + safety screen**: non-empty, <= 1000 chars, existing safety guardrail.
2. **Medical-claim gate**: nine signals -> logistic stacker -> `ACCEPT` / `REJECT` / `UNSURE`
   (Gemini tie-breaker when configured, otherwise `UNSURE` is rejected: fail-closed).
3. **Retrieval by mode**: static = FAISS `fc_index` + `abs_index` (top-10 each, 768-d MedCPT query vector);
   live = PubMed E-utilities (top-25 abstracts); hybrid = both, de-duplicated.
4. **MonoT5 rerank**: keep 3 fact-checks + 3 abstracts (live: 5).
5. **Pair classifier**: DeBERTa-v3-base + BiomedBERT, probabilities averaged -> P(SUPPORT, CONTRADICT, NEI).
6. **Meta-model by mode**: static `static.joblib` (18 features), live `live.joblib` (6), hybrid
   `hybrid_stack.joblib` (7 = static probs 3 + live probs 3 + best live rerank 1). Feature builders are in
   `app/pipeline/features.py` and each load checks `n_features_in_`.
7. **Abstain**: confidence < 0.5859 -> `needsReview = true`.
8. **Explanation**: Gemini if `GEMINI_API_KEY` is set, otherwise the template writer; every sentence is audited
   (NLI entailment >= 0.70 + certainty guard); suggested rewordings are re-verified in static mode.

If PubMed is unreachable, live and hybrid fall back to static and say so in `notes`.

## Artifact layout

```
<MEDVERIFY_ARTIFACT_ROOT>/            default: medverify-ai-backend/artifacts
  models/deberta/  models/biomed/     fine-tuned pair classifiers
  vector_db/   fc_index.faiss abs_index.faiss fc_ids.npy abs_ids.npy records.jsonl
  meta_models/ static.joblib live.joblib hybrid_stack.joblib
  gate/        probe.joblib stacker.joblib gate_config.json  [lexicon.json]
```

Each folder can be moved with `MEDVERIFY_DEBERTA_DIR`, `MEDVERIFY_BIOMED_DIR`, `MEDVERIFY_VECTOR_DB_DIR`,
`MEDVERIFY_META_DIR`, `MEDVERIFY_GATE_DIR`. MedCPT encoders, MonoT5 and the gate NLI model download by name on
first start (internet access to Hugging Face needed once).

## Run

```bash
cd medverify-ai-backend
pip install -r requirements.txt        # scikit-learn must match the version that wrote the .joblib files
python scripts/check_artifacts.py      # validates dims, feature counts, files; no models are loaded
uvicorn app.main:app --port 8000       # loads the pipeline in a background thread
curl localhost:8000/health/pipeline    # 200 when ready, 503 + reason otherwise
pytest tests                           # logic tests; do not need the artifacts
```

Environment: `GEMINI_API_KEY` (optional), `NCBI_API_KEY` + `NCBI_EMAIL` (recommended), `JWT_SECRET_KEY`
(required in production), `MEDVERIFY_GATE_STRICTNESS` = `strict|balanced|lenient`,
`MEDVERIFY_SKIP_PIPELINE_LOAD=1` to start the API without models.

## Things to confirm against your notebook (cannot be verified without the artifacts)

* **Feature order inside the 18-vector.** `features.py` follows the order described in the report. If the
  notebook concatenated blocks differently, change it there only. A wrong order does not crash: it silently
  degrades accuracy. Compare `static_features` output with the notebook for one claim.
* **Gate lexicon.** Ship the notebook's strong/weak term lists as `gate/lexicon.json`.
* **MedCPT query-encoder settings** (CLS pooling, max 64 tokens, L2-normalised) are implemented as in the report.
* **Frontend verdict labels:** `Supported`, `Contradicted`, `Insufficient Evidence`.
