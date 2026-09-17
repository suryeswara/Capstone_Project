---
name: medverify-ai-pipeline
description: Deep-dive into MedVerify AI's AI inference pipeline — disease classifier, NLI consensus engine, explanation generator, BioScope certainty guard, population applicability scoring, and the orchestrator that wires them all.
---

# MedVerify AI — AI Inference Pipeline

## Stage 6: Disease Classifier

**Model**: Fine-tuned DistilBERT (saved at `models/biobert_disease_classifier/`)  
**File**: `medverify-ai-backend/app/services/orchestrator.py`

### Labels
```python
LABEL_MAP = {0: "Diabetes", 1: "Cardiovascular Disease", 2: "Vaccination"}
```

### How it works
1. Tokenize claim → `AutoTokenizer.from_pretrained(CLASSIFIER_MODEL_DIR)`
2. Run inference → `AutoModelForSequenceClassification`
3. Softmax logits → pick argmax label
4. If model missing → keyword fallback (searches for disease keywords in text)

### Training
```bash
python scripts/train_disease_classifier.py
```
Outputs to `models/biobert_disease_classifier/`

---

## Stage 7: Hybrid Retrieval

See the `medverify-vector-db` skill for full details on FAISS.

**Key API**: `HybridRetrievalEngine.retrieve_and_rank(claim_text, disease_category)`

Returns a list of `EvidenceItem` dicts, each with:
- `source`: `"FAISS_STATIC"` or `"PUBMED_LIVE"`
- `cosine_similarity`: float (FAISS only; 0.0 for PubMed)
- `reliability_score`: R_i ∈ [0, 1]
- `stance`: pre-label (refined by Stage 8)
- `source_tier`, `pub_year`, `disease_category`, `doc_id`, `url`

---

## Stage 8: Consensus Engine

**File**: `medverify-ai-backend/app/services/consensus_engine.py`  
**NLI Model**: `cross-encoder/nli-deberta-v3-small`

### Step 1: NLI Stance Detection (`NLIStanceDetector`)
For each evidence item, runs:
```
premise = evidence chunk_text
hypothesis = claim_text
→ HuggingFace zero-shot-classification pipeline
→ label: entailment→"supporting" | contradiction→"contradicting" | neutral→"neutral"
```

### Step 2: Weighted Consensus Score
```
ConsensusScore = Σ(R_i × StanceValue_i) / Σ(R_i)
StanceValue: supporting=+1, contradicting=-1, neutral=0
```

### Step 3: Credibility Score → Verdict Mapping
```
CredibilityScore = normalize(ConsensusScore) * 100

Score ≥ 70  →  "Supported"
Score ≤ 30  →  "Contradicted"
30 < Score < 70 → "Mixed" or "Insufficient Evidence"
```

### BioScope Certainty Inflation Guard (`FaithfulnessVerifier`)
Uses hedge cue dictionaries to detect certainty level:
- **weak**: may, might, could, suggests, associated with, …
- **moderate**: indicates, shows, reduces risk of, …
- **strong**: causes, proves, cures, prevents, definitively, …

Prevents the explanation from making stronger claims than the evidence supports.

---

## Stage 10: Explanation Generator

**File**: `medverify-ai-backend/app/services/explanation_generator.py`

### Process
1. **Draft sentences**: Template-based grounded sentences referencing `[ev-1]`, `[ev-2]`, etc.
2. **Dual-pass faithfulness verification** per sentence:
   - **Pass 1 (NLI)**: Entailment check against cited evidence; threshold ≥ 0.70 to mark "verified"
   - **Pass 2 (BioScope)**: Explanation certainty level must NOT exceed evidence certainty level
3. **Assign sentence status**: `"verified"` | `"unsupported"` | `"contradiction"`
4. **Compute faithfulnessConfidence**: % of verified sentences × 100

### Output (ExplanationSentenceDTO)
```json
{
  "sentence": "Studies show statins reduce LDL cholesterol [ev-1].",
  "status": "verified",
  "cited_evidence_ids": ["ev-1"],
  "certainty_level": "moderate",
  "faithfulness_score": 0.82
}
```

---

## Population Applicability Engine

**Directory**: `medverify-ai-backend/app/services/population/`

### Purpose
Adjusts evidence reliability based on whether the study population matches the claim's implied population.

### Scoring Modes (env var `POPULATION_SCORING_MODE`)
| Mode | What it weighs |
|------|---------------|
| `v1_age_only` | Age overlap ratio only |
| `v2_age_sex` | Age + sex compatibility |
| `v3_full` | Age + sex + region |

### Score P_i ∈ [0, 1]
- MATCHED → high P_i (not 1.0; maintains uncertainty)
- MISMATCHED → low P_i (not 0.0; doesn't fully discard evidence)
- UNKNOWN → neutral P_i

### Activation
Controlled via env var: `CONSENSUS_MODE=population` (default: `baseline`)

### Key files
| File | Role |
|------|------|
| `claim_population.py` | Extract age/sex/region from claim text |
| `study_population.py` | Extract population from evidence title/abstract |
| `matcher.py` | Compare claim vs study population → MatchResult |
| `applicability.py` | Convert MatchResult → P_i score |
| `normalizer.py` | Standardize age ranges, sex terms |
| `schemas.py` | `PopulationProfile`, `MatchResult`, `ApplicabilityResult` DTOs |

---

## Orchestrator (`orchestrator.py`)

**Class**: `MedVerifyOrchestrator` (singleton via `get_orchestrator()`)

### Lazy Initialization (loads on first API call)
1. Disease classifier tokenizer + model
2. `HybridRetrievalEngine` (FAISS + PubMed)
3. `VerificationPipeline` (NLI DeBERTa)
4. `FaithfulnessVerifier`
5. `ExplanationGenerator`

### Safety Refusal Patterns
Personal medical advice queries are rejected before pipeline runs:
```python
SAFETY_REFUSAL_PATTERNS = [
    ("i have", "should i take"),
    ("i have", "chest pain"),
    ("my doctor", "should i"),
    ...
]
```

### Graceful Degradation
- No classifier model → keyword-based disease detection fallback
- No FAISS index → retrieval disabled, PubMed only
- PubMed timeout → FAISS only

---

## SSE Streaming Progress

The verification runs as a FastAPI `BackgroundTask`. Progress is persisted to DB and polled via Server-Sent Events (SSE):

```
GET /api/verifications/{id}/stream
→ text/event-stream
→ sends status updates every ~1s
→ final event includes full VerificationReportDTO
```
