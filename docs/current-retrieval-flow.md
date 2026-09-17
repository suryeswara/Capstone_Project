# MedVerify AI — Current Retrieval & Verification Pipeline Flow

## Full Call Chain (Claim Input → Verdict Output)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  FRONTEND: User submits claim text via React UI                             │
│  src/lib/api.ts → POST /api/claims                                         │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  API ENDPOINT: submit_claim()                                               │
│  File: app/api/verifications.py:192-247                                     │
│                                                                              │
│  1. Generate claim_id (clm-xxx) and verification_id (ver-xxx)               │
│  2. Quick disease classification via orchestrator.classify_disease()         │
│  3. Persist ClaimModel + VerificationModel (status=CREATED) to DB           │
│  4. Launch background task: run_real_verification_pipeline()                │
│  5. Return 202 Accepted with verificationId + pollUrl                       │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  BACKGROUND PIPELINE: run_real_verification_pipeline()                       │
│  File: app/api/verifications.py:42-183                                      │
│                                                                              │
│  State Machine Progression:                                                  │
│  CREATED → EXTRACTING(10%) → CLASSIFYING(25%) → RETRIEVING(45%)            │
│         → RANKING(65%) → CONSENSUS(80%) → GENERATING(90%) → VERIFYING(95%) │
│                                                                              │
│  Safety check (fast path):                                                   │
│    orchestrator.check_safety_refusal(raw_text) → REFUSED_SAFETY             │
│                                                                              │
│  Real AI work triggers at CONSENSUS stage:                                   │
│    result = orchestrator.verify(raw_text, disease_category_override)         │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  ORCHESTRATOR: MedVerifyOrchestrator.verify()                                │
│  File: app/services/orchestrator.py:159-328                                 │
│                                                                              │
│  Step 1: Safety Check                                                        │
│    check_safety_refusal() [orchestrator.py:116-125]                          │
│    Pattern matching for personal diagnostic queries                           │
│    → Returns REFUSED_SAFETY if matched                                       │
│                                                                              │
│  Step 2: Disease Classification (Stage 6)                                    │
│    classify_disease() [orchestrator.py:131-153]                              │
│    Model: dmis-lab/biobert-base-cased-v1.2 (fine-tuned on PubMed+PMC) or keyword fallback           │
│    Output: "Diabetes" | "Cardiovascular Disease" | "Vaccination"             │
│                                                                              │
│  Step 3: Hybrid Retrieval & Evidence Ranking (Stage 7)                       │
│    retrieval_engine.retrieve_and_rank() [retrieval_engine.py:231-274]        │
│    → Details below                                                           │
│                                                                              │
│  Step 4: NLI Stance Detection & Consensus (Stage 8)                          │
│    verification_pipeline.verify_claim() [consensus_engine.py:331-386]        │
│    → Details below                                                           │
│                                                                              │
│  Step 5: Build evidence citation objects                                     │
│    [orchestrator.py:224-243]                                                 │
│                                                                              │
│  Step 6: Explanation Generation & Faithfulness (Stage 10)                    │
│    explanation_generator.generate_and_verify_explanation()                    │
│    [explanation_generator.py:35-128]                                         │
│    → Details below                                                           │
│                                                                              │
│  Step 7: Build credibility breakdown & return result dict                    │
│    [orchestrator.py:270-328]                                                 │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Sub-Pipeline: Hybrid Retrieval (Stage 7)

```
retrieve_and_rank(claim_text, disease_category)
    File: app/services/retrieval_engine.py:231-274

    ┌─────────────────────────────────────────┐
    │ 1. FAISS Static Retrieval               │
    │    FAISSRetriever.search()              │
    │    [retrieval_engine.py:89-116]         │
    │                                          │
    │    Model: all-MiniLM-L6-v2 (384-dim)    │
    │    Index: vector_store/faiss_index.bin   │
    │    Metadata: vector_metadata.json        │
    │    Top-K: 5 (with disease filter)        │
    │                                          │
    │    Returns per result:                    │
    │      source: "FAISS_STATIC"              │
    │      title, chunk_text                   │
    │      source_tier, pub_year               │
    │      cosine_similarity                   │
    │      reliability_score (pre-computed)     │
    └────────────────────┬────────────────────┘
                         │
    ┌────────────────────┴────────────────────┐
    │ 2. PubMed Live Retrieval                │
    │    PubMedRetriever.search()             │
    │    [retrieval_engine.py:130-214]        │
    │                                          │
    │    Step A: esearch.fcgi → get PMIDs      │
    │    Step B: esummary.fcgi → get metadata  │
    │    Max results: 3                        │
    │                                          │
    │    Source tier detection from pubtype:    │
    │      "meta-analysis" → Meta-Analysis     │
    │      "rct" → RCT                         │
    │      "review" → Cohort/Observational     │
    │      else → "PubMed Article"             │
    │                                          │
    │    Returns per result:                    │
    │      source: "PUBMED_LIVE"               │
    │      title, chunk_text (title only!)     │
    │      source_tier, pub_year, doi, url     │
    │      reliability_score (calculated)      │
    │      stance: "supporting" (default)      │
    └────────────────────┬────────────────────┘
                         │
    ┌────────────────────┴────────────────────┐
    │ 3. Merge & Deduplicate by doc_id        │
    │    [retrieval_engine.py:253-260]        │
    │                                          │
    │ 4. Recalculate R_i for all evidence     │
    │    calculate_reliability_score()         │
    │    [retrieval_engine.py:48-65]          │
    │    R_i = Tier × e^(−λ×age) × PeerReview│
    │                                          │
    │ 5. Sort by R_i descending               │
    │    [retrieval_engine.py:272]            │
    └─────────────────────────────────────────┘
```

### Reliability Score Constants

| Source Tier | Base Weight |
| :--- | :---: |
| WHO Guideline | 1.00 |
| CDC Guideline | 0.98 |
| Systematic Review / Meta-Analysis | 0.95 |
| Randomized Controlled Trial (RCT) | 0.90 |
| PubMed Article | 0.80 |
| Cohort Study / Observational | 0.75 |
| Case Report | 0.55 |
| Preprint | 0.30 |

### Disease-Specific Decay Rates (λ)

| Disease | λ |
| :--- | :---: |
| Vaccination | 0.10 |
| Cardiovascular Disease | 0.05 |
| Diabetes | 0.03 |

---

## Detailed Sub-Pipeline: Consensus Engine (Stage 8)

```
verification_pipeline.verify_claim(claim_text, ranked_evidence)
    File: app/services/consensus_engine.py:331-386

    ┌─────────────────────────────────────────┐
    │ 1. NLI Stance Detection (per evidence)  │
    │    NLIStanceDetector.detect_stance()    │
    │    [consensus_engine.py:133-172]        │
    │                                          │
    │    Model: cross-encoder/nli-deberta-v3  │
    │    Input: {text: evidence, text_pair:    │
    │            claim}                        │
    │    Output per evidence:                  │
    │      stance: supporting|contradicting|   │
    │              neutral                     │
    │      stance_value: +1 | -1 | 0          │
    │      entailment_prob, contradiction_prob │
    │      confidence (highest prob)           │
    └────────────────────┬────────────────────┘
                         │
    ┌────────────────────┴────────────────────┐
    │ 2. Weighted Consensus Calculation        │
    │    ConsensusEngine.calculate_weighted_   │
    │    consensus()                            │
    │    [consensus_engine.py:243-297]         │
    │                                          │
    │    Formula:                               │
    │    C = Σ(R_i × S_i) / Σ(R_i)            │
    │      where S_i ∈ {+1, -1, 0}            │
    │                                          │
    │    Credibility = (C + 1.0) × 50.0        │
    │      maps [-1,+1] → [0,100]             │
    │                                          │
    │    Verdict thresholds:                    │
    │      C >= 0.30  → SUPPORTED              │
    │      C >= 0.10  → LIKELY_SUPPORTED       │
    │      C >= -0.10 → INCONCLUSIVE           │
    │      C >= -0.30 → LIKELY_REFUTED         │
    │      C < -0.30  → REFUTED                │
    └─────────────────────────────────────────┘
```

---

## Detailed Sub-Pipeline: Explanation & Faithfulness (Stage 10)

```
explanation_generator.generate_and_verify_explanation()
    File: app/services/explanation_generator.py:35-128

    ┌─────────────────────────────────────────┐
    │ 1. Draft Grounded Sentences              │
    │    _draft_explanation_sentences()        │
    │    [explanation_generator.py:130-204]    │
    │                                          │
    │    Template-based per verdict:            │
    │      Supported → 2-3 sentences with      │
    │        [ev-1], [ev-2] citations          │
    │      Contradicted → 2-3 refuting         │
    │      Inconclusive → 2 mixed              │
    └────────────────────┬────────────────────┘
                         │
    ┌────────────────────┴────────────────────┐
    │ 2. Dual-Pass Faithfulness Verification   │
    │    Per sentence:                          │
    │                                          │
    │    Pass 1: NLI Entailment Check          │
    │      stance_detector.detect_stance()     │
    │      threshold: entailment_prob >= 0.70  │
    │                                          │
    │    Pass 2: BioScope Certainty Guard      │
    │      get_certainty_level() on evidence   │
    │      get_certainty_level() on sentence   │
    │      FAIL if sentence_level > evidence   │
    │      (catches L1→L3 inflation)           │
    │                                          │
    │    Status assignment:                     │
    │      Both pass → "verified"              │
    │      Contradiction > 0.40 → "contra..."  │
    │      else → "unsupported"                │
    └────────────────────┬────────────────────┘
                         │
    ┌────────────────────┴────────────────────┐
    │ 3. faithfulnessConfidence Score          │
    │    = (verified_count / total) × 100      │
    └─────────────────────────────────────────┘
```

---

## Data Flow: Evidence Object Schema

At each pipeline stage, the evidence dict accumulates fields:

```
After Retrieval (Stage 7):
  source, title, chunk_text, source_tier, pub_year,
  disease_category, doc_id, doi, url,
  cosine_similarity, reliability_score, stance (default)

After NLI (Stage 8):
  + stance (refined), stance_value, nli_confidence,
    entailment_prob, contradiction_prob, neutral_prob, rank

After Orchestrator (Step 5):
  → EvidenceCitation object:
    id, title, source_type, pub_year, doi,
    reliability_score, stance, abstract_chunk, url,
    source_channel, nli_confidence, entailment_prob,
    contradiction_prob, cosine_similarity
```

---

## Key Insertion Point for Population Applicability

The population analysis should be inserted **between Retrieval (Step 3) and NLI Consensus (Step 4)** in `orchestrator.py`:

```
Current orchestrator.verify():
  Step 1: Safety Check
  Step 2: Disease Classification
  Step 3: Hybrid Retrieval & Ranking          ← evidence retrieved here
  ──────────────────────────────────────────── ← INSERT POPULATION HERE
  Step 4: NLI Stance Detection & Consensus    ← evidence weighted here
  Step 5: Build citation objects
  Step 6: Explanation Generation
  Step 7: Credibility breakdown

Proposed:
  Step 3: Hybrid Retrieval & Ranking
  Step 3.5: Claim Population Extraction       ← NEW
  Step 3.6: Study Population Extraction       ← NEW (per evidence item)
  Step 3.7: Population Matching               ← NEW
  Step 3.8: Applicability Score (P_i)         ← NEW
  Step 4: NLI Stance Detection & Consensus    ← now uses R_i × P_i × S_i
```
