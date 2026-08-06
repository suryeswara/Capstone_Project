# MedVerify AI — Stage 1 API Contract & Architecture Specification

**Document Version:** 1.0.0 (Frozen)  
**Status:** Approved for Implementation  
**Target Systems:** Frontend (`medverify-ai-frontend`), API Gateway (`Spring Boot` / `FastAPI`), AI Microservice (`FastAPI`)  

---

## 1. Shared Verification State Machine

The verification process executes asynchronously as a formal 10-state state machine.

```
 [CREATED] ──► [EXTRACTING] ──► [CLASSIFYING] ──► [RETRIEVING] ──► [RANKING]
                                                                        │
 [COMPLETED] ◄── [VERIFYING] ◄── [GENERATING] ◄── [CONSENSUS] ◄─────────┘
      │
      ├── (On Error) ──► [FAILED]
      └── (On Diagnostic/Treatment Query) ──► [REFUSED_SAFETY]
```

### State Machine Enum (`VerificationState`)
* `CREATED`: Claim received by API gateway, initial DB record generated.
* `EXTRACTING`: LLM extracting atomic verifiable claim from raw input text.
* `CLASSIFYING`: BioBERT classifier mapping claim to disease category (*Diabetes*, *Cardiovascular Disease*, *Vaccination*).
* `RETRIEVING`: Hybrid RAG fetching candidate passages from FAISS vector store + live PubMed/PMC API.
* `RANKING`: Evidence Ranking Engine calculating reliability score $R_i$ for each retrieved passage.
* `CONSENSUS`: Medical Consensus Engine computing reliability-weighted stance score.
* `GENERATING`: LLM generating evidence-grounded explanation with sentence-level citations.
* `VERIFYING`: Faithfulness Verification Engine evaluating NLI entailment and BioScope certainty cues.
* `COMPLETED`: Pipeline finished successfully; verdict, confidence score, and verified report ready.
* `FAILED`: Processing failed due to timeout, unhandled exception, or API circuit breaker.
* `REFUSED_SAFETY`: Input flagged as diagnostic/treatment advice query; safe medical referral emitted.

---

## 2. Mandatory Version Metadata Object (`VersionMetadataDTO`)

To preserve scientific reproducibility and auditability, every verification record stored in PostgreSQL or returned to the client **must** include the following schema:

```json
{
  "modelVersion": "qwen3-8b-instruct-v1.2",
  "diseaseClassifierVersion": "biobert-v2.1-finetuned",
  "embeddingVersion": "bge-large-en-v1.5",
  "knowledgeBaseVersion": "kb-2026-q3-v1.0",
  "rankingFormulaVersion": "rwrav-v1.0-calibrated",
  "promptVersion": "grounded-explanation-v3",
  "faithfulnessModelVersion": "mednli-deberta-v1.1-bioscope"
}
```

---

## 3. OpenAPI 3.0 REST Endpoints Specification

### 3.1 Endpoint 1: Submit Claim (`POST /api/claims`)

**Request Payload (`SubmitClaimRequestDTO`):**
```json
{
  "rawText": "Statins reduce the risk of recurrent heart attacks in people with existing heart disease.",
  "diseaseCategory": "Cardiovascular Disease", // Optional override; auto-classified if null
  "language": "en"
}
```

**Response Payload (`SubmitClaimResponseDTO` - 202 Accepted):**
```json
{
  "claimId": "clm-8f92a11b",
  "verificationId": "ver-79a02c34",
  "status": "CREATED",
  "submittedAt": "2026-08-04T15:15:00Z",
  "pollUrl": "/api/verifications/ver-79a02c34"
}
```

---

### 3.2 Endpoint 2: Poll State Machine Status (`GET /api/verifications/{verificationId}`)

**Response Payload (`VerificationStatusResponseDTO` - 200 OK):**
```json
{
  "verificationId": "ver-79a02c34",
  "claimId": "clm-8f92a11b",
  "rawText": "Statins reduce the risk of recurrent heart attacks in people with existing heart disease.",
  "status": "RANKING",
  "progressPercentage": 50,
  "currentStepLabel": "Ranking evidence quality by source tier and recency...",
  "updatedAt": "2026-08-04T15:15:02Z",
  "isTerminal": false
}
```

---

### 3.3 Endpoint 3: Fetch Full Verified Report (`GET /api/verifications/{verificationId}/report`)

**Response Payload (`VerificationReportDTO` - 200 OK):**
```json
{
  "verificationId": "ver-79a02c34",
  "claimId": "clm-8f92a11b",
  "rawText": "Statins reduce the risk of recurrent heart attacks in people with existing heart disease.",
  "extractedClaim": "Statin therapy reduces recurrent myocardial infarction risk in established cardiovascular disease patients.",
  "diseaseCategory": "Cardiovascular Disease",
  "status": "COMPLETED",
  "completedAt": "2026-08-04T15:15:04Z",
  "verdict": "Supported",
  "credibility": {
    "overall": 92.5,
    "evidenceConfidence": 95.0,
    "consensusConfidence": 94.2,
    "sourceQuality": 88.0,
    "faithfulnessConfidence": 100.0
  },
  "consensus": {
    "weightedConsensusScore": 0.885,
    "totalReliabilityWeight": 4.52,
    "rawCounts": {
      "supporting": 5,
      "contradicting": 0,
      "neutral": 1
    }
  },
  "explanation": [
    {
      "sentenceId": "s-1",
      "text": "Clinical trial meta-analyses confirm that statin therapy significantly decreases recurrent cardiovascular events.",
      "status": "verified",
      "nliConfidence": 0.96,
      "certaintyLevel": 2,
      "citedEvidenceIds": ["ev-1", "ev-2"]
    },
    {
      "sentenceId": "s-2",
      "text": "Statins work primarily by lowering LDL cholesterol levels.",
      "status": "verified",
      "nliConfidence": 0.92,
      "certaintyLevel": 2,
      "citedEvidenceIds": ["ev-3"]
    }
  ],
  "evidence": [
    {
      "id": "ev-1",
      "title": "Efficacy and safety of statin therapy: a systematic review and meta-analysis",
      "sourceType": "Systematic Review / Meta-Analysis",
      "authors": "Baigent C, et al.",
      "pubYear": 2024,
      "doi": "10.1016/S0140-6736(05)67394-1",
      "reliabilityScore": 0.945,
      "stance": "supporting",
      "abstractChunk": "Statin therapy produced a 21% relative reduction in major vascular events per 1.0 mmol/L reduction in LDL cholesterol...",
      "url": "https://pubmed.ncbi.nlm.nih.gov/16214597/"
    }
  ],
  "versionMetadata": {
    "modelVersion": "qwen3-8b-instruct-v1.2",
    "diseaseClassifierVersion": "biobert-v2.1-finetuned",
    "embeddingVersion": "bge-large-en-v1.5",
    "knowledgeBaseVersion": "kb-2026-q3-v1.0",
    "rankingFormulaVersion": "rwrav-v1.0-calibrated",
    "promptVersion": "grounded-explanation-v3",
    "faithfulnessModelVersion": "mednli-deberta-v1.1-bioscope"
  }
}
```

---

### 3.4 Endpoint 4: Medical Safety Refusal Payload (Diagnostic/Treatment Query)

If an input is flagged as seeking clinical diagnosis or treatment (e.g. *"I have chest pain, how much statin should I take?"*), the backend responds with `status: "REFUSED_SAFETY"`:

```json
{
  "verificationId": "ver-99b11e22",
  "status": "REFUSED_SAFETY",
  "safetyReason": "DIAGNOSTIC_OR_TREATMENT_ADVICE_SEEKING",
  "disclaimer": "MedVerify AI is an academic claim verification platform and cannot provide personal medical diagnosis, triage, or treatment guidance.",
  "refusalMessage": "Your query appears to seek personal medical treatment advice. If you are experiencing acute health symptoms such as chest pain or shortness of breath, please contact emergency medical services or a licensed physician immediately.",
  "emergencyResources": [
    {"name": "Emergency Healthcare Services", "contact": "911 / 112 / Regional Emergency"},
    {"name": "Find a Physician", "url": "https://www.who.int/health-topics"}
  ]
}
```

---

## 4. Inter-Service Communication Contracts (Gateway $\leftrightarrow$ AI Microservice)

The API Gateway communicates with the internal FastAPI AI Microservice over HTTP/gRPC using the following internal endpoint spec:

* `POST /internal/ai/v1/extract-claim` $\rightarrow$ Accepts raw text, returns extracted claim string.
* `POST /internal/ai/v1/classify-disease` $\rightarrow$ Accepts claim string, returns disease category + confidence score.
* `POST /internal/ai/v1/retrieve-and-rank` $\rightarrow$ Accepts claim + disease category, performs FAISS vector search + PubMed API query, calculates reliability scores $R_i$, returns ranked DTO.
* `POST /internal/ai/v1/verify-faithfulness` $\rightarrow$ Accepts explanation sentences + cited evidence passages, performs NLI entailment check + BioScope certainty check, returns sentence-level audit.

---

## 5. Stage 1 Sign-Off Checklist

- [x] REST API contracts specified (`/api/claims`, `/api/verifications/{id}`, `/api/verifications/{id}/report`).
- [x] Shared 10-state verification state machine enum defined.
- [x] Version metadata object specified (`VersionMetadataDTO`).
- [x] Medical safety refusal schema defined.
- [x] Inter-service AI DTOs specified.

**Stage 1 is complete and frozen.** Backend development (Stage 2 & Stage 3 Mock API) can now begin without ambiguity.
