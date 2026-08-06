"""
MedVerify AI - Verifications API Router

Stage 9: Real AI Pipeline Integration
Replaces Stage 3 mock pipeline with the unified MedVerify Orchestrator
that wires: Disease Classifier -> Hybrid Retrieval -> Consensus NLI -> Verdict
"""

import uuid
import json
import asyncio
import logging
from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.db.models import UserModel, ClaimModel, VerificationModel, EvidenceCitationModel, VerificationStatusEnum, VerdictEnum
from app.schemas.dto import (
    SubmitClaimRequestDTO,
    SubmitClaimResponseDTO,
    VerificationStatusResponseDTO,
    VerificationReportDTO,
    CredibilityBreakdownDTO,
    ConsensusDTO,
    ExplanationSentenceDTO,
    EvidenceItemDTO,
    VersionMetadataDTO
)
from app.services.orchestrator import get_orchestrator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Verifications"])


# ---------------------------------------------------------------------------
# REAL AI VERIFICATION PIPELINE (replaces Stage 3 mock)
# ---------------------------------------------------------------------------

async def run_real_verification_pipeline(verification_id: str, raw_text: str, disease_category_override: str = None):
    """
    Background task that executes the real AI verification pipeline
    with state-machine progress updates persisted to the database.
    """
    orchestrator = get_orchestrator()

    # Define pipeline stages with progress percentages
    stages = [
        (VerificationStatusEnum.EXTRACTING, 10, "Extracting verifiable atomic claim assertion..."),
        (VerificationStatusEnum.CLASSIFYING, 25, "Classifying medical domain using fine-tuned DistilBERT..."),
        (VerificationStatusEnum.RETRIEVING, 45, "Retrieving evidence from FAISS vector store & live PubMed API..."),
        (VerificationStatusEnum.RANKING, 65, "Ranking evidence quality by source tier and recency..."),
        (VerificationStatusEnum.CONSENSUS, 80, "Analyzing reliability-weighted medical consensus via NLI..."),
        (VerificationStatusEnum.GENERATING, 90, "Generating grounded explanation with sentence citations..."),
        (VerificationStatusEnum.VERIFYING, 95, "Evaluating faithfulness (NLI entailment + BioScope certainty)..."),
    ]

    # Safety refusal check (fast path)
    if orchestrator.check_safety_refusal(raw_text):
        db = next(get_db())
        try:
            ver = db.query(VerificationModel).filter(VerificationModel.id == verification_id).first()
            if ver:
                ver.status = VerificationStatusEnum.REFUSED_SAFETY
                ver.progress_percentage = 100
                ver.current_step_label = "Refused: Input seeks personal diagnostic/treatment advice."
                db.commit()
        finally:
            db.close()
        return

    # Update progress through pipeline stages (real processing happens during CONSENSUS)
    for i, (status_enum, progress, label) in enumerate(stages):
        db = next(get_db())
        try:
            ver = db.query(VerificationModel).filter(VerificationModel.id == verification_id).first()
            if ver:
                ver.status = status_enum
                ver.progress_percentage = progress
                ver.current_step_label = label
                db.commit()
        finally:
            db.close()

        # Real AI inference happens during CONSENSUS stage
        if status_enum == VerificationStatusEnum.CONSENSUS:
            # Run the real AI pipeline (this is the heavy computation)
            result = orchestrator.verify(raw_text, disease_category_override)
        else:
            # Brief pause for other stages to show progress in UI
            await asyncio.sleep(0.2)

    # Final stage: persist results
    db = next(get_db())
    try:
        ver = db.query(VerificationModel).filter(VerificationModel.id == verification_id).first()
        claim = db.query(ClaimModel).filter(ClaimModel.id == ver.claim_id).first() if ver else None

        if ver and result.get("status") == "COMPLETED":
            # Update claim with classified disease category
            if claim:
                claim.disease_category = result.get("disease_category", "Diabetes")

            # Map verdict string to VerdictEnum
            verdict_str = result.get("verdict", "Insufficient Evidence")
            verdict_enum_map = {
                "Supported": VerdictEnum.SUPPORTED,
                "Contradicted": VerdictEnum.CONTRADICTED,
                "Insufficient Evidence": VerdictEnum.INSUFFICIENT,
                "Mixed": VerdictEnum.MIXED,
            }
            ver.verdict = verdict_enum_map.get(verdict_str, VerdictEnum.INSUFFICIENT)
            ver.credibility_score = result.get("credibility_score", 50.0)
            ver.credibility_breakdown = result.get("credibility_breakdown")
            ver.consensus_summary = result.get("consensus_summary")
            ver.version_metadata = result.get("version_metadata")

            # Store Stage 10 dual-pass verified explanation sentences
            explanation_sentences = result.get("explanation_sentences", [])
            evidence_citations = result.get("evidence_citations", [])

            if not explanation_sentences and evidence_citations:
                for i, ev in enumerate(evidence_citations[:3]):
                    explanation_sentences.append({
                        "sentenceId": f"s-{i+1}",
                        "text": f"Evidence from {ev.get('source_type', 'PubMed')}: {ev.get('title', '')}",
                        "status": "verified" if ev.get("stance") == "supporting" else (
                            "contradiction" if ev.get("stance") == "contradicting" else "unsupported"
                        ),
                        "nliConfidence": ev.get("nli_confidence", 0.0),
                        "certaintyLevel": 2,
                        "citedEvidenceIds": [ev.get("id", "")],
                    })
            ver.explanation_json = explanation_sentences

            # Persist evidence citations to DB
            for ev in evidence_citations:
                citation = EvidenceCitationModel(
                    id=ev["id"],
                    verification_id=verification_id,
                    title=ev.get("title", ""),
                    source_type=ev.get("source_type", "PubMed Article"),
                    authors=None,
                    pub_year=ev.get("pub_year"),
                    doi=ev.get("doi", ""),
                    reliability_score=ev.get("reliability_score", 0.5),
                    stance=ev.get("stance", "neutral"),
                    abstract_chunk=ev.get("abstract_chunk", ""),
                    url=ev.get("url", ""),
                )
                db.add(citation)

            ver.status = VerificationStatusEnum.COMPLETED
            ver.progress_percentage = 100
            ver.current_step_label = f"Verification completed. Verdict: {verdict_str} ({ver.credibility_score:.1f}%)"
            ver.completed_at = datetime.utcnow()

            db.commit()
            logger.info(
                f"[Pipeline] Verification {verification_id} completed: "
                f"verdict={verdict_str}, credibility={ver.credibility_score:.1f}%, "
                f"evidence={len(evidence_citations)}, elapsed={result.get('elapsed_seconds', 0):.1f}s"
            )
        else:
            ver.status = VerificationStatusEnum.FAILED
            ver.progress_percentage = 100
            ver.current_step_label = "Verification failed due to pipeline error."
            db.commit()
    except Exception as e:
        logger.error(f"[Pipeline] Error persisting verification results: {e}")
        try:
            ver = db.query(VerificationModel).filter(VerificationModel.id == verification_id).first()
            if ver:
                ver.status = VerificationStatusEnum.FAILED
                ver.progress_percentage = 100
                ver.current_step_label = f"Pipeline error: {str(e)[:100]}"
                db.commit()
        except Exception:
            pass
    finally:
        db.close()


# ---------------------------------------------------------------------------
# API ENDPOINTS
# ---------------------------------------------------------------------------

from app.api.auth import get_current_user_optional

@router.post("/claims", response_model=SubmitClaimResponseDTO, status_code=status.HTTP_202_ACCEPTED)
def submit_claim(
    payload: SubmitClaimRequestDTO,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: Optional[UserModel] = Depends(get_current_user_optional),
):
    """
    Submit a medical claim for verification.
    Returns immediately with a verification ID for polling.
    The real AI pipeline runs asynchronously in the background.
    """
    claim_id = f"clm-{uuid.uuid4().hex[:8]}"
    verification_id = f"ver-{uuid.uuid4().hex[:8]}"

    # Quick disease category detection for initial DB record
    orchestrator = get_orchestrator()
    initial_disease = payload.diseaseCategory or orchestrator.classify_disease(payload.rawText)

    # Persist claim linked to current_user if authenticated
    claim = ClaimModel(
        id=claim_id,
        user_id=current_user.id if current_user else None,
        raw_text=payload.rawText,
        extracted_claim=payload.rawText,
        disease_category=initial_disease,
        language=payload.language
    )
    db.add(claim)

    # Persist verification in CREATED status
    ver = VerificationModel(
        id=verification_id,
        claim_id=claim_id,
        status=VerificationStatusEnum.CREATED,
        progress_percentage=0,
        current_step_label="Verification request accepted. Queuing AI pipeline..."
    )
    db.add(ver)
    db.commit()

    # Launch real AI pipeline in background
    background_tasks.add_task(
        run_real_verification_pipeline,
        verification_id,
        payload.rawText,
        payload.diseaseCategory,
    )

    return SubmitClaimResponseDTO(
        claimId=claim_id,
        verificationId=verification_id,
        status="CREATED",
        submittedAt=datetime.utcnow().isoformat() + "Z",
        pollUrl=f"/api/verifications/{verification_id}"
    )


@router.get("/verifications/{verificationId}", response_model=VerificationStatusResponseDTO)
def get_verification_status(verificationId: str, db: Session = Depends(get_db)):
    """Poll verification status by ID."""
    ver = db.query(VerificationModel).filter(VerificationModel.id == verificationId).first()
    if not ver:
        raise HTTPException(status_code=404, detail="Verification record not found")

    return VerificationStatusResponseDTO(
        verificationId=ver.id,
        claimId=ver.claim_id,
        rawText=ver.claim.raw_text if ver.claim else "",
        status=ver.status.value,
        progressPercentage=ver.progress_percentage,
        currentStepLabel=ver.current_step_label,
        updatedAt=ver.updated_at.isoformat() + "Z",
        isTerminal=ver.status in [
            VerificationStatusEnum.COMPLETED,
            VerificationStatusEnum.FAILED,
            VerificationStatusEnum.REFUSED_SAFETY,
        ]
    )


@router.get("/verifications/{verificationId}/stream")
async def stream_verification_status(verificationId: str):
    """
    Server-Sent Events (SSE) endpoint that streams real-time state machine progress updates
    to the frontend as the AI verification pipeline executes.
    """
    async def event_generator():
        last_status = None
        last_progress = -1
        while True:
            db = next(get_db())
            try:
                ver = db.query(VerificationModel).filter(VerificationModel.id == verificationId).first()
                if not ver:
                    yield f"event: error\ndata: {json.dumps({'error': 'Verification record not found'})}\n\n"
                    break

                status_val = ver.status.value
                progress_val = ver.progress_percentage
                is_terminal = ver.status in [
                    VerificationStatusEnum.COMPLETED,
                    VerificationStatusEnum.FAILED,
                    VerificationStatusEnum.REFUSED_SAFETY,
                ]

                # Emit status update if progress/status changed or terminal
                if status_val != last_status or progress_val != last_progress or is_terminal:
                    event_payload = {
                        "verificationId": ver.id,
                        "claimId": ver.claim_id,
                        "rawText": ver.claim.raw_text if ver.claim else "",
                        "status": status_val,
                        "progressPercentage": progress_val,
                        "currentStepLabel": ver.current_step_label,
                        "updatedAt": ver.updated_at.isoformat() + "Z",
                        "isTerminal": is_terminal
                    }
                    yield f"event: status_update\ndata: {json.dumps(event_payload)}\n\n"
                    last_status = status_val
                    last_progress = progress_val

                if is_terminal:
                    break
            finally:
                db.close()

            await asyncio.sleep(0.25)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/verifications/{verificationId}/report", response_model=VerificationReportDTO)
def get_verification_report(verificationId: str, db: Session = Depends(get_db)):
    """Get full verification report with evidence and credibility breakdown."""
    ver = db.query(VerificationModel).filter(VerificationModel.id == verificationId).first()
    if not ver:
        raise HTTPException(status_code=404, detail="Verification record not found")

    citations = db.query(EvidenceCitationModel).filter(
        EvidenceCitationModel.verification_id == verificationId
    ).all()

    evidence_dtos = [
        EvidenceItemDTO(
            id=c.id,
            title=c.title,
            sourceType=c.source_type,
            authors=c.authors,
            pubYear=c.pub_year,
            doi=c.doi,
            reliabilityScore=c.reliability_score,
            stance=c.stance,
            abstractChunk=c.abstract_chunk,
            url=c.url
        ) for c in citations
    ]

    explanation_dtos = [
        ExplanationSentenceDTO(**item) for item in (ver.explanation_json or [])
    ]

    version_meta = None
    if ver.version_metadata:
        version_meta = VersionMetadataDTO(**ver.version_metadata)

    return VerificationReportDTO(
        verificationId=ver.id,
        claimId=ver.claim_id,
        rawText=ver.claim.raw_text if ver.claim else "",
        extractedClaim=ver.claim.extracted_claim if ver.claim else "",
        diseaseCategory=ver.claim.disease_category if ver.claim else "Diabetes",
        status=ver.status.value,
        completedAt=ver.completed_at.isoformat() + "Z" if ver.completed_at else None,
        verdict=ver.verdict.value if ver.verdict else "Insufficient Evidence",
        credibility=CredibilityBreakdownDTO(**ver.credibility_breakdown) if ver.credibility_breakdown else None,
        consensus=ConsensusDTO(**ver.consensus_summary) if ver.consensus_summary else None,
        explanation=explanation_dtos,
        evidence=evidence_dtos,
        versionMetadata=version_meta,
    )


@router.get("/verifications", response_model=List[VerificationStatusResponseDTO])
def list_verifications(limit: int = 10, db: Session = Depends(get_db)):
    """List recent verifications ordered by creation date."""
    verifications = db.query(VerificationModel).order_by(
        VerificationModel.created_at.desc()
    ).limit(limit).all()

    return [
        VerificationStatusResponseDTO(
            verificationId=v.id,
            claimId=v.claim_id,
            rawText=v.claim.raw_text if v.claim else "",
            status=v.status.value,
            progressPercentage=v.progress_percentage,
            currentStepLabel=v.current_step_label,
            updatedAt=v.updated_at.isoformat() + "Z",
            isTerminal=v.status in [
                VerificationStatusEnum.COMPLETED,
                VerificationStatusEnum.FAILED,
                VerificationStatusEnum.REFUSED_SAFETY,
            ]
        ) for v in verifications
    ]
