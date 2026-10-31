"""
MedVerify AI - Verifications API Router

Runs the inference pipeline (gate -> retrieval -> rerank -> pair classification
-> meta-model -> abstain check -> explanation) in a worker thread and persists
progress and results for polling / SSE.

Verification modes: static (local FAISS), live (PubMed), hybrid (both).
"""

import uuid
import json
import asyncio
import logging
from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status, UploadFile, File, Form
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
    ExplanationSentenceDTO,
    EvidenceItemDTO,
)
from app.pipeline import Mode, PipelineError
from app.pipeline.orchestrator import get_pipeline
from app.services.image_claim_extractor import extract_claim_from_image

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Verifications"])

TERMINAL_STATES = [
    VerificationStatusEnum.COMPLETED,
    VerificationStatusEnum.FAILED,
    VerificationStatusEnum.REFUSED_SAFETY,
    VerificationStatusEnum.REJECTED_NON_MEDICAL,
]

# pipeline stage key -> (status, progress %). The status enum values pre-date the
# rewrite; their meaning now is: EXTRACTING = validate + gate, RANKING = rerank,
# CLASSIFYING = pair classifier, CONSENSUS = meta-model + abstain check.
STAGE_STATUS = {
    "validate": (VerificationStatusEnum.EXTRACTING, 5),
    "gate": (VerificationStatusEnum.EXTRACTING, 15),
    "retrieve": (VerificationStatusEnum.RETRIEVING, 30),
    "rerank": (VerificationStatusEnum.RANKING, 50),
    "classify": (VerificationStatusEnum.CLASSIFYING, 65),
    "meta": (VerificationStatusEnum.CONSENSUS, 80),
    "explain": (VerificationStatusEnum.GENERATING, 90),
}

SOURCE_DISPLAY = {
    "PUBHEALTH": "PUBHEALTH fact-check",
    "HEALTHVER": "HealthVer abstract",
    "PUBMED": "PubMed abstract",
}

VERDICT_ENUM = {
    "Supported": VerdictEnum.SUPPORTED,
    "Contradicted": VerdictEnum.CONTRADICTED,
    "Insufficient Evidence": VerdictEnum.INSUFFICIENT,
}


def _update(verification_id: str, **fields) -> None:
    db = next(get_db())
    try:
        ver = db.query(VerificationModel).filter(VerificationModel.id == verification_id).first()
        if ver:
            for k, v in fields.items():
                setattr(ver, k, v)
            db.commit()
    finally:
        db.close()


def run_verification_pipeline(verification_id: str, raw_text: str, mode_value: str) -> None:
    """
    Background task (plain ``def`` so Starlette runs it in a worker thread and the
    event loop -- and therefore SSE streaming -- stays responsive during inference).
    """
    pipeline = get_pipeline()

    def progress(stage: str, label: str) -> None:
        st, pct = STAGE_STATUS[stage]
        _update(verification_id, status=st, progress_percentage=pct, current_step_label=label)

    try:
        result = pipeline.verify(raw_text, Mode.parse(mode_value), progress)
    except (PipelineError, ValueError) as exc:
        logger.error("[pipeline] %s failed: %s", verification_id, exc)
        _update(verification_id, status=VerificationStatusEnum.FAILED, progress_percentage=100,
                current_step_label=f"Pipeline error: {str(exc)[:200]}")
        return
    except Exception as exc:  # never leave a verification stuck in a non-terminal state
        logger.exception("[pipeline] unexpected failure for %s", verification_id)
        _update(verification_id, status=VerificationStatusEnum.FAILED, progress_percentage=100,
                current_step_label=f"Unexpected error: {type(exc).__name__}")
        return

    _persist_result(verification_id, result)


def _persist_result(verification_id: str, result: dict) -> None:
    db = next(get_db())
    try:
        ver = db.query(VerificationModel).filter(VerificationModel.id == verification_id).first()
        if not ver:
            return
        status_name = result["status"]
        ver.mode = result.get("mode")
        ver.mode_effective = result.get("mode_effective")
        ver.gate_result = result.get("gate")

        if status_name == "REFUSED_SAFETY":
            ver.status = VerificationStatusEnum.REFUSED_SAFETY
            ver.current_step_label = f"Refused: {result.get('reason', '')[:200]}"
            ver.pipeline_trace = {"disclaimer": result.get("disclaimer"), "safety": result.get("safety_action")}
        elif status_name == "REJECTED_NON_MEDICAL":
            ver.status = VerificationStatusEnum.REJECTED_NON_MEDICAL
            ver.current_step_label = f"Not a medical claim: {result.get('reason', '')[:200]}"
        else:
            expl = result["explanation"]
            ver.verdict = VERDICT_ENUM[result["verdict"]]
            ver.credibility_score = round(100 * result["confidence"], 1)
            ver.system_confidence = ver.credibility_score
            ver.faithfulness_score = expl["faithfulness_confidence"]
            ver.credibility_breakdown = result["credibility_breakdown"]
            ver.version_metadata = result["versions"]
            ver.probabilities = result["probabilities"]
            ver.needs_review = result["needs_review"]
            ver.explanation_json = [
                {
                    "sentenceId": x["id"], "text": x["text"], "status": x["status"], "role": x["role"],
                    "nliConfidence": x["confidence"], "certaintyLevel": x["certainty"],
                    "citedEvidenceIds": x["evidence_ids"],
                }
                for x in expl["sentences"]
            ]
            ver.pipeline_trace = {
                "notes": result["notes"],
                "trace": result["trace"],
                "suggestions": result["suggestions"],
                "reviewThreshold": result["review_threshold"],
                "disclaimer": result["disclaimer"],
                "explanationSummary": {
                    "writer": expl["source"], "why": expl["why"], "caveats": expl["caveats"],
                    "nextSteps": expl["next_steps"], "strength": expl["strength"],
                },
            }
            for e in result["evidence"]:
                db.add(EvidenceCitationModel(
                    id=e["id"], verification_id=verification_id, title=e["title"] or "(untitled)",
                    source_type=SOURCE_DISPLAY.get(e["source"], e["source"]), pub_year=e["year"],
                    pmid=e["pmid"], doi=e["doi"], similarity=e["dense_sim"] or 0.0,
                    reliability_score=e["rerank"], stance=e["stance"], abstract_chunk=e["text"],
                    url=e["url"], pool=e["pool"], stance_probs=e["probs"],
                ))
            ver.status = VerificationStatusEnum.COMPLETED
            flag = " \u2014 needs review" if result["needs_review"] else ""
            ver.current_step_label = (
                f"Verdict: {result['verdict']} ({ver.credibility_score:.1f}% confidence){flag}"
            )
        ver.progress_percentage = 100
        ver.completed_at = datetime.utcnow()
        db.commit()
        logger.info("[pipeline] %s -> %s in %ss", verification_id, status_name, result.get("elapsed_seconds"))
    except Exception:
        logger.exception("[pipeline] could not persist %s", verification_id)
        db.rollback()
        _update(verification_id, status=VerificationStatusEnum.FAILED, progress_percentage=100,
                current_step_label="Could not store the verification result.")
    finally:
        db.close()


def _require_pipeline(mode_value: str) -> Mode:
    """Validate the mode and refuse (503) rather than fabricate when models are not loaded."""
    try:
        mode = Mode.parse(mode_value)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    p = get_pipeline()
    if not p.loaded:
        detail = ("The inference pipeline failed to load: " + p.load_error) if p.load_error \
            else "The inference pipeline is still loading. Please retry in a moment."
        raise HTTPException(status_code=503, detail=detail)
    return mode


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
    """Queue a text claim. The pipeline runs in the background; poll or stream for progress."""
    mode = _require_pipeline(payload.mode)
    return _queue(payload.rawText, mode, payload.language, "Verification request accepted. Starting pipeline...",
                  background_tasks, db, current_user)


@router.post("/claims/image", response_model=SubmitClaimResponseDTO, status_code=status.HTTP_202_ACCEPTED)
async def submit_claim_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    mode: str = Form("static"),
    diseaseCategory: Optional[str] = Form(None),  # deprecated, ignored
    db: Session = Depends(get_db),
    current_user: Optional[UserModel] = Depends(get_current_user_optional),
):
    """Queue a claim extracted by OCR from an uploaded screenshot / infographic."""
    mode_enum = _require_pipeline(mode)
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    extraction = extract_claim_from_image(content, filename=file.filename or "upload.jpg")
    return _queue(extraction["extracted_claim"], mode_enum, "en",
                  f"Image processed ({file.filename}). Starting pipeline...", background_tasks, db, current_user)


def _queue(text: str, mode: Mode, language: str, label: str, background_tasks: BackgroundTasks,
           db: Session, current_user: Optional[UserModel]) -> SubmitClaimResponseDTO:
    claim_id = f"clm-{uuid.uuid4().hex[:8]}"
    verification_id = f"ver-{uuid.uuid4().hex[:8]}"
    db.add(ClaimModel(
        id=claim_id, user_id=current_user.id if current_user else None, raw_text=text,
        extracted_claim=text, disease_category=None, language=language,
    ))
    db.add(VerificationModel(
        id=verification_id, claim_id=claim_id, status=VerificationStatusEnum.CREATED,
        progress_percentage=0, current_step_label=label, mode=mode.value,
    ))
    db.commit()
    background_tasks.add_task(run_verification_pipeline, verification_id, text, mode.value)
    return SubmitClaimResponseDTO(
        claimId=claim_id, verificationId=verification_id, status="CREATED",
        submittedAt=datetime.utcnow().isoformat() + "Z", pollUrl=f"/api/verifications/{verification_id}",
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
        isTerminal=ver.status in TERMINAL_STATES
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
                is_terminal = ver.status in TERMINAL_STATES

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


@router.get("/verifications/{verificationId}/status", response_model=VerificationStatusResponseDTO)
def get_verification_status_alias(verificationId: str, db: Session = Depends(get_db)):
    """Poll verification status by ID (alias matching Section 33)."""
    return get_verification_status(verificationId, db)


def _evidence_dto(c: EvidenceCitationModel) -> EvidenceItemDTO:
    return EvidenceItemDTO(
        id=c.id, title=c.title, sourceType=c.source_type, pool=c.pool, authors=c.authors, pubYear=c.pub_year,
        pmid=c.pmid, doi=c.doi, similarity=c.similarity or 0.0, relevanceScore=c.reliability_score or 0.0,
        reliabilityScore=c.reliability_score or 0.0, stance=c.stance, stanceProbs=c.stance_probs,
        abstractChunk=c.abstract_chunk, url=c.url,
    )


@router.get("/verifications/{verificationId}/evidence", response_model=List[EvidenceItemDTO])
def get_verification_evidence(verificationId: str, db: Session = Depends(get_db)):
    """Evidence the verdict was based on (empty for rejected / refused claims)."""
    ver = db.query(VerificationModel).filter(VerificationModel.id == verificationId).first()
    if not ver:
        raise HTTPException(status_code=404, detail="Verification record not found")
    citations = db.query(EvidenceCitationModel).filter(EvidenceCitationModel.verification_id == verificationId).all()
    return [_evidence_dto(c) for c in citations]


@router.get("/verifications/{verificationId}/explanation", response_model=List[ExplanationSentenceDTO])
def get_verification_explanation(verificationId: str, db: Session = Depends(get_db)):
    """Explanation sentences with their faithfulness-audit status."""
    ver = db.query(VerificationModel).filter(VerificationModel.id == verificationId).first()
    if not ver:
        raise HTTPException(status_code=404, detail="Verification record not found")
    return [ExplanationSentenceDTO(**item) for item in (ver.explanation_json or [])]


@router.get("/verifications/{verificationId}/report", response_model=VerificationReportDTO)
def get_verification_report(verificationId: str, db: Session = Depends(get_db)):
    """Full report: verdict, class probabilities, review flag, evidence, explanation, suggestions."""
    ver = db.query(VerificationModel).filter(VerificationModel.id == verificationId).first()
    if not ver:
        raise HTTPException(status_code=404, detail="Verification record not found")
    citations = db.query(EvidenceCitationModel).filter(EvidenceCitationModel.verification_id == verificationId).all()
    trace = ver.pipeline_trace or {}
    return VerificationReportDTO(
        verificationId=ver.id,
        claimId=ver.claim_id,
        rawText=ver.claim.raw_text if ver.claim else "",
        extractedClaim=ver.claim.extracted_claim if ver.claim else "",
        diseaseCategory=None,
        status=ver.status.value,
        completedAt=ver.completed_at.isoformat() + "Z" if ver.completed_at else None,
        verdict=ver.verdict.value if ver.verdict else None,
        systemConfidence=ver.credibility_score,
        credibility=CredibilityBreakdownDTO(**ver.credibility_breakdown) if ver.credibility_breakdown else None,
        mode=ver.mode,
        modeEffective=ver.mode_effective,
        probabilities=ver.probabilities,
        needsReview=ver.needs_review,
        reviewThreshold=trace.get("reviewThreshold"),
        gate=ver.gate_result,
        notes=trace.get("notes", []),
        explanationSummary=trace.get("explanationSummary"),
        suggestions=trace.get("suggestions", []),
        disclaimer=trace.get("disclaimer"),
        trace=trace.get("trace"),
        explanation=[ExplanationSentenceDTO(**item) for item in (ver.explanation_json or [])],
        evidence=[_evidence_dto(c) for c in citations],
        versionMetadata=ver.version_metadata,
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
            isTerminal=v.status in TERMINAL_STATES
        ) for v in verifications
    ]
