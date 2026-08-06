"""
MedVerify AI - Authentication & User Profile API Router

Endpoints:
- POST /api/auth/register  : Register new user in DB
- POST /api/auth/login     : Authenticate user & return JWT token
- GET  /api/auth/me        : Fetch user profile & database verification statistics
- GET  /api/auth/history   : Fetch current user's claim verification history from DB
"""

import uuid
import logging
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Header, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.db.models import UserModel, ClaimModel, VerificationModel, VerdictEnum
from app.schemas.dto import (
    UserRegisterDTO,
    UserLoginDTO,
    UserResponseDTO,
    TokenDTO,
    UserProfileDTO,
    VerificationStatusResponseDTO,
)
from app.core.security import (
    hash_password,
    verify_password,
    create_access_token,
    decode_access_token,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


# ---------------------------------------------------------------------------
# DEPENDENCY: GET CURRENT USER FROM JWT BEARER TOKEN
# ---------------------------------------------------------------------------

def get_current_user(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
) -> UserModel:
    """Extract and verify JWT token from Authorization header (Bearer <token>)."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = authorization.split(" ")[1]
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token invalid or expired",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = payload.get("sub")
    user = db.query(UserModel).filter(UserModel.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    return user


def get_current_user_optional(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
) -> Optional[UserModel]:
    """Optional version of get_current_user for endpoints allowing anonymous access."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    try:
        token = authorization.split(" ")[1]
        payload = decode_access_token(token)
        if not payload:
            return None
        user_id = payload.get("sub")
        return db.query(UserModel).filter(UserModel.id == user_id).first()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------------------------

@router.post("/register", response_model=TokenDTO, status_code=status.HTTP_201_CREATED)
def register(payload: UserRegisterDTO, db: Session = Depends(get_db)):
    """Register a new user account and save to DB."""
    email_clean = payload.email.strip().lower()

    # Check if email already exists
    existing = db.query(UserModel).filter(UserModel.email == email_clean).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An account with this email already exists",
        )

    user_id = f"usr-{uuid.uuid4().hex[:8]}"
    hashed_pwd = hash_password(payload.password)

    user = UserModel(
        id=user_id,
        email=email_clean,
        hashed_password=hashed_pwd,
        full_name=payload.fullName.strip(),
        role=payload.role or "user",
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    logger.info(f"[Auth] User registered successfully: id={user.id}, email={user.email}")

    # Generate JWT Token
    access_token = create_access_token({"sub": user.id, "email": user.email, "role": user.role})

    user_dto = UserResponseDTO(
        id=user.id,
        email=user.email,
        fullName=user.full_name,
        role=user.role,
        createdAt=user.created_at.isoformat() + "Z",
    )

    return TokenDTO(accessToken=access_token, tokenType="bearer", user=user_dto)


@router.post("/login", response_model=TokenDTO)
def login(payload: UserLoginDTO, db: Session = Depends(get_db)):
    """Authenticate user credentials and return JWT token."""
    email_clean = payload.email.strip().lower()
    user = db.query(UserModel).filter(UserModel.email == email_clean).first()

    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )

    logger.info(f"[Auth] User logged in: id={user.id}, email={user.email}")

    access_token = create_access_token({"sub": user.id, "email": user.email, "role": user.role})

    user_dto = UserResponseDTO(
        id=user.id,
        email=user.email,
        fullName=user.full_name,
        role=user.role,
        createdAt=user.created_at.isoformat() + "Z",
    )

    return TokenDTO(accessToken=access_token, tokenType="bearer", user=user_dto)


@router.get("/me", response_model=UserProfileDTO)
def get_user_profile(
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Fetch current user profile and account statistics from DB."""
    # Count claims linked to this user
    user_claims = db.query(ClaimModel).filter(ClaimModel.user_id == current_user.id).all()
    user_claim_ids = [c.id for c in user_claims]

    total_submitted = len(user_claims)

    total_supported = 0
    total_contradicted = 0

    if user_claim_ids:
        verifications = db.query(VerificationModel).filter(
            VerificationModel.claim_id.in_(user_claim_ids)
        ).all()
        for v in verifications:
            if v.verdict == VerdictEnum.SUPPORTED:
                total_supported += 1
            elif v.verdict == VerdictEnum.CONTRADICTED:
                total_contradicted += 1

    user_dto = UserResponseDTO(
        id=current_user.id,
        email=current_user.email,
        fullName=current_user.full_name,
        role=current_user.role,
        createdAt=current_user.created_at.isoformat() + "Z",
    )

    return UserProfileDTO(
        user=user_dto,
        totalVerificationsSubmitted=total_submitted,
        totalClaimsSupported=total_supported,
        totalClaimsContradicted=total_contradicted,
        memberSince=current_user.created_at.strftime("%B %Y"),
    )


@router.get("/history", response_model=List[VerificationStatusResponseDTO])
def get_user_history(
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Fetch claim verifications submitted by the current logged-in user."""
    user_claims = db.query(ClaimModel).filter(ClaimModel.user_id == current_user.id).all()
    claim_ids = [c.id for c in user_claims]

    if not claim_ids:
        return []

    verifications = db.query(VerificationModel).filter(
        VerificationModel.claim_id.in_(claim_ids)
    ).order_by(VerificationModel.created_at.desc()).all()

    return [
        VerificationStatusResponseDTO(
            verificationId=v.id,
            claimId=v.claim_id,
            rawText=v.claim.raw_text if v.claim else "",
            status=v.status.value,
            progressPercentage=v.progress_percentage,
            currentStepLabel=v.current_step_label,
            updatedAt=v.updated_at.isoformat() + "Z",
            isTerminal=v.status.value in ["COMPLETED", "FAILED", "REFUSED_SAFETY"],
        )
        for v in verifications
    ]
