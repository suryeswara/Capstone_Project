from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Dict, Any
from datetime import datetime

# ---------------------------------------------------------------------------
# AUTHENTICATION & USER PROFILE DTOS
# ---------------------------------------------------------------------------

class UserRegisterDTO(BaseModel):
    email: str = Field(..., example="researcher@medverify.ai")
    password: str = Field(..., min_length=6, example="SecurePass123!")
    fullName: str = Field(..., example="Dr. Sarah Lin")
    role: Optional[str] = Field("user", example="researcher")

class UserLoginDTO(BaseModel):
    email: str = Field(..., example="researcher@medverify.ai")
    password: str = Field(..., example="SecurePass123!")

class UserResponseDTO(BaseModel):
    id: str
    email: str
    fullName: str
    role: str
    createdAt: str

class TokenDTO(BaseModel):
    accessToken: str
    tokenType: str = "bearer"
    user: UserResponseDTO

class UserProfileDTO(BaseModel):
    user: UserResponseDTO
    totalVerificationsSubmitted: int
    totalClaimsSupported: int
    totalClaimsContradicted: int
    memberSince: str

# ---------------------------------------------------------------------------
# CLAIM & VERIFICATION DTOS
# ---------------------------------------------------------------------------

class SubmitClaimRequestDTO(BaseModel):
    rawText: str = Field(..., example="Statins reduce the risk of recurrent heart attacks.")
    diseaseCategory: Optional[str] = Field(None, example="Cardiovascular Disease")
    language: str = Field("en", example="en")

class SubmitClaimResponseDTO(BaseModel):
    claimId: str
    verificationId: str
    status: str
    submittedAt: str
    pollUrl: str

class VerificationStatusResponseDTO(BaseModel):
    verificationId: str
    claimId: str
    rawText: str
    status: str
    progressPercentage: int
    currentStepLabel: Optional[str] = None
    updatedAt: str
    isTerminal: bool

class VersionMetadataDTO(BaseModel):
    modelVersion: str
    diseaseClassifierVersion: str
    embeddingVersion: str
    knowledgeBaseVersion: str
    rankingFormulaVersion: str
    promptVersion: str
    faithfulnessModelVersion: str

class CredibilityBreakdownDTO(BaseModel):
    overall: float
    evidenceConfidence: float
    consensusConfidence: float
    sourceQuality: float
    faithfulnessConfidence: float

class ConsensusDTO(BaseModel):
    weightedConsensusScore: float
    totalReliabilityWeight: float
    rawCounts: Dict[str, int]

class ExplanationSentenceDTO(BaseModel):
    sentenceId: str
    text: str
    status: str  # 'verified', 'unsupported', 'contradiction'
    nliConfidence: float
    certaintyLevel: int
    citedEvidenceIds: List[str]

class EvidenceItemDTO(BaseModel):
    id: str
    title: str
    sourceType: str
    authors: Optional[str] = None
    pubYear: Optional[int] = None
    doi: Optional[str] = None
    reliabilityScore: float
    stance: str
    abstractChunk: Optional[str] = None
    url: Optional[str] = None

class VerificationReportDTO(BaseModel):
    verificationId: str
    claimId: str
    rawText: str
    extractedClaim: Optional[str] = None
    diseaseCategory: Optional[str] = None
    status: str
    completedAt: Optional[str] = None
    verdict: Optional[str] = None
    credibility: Optional[CredibilityBreakdownDTO] = None
    consensus: Optional[ConsensusDTO] = None
    explanation: List[ExplanationSentenceDTO] = []
    evidence: List[EvidenceItemDTO] = []
    versionMetadata: Optional[VersionMetadataDTO] = None
