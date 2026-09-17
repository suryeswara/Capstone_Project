import enum
from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, ForeignKey, Text, JSON, Enum
from sqlalchemy.orm import relationship
from app.db.session import Base

class VerificationStatusEnum(str, enum.Enum):
    CREATED = "CREATED"
    EXTRACTING = "EXTRACTING"
    CLASSIFYING = "CLASSIFYING"
    RETRIEVING = "RETRIEVING"
    RANKING = "RANKING"
    CONSENSUS = "CONSENSUS"
    GENERATING = "GENERATING"
    VERIFYING = "VERIFYING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    REFUSED_SAFETY = "REFUSED_SAFETY"

class VerdictEnum(str, enum.Enum):
    SUPPORTED = "Supported"
    CONTRADICTED = "Contradicted"
    INSUFFICIENT = "Insufficient Evidence"
    MIXED = "Mixed"

class UserModel(Base):
    __tablename__ = "users"

    id = Column(String(50), primary_key=True, index=True)  # e.g. 'usr-xxx'
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False)
    role = Column(String(50), default="user")  # 'user', 'researcher', 'admin'
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    claims = relationship("ClaimModel", back_populates="user")

class DiseaseCategoryModel(Base):
    __tablename__ = "disease_categories"

    id = Column(String(50), primary_key=True, index=True)  # e.g. 'diabetes', 'cardiovascular', 'vaccination'
    name = Column(String(100), nullable=False)
    active_phase = Column(Integer, default=1)
    classifier_model_version = Column(String(100), nullable=False)

class ClaimModel(Base):
    __tablename__ = "claims"

    id = Column(String(50), primary_key=True, index=True)  # e.g. 'clm-xxx'
    user_id = Column(String(50), ForeignKey("users.id"), nullable=True)
    raw_text = Column(Text, nullable=False)
    extracted_claim = Column(Text, nullable=True)
    disease_category = Column(String(50), ForeignKey("disease_categories.id"), nullable=True)
    language = Column(String(10), default="en")
    submitted_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("UserModel", back_populates="claims")
    verifications = relationship("VerificationModel", back_populates="claim")

class VerificationModel(Base):
    __tablename__ = "verifications"

    id = Column(String(50), primary_key=True, index=True)  # e.g. 'ver-xxx'
    claim_id = Column(String(50), ForeignKey("claims.id"), nullable=False)
    status = Column(Enum(VerificationStatusEnum), default=VerificationStatusEnum.CREATED, nullable=False)
    progress_percentage = Column(Integer, default=0)
    current_step_label = Column(String(255), nullable=True)

    verdict = Column(Enum(VerdictEnum), nullable=True)
    credibility_score = Column(Float, nullable=True)
    faithfulness_score = Column(Float, nullable=True)
    system_confidence = Column(Float, nullable=True)
    credibility_breakdown = Column(JSON, nullable=True)
    consensus_summary = Column(JSON, nullable=True)
    population_analysis = Column(JSON, nullable=True)
    explanation_json = Column(JSON, nullable=True)
    version_metadata = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    claim = relationship("ClaimModel", back_populates="verifications")
    citations = relationship("EvidenceCitationModel", back_populates="verification")

class EvidenceCitationModel(Base):
    __tablename__ = "evidence_citations"

    id = Column(String(50), primary_key=True, index=True)  # e.g. 'ev-xxx'
    verification_id = Column(String(50), ForeignKey("verifications.id"), nullable=False)
    title = Column(Text, nullable=False)
    source_type = Column(String(100), nullable=False)
    authors = Column(String(255), nullable=True)
    pub_year = Column(Integer, nullable=True)
    pmid = Column(String(50), nullable=True)
    doi = Column(String(100), nullable=True)
    similarity = Column(Float, default=0.0)
    reliability_score = Column(Float, nullable=False)  # R_i
    applicability_score = Column(Float, default=1.0)  # P_i
    final_weight = Column(Float, default=0.5)         # W_i = R_i * P_i
    p_age = Column(Float, default=0.5)
    p_sex = Column(Float, default=0.5)
    p_condition = Column(Float, default=0.5)
    p_region = Column(Float, default=0.5)
    population_match_type = Column(String(50), default="UNKNOWN")
    stance = Column(String(20), nullable=False)  # 'supporting', 'contradicting', 'neutral'
    abstract_chunk = Column(Text, nullable=True)
    url = Column(String(500), nullable=True)

    verification = relationship("VerificationModel", back_populates="citations")

class ModelVersionModel(Base):
    __tablename__ = "model_versions"

    id = Column(String(50), primary_key=True, index=True)
    component = Column(String(100), nullable=False)   # 'disease_classifier', 'retrieval_encoder', etc.
    model_name = Column(String(150), nullable=False)  # 'BioBERT', 'MedCPT', 'DeBERTa-v3'
    version = Column(String(50), nullable=False)      # '1.0.0', 'R1.0'
    hash_signature = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

