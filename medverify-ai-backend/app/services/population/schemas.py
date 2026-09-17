"""
MedVerify AI — Population Applicability: Core Data Schemas

Defines the PopulationProfile, MatchResult, and ApplicabilityResult
data classes used throughout the population analysis pipeline.

Design decisions:
  - PopulationProfile uses Optional fields to distinguish "not specified"
    from "explicitly stated as all/any".
  - extraction_source tracks provenance for audit & error analysis.
  - MatchResult uses a 4-category enum rather than binary to support
    the continuous applicability scoring required by the research plan.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List
from enum import Enum


class ExtractionMethod(str, Enum):
    """How the population information was obtained."""
    REGEX = "regex"                       # Rule-based pattern matching
    NLP = "nlp"                           # NLP/transformer extraction
    CLINICAL_TRIALS_API = "clinical_trials_api"  # ClinicalTrials.gov structured data
    PUBMED_METADATA = "pubmed_metadata"   # PubMed article metadata
    UNKNOWN = "unknown"                   # Could not determine


class PopulationSpecificity(str, Enum):
    """Whether the population was explicitly stated, implied, or absent."""
    EXPLICIT = "explicit"     # "adults aged 65 and older"
    IMPLICIT = "implicit"     # "elderly patients" (requires normalization)
    UNKNOWN = "unknown"       # No population information found


class MatchType(str, Enum):
    """Population compatibility classification."""
    MATCHED = "MATCHED"         # Study population fully covers claim population
    PARTIAL = "PARTIAL"         # Partial overlap (e.g., claim 65+, study 60-75)
    MISMATCHED = "MISMATCHED"   # No overlap (e.g., claim 65+, study 18-40)
    UNKNOWN = "UNKNOWN"         # Insufficient data to determine


@dataclass
class PopulationProfile:
    """
    Represents the population characteristics of a claim or study.

    Fields use None to indicate "not specified" vs explicit values.
    For example:
      - age_min=65, age_max=None → "65 and older"
      - sex="all" → explicitly stated as including all sexes
      - sex=None → sex not mentioned
    """
    age_min: Optional[int] = None
    age_max: Optional[int] = None
    sex: Optional[str] = None           # "male", "female", "all", None
    region: Optional[str] = None        # Country or region string
    specificity: PopulationSpecificity = PopulationSpecificity.UNKNOWN
    confidence: float = 0.0             # 0.0 to 1.0
    extraction_method: ExtractionMethod = ExtractionMethod.UNKNOWN
    raw_text_snippet: Optional[str] = None  # The source text fragment

    def has_age_info(self) -> bool:
        """Returns True if any age information is available."""
        return self.age_min is not None or self.age_max is not None

    def has_sex_info(self) -> bool:
        """Returns True if sex information is available."""
        return self.sex is not None

    def has_region_info(self) -> bool:
        """Returns True if region information is available."""
        return self.region is not None

    def has_any_info(self) -> bool:
        """Returns True if any population information is available."""
        return self.has_age_info() or self.has_sex_info() or self.has_region_info()

    def to_dict(self) -> dict:
        """Convert to serializable dictionary."""
        return {
            "age_min": self.age_min,
            "age_max": self.age_max,
            "sex": self.sex,
            "region": self.region,
            "specificity": self.specificity.value,
            "confidence": round(self.confidence, 4),
            "extraction_method": self.extraction_method.value,
            "raw_text_snippet": self.raw_text_snippet,
        }


@dataclass
class MatchResult:
    """
    Result of comparing a claim population against a study population.
    """
    match_type: MatchType = MatchType.UNKNOWN
    age_overlap_ratio: float = 0.0      # 0.0 (no overlap) to 1.0 (full overlap)
    sex_compatible: Optional[bool] = None
    region_compatible: Optional[bool] = None
    condition_compatible: Optional[bool] = None
    
    # Explicit 4-dimension scores for Stage 8 frozen formula:
    # 1.0 = compatible, 0.5 = neutral / unspecified, 0.0 = incompatible
    p_age: float = 0.5
    p_sex: float = 0.5
    p_condition: float = 0.5
    p_region: float = 0.5
    details: str = ""                    # Human-readable explanation

    def to_dict(self) -> dict:
        return {
            "match_type": self.match_type.value,
            "age_overlap_ratio": round(self.age_overlap_ratio, 4),
            "sex_compatible": self.sex_compatible,
            "region_compatible": self.region_compatible,
            "condition_compatible": self.condition_compatible,
            "p_age": round(self.p_age, 4),
            "p_sex": round(self.p_sex, 4),
            "p_condition": round(self.p_condition, 4),
            "p_region": round(self.p_region, 4),
            "details": self.details,
        }


@dataclass
class ApplicabilityResult:
    """
    Final population applicability score P_i in [0, 1] for a single
    claim-evidence pair, combining age, sex, condition, and region compatibility.
    """
    score: float = 0.5                  # P_total in [0.0, 1.0]
    p_age: float = 0.5                  # A_i
    p_sex: float = 0.5                  # S_i
    p_condition: float = 0.5            # C_i
    p_region: float = 0.5               # G_i
    claim_population: Optional[PopulationProfile] = None
    study_population: Optional[PopulationProfile] = None
    match_result: Optional[MatchResult] = None
    scoring_mode: str = "v3_frozen_four_dimension"

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "p_total": round(self.score, 4),
            "p_age": round(self.p_age, 4),
            "p_sex": round(self.p_sex, 4),
            "p_condition": round(self.p_condition, 4),
            "p_region": round(self.p_region, 4),
            "claim_population": self.claim_population.to_dict() if self.claim_population else None,
            "study_population": self.study_population.to_dict() if self.study_population else None,
            "match_result": self.match_result.to_dict() if self.match_result else None,
            "scoring_mode": self.scoring_mode,
        }

