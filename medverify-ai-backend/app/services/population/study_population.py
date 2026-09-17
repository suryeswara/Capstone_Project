"""
MedVerify AI — Population Applicability: Study Population Extractor

Extracts population characteristics from PubMed evidence abstracts and
article metadata. Uses a priority chain:

    1. ClinicalTrials.gov structured metadata (when NCT ID present)
    2. PubMed abstract text (regex extraction)
    3. UNKNOWN (never invents missing data)

Design note:
    The extractor intentionally prefers structured data from
    ClinicalTrials.gov over free-text extraction because structured
    eligibility criteria are more reliable and auditable.
"""

import re
import logging
from typing import Optional, Dict

from app.services.population.schemas import (
    PopulationProfile,
    PopulationSpecificity,
    ExtractionMethod,
)
from app.services.population.normalizer import (
    normalize_age_group,
    normalize_sex,
    normalize_region,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NCT ID EXTRACTION PATTERN
# ---------------------------------------------------------------------------

NCT_PATTERN = re.compile(r'NCT\d{8,}', re.IGNORECASE)

# ---------------------------------------------------------------------------
# ABSTRACT POPULATION EXTRACTION PATTERNS
# ---------------------------------------------------------------------------

# Patterns commonly found in PubMed abstracts describing study populations
STUDY_AGE_PATTERNS = [
    # "participants aged 40-65 years"
    r'(?:participants?|patients?|subjects?|individuals?|adults?|persons?)\s+aged?\s+(\d{1,3})\s*[-–to]+\s*(\d{1,3})',
    # "mean age 52.3 ± 7.2 years" → approximate range
    r'mean\s+age\s+(?:was\s+)?(\d{1,3}(?:\.\d+)?)',
    # "age range: 40-70"
    r'age\s+range[:\s]+(\d{1,3})\s*[-–to]+\s*(\d{1,3})',
    # "≥65 years" in context
    r'(?:[≥>]=?\s*)(\d{1,3})\s*(?:years?|yrs?)',
    # "18 years or older"
    r'(\d{1,3})\s+years?\s+(?:or\s+)?older',
    # "between 40 and 65 years"
    r'between\s+(\d{1,3})\s+and\s+(\d{1,3})\s*(?:years?|yrs?)',
]

STUDY_SIZE_PATTERNS = [
    # "N=500", "n = 1234"
    r'[Nn]\s*=\s*(\d+)',
    # "500 participants", "1234 patients"
    r'(\d{2,})\s+(?:participants?|patients?|subjects?|individuals?|adults?)',
]


def extract_study_population(
    evidence_dict: Dict,
    clinical_trials_lookup: Optional[callable] = None,
) -> PopulationProfile:
    """
    Extract population characteristics from a piece of retrieved evidence.

    Priority chain:
        1. ClinicalTrials.gov API (if NCT ID found and lookup function provided)
        2. Abstract/chunk text regex extraction
        3. UNKNOWN

    Args:
        evidence_dict: Evidence dict from the retrieval engine, with keys like
                       'chunk_text', 'title', 'doc_id', 'abstract_chunk', etc.
        clinical_trials_lookup: Optional callable that takes an NCT ID string
                                and returns a PopulationProfile or None.

    Returns:
        PopulationProfile with study population characteristics.
    """
    # Combine available text fields for extraction
    text_fields = [
        evidence_dict.get("abstract_chunk", ""),
        evidence_dict.get("chunk_text", ""),
        evidence_dict.get("title", ""),
    ]
    combined_text = " ".join(t for t in text_fields if t).strip()
    doc_id = evidence_dict.get("doc_id", "")

    # -----------------------------------------------------------------------
    # PRIORITY 1: ClinicalTrials.gov structured data
    # -----------------------------------------------------------------------

    nct_match = NCT_PATTERN.search(combined_text) or NCT_PATTERN.search(doc_id)
    if nct_match and clinical_trials_lookup:
        nct_id = nct_match.group(0).upper()
        ct_profile = clinical_trials_lookup(nct_id)
        if ct_profile and ct_profile.has_any_info():
            logger.info(f"[StudyPopulation] Using ClinicalTrials.gov data for {nct_id}")
            return ct_profile

    # -----------------------------------------------------------------------
    # PRIORITY 2: Abstract text regex extraction
    # -----------------------------------------------------------------------

    return _extract_from_abstract(combined_text)


def _extract_from_abstract(text: str) -> PopulationProfile:
    """
    Extract population information from abstract/chunk text using regex patterns.
    """
    if not text:
        return PopulationProfile(
            extraction_method=ExtractionMethod.UNKNOWN,
            specificity=PopulationSpecificity.UNKNOWN,
        )

    age_min = None
    age_max = None
    sex = None
    region = None
    specificity = PopulationSpecificity.UNKNOWN
    confidence_components = []
    raw_snippet = None

    # -----------------------------------------------------------------------
    # AGE EXTRACTION from abstract
    # -----------------------------------------------------------------------

    for pattern in STUDY_AGE_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            groups = match.groups()
            if len(groups) == 2:
                try:
                    age_min = int(float(groups[0]))
                    age_max = int(float(groups[1]))
                    specificity = PopulationSpecificity.EXPLICIT
                    confidence_components.append(0.85)
                    raw_snippet = text[max(0, match.start()-5):match.end()+20].strip()
                    break
                except (ValueError, TypeError):
                    pass
            elif len(groups) == 1:
                try:
                    val = int(float(groups[0]))
                    # "mean age X" → approximate ±15 year range
                    if "mean" in text[max(0, match.start()-10):match.start()].lower():
                        age_min = max(0, val - 15)
                        age_max = val + 15
                        specificity = PopulationSpecificity.IMPLICIT
                        confidence_components.append(0.60)  # Lower confidence for mean-based
                    else:
                        # "≥X years" or "X years or older"
                        age_min = val
                        age_max = None
                        specificity = PopulationSpecificity.EXPLICIT
                        confidence_components.append(0.85)
                    raw_snippet = text[max(0, match.start()-5):match.end()+20].strip()
                    break
                except (ValueError, TypeError):
                    pass

    # Fallback: try named age groups from normalizer
    if age_min is None:
        norm_min, norm_max, is_implicit = normalize_age_group(text)
        if norm_min is not None or norm_max is not None:
            age_min = norm_min
            age_max = norm_max
            specificity = PopulationSpecificity.IMPLICIT if is_implicit else PopulationSpecificity.EXPLICIT
            confidence_components.append(0.70 if is_implicit else 0.85)

    # -----------------------------------------------------------------------
    # SEX EXTRACTION from abstract
    # -----------------------------------------------------------------------

    sex_result = normalize_sex(text)
    if sex_result:
        sex = sex_result
        confidence_components.append(0.85)

    # -----------------------------------------------------------------------
    # REGION EXTRACTION from abstract
    # -----------------------------------------------------------------------

    region_result = normalize_region(text)
    if region_result:
        region = region_result
        confidence_components.append(0.80)

    # -----------------------------------------------------------------------
    # CONFIDENCE
    # -----------------------------------------------------------------------

    confidence = 0.0
    if confidence_components:
        confidence = sum(confidence_components) / len(confidence_components)

    profile = PopulationProfile(
        age_min=age_min,
        age_max=age_max,
        sex=sex,
        region=region,
        specificity=specificity,
        confidence=round(confidence, 4),
        extraction_method=ExtractionMethod.REGEX if confidence > 0 else ExtractionMethod.UNKNOWN,
        raw_text_snippet=raw_snippet,
    )

    if profile.has_any_info():
        logger.info(
            f"[StudyPopulation] Extracted: age={age_min}-{age_max}, "
            f"sex={sex}, region={region}, confidence={confidence:.2f}"
        )
    else:
        logger.debug("[StudyPopulation] No population info found in abstract text.")

    return profile
