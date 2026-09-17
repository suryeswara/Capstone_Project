"""
MedVerify AI — Population Applicability: Claim Population Extractor

Extracts population characteristics (age, sex, region) from a user's
medical claim text using rule-based regex + NLP pattern matching.

Examples:
  "Drug X reduces blood pressure in people over 65."
    → PopulationProfile(age_min=65, specificity=EXPLICIT, confidence=0.90)

  "MMR vaccine is safe for children."
    → PopulationProfile(age_min=0, age_max=17, specificity=IMPLICIT, confidence=0.85)

  "Statins reduce heart attack risk."
    → PopulationProfile(specificity=UNKNOWN, confidence=0.0)

Design decisions:
  - Uses regex-based extraction (deterministic, auditable, no hallucination risk)
  - Confidence scoring reflects extraction quality and specificity
  - Never invents population data that isn't in the claim text
"""

import re
import logging
from typing import Optional

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
# CLAIM POPULATION EXTRACTION PATTERNS
# ---------------------------------------------------------------------------

# Patterns that indicate a specific population is being referenced
POPULATION_INDICATOR_PATTERNS = [
    # Explicit age references
    r'(?:in|for|among|of)\s+(?:people|patients?|adults?|individuals?|persons?|subjects?)\s+(?:aged?|over|under|above|below|older|younger)',
    r'(?:in|for|among|of)\s+(?:the\s+)?(?:elderly|geriatric|pediatric|children|adolescents?|neonates?|infants?|seniors?|older\s+adults?)',
    r'(?:aged?\s+)?\d{1,3}\s*[-–+]',
    r'(?:≥|>=|>|≤|<=|<)\s*\d{1,3}',

    # Sex-specific references
    r'(?:in|for|among|of)\s+(?:men|women|males?|females?|boys?|girls?)',
    r'(?:postmenopausal|premenopausal|pregnant)\s+(?:women|females?|patients?)',

    # Region-specific references
    r'(?:in|from|across)\s+(?:the\s+)?(?:united\s+states|usa|uk|europe|asia|china|india|japan|africa|australia|canada)',
]


def extract_claim_population(claim_text: str) -> PopulationProfile:
    """
    Extract population characteristics from a medical claim text.

    Args:
        claim_text: The user's medical claim string.

    Returns:
        PopulationProfile with extracted age, sex, region, and confidence.
    """
    text_lower = claim_text.lower().strip()

    # Initialize profile
    age_min = None
    age_max = None
    sex = None
    region = None
    specificity = PopulationSpecificity.UNKNOWN
    confidence = 0.0
    raw_snippet = None
    confidence_components = []

    # -----------------------------------------------------------------------
    # 1. AGE EXTRACTION
    # -----------------------------------------------------------------------

    age_min_result, age_max_result, is_implicit = normalize_age_group(claim_text)

    if age_min_result is not None or age_max_result is not None:
        age_min = age_min_result
        age_max = age_max_result

        if is_implicit:
            specificity = PopulationSpecificity.IMPLICIT
            confidence_components.append(0.80)  # Implicit age group
        else:
            specificity = PopulationSpecificity.EXPLICIT
            confidence_components.append(0.95)  # Explicit numeric age

        # Extract the matching snippet for audit trail
        raw_snippet = _extract_population_snippet(claim_text)

    # -----------------------------------------------------------------------
    # 2. SEX EXTRACTION
    # -----------------------------------------------------------------------

    sex_result = normalize_sex(claim_text)
    if sex_result:
        sex = sex_result
        if specificity == PopulationSpecificity.UNKNOWN:
            specificity = PopulationSpecificity.EXPLICIT
        confidence_components.append(0.90)

    # -----------------------------------------------------------------------
    # 3. REGION EXTRACTION
    # -----------------------------------------------------------------------

    region_result = normalize_region(claim_text)
    if region_result:
        region = region_result
        if specificity == PopulationSpecificity.UNKNOWN:
            specificity = PopulationSpecificity.EXPLICIT
        confidence_components.append(0.85)

    # -----------------------------------------------------------------------
    # 4. CONFIDENCE SCORE COMPUTATION
    # -----------------------------------------------------------------------

    if confidence_components:
        # Average confidence across all extracted components
        confidence = sum(confidence_components) / len(confidence_components)
    else:
        confidence = 0.0

    profile = PopulationProfile(
        age_min=age_min,
        age_max=age_max,
        sex=sex,
        region=region,
        specificity=specificity,
        confidence=round(confidence, 4),
        extraction_method=ExtractionMethod.REGEX,
        raw_text_snippet=raw_snippet,
    )

    if profile.has_any_info():
        logger.info(
            f"[ClaimPopulation] Extracted: age={age_min}-{age_max}, "
            f"sex={sex}, region={region}, specificity={specificity.value}, "
            f"confidence={confidence:.2f}"
        )
    else:
        logger.info("[ClaimPopulation] No population information found in claim.")

    return profile


def _extract_population_snippet(text: str, max_len: int = 100) -> Optional[str]:
    """
    Extract the text fragment that contains population information.
    Used for audit trail and error analysis.
    """
    text_lower = text.lower()

    for pattern in POPULATION_INDICATOR_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            start = max(0, match.start() - 10)
            end = min(len(text), match.end() + 40)
            return text[start:end].strip()

    return None
