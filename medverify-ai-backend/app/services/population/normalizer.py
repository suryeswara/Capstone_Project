"""
MedVerify AI — Population Applicability: Age & Sex Normalizer

Normalizes natural-language population references into structured
PopulationProfile fields.

Examples:
  "elderly patients"       → age_min=65, specificity=IMPLICIT
  "adults aged 40-60"      → age_min=40, age_max=60, specificity=EXPLICIT
  "women over 50"          → age_min=50, sex="female", specificity=EXPLICIT
  "pediatric population"   → age_min=0, age_max=17, specificity=IMPLICIT
  "men and women"          → sex="all", specificity=EXPLICIT

Design note:
  The normalizer intentionally uses conservative, well-established
  age-group mappings from WHO/CDC guidelines. Ambiguous terms
  (e.g., "middle-aged") use broad ranges rather than precise cutoffs.
"""

import re
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AGE GROUP NORMALIZATION MAPPINGS
# Based on WHO/CDC standard age group definitions
# ---------------------------------------------------------------------------

AGE_GROUP_MAP = {
    # Pediatric
    "neonates": (0, 0),
    "neonatal": (0, 0),
    "infants": (0, 1),
    "infant": (0, 1),
    "toddlers": (1, 3),
    "toddler": (1, 3),
    "children": (0, 17),
    "child": (0, 17),
    "pediatric": (0, 17),
    "paediatric": (0, 17),
    "adolescents": (12, 17),
    "adolescent": (12, 17),
    "teenagers": (13, 19),
    "teens": (13, 19),

    # Adult categories
    "young adults": (18, 39),
    "young adult": (18, 39),
    "adults": (18, None),
    "adult": (18, None),
    "middle-aged": (40, 64),
    "middle aged": (40, 64),

    # Older adults / Geriatric
    "elderly": (65, None),
    "older adults": (65, None),
    "older adult": (65, None),
    "geriatric": (65, None),
    "senior": (65, None),
    "seniors": (65, None),
    "aged": (65, None),
    "octogenarian": (80, 89),
    "nonagenarian": (90, 99),
    "centenarian": (100, None),

    # General population
    "all ages": (0, None),
    "general population": (0, None),
}

# ---------------------------------------------------------------------------
# SEX / GENDER NORMALIZATION MAPPINGS
# Note: This normalizer maps to biological sex categories used in clinical
# trial eligibility. The distinction between sex and gender is acknowledged
# but clinical trials typically report biological sex.
# ---------------------------------------------------------------------------

SEX_NORMALIZATION_MAP = {
    # Male
    "male": "male",
    "males": "male",
    "men": "male",
    "man": "male",
    "boys": "male",
    "boy": "male",

    # Female
    "female": "female",
    "females": "female",
    "women": "female",
    "woman": "female",
    "girls": "female",
    "girl": "female",
    "pregnant women": "female",
    "postmenopausal": "female",
    "premenopausal": "female",

    # All / Both
    "both sexes": "all",
    "both genders": "all",
    "all sexes": "all",
    "all genders": "all",
    "men and women": "all",
    "women and men": "all",
    "male and female": "all",
    "female and male": "all",
    "mixed": "all",
}

# ---------------------------------------------------------------------------
# REGION NORMALIZATION
# ---------------------------------------------------------------------------

REGION_NORMALIZATION_MAP = {
    "us": "USA",
    "usa": "USA",
    "united states": "USA",
    "america": "USA",
    "american": "USA",
    "uk": "United Kingdom",
    "united kingdom": "United Kingdom",
    "britain": "United Kingdom",
    "british": "United Kingdom",
    "europe": "Europe",
    "european": "Europe",
    "asia": "Asia",
    "asian": "Asia",
    "china": "China",
    "chinese": "China",
    "japan": "Japan",
    "japanese": "Japan",
    "india": "India",
    "indian": "India",
    "africa": "Africa",
    "african": "Africa",
    "south america": "South America",
    "latin america": "Latin America",
    "australia": "Australia",
    "australian": "Australia",
    "canada": "Canada",
    "canadian": "Canada",
    "global": "Global",
    "worldwide": "Global",
    "multinational": "Global",
    "multicenter": "Global",
    "multi-center": "Global",
}


def normalize_age_group(text: str) -> Tuple[Optional[int], Optional[int], bool]:
    """
    Normalize an age group reference to (age_min, age_max, is_implicit).

    Returns:
        Tuple of (age_min, age_max, is_implicit)
        is_implicit=True means the mapping was from a named group (e.g. "elderly")
        rather than an explicit numeric range.

    Priority: Explicit numeric patterns are checked FIRST so that
    "adults aged 40-60" returns (40, 60, False) rather than matching
    the named group "adults" → (18, None, True).
    """
    text_lower = text.lower().strip()

    # 1. Try EXPLICIT numeric age ranges FIRST (higher priority)
    #    Patterns: "aged 40-60", "age 65+", "≥65 years", "40 to 60 years",
    #              "over 65", "under 18", "between 40 and 60"

    # Pattern: "aged X-Y", "age X-Y", "ages X-Y"
    match = re.search(r'(?:aged?|ages?)\s*(\d{1,3})\s*[-\u2013to]+\s*(\d{1,3})', text_lower)
    if match:
        return (int(match.group(1)), int(match.group(2)), False)

    # Pattern: "between X and Y"
    match = re.search(r'between\s+(\d{1,3})\s+and\s+(\d{1,3})', text_lower)
    if match:
        return (int(match.group(1)), int(match.group(2)), False)

    # Pattern: "X to Y years"
    match = re.search(r'(\d{1,3})\s+to\s+(\d{1,3})\s*(?:years?|yrs?)', text_lower)
    if match:
        return (int(match.group(1)), int(match.group(2)), False)

    # Pattern: "≥X", ">=X", "over X", "older than X", "at least X"
    match = re.search(r'(?:[\u2265>]=?\s*|over\s+|older\s+than\s+|at\s+least\s+|above\s+)(\d{1,3})', text_lower)
    if match:
        return (int(match.group(1)), None, False)

    # Pattern: "X+" or "X and older" or "X and above"
    match = re.search(r'(\d{1,3})\s*\+', text_lower)
    if match:
        return (int(match.group(1)), None, False)

    match = re.search(r'(\d{1,3})\s+and\s+(?:older|above|over)', text_lower)
    if match:
        return (int(match.group(1)), None, False)

    # Pattern: "≤X", "<=X", "under X", "younger than X", "below X"
    match = re.search(r'(?:[\u2264<]=?\s*|under\s+|younger\s+than\s+|below\s+)(\d{1,3})', text_lower)
    if match:
        return (0, int(match.group(1)), False)

    # 2. Fall back to named age groups (implicit) only if no numeric match
    for group_name, (age_min, age_max) in AGE_GROUP_MAP.items():
        if group_name in text_lower:
            return (age_min, age_max, True)

    return (None, None, False)


def normalize_sex(text: str) -> Optional[str]:
    """
    Normalize sex/gender references to standardized values.
    Uses word-boundary matching to prevent false positives
    (e.g., 'treatment' should NOT match 'men').

    Returns: "male", "female", "all", or None if not found.
    """
    text_lower = text.lower().strip()

    # Check multi-word phrases first (longer matches take priority)
    for phrase in sorted(SEX_NORMALIZATION_MAP.keys(), key=len, reverse=True):
        # Use word-boundary regex to avoid substring matches
        pattern = r'\b' + re.escape(phrase) + r'\b'
        if re.search(pattern, text_lower):
            return SEX_NORMALIZATION_MAP[phrase]

    return None


def normalize_region(text: str) -> Optional[str]:
    """
    Normalize region/country references to standardized values.

    Returns: Normalized region string or None if not found.
    """
    text_lower = text.lower().strip()

    for phrase in sorted(REGION_NORMALIZATION_MAP.keys(), key=len, reverse=True):
        if phrase in text_lower:
            return REGION_NORMALIZATION_MAP[phrase]

    return None


def normalize_clinical_trials_age(age_str: str) -> Optional[int]:
    """
    Parse ClinicalTrials.gov age strings like "65 Years", "18 Years", "N/A".

    Returns: Integer age or None.
    """
    if not age_str or age_str.lower() in ("n/a", "na", "not applicable", ""):
        return None

    match = re.search(r'(\d{1,3})', age_str)
    if match:
        return int(match.group(1))

    return None


def normalize_clinical_trials_sex(sex_str: str) -> Optional[str]:
    """
    Parse ClinicalTrials.gov sex strings like "All", "Female", "Male".

    Returns: "male", "female", "all", or None.
    """
    if not sex_str:
        return None

    sex_lower = sex_str.lower().strip()
    if sex_lower in ("all", "both"):
        return "all"
    elif sex_lower in ("male", "men"):
        return "male"
    elif sex_lower in ("female", "women"):
        return "female"

    return None
