"""
MedVerify AI — Population Applicability: Population Matcher

Compares a claim's target population against a study's actual population
to determine compatibility: MATCHED, PARTIAL, MISMATCHED, or UNKNOWN.

The matcher computes an age_overlap_ratio ∈ [0,1] that measures the degree
of overlap between the claim's target age range and the study's enrolled
age range. This ratio feeds into the continuous applicability score P_i.

Design note:
    - Age matching uses interval overlap analysis.
    - Sex matching is binary (compatible or not).
    - Region matching is binary (same region or not).
    - UNKNOWN is used when either claim or study lacks the relevant data.
"""

import logging
from typing import Optional

from app.services.population.schemas import (
    PopulationProfile,
    MatchResult,
    MatchType,
)

logger = logging.getLogger(__name__)

# Default age bounds when one side is open-ended
DEFAULT_AGE_MIN = 0
DEFAULT_AGE_MAX = 120


def match_populations(
    claim_pop: PopulationProfile,
    study_pop: PopulationProfile,
) -> MatchResult:
    """
    Compare claim population against study population.

    Args:
        claim_pop: Population profile extracted from the claim text.
        study_pop: Population profile extracted from the evidence.

    Returns:
        MatchResult with match_type, age_overlap_ratio, and compatibility flags.
    """
    # If neither has any population info, return UNKNOWN
    if not claim_pop.has_any_info() and not study_pop.has_any_info():
        return MatchResult(
            match_type=MatchType.UNKNOWN,
            age_overlap_ratio=0.0,
            details="Neither claim nor study specifies population characteristics.",
        )

    # If claim has no population info, consider study universally applicable
    if not claim_pop.has_any_info():
        return MatchResult(
            match_type=MatchType.UNKNOWN,
            age_overlap_ratio=0.0,
            details="Claim does not specify a target population.",
        )

    # If study has no population info, we cannot determine applicability
    if not study_pop.has_any_info():
        return MatchResult(
            match_type=MatchType.UNKNOWN,
            age_overlap_ratio=0.0,
            details="Study does not report population characteristics.",
        )

    # -----------------------------------------------------------------------
    # AGE MATCHING
    # -----------------------------------------------------------------------
    age_overlap = _compute_age_overlap(claim_pop, study_pop)

    # -----------------------------------------------------------------------
    # SEX MATCHING
    # -----------------------------------------------------------------------
    sex_compat = _check_sex_compatibility(claim_pop.sex, study_pop.sex)

    # -----------------------------------------------------------------------
    # REGION MATCHING
    # -----------------------------------------------------------------------
    region_compat = _check_region_compatibility(claim_pop.region, study_pop.region)

    # -----------------------------------------------------------------------
    # DETERMINE 4-DIMENSION SCORES (Stage 8 Frozen Protocol)
    # 1.0 = compatible, 0.5 = unspecified / neutral / partial, 0.0 = incompatible
    # -----------------------------------------------------------------------
    if not claim_pop.has_age_info() or not study_pop.has_age_info():
        p_age = 0.5
    elif age_overlap >= 0.70:
        p_age = 1.0
    elif age_overlap > 0.0:
        p_age = 0.5
    else:
        p_age = 0.0

    if sex_compat is None:
        p_sex = 0.5
    elif sex_compat is True:
        p_sex = 1.0
    else:
        p_sex = 0.0

    if region_compat is None:
        p_region = 0.5
    elif region_compat is True:
        p_region = 1.0
    else:
        p_region = 0.0

    # Condition dimension (default 1.0 if matching disease domain, 0.5 if unspecified)
    p_condition = 1.0

    # -----------------------------------------------------------------------
    # DETERMINE OVERALL MATCH TYPE
    # -----------------------------------------------------------------------
    match_type, details = _determine_match_type(
        age_overlap=age_overlap,
        sex_compatible=sex_compat,
        region_compatible=region_compat,
        claim_pop=claim_pop,
        study_pop=study_pop,
    )

    result = MatchResult(
        match_type=match_type,
        age_overlap_ratio=round(age_overlap, 4),
        sex_compatible=sex_compat,
        region_compatible=region_compat,
        condition_compatible=True,
        p_age=p_age,
        p_sex=p_sex,
        p_condition=p_condition,
        p_region=p_region,
        details=details,
    )

    logger.info(
        f"[PopulationMatcher] {match_type.value}: "
        f"A_i={p_age}, S_i={p_sex}, C_i={p_condition}, G_i={p_region}, "
        f"age_overlap={age_overlap:.2f}, sex={sex_compat}, region={region_compat}"
    )

    return result



def _compute_age_overlap(
    claim_pop: PopulationProfile,
    study_pop: PopulationProfile,
) -> float:
    """
    Compute the ratio of overlap between claim and study age ranges.

    Returns a value in [0.0, 1.0]:
      1.0 = Study fully covers the claim's age range
      0.0 = No overlap at all
      0.0-1.0 = Partial overlap

    The overlap is measured as:
      intersection_length / claim_range_length
    """
    if not claim_pop.has_age_info() or not study_pop.has_age_info():
        return 0.0  # Cannot compute without age data

    # Resolve open-ended ranges
    c_min = claim_pop.age_min if claim_pop.age_min is not None else DEFAULT_AGE_MIN
    c_max = claim_pop.age_max if claim_pop.age_max is not None else DEFAULT_AGE_MAX

    s_min = study_pop.age_min if study_pop.age_min is not None else DEFAULT_AGE_MIN
    s_max = study_pop.age_max if study_pop.age_max is not None else DEFAULT_AGE_MAX

    # Ensure valid ranges
    if c_min > c_max:
        c_min, c_max = c_max, c_min
    if s_min > s_max:
        s_min, s_max = s_max, s_min

    # Compute interval intersection
    overlap_start = max(c_min, s_min)
    overlap_end = min(c_max, s_max)

    if overlap_start > overlap_end:
        return 0.0  # No overlap

    overlap_length = overlap_end - overlap_start
    claim_length = c_max - c_min

    if claim_length == 0:
        # Point age (e.g., exactly 65)
        # Check if the study range contains this point
        return 1.0 if s_min <= c_min <= s_max else 0.0

    return min(1.0, overlap_length / claim_length)


def _check_sex_compatibility(
    claim_sex: Optional[str],
    study_sex: Optional[str],
) -> Optional[bool]:
    """
    Check if study sex enrollment is compatible with claim population.

    Returns:
      True = Compatible
      False = Incompatible (e.g., claim "women", study "male only")
      None = Cannot determine (one or both not specified)
    """
    if claim_sex is None or study_sex is None:
        return None  # Insufficient data

    # "all" is compatible with everything
    if claim_sex == "all" or study_sex == "all":
        return True

    return claim_sex == study_sex


def _check_region_compatibility(
    claim_region: Optional[str],
    study_region: Optional[str],
) -> Optional[bool]:
    """
    Check if study region matches claim population.

    Returns:
      True = Same region or one is "Global"
      False = Different specific regions
      None = Cannot determine
    """
    if claim_region is None or study_region is None:
        return None

    if claim_region == "Global" or study_region == "Global":
        return True

    return claim_region.lower() == study_region.lower()


def _determine_match_type(
    age_overlap: float,
    sex_compatible: Optional[bool],
    region_compatible: Optional[bool],
    claim_pop: PopulationProfile,
    study_pop: PopulationProfile,
) -> tuple:
    """
    Determine overall match type based on component compatibilities.
    Returns (MatchType, details_string).
    """
    details_parts = []

    # Age assessment
    if claim_pop.has_age_info() and study_pop.has_age_info():
        if age_overlap >= 0.80:
            details_parts.append(f"Age: HIGH overlap ({age_overlap:.0%})")
        elif age_overlap >= 0.30:
            details_parts.append(f"Age: PARTIAL overlap ({age_overlap:.0%})")
        elif age_overlap > 0:
            details_parts.append(f"Age: LOW overlap ({age_overlap:.0%})")
        else:
            details_parts.append("Age: NO overlap (complete mismatch)")

    # Sex assessment
    if sex_compatible is True:
        details_parts.append("Sex: compatible")
    elif sex_compatible is False:
        details_parts.append("Sex: INCOMPATIBLE")

    # Region assessment
    if region_compatible is True:
        details_parts.append("Region: compatible")
    elif region_compatible is False:
        details_parts.append("Region: DIFFERENT")

    details = "; ".join(details_parts) if details_parts else "No comparable dimensions."

    # Decision logic:
    # Hard mismatch on sex → MISMATCHED
    if sex_compatible is False:
        return MatchType.MISMATCHED, details

    # Age-based decision
    if claim_pop.has_age_info() and study_pop.has_age_info():
        if age_overlap >= 0.80:
            return MatchType.MATCHED, details
        elif age_overlap >= 0.20:
            return MatchType.PARTIAL, details
        elif age_overlap > 0:
            return MatchType.PARTIAL, details
        else:
            return MatchType.MISMATCHED, details

    # If only sex/region info and they're compatible
    if sex_compatible is True or region_compatible is True:
        return MatchType.MATCHED, details

    return MatchType.UNKNOWN, details
