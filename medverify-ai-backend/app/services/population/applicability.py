"""
MedVerify AI — Population Applicability: Applicability Score Engine

Converts a MatchResult into a continuous applicability score P_i ∈ [0, 1].

The exact mapping from match category to score is an EXPERIMENTAL VARIABLE.
Multiple scoring modes are provided to support the ablation study:

  v1_age_only:   P_i depends only on age_overlap_ratio
  v2_age_sex:    P_i depends on age + sex compatibility
  v3_full:       P_i depends on age + sex + region

Design note (from research plan):
  "Do NOT immediately decide that MATCH = 1 and MISMATCH = 0.
   The literature warns against simplistic binary population filtering.
   Start with P ∈ [0,1] but make the mapping an experimental variable."

The scoring functions use a sigmoid-shaped mapping to avoid hard cutoffs:
  - MATCHED evidence gets high but not perfect P_i (allows some uncertainty)
  - MISMATCHED evidence gets low but non-zero P_i (doesn't fully discard)
  - UNKNOWN evidence gets a neutral P_i (separately evaluated)
"""

import os
import logging
from typing import Optional

from app.services.population.schemas import (
    PopulationProfile,
    MatchResult,
    MatchType,
    ApplicabilityResult,
)
from app.services.population.claim_population import extract_claim_population
from app.services.population.study_population import extract_study_population
from app.services.population.matcher import match_populations

logger = logging.getLogger(__name__)

# Environment variable to switch consensus mode
CONSENSUS_MODE = os.getenv("CONSENSUS_MODE", "baseline")  # "baseline" or "population"
SCORING_MODE = os.getenv("POPULATION_SCORING_MODE", "v1_age_only")


# ---------------------------------------------------------------------------
# SCORING CONFIGURATIONS (Experimental Variables)
# ---------------------------------------------------------------------------

# V1: Age-only scoring
V1_AGE_SCORES = {
    MatchType.MATCHED: 0.95,      # Study covers claim age range
    MatchType.PARTIAL: 0.60,      # Partial age overlap
    MatchType.MISMATCHED: 0.15,   # No age overlap (low but non-zero)
    MatchType.UNKNOWN: 0.50,      # Neutral when unknown
}

# V2: Age + Sex scoring
V2_SEX_PENALTY = 0.30   # Penalty factor when sex is incompatible
V2_SEX_BONUS = 0.05     # Small bonus when sex is explicitly compatible

# V3: Full (Age + Sex + Region) scoring
V3_REGION_PENALTY = 0.10   # Mild penalty for different region
V3_REGION_BONUS = 0.03     # Small bonus for matching region


def compute_applicability(
    claim_population: PopulationProfile,
    study_population: PopulationProfile,
    match_result: MatchResult,
    scoring_mode: Optional[str] = None,
) -> ApplicabilityResult:
    """
    Compute the population applicability score P_i in [0, 1] using
    the frozen Stage 8 formula:
    P_i = (A_i + S_i + C_i + G_i) / 4.0
    where:
      A_i = Age compatibility (1.0, 0.5, 0.0)
      S_i = Sex compatibility (1.0, 0.5, 0.0)
      C_i = Condition compatibility (1.0, 0.5, 0.0)
      G_i = Region compatibility (1.0, 0.5, 0.0)
    """
    mode = scoring_mode or SCORING_MODE

    # Extract 4-dimension scores from match result
    p_age = match_result.p_age
    p_sex = match_result.p_sex
    p_condition = match_result.p_condition
    p_region = match_result.p_region

    if mode == "v1_age_only":
        score = _score_v1_age_only(match_result)
    elif mode == "v2_age_sex":
        score = _score_v2_age_sex(match_result)
    elif mode in ["v3_full", "v3_frozen_four_dimension", "frozen_stage_8"]:
        # Frozen formula: 4-dimension arithmetic average
        score = (p_age + p_sex + p_condition + p_region) / 4.0
    else:
        score = (p_age + p_sex + p_condition + p_region) / 4.0

    # Clamp to [0, 1]
    score = max(0.0, min(1.0, score))

    result = ApplicabilityResult(
        score=round(score, 4),
        p_age=round(p_age, 4),
        p_sex=round(p_sex, 4),
        p_condition=round(p_condition, 4),
        p_region=round(p_region, 4),
        claim_population=claim_population,
        study_population=study_population,
        match_result=match_result,
        scoring_mode=mode,
    )

    logger.info(
        f"[Applicability] P_total={score:.4f} (A_i={p_age}, S_i={p_sex}, C_i={p_condition}, G_i={p_region}), "
        f"mode={mode}, match={match_result.match_type.value}"
    )

    return result



def _score_v1_age_only(match_result: MatchResult) -> float:
    """
    V1 scoring: Only age overlap matters.

    Uses the base category score, refined by the actual overlap ratio
    for PARTIAL matches.
    """
    base_score = V1_AGE_SCORES[match_result.match_type]

    if match_result.match_type == MatchType.PARTIAL:
        # Refine with actual overlap: interpolate between MISMATCHED and MATCHED
        low = V1_AGE_SCORES[MatchType.MISMATCHED]
        high = V1_AGE_SCORES[MatchType.MATCHED]
        base_score = low + (high - low) * match_result.age_overlap_ratio

    return base_score


def _score_v2_age_sex(match_result: MatchResult) -> float:
    """
    V2 scoring: Age overlap + sex compatibility.
    """
    score = _score_v1_age_only(match_result)

    if match_result.sex_compatible is False:
        score = max(0.0, score - V2_SEX_PENALTY)
    elif match_result.sex_compatible is True:
        score = min(1.0, score + V2_SEX_BONUS)

    return score


def _score_v3_full(match_result: MatchResult) -> float:
    """
    V3 scoring: Age overlap + sex + region compatibility.
    """
    score = _score_v2_age_sex(match_result)

    if match_result.region_compatible is False:
        score = max(0.0, score - V3_REGION_PENALTY)
    elif match_result.region_compatible is True:
        score = min(1.0, score + V3_REGION_BONUS)

    return score


# ---------------------------------------------------------------------------
# CONVENIENCE: Full Pipeline (extract + match + score) for One Evidence Item
# ---------------------------------------------------------------------------

def analyze_population_applicability(
    claim_text: str,
    evidence_dict: dict,
    scoring_mode: Optional[str] = None,
    clinical_trials_lookup: Optional[callable] = None,
) -> ApplicabilityResult:
    """
    Full pipeline: Extract claim population, extract study population,
    match them, and compute applicability score P_i.

    This is the main entry point for integrating population applicability
    into the orchestrator.

    Args:
        claim_text: The user's medical claim.
        evidence_dict: Evidence dict from retrieval engine.
        scoring_mode: Override for scoring mode.
        clinical_trials_lookup: Optional ClinicalTrials.gov lookup function.

    Returns:
        ApplicabilityResult with P_i score and full provenance chain.
    """
    # 1. Extract claim population
    claim_pop = extract_claim_population(claim_text)

    # 2. Extract study population
    study_pop = extract_study_population(
        evidence_dict=evidence_dict,
        clinical_trials_lookup=clinical_trials_lookup,
    )

    # 3. Match populations
    match_result = match_populations(claim_pop, study_pop)

    # 4. Compute applicability score
    return compute_applicability(
        claim_population=claim_pop,
        study_population=study_pop,
        match_result=match_result,
        scoring_mode=scoring_mode,
    )
