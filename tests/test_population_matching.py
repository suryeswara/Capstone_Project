"""
MedVerify AI — Unit Tests: Population Matching & Applicability Scoring

Tests the population matcher and applicability score engine across
diverse claim–study population combinations.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "medverify-ai-backend"))

from app.services.population.schemas import (
    PopulationProfile,
    PopulationSpecificity,
    ExtractionMethod,
    MatchType,
)
from app.services.population.matcher import match_populations
from app.services.population.applicability import compute_applicability


def test_population_matching():
    """Test population matching with various claim–study combinations."""
    print("=" * 70)
    print("TEST 1: Population Matching")
    print("=" * 70)

    test_cases = [
        {
            "description": "Exact age match: Claim 65+, Study 65+",
            "claim": PopulationProfile(age_min=65, age_max=None, specificity=PopulationSpecificity.EXPLICIT, confidence=0.90),
            "study": PopulationProfile(age_min=65, age_max=None, specificity=PopulationSpecificity.EXPLICIT, confidence=0.85),
            "expected_match": MatchType.MATCHED,
        },
        {
            "description": "Complete mismatch: Claim 65+, Study 18-40",
            "claim": PopulationProfile(age_min=65, age_max=None, specificity=PopulationSpecificity.EXPLICIT, confidence=0.90),
            "study": PopulationProfile(age_min=18, age_max=40, specificity=PopulationSpecificity.EXPLICIT, confidence=0.85),
            "expected_match": MatchType.MISMATCHED,
        },
        {
            "description": "Partial overlap: Claim 65+, Study 60-75",
            "claim": PopulationProfile(age_min=65, age_max=None, specificity=PopulationSpecificity.EXPLICIT, confidence=0.90),
            "study": PopulationProfile(age_min=60, age_max=75, specificity=PopulationSpecificity.EXPLICIT, confidence=0.85),
            "expected_match": MatchType.PARTIAL,
        },
        {
            "description": "Study covers claim: Claim 65+, Study 50+",
            "claim": PopulationProfile(age_min=65, age_max=None, specificity=PopulationSpecificity.EXPLICIT, confidence=0.90),
            "study": PopulationProfile(age_min=50, age_max=None, specificity=PopulationSpecificity.EXPLICIT, confidence=0.85),
            "expected_match": MatchType.MATCHED,
        },
        {
            "description": "Sex mismatch: Claim women, Study male-only",
            "claim": PopulationProfile(sex="female", specificity=PopulationSpecificity.EXPLICIT, confidence=0.90),
            "study": PopulationProfile(age_min=50, age_max=70, sex="male", specificity=PopulationSpecificity.EXPLICIT, confidence=0.85),
            "expected_match": MatchType.MISMATCHED,
        },
        {
            "description": "Both unknown populations",
            "claim": PopulationProfile(),
            "study": PopulationProfile(),
            "expected_match": MatchType.UNKNOWN,
        },
        {
            "description": "Claim has no population → UNKNOWN",
            "claim": PopulationProfile(),
            "study": PopulationProfile(age_min=18, age_max=65, specificity=PopulationSpecificity.EXPLICIT, confidence=0.85),
            "expected_match": MatchType.UNKNOWN,
        },
        {
            "description": "Pediatric claim vs Adult study: Claim 0-17, Study 40-60",
            "claim": PopulationProfile(age_min=0, age_max=17, specificity=PopulationSpecificity.IMPLICIT, confidence=0.80),
            "study": PopulationProfile(age_min=40, age_max=60, specificity=PopulationSpecificity.EXPLICIT, confidence=0.85),
            "expected_match": MatchType.MISMATCHED,
        },
    ]

    passed = 0
    for tc in test_cases:
        result = match_populations(tc["claim"], tc["study"])
        ok = result.match_type == tc["expected_match"]
        status = "PASS" if ok else "FAIL"
        print(f"\n  [{status}] {tc['description']}")
        print(f"         match_type={result.match_type.value}, "
              f"age_overlap={result.age_overlap_ratio:.2f}")
        print(f"         details: {result.details}")

        if not ok:
            print(f"         ✗ Expected {tc['expected_match'].value}, got {result.match_type.value}")
        else:
            passed += 1

    print(f"\n  Population Matching: {passed}/{len(test_cases)} passed")
    return passed, len(test_cases)


def test_applicability_scoring():
    """Test applicability scoring across different modes."""
    print("\n" + "=" * 70)
    print("TEST 2: Applicability Scoring (P_i)")
    print("=" * 70)

    test_cases = [
        {
            "description": "MATCHED → high P_i",
            "claim": PopulationProfile(age_min=65, specificity=PopulationSpecificity.EXPLICIT, confidence=0.90),
            "study": PopulationProfile(age_min=65, specificity=PopulationSpecificity.EXPLICIT, confidence=0.85),
            "mode": "v1_age_only",
            "expect_min": 0.85,
            "expect_max": 1.00,
        },
        {
            "description": "MISMATCHED → low P_i (but non-zero)",
            "claim": PopulationProfile(age_min=65, specificity=PopulationSpecificity.EXPLICIT, confidence=0.90),
            "study": PopulationProfile(age_min=18, age_max=40, specificity=PopulationSpecificity.EXPLICIT, confidence=0.85),
            "mode": "v1_age_only",
            "expect_min": 0.05,
            "expect_max": 0.30,
        },
        {
            "description": "UNKNOWN → neutral P_i",
            "claim": PopulationProfile(),
            "study": PopulationProfile(),
            "mode": "v1_age_only",
            "expect_min": 0.40,
            "expect_max": 0.60,
        },
        {
            "description": "v2: MATCHED age + compatible sex → high P_i",
            "claim": PopulationProfile(age_min=65, sex="female", specificity=PopulationSpecificity.EXPLICIT, confidence=0.90),
            "study": PopulationProfile(age_min=60, age_max=None, sex="all", specificity=PopulationSpecificity.EXPLICIT, confidence=0.85),
            "mode": "v2_age_sex",
            "expect_min": 0.70,
            "expect_max": 1.00,
        },
        {
            "description": "v2: MATCHED age + incompatible sex → penalized P_i",
            "claim": PopulationProfile(age_min=65, sex="female", specificity=PopulationSpecificity.EXPLICIT, confidence=0.90),
            "study": PopulationProfile(age_min=65, age_max=80, sex="male", specificity=PopulationSpecificity.EXPLICIT, confidence=0.85),
            "mode": "v2_age_sex",
            "expect_min": 0.0,
            "expect_max": 0.40,
        },
    ]

    passed = 0
    for tc in test_cases:
        match_result = match_populations(tc["claim"], tc["study"])
        result = compute_applicability(
            claim_population=tc["claim"],
            study_population=tc["study"],
            match_result=match_result,
            scoring_mode=tc["mode"],
        )

        in_range = tc["expect_min"] <= result.score <= tc["expect_max"]
        status = "PASS" if in_range else "FAIL"
        print(f"\n  [{status}] {tc['description']}")
        print(f"         P_i={result.score:.4f} (expected [{tc['expect_min']:.2f}, {tc['expect_max']:.2f}])")
        print(f"         match_type={match_result.match_type.value}, mode={tc['mode']}")

        if in_range:
            passed += 1
        else:
            print(f"         ✗ Score {result.score:.4f} outside expected range")

    print(f"\n  Applicability Scoring: {passed}/{len(test_cases)} passed")
    return passed, len(test_cases)


def test_quality_overrides_applicability():
    """
    Critical test: Verify that reliability ≠ applicability.
    A high-quality RCT on young adults should still have HIGH reliability
    but LOW applicability when the claim targets elderly patients.
    """
    print("\n" + "=" * 70)
    print("TEST 3: Quality ≠ Applicability (Research Principle)")
    print("=" * 70)

    claim = PopulationProfile(
        age_min=65, specificity=PopulationSpecificity.EXPLICIT, confidence=0.90
    )
    study = PopulationProfile(
        age_min=18, age_max=40, specificity=PopulationSpecificity.EXPLICIT, confidence=0.85
    )

    match_result = match_populations(claim, study)
    applicability = compute_applicability(claim, study, match_result, scoring_mode="v1_age_only")

    # Simulated reliability (this study IS high quality, just mismatched population)
    reliability_score = 0.90  # RCT-level

    print(f"\n  Claim: Adults 65+")
    print(f"  Study: RCT on Adults 18-40")
    print(f"  Reliability (R_i): {reliability_score}")
    print(f"  Applicability (P_i): {applicability.score}")
    print(f"  Match Type: {match_result.match_type.value}")
    print(f"  Age Overlap: {match_result.age_overlap_ratio:.2f}")

    ok = (
        reliability_score >= 0.85 and  # High quality
        applicability.score <= 0.30 and  # Low applicability
        match_result.match_type == MatchType.MISMATCHED
    )

    status = "PASS" if ok else "FAIL"
    print(f"\n  [{status}] R_i={reliability_score:.2f} (HIGH quality), "
          f"P_i={applicability.score:.4f} (LOW applicability)")
    print(f"  This confirms: Reliability ≠ Applicability")

    return 1 if ok else 0, 1


if __name__ == "__main__":
    print("=" * 70)
    print("MEDVERIFY AI — POPULATION MATCHING & APPLICABILITY UNIT TESTS")
    print("=" * 70)

    total_passed = 0
    total_tests = 0

    p, t = test_population_matching()
    total_passed += p
    total_tests += t

    p, t = test_applicability_scoring()
    total_passed += p
    total_tests += t

    p, t = test_quality_overrides_applicability()
    total_passed += p
    total_tests += t

    print("\n" + "=" * 70)
    print(f"TOTAL: {total_passed}/{total_tests} tests passed")
    print("=" * 70)

    if total_passed == total_tests:
        print("ALL TESTS PASSED ✅")
    else:
        print(f"FAILURES: {total_tests - total_passed} test(s) failed ❌")
