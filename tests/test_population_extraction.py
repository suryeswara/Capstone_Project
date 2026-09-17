"""
MedVerify AI — Unit Tests: Population Extraction

Tests the claim_population and study_population extractors with
diverse medical claim and evidence text inputs.
"""

import os
import sys

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "medverify-ai-backend"))

from app.services.population.schemas import PopulationSpecificity, ExtractionMethod
from app.services.population.claim_population import extract_claim_population
from app.services.population.study_population import extract_study_population
from app.services.population.normalizer import normalize_age_group, normalize_sex, normalize_region


def test_normalizer():
    """Test the normalization functions."""
    print("=" * 70)
    print("TEST 1: Normalizer Functions")
    print("=" * 70)

    # Age group normalization
    age_tests = [
        ("elderly patients", 65, None, True),
        ("children under 12", 0, 12, False),
        ("adults aged 40-60", 40, 60, False),
        ("people over 65", 65, None, False),
        ("patients 65+", 65, None, False),
        ("≥18 years", 18, None, False),
        ("pediatric population", 0, 17, True),
        ("between 40 and 65 years", 40, 65, False),
    ]

    passed = 0
    for text, exp_min, exp_max, exp_implicit in age_tests:
        age_min, age_max, is_implicit = normalize_age_group(text)
        ok = (age_min == exp_min and age_max == exp_max)
        status = "PASS" if ok else "FAIL"
        if not ok:
            print(f"  [{status}] '{text}' → age={age_min}-{age_max} (expected {exp_min}-{exp_max})")
        else:
            print(f"  [{status}] '{text}' → age={age_min}-{age_max}")
            passed += 1

    print(f"\n  Age Normalization: {passed}/{len(age_tests)} passed")

    # Sex normalization
    sex_tests = [
        ("men over 50", "male"),
        ("postmenopausal women", "female"),
        ("male and female participants", "all"),
        ("general population", None),
    ]
    sex_passed = 0
    for text, expected in sex_tests:
        result = normalize_sex(text)
        ok = result == expected
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] sex('{text}') → {result} (expected {expected})")
        if ok:
            sex_passed += 1

    print(f"\n  Sex Normalization: {sex_passed}/{len(sex_tests)} passed")

    # Region normalization
    region_tests = [
        ("conducted in the United States", "USA"),
        ("Japanese population", "Japan"),
        ("multinational study", "Global"),
        ("local hospital", None),
    ]
    region_passed = 0
    for text, expected in region_tests:
        result = normalize_region(text)
        ok = result == expected
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] region('{text}') → {result} (expected {expected})")
        if ok:
            region_passed += 1

    print(f"\n  Region Normalization: {region_passed}/{len(region_tests)} passed")
    return passed + sex_passed + region_passed, len(age_tests) + len(sex_tests) + len(region_tests)


def test_claim_population_extraction():
    """Test population extraction from claim texts."""
    print("\n" + "=" * 70)
    print("TEST 2: Claim Population Extraction")
    print("=" * 70)

    test_cases = [
        {
            "claim": "Drug X reduces blood pressure in people over 65.",
            "expect_age_min": 65,
            "expect_age_max": None,
            "expect_sex": None,
            "expect_has_info": True,
            "expect_specificity": PopulationSpecificity.EXPLICIT,
        },
        {
            "claim": "MMR vaccine is safe for children.",
            "expect_age_min": 0,
            "expect_age_max": 17,
            "expect_sex": None,
            "expect_has_info": True,
            "expect_specificity": PopulationSpecificity.IMPLICIT,
        },
        {
            "claim": "Statins reduce recurrent heart attack risk.",
            "expect_age_min": None,
            "expect_age_max": None,
            "expect_sex": None,
            "expect_has_info": False,
            "expect_specificity": PopulationSpecificity.UNKNOWN,
        },
        {
            "claim": "Aspirin daily prevents cardiac arrest in elderly women.",
            "expect_age_min": 65,
            "expect_age_max": None,
            "expect_sex": "female",
            "expect_has_info": True,
            "expect_specificity": PopulationSpecificity.IMPLICIT,
        },
        {
            "claim": "SGLT2 inhibitors improve glycemic control in adults aged 40-60.",
            "expect_age_min": 40,
            "expect_age_max": 60,
            "expect_sex": None,
            "expect_has_info": True,
            "expect_specificity": PopulationSpecificity.EXPLICIT,
        },
        {
            "claim": "Metformin is the first-line treatment for type 2 diabetes.",
            "expect_age_min": None,
            "expect_age_max": None,
            "expect_sex": None,
            "expect_has_info": False,
            "expect_specificity": PopulationSpecificity.UNKNOWN,
        },
    ]

    passed = 0
    for tc in test_cases:
        profile = extract_claim_population(tc["claim"])

        checks = [
            ("age_min", profile.age_min, tc["expect_age_min"]),
            ("age_max", profile.age_max, tc["expect_age_max"]),
            ("sex", profile.sex, tc["expect_sex"]),
            ("has_info", profile.has_any_info(), tc["expect_has_info"]),
        ]

        all_ok = all(actual == expected for _, actual, expected in checks)

        status = "PASS" if all_ok else "FAIL"
        print(f"\n  [{status}] Claim: \"{tc['claim'][:60]}...\"")
        print(f"         age={profile.age_min}-{profile.age_max}, sex={profile.sex}, "
              f"specificity={profile.specificity.value}, confidence={profile.confidence:.2f}")

        if not all_ok:
            for name, actual, expected in checks:
                if actual != expected:
                    print(f"         ✗ {name}: got {actual}, expected {expected}")
        else:
            passed += 1

    print(f"\n  Claim Extraction: {passed}/{len(test_cases)} passed")
    return passed, len(test_cases)


def test_study_population_extraction():
    """Test population extraction from PubMed evidence dicts."""
    print("\n" + "=" * 70)
    print("TEST 3: Study Population Extraction")
    print("=" * 70)

    test_cases = [
        {
            "evidence": {
                "chunk_text": "This randomized trial enrolled 500 participants aged 65 to 80 years with type 2 diabetes.",
                "title": "SGLT2 Inhibitor Trial in Elderly Diabetics",
                "doc_id": "pmid-12345",
            },
            "expect_age_min": 65,
            "expect_age_max": 80,
            "expect_has_info": True,
        },
        {
            "evidence": {
                "chunk_text": "A cohort of 1200 women with cardiovascular disease risk factors were followed for 5 years.",
                "title": "Cardiovascular Risk in Women",
                "doc_id": "pmid-67890",
            },
            "expect_sex": "female",
            "expect_has_info": True,
        },
        {
            "evidence": {
                "chunk_text": "PubMed Article (PMID:99999): Metformin improves glycemic control",
                "title": "Metformin improves glycemic control",
                "doc_id": "pmid-99999",
            },
            "expect_has_info": False,
        },
        {
            "evidence": {
                "chunk_text": "The study included children between 2 and 12 years of age in the United States.",
                "title": "Pediatric Vaccine Study",
                "doc_id": "pmid-11111",
            },
            "expect_age_min": 2,
            "expect_age_max": 12,
            "expect_has_info": True,
        },
    ]

    passed = 0
    for tc in test_cases:
        profile = extract_study_population(tc["evidence"])

        has_info_ok = profile.has_any_info() == tc["expect_has_info"]
        age_min_ok = True
        age_max_ok = True
        sex_ok = True

        if "expect_age_min" in tc:
            age_min_ok = profile.age_min == tc["expect_age_min"]
        if "expect_age_max" in tc:
            age_max_ok = profile.age_max == tc["expect_age_max"]
        if "expect_sex" in tc:
            sex_ok = profile.sex == tc["expect_sex"]

        all_ok = has_info_ok and age_min_ok and age_max_ok and sex_ok
        status = "PASS" if all_ok else "FAIL"
        print(f"\n  [{status}] Evidence: \"{tc['evidence']['title'][:50]}...\"")
        print(f"         age={profile.age_min}-{profile.age_max}, sex={profile.sex}, "
              f"confidence={profile.confidence:.2f}")

        if all_ok:
            passed += 1
        else:
            if not has_info_ok:
                print(f"         ✗ has_info: got {profile.has_any_info()}, expected {tc['expect_has_info']}")
            if not age_min_ok:
                print(f"         ✗ age_min: got {profile.age_min}, expected {tc.get('expect_age_min')}")
            if not age_max_ok:
                print(f"         ✗ age_max: got {profile.age_max}, expected {tc.get('expect_age_max')}")
            if not sex_ok:
                print(f"         ✗ sex: got {profile.sex}, expected {tc.get('expect_sex')}")

    print(f"\n  Study Extraction: {passed}/{len(test_cases)} passed")
    return passed, len(test_cases)


if __name__ == "__main__":
    print("=" * 70)
    print("MEDVERIFY AI — POPULATION EXTRACTION UNIT TESTS")
    print("=" * 70)

    total_passed = 0
    total_tests = 0

    p, t = test_normalizer()
    total_passed += p
    total_tests += t

    p, t = test_claim_population_extraction()
    total_passed += p
    total_tests += t

    p, t = test_study_population_extraction()
    total_passed += p
    total_tests += t

    print("\n" + "=" * 70)
    print(f"TOTAL: {total_passed}/{total_tests} tests passed")
    print("=" * 70)

    if total_passed == total_tests:
        print("ALL TESTS PASSED ✅")
    else:
        print(f"FAILURES: {total_tests - total_passed} test(s) failed ❌")
