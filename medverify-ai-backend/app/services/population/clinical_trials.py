"""
MedVerify AI — Population Applicability: ClinicalTrials.gov NCT Lookup

Queries the ClinicalTrials.gov API v2 to retrieve structured eligibility
criteria for clinical trials referenced by NCT ID in PubMed evidence.

Features:
  - Fetches: MinimumAge, MaximumAge, Sex, LocationCountry
  - Local JSON file cache to avoid redundant API calls
  - Timeout handling with graceful fallback

Design note:
  ClinicalTrials.gov provides the most reliable structured population
  data, so it takes highest priority in the extraction chain.
"""

import os
import json
import logging
import urllib.request
import urllib.error
from typing import Optional, Dict

from app.services.population.schemas import (
    PopulationProfile,
    PopulationSpecificity,
    ExtractionMethod,
)
from app.services.population.normalizer import (
    normalize_clinical_trials_age,
    normalize_clinical_trials_sex,
    normalize_region,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CACHE CONFIGURATION
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CACHE_DIR = os.path.join(PROJECT_ROOT, "..", "med_datasets", "cache")
CACHE_FILE = os.path.join(CACHE_DIR, "clinical_trials_cache.json")

# ClinicalTrials.gov API v2 endpoint
CT_API_BASE = "https://clinicaltrials.gov/api/v2/studies"
CT_TIMEOUT = 8  # seconds


def _load_cache() -> Dict:
    """Load the NCT cache from disk."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def _save_cache(cache: Dict):
    """Persist the NCT cache to disk."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except IOError as e:
        logger.warning(f"Could not save NCT cache: {e}")


def lookup_clinical_trial(nct_id: str) -> Optional[PopulationProfile]:
    """
    Look up structured eligibility criteria for a clinical trial by NCT ID.

    Architecture:
      NCT ID
        ↓
      Cache?
        ├── YES → return cached PopulationProfile
        └── NO
              ↓
        ClinicalTrials.gov API v2
              ↓
        Parse eligibility criteria
              ↓
        Store in cache
              ↓
        Return PopulationProfile

    Args:
        nct_id: ClinicalTrials.gov NCT identifier (e.g., "NCT01234567").

    Returns:
        PopulationProfile with structured eligibility data, or None if lookup fails.
    """
    nct_id = nct_id.upper().strip()

    # -----------------------------------------------------------------------
    # CHECK CACHE
    # -----------------------------------------------------------------------
    cache = _load_cache()
    if nct_id in cache:
        cached = cache[nct_id]
        if cached is None:
            # Previously failed lookup — don't retry
            logger.debug(f"[ClinicalTrials] {nct_id}: cached as not found, skipping.")
            return None

        logger.info(f"[ClinicalTrials] {nct_id}: returning cached eligibility data.")
        return PopulationProfile(
            age_min=cached.get("age_min"),
            age_max=cached.get("age_max"),
            sex=cached.get("sex"),
            region=cached.get("region"),
            specificity=PopulationSpecificity.EXPLICIT,
            confidence=0.95,
            extraction_method=ExtractionMethod.CLINICAL_TRIALS_API,
            raw_text_snippet=cached.get("raw_criteria"),
        )

    # -----------------------------------------------------------------------
    # QUERY API
    # -----------------------------------------------------------------------
    try:
        url = f"{CT_API_BASE}/{nct_id}?fields=protocolSection.eligibilityModule,protocolSection.contactsLocationsModule"
        req = urllib.request.Request(url, headers={
            "User-Agent": "MedVerifyAI/1.0",
            "Accept": "application/json",
        })

        with urllib.request.urlopen(req, timeout=CT_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        # Parse eligibility module
        protocol = data.get("protocolSection", {})
        eligibility = protocol.get("eligibilityModule", {})

        min_age_str = eligibility.get("minimumAge", "")
        max_age_str = eligibility.get("maximumAge", "")
        sex_str = eligibility.get("sex", "")
        criteria_text = eligibility.get("eligibilityCriteria", "")

        age_min = normalize_clinical_trials_age(min_age_str)
        age_max = normalize_clinical_trials_age(max_age_str)
        sex = normalize_clinical_trials_sex(sex_str)

        # Extract primary location country
        locations = protocol.get("contactsLocationsModule", {}).get("locations", [])
        region = None
        if locations:
            country = locations[0].get("country", "")
            if country:
                region = normalize_region(country) or country

        # Store in cache
        cache_entry = {
            "age_min": age_min,
            "age_max": age_max,
            "sex": sex,
            "region": region,
            "raw_criteria": criteria_text[:500] if criteria_text else None,
        }
        cache[nct_id] = cache_entry
        _save_cache(cache)

        logger.info(
            f"[ClinicalTrials] {nct_id}: age={age_min}-{age_max}, "
            f"sex={sex}, region={region}"
        )

        return PopulationProfile(
            age_min=age_min,
            age_max=age_max,
            sex=sex,
            region=region,
            specificity=PopulationSpecificity.EXPLICIT,
            confidence=0.95,
            extraction_method=ExtractionMethod.CLINICAL_TRIALS_API,
            raw_text_snippet=criteria_text[:200] if criteria_text else None,
        )

    except urllib.error.HTTPError as e:
        logger.warning(f"[ClinicalTrials] {nct_id}: HTTP {e.code} — trial not found.")
        cache[nct_id] = None  # Mark as not found
        _save_cache(cache)
        return None

    except (urllib.error.URLError, TimeoutError) as e:
        logger.warning(f"[ClinicalTrials] {nct_id}: Network error — {e}")
        # Don't cache network errors (transient)
        return None

    except Exception as e:
        logger.warning(f"[ClinicalTrials] {nct_id}: Unexpected error — {e}")
        return None
