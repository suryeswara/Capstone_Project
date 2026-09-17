"""
MedVerify AI — Final Evaluation Suite (experiment-v2) Data Loader
=================================================================
Utilities for loading, validating, and filtering dataset splits.
"""

import json
import os
import re
from typing import Dict, List, Optional, Tuple
from .config import (
    TRAIN_FROZEN_PATH,
    VAL_FROZEN_PATH,
    TEST_FROZEN_PATH,
    SPLITS_MANIFEST_PATH,
    COAID_MANIFEST_PATH,
    PUBHEALTH_VERIFIED_PATH,
    FAITHFULNESS_BENCHMARK_PATH,
    PUBHEALTH_TO_VERDICT_MAP,
    SPEC_TO_VERDICT_MAP,
    DISEASE_CATEGORIES,
)


def normalize_text(text: str) -> str:
    """Normalize text for consistent comparison and deduplication."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[\"\']', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def load_json(filepath: str):
    """Load JSON from filepath."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, filepath: str, indent: int = 2):
    """Save data to JSON with UTF-8 encoding."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def get_normalized_verdict(raw_verdict: str) -> str:
    """Map raw verdict strings to standardized 4-class verdict space."""
    if not raw_verdict:
        return "Insufficient"
    raw_clean = str(raw_verdict).strip().lower()
    return PUBHEALTH_TO_VERDICT_MAP.get(raw_clean, "Insufficient")


def load_frozen_splits() -> Dict[str, List[Dict]]:
    """Load all three frozen splits (train, val, test)."""
    splits = {
        "train": load_json(TRAIN_FROZEN_PATH),
        "val": load_json(VAL_FROZEN_PATH),
        "test": load_json(TEST_FROZEN_PATH),
    }
    return splits


def load_test_claims(sample_size: Optional[int] = None) -> List[Dict]:
    """
    Load frozen test set claims with standardized verdict labels.
    If sample_size is provided, returns first N claims deterministically.
    """
    records = load_json(TEST_FROZEN_PATH)
    cleaned = []
    for r in records:
        verdict = get_normalized_verdict(r.get("fact_check_verdict", ""))
        cleaned.append({
            "claim_id": r.get("claim_id", ""),
            "claim_text": r.get("claim_text", "").strip(),
            "disease_category": r.get("disease_category", ""),
            "label_idx": r.get("label", -1),
            "raw_verdict": r.get("fact_check_verdict", ""),
            "standard_verdict": verdict,
            "dataset_source": r.get("dataset_source", "PubHealth"),
            "split": r.get("split", "test"),
        })
    if sample_size and sample_size < len(cleaned):
        return cleaned[:sample_size]
    return cleaned


def load_coaid_claims(vaccination_only: bool = False) -> List[Dict]:
    """Load real CoAID claims from the processed manifest."""
    manifest = load_json(COAID_MANIFEST_PATH)
    coaid_claims = [r for r in manifest if r.get("dataset_source") == "CoAID"]
    cleaned = []
    for r in coaid_claims:
        if vaccination_only and r.get("disease_category") != "Vaccination":
            continue
        verdict = get_normalized_verdict(r.get("fact_check_verdict", ""))
        cleaned.append({
            "claim_id": r.get("claim_id"),
            "claim_text": r.get("claim_text", "").strip().strip('"'),
            "disease_category": r.get("disease_category", "Vaccination"),
            "raw_verdict": r.get("fact_check_verdict", ""),
            "standard_verdict": verdict,
            "dataset_source": "CoAID",
            "evidence_sources": r.get("evidence_sources", []),
        })
    return cleaned


def load_faithfulness_benchmark() -> List[Dict]:
    """Load the 200-item faithfulness and BioScope certainty benchmark."""
    data = load_json(FAITHFULNESS_BENCHMARK_PATH)
    return data.get("dataset", [])
