"""
MedVerify AI — Final Evaluation Suite (experiment-v2) Configuration
===================================================================
Central configuration for all 16 evaluation modules, defining paths,
label spaces, clinical reliability hierarchies, and threshold parameters.
"""

import os
from typing import Dict, List

# ---------------------------------------------------------------------------
# DIRECTORY PATHS
# ---------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
BACKEND_DIR = os.path.join(PROJECT_ROOT, "medverify-ai-backend")

DATASETS_DIR = os.path.join(PROJECT_ROOT, "med_datasets")
SPLITS_DIR = os.path.join(DATASETS_DIR, "splits")
EVAL_DATASETS_DIR = os.path.join(DATASETS_DIR, "evaluation")
PROCESSED_DATASETS_DIR = os.path.join(DATASETS_DIR, "processed")

EXP_DATASETS_DIR = os.path.join(CURRENT_DIR, "datasets")
RESULTS_DIR = os.path.join(CURRENT_DIR, "results")

# Ensure required output directories exist
os.makedirs(EXP_DATASETS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# FILE PATHS
# ---------------------------------------------------------------------------
TRAIN_FROZEN_PATH = os.path.join(SPLITS_DIR, "train_frozen.json")
VAL_FROZEN_PATH = os.path.join(SPLITS_DIR, "val_frozen.json")
TEST_FROZEN_PATH = os.path.join(SPLITS_DIR, "test_frozen.json")
SPLITS_MANIFEST_PATH = os.path.join(SPLITS_DIR, "splits_manifest.json")

PUBHEALTH_VERIFIED_PATH = os.path.join(PROCESSED_DATASETS_DIR, "pubhealth_verified_claims.json")
COAID_MANIFEST_PATH = os.path.join(PROCESSED_DATASETS_DIR, "phase1_disease_claims_manifest.json")
FAITHFULNESS_BENCHMARK_PATH = os.path.join(EVAL_DATASETS_DIR, "faithfulness_benchmark_200.json")

VECTOR_STORE_DIR = os.path.join(PROJECT_ROOT, "vector_store")
FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "faiss_index.bin")
VECTOR_META_PATH = os.path.join(VECTOR_STORE_DIR, "vector_metadata.json")

BIOBERT_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "biobert_disease_classifier")

# ---------------------------------------------------------------------------
# DISEASE CATEGORIES (22 Phase 1 Categories)
# ---------------------------------------------------------------------------
DISEASE_CATEGORIES: List[str] = [
    "COVID-19",
    "General Cancer",
    "Influenza",
    "Reproductive Health / Abortion",
    "Breast Cancer",
    "HIV/AIDS",
    "Ebola Virus",
    "Alzheimers Disease",
    "Prostate Cancer",
    "Diabetes",
    "Obesity & Weight Management",
    "Opioids & Pain Management",
    "Measles / MMR",
    "Depression",
    "Heart Disease",
    "Smoking & Tobacco",
    "Pregnancy & Maternal Health",
    "Heart Attack",
    "Stroke",
    "Lung Cancer",
    "Autism Spectrum Disorder",
    "Vaccination",
]

# ---------------------------------------------------------------------------
# LABEL NORMALIZATION & VERDICT CLASSES
# ---------------------------------------------------------------------------
# Target 4-class verdict space
VERDICT_CLASSES = ["Supported", "Contradicted", "Insufficient", "Uncertain"]

SPEC_TO_VERDICT_MAP = {
    "TRUE": "Supported",
    "FALSE": "Contradicted",
    "MIXTURE": "Uncertain",
    "UNPROVEN": "Insufficient",
}

PUBHEALTH_TO_VERDICT_MAP = {
    "true": "Supported",
    "True": "Supported",
    "TRUE": "Supported",
    "supported": "Supported",
    "Supported": "Supported",
    "false": "Contradicted",
    "False": "Contradicted",
    "FALSE": "Contradicted",
    "contradicted": "Contradicted",
    "Contradicted": "Contradicted",
    "mixture": "Uncertain",
    "Mixture": "Uncertain",
    "MIXTURE": "Uncertain",
    "mixed": "Uncertain",
    "Mixed": "Uncertain",
    "unproven": "Insufficient",
    "Unproven": "Insufficient",
    "UNPROVEN": "Insufficient",
    "insufficient": "Insufficient",
}

# ---------------------------------------------------------------------------
# EVIDENCE RELIABILITY HIERARCHY (R_i)
# ---------------------------------------------------------------------------
SOURCE_TIERS: Dict[str, float] = {
    "WHO Guideline": 1.00,
    "CDC Guideline": 0.98,
    "WHO/CDC Guideline": 0.99,
    "Systematic Review / Meta-Analysis": 0.95,
    "Randomized Controlled Trial (RCT)": 0.90,
    "Cohort Study / Observational": 0.75,
    "Case Report": 0.55,
    "Preprint (bioRxiv/medRxiv)": 0.30,
    "PubMed Article": 0.80,
    "Unverified Blog / Opinion": 0.05,
}

DISEASE_DECAY_RATES: Dict[str, float] = {
    "Vaccination": 0.10,
    "Cardiovascular Disease": 0.05,
    "Heart Disease": 0.05,
    "Diabetes": 0.03,
    "COVID-19": 0.12,
}

# ---------------------------------------------------------------------------
# CONSENSUS SCORING PARAMETERS
# ---------------------------------------------------------------------------
# Consensus score in [-1.0, 1.0] thresholds:
#   C >= 0.25         -> Supported
#   -0.20 < C < 0.25  -> Insufficient / Uncertain (depending on neutral mass & polarization)
#   C <= -0.20        -> Contradicted
CONSENSUS_THRESHOLDS = {
    "SUPPORTED": 0.25,
    "CONTRADICTED": -0.20,
    "NEUTRAL_MASS_CUTOFF": 0.70,
}

RANDOM_SEED = 42
DEFAULT_EVAL_SAMPLE = None  # None = full dataset (e.g. 614 test claims)
