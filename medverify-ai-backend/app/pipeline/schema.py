"""Shared constants and data containers for the MedVerify inference pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

# Canonical class order used everywhere inside the pipeline. Classifier outputs
# and meta-model outputs are re-ordered into this order at load time, so no
# other module ever depends on an artifact's internal label ids.
CLASSES = ("SUPPORT", "CONTRADICT", "NEI")

STANCE_BY_CLASS = {"SUPPORT": "supporting", "CONTRADICT": "contradicting", "NEI": "neutral"}
VERDICT_DISPLAY = {"SUPPORT": "Supported", "CONTRADICT": "Contradicted", "NEI": "Insufficient Evidence"}

# PUBHEALTH / HealthVer raw labels -> 3-class labels (report section 1).
RAW_LABEL_TO_CLASS = {
    "true": "SUPPORT",
    "support": "SUPPORT",
    "supports": "SUPPORT",
    "supported": "SUPPORT",
    "false": "CONTRADICT",
    "contradict": "CONTRADICT",
    "contradiction": "CONTRADICT",
    "refute": "CONTRADICT",
    "refutes": "CONTRADICT",
    "mixture": "NEI",
    "unproven": "NEI",
    "nei": "NEI",
    "neutral": "NEI",
}


class Mode(str, Enum):
    STATIC = "static"
    LIVE = "live"
    HYBRID = "hybrid"

    @classmethod
    def parse(cls, value: Optional[str]) -> "Mode":
        if value is None or value == "":
            return cls.STATIC
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            raise ValueError(f"Unknown verification mode {value!r}; expected static, live or hybrid") from exc


class Pool(str, Enum):
    FACT_CHECK = "fc"   # PUBHEALTH fact-checks (local)
    ABSTRACT = "abs"    # HealthVer research abstracts (local)
    LIVE = "live"       # PubMed abstracts fetched at query time


class PipelineError(RuntimeError):
    """Raised when the pipeline cannot run without fabricating results."""


class ArtifactError(PipelineError):
    """A required artifact is missing or does not have the expected shape."""


@dataclass
class Evidence:
    uid: str
    pool: Pool
    source: str                 # "PUBHEALTH" | "HEALTHVER" | "PUBMED"
    title: str
    text: str                   # claim-aware evidence text fed to the pair classifier
    url: str = ""
    year: Optional[int] = None
    pmid: Optional[str] = None
    doi: Optional[str] = None
    dense_sim: Optional[float] = None   # inner product with the claim vector (static only)
    rerank: float = 0.0                 # MonoT5 P("true")
    probs: Optional[np.ndarray] = None  # pair-classifier probs in CLASSES order
    record_label: Optional[str] = None  # 3-class label of a local fact-check record
    ev_id: str = ""                     # per-verification id shown to the user (ev-xxxxxx)

    @property
    def stance_class(self) -> str:
        if self.probs is None:
            raise PipelineError("Evidence has not been classified yet")
        return CLASSES[int(np.argmax(self.probs))]


@dataclass
class GateResult:
    accepted: bool
    p_medical: float
    decision: str               # ACCEPT | REJECT | UNSURE_ACCEPT | UNSURE_REJECT
    reason: str
    strictness: str
    tau_lo: Optional[float] = None
    tau_hi: Optional[float] = None
    signals: Dict[str, float] = field(default_factory=dict)
    tie_breaker: Optional[str] = None  # "gemini" if used


@dataclass
class MetaPrediction:
    probs: np.ndarray           # CLASSES order
    mode_requested: Mode
    mode_effective: Mode
    feature_lengths: Dict[str, int]
    notes: List[str] = field(default_factory=list)
    static_probs: Optional[np.ndarray] = None
    live_probs: Optional[np.ndarray] = None

    @property
    def label(self) -> str:
        return CLASSES[int(np.argmax(self.probs))]

    @property
    def confidence(self) -> float:
        return float(np.max(self.probs))
