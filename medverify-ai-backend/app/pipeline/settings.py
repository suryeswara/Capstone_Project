"""
MedVerify inference pipeline settings.

Everything is overridable through environment variables so the same code runs
locally, in Docker, and on a server. Nothing here retrains or rebuilds an
artifact: the pipeline only *loads* the final v4 artifacts.

Expected artifact layout (see docs/INTEGRATION.md):

    <ARTIFACT_ROOT>/
        models/deberta/            fine-tuned DeBERTa-v3-base pair classifier
        models/biomed/             fine-tuned BiomedBERT pair classifier
        vector_db/                 fc_index.faiss, abs_index.faiss,
                                   fc_ids.npy, abs_ids.npy, records.jsonl
        meta_models/               static.joblib, live.joblib, hybrid_stack.joblib
        gate/                      probe.joblib, stacker.joblib, gate_config.json
                                   (optional: lexicon.json)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parents[2]


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    return default if v is None else v.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class PipelineSettings:
    # ---- artifact locations -------------------------------------------------
    artifact_root: Path = field(
        default_factory=lambda: Path(os.getenv("MEDVERIFY_ARTIFACT_ROOT", _BACKEND_ROOT / "artifacts"))
    )

    @property
    def deberta_dir(self) -> Path:
        return Path(os.getenv("MEDVERIFY_DEBERTA_DIR", self.artifact_root / "models" / "deberta"))

    @property
    def biomed_dir(self) -> Path:
        return Path(os.getenv("MEDVERIFY_BIOMED_DIR", self.artifact_root / "models" / "biomed"))

    @property
    def vector_db_dir(self) -> Path:
        return Path(os.getenv("MEDVERIFY_VECTOR_DB_DIR", self.artifact_root / "vector_db"))

    @property
    def meta_dir(self) -> Path:
        return Path(os.getenv("MEDVERIFY_META_DIR", self.artifact_root / "meta_models"))

    @property
    def gate_dir(self) -> Path:
        return Path(os.getenv("MEDVERIFY_GATE_DIR", self.artifact_root / "gate"))

    # ---- model names downloaded by name at run time -------------------------
    query_encoder: str = "ncbi/MedCPT-Query-Encoder"
    reranker: str = "castorini/monot5-base-med-msmarco"
    gate_nli: str = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
    gate_nli_fallback: str = "facebook/bart-large-mnli"

    # ---- retrieval / rerank sizes (report sections 4.2-4.4) ------------------
    query_max_tokens: int = 64
    k_fc: int = 10
    k_abs: int = 10
    keep_fc: int = 3
    keep_abs: int = 3
    live_fetch: int = 25
    keep_live: int = 5
    live_min_abstract_chars: int = 150
    pair_max_length: int = 384
    evidence_max_words: int = 150

    # ---- decision rules ------------------------------------------------------
    abstain_threshold: float = 0.5859  # report 4.6: chosen on validation
    faithfulness_threshold: float = 0.70
    gate_strictness: str = field(default_factory=lambda: os.getenv("MEDVERIFY_GATE_STRICTNESS", "strict"))
    gate_min_words: int = 3

    # ---- external services ---------------------------------------------------
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_models: tuple = field(
        default_factory=lambda: tuple(
            m.strip() for m in os.getenv("GEMINI_MODELS", "gemini-2.5-flash,gemini-2.0-flash").split(",") if m.strip()
        )
    )
    ncbi_api_key: str = field(default_factory=lambda: os.getenv("NCBI_API_KEY", ""))
    ncbi_email: str = field(default_factory=lambda: os.getenv("NCBI_EMAIL", ""))
    http_timeout: float = 20.0

    # ---- behaviour flags -----------------------------------------------------
    faithfulness_audit: bool = field(default_factory=lambda: _env_bool("MEDVERIFY_FAITHFULNESS_AUDIT", True))
    validate_suggestions: bool = field(default_factory=lambda: _env_bool("MEDVERIFY_VALIDATE_SUGGESTIONS", True))
    device: str = field(default_factory=lambda: os.getenv("MEDVERIFY_DEVICE", "auto"))


def get_settings() -> PipelineSettings:
    return PipelineSettings()
