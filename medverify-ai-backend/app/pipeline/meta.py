"""
Meta-model routing: one verification mode -> one claim-level probability vector.

    static   evidence: local KB            model: static.joblib        (18 features)
    live     evidence: PubMed only         model: live.joblib          (6 features)
    hybrid   evidence: both                model: static + live -> hybrid_stack.joblib (7 features)

If PubMed is unreachable or returns nothing, live and hybrid fall back to the
static path and say so in ``notes`` (report section 5).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from . import features as F
from .schema import CLASSES, ArtifactError, Evidence, MetaPrediction, Mode, PipelineError

logger = logging.getLogger(__name__)


def _class_order(model, where: str) -> list:
    """Column indices of ``predict_proba`` that give CLASSES order."""
    cls = list(getattr(model, "classes_", []))
    if len(cls) != len(CLASSES):
        raise ArtifactError(f"{where}: classes_={cls!r}; expected 3 classes")
    if all(isinstance(c, (int, np.integer)) for c in cls):
        if sorted(int(c) for c in cls) != [0, 1, 2]:
            raise ArtifactError(f"{where}: integer classes_ {cls} must be 0,1,2 (SUPPORT, CONTRADICT, NEI)")
        return [[int(c) for c in cls].index(i) for i in range(3)]
    from .models import _LABEL_ALIASES

    try:
        canon = [_LABEL_ALIASES[str(c).strip().upper()] for c in cls]
    except KeyError as exc:
        raise ArtifactError(f"{where}: unrecognised class label {exc} in classes_={cls}") from exc
    if sorted(canon) != sorted(CLASSES):
        raise ArtifactError(f"{where}: classes_={cls} do not cover {CLASSES}")
    return [canon.index(c) for c in CLASSES]


class MetaModels:
    EXPECTED = {"static": F.N_STATIC, "live": F.N_LIVE, "hybrid_stack": F.N_HYBRID}

    def __init__(self, directory: Path):
        import joblib

        d = Path(directory)
        self.models, self.order = {}, {}
        for name, n_expected in self.EXPECTED.items():
            path = d / f"{name}.joblib"
            if not path.exists():
                raise ArtifactError(f"meta-model not found: {path}")
            m = joblib.load(path)
            n_in = getattr(m, "n_features_in_", None)
            if n_in is None and hasattr(m, "steps"):  # sklearn Pipeline
                n_in = getattr(m.steps[0][1], "n_features_in_", None)
            if n_in is not None and int(n_in) != n_expected:
                raise ArtifactError(f"{path.name} expects {n_in} features, the pipeline builds {n_expected}")
            self.models[name] = m
            self.order[name] = _class_order(m, path.name)

    def _proba(self, name: str, x: np.ndarray) -> np.ndarray:
        raw = self.models[name].predict_proba(x.reshape(1, -1))[0]
        p = raw[self.order[name]]
        s = float(p.sum())
        return p / s if s > 0 else p

    # ------------------------------------------------------------------
    def predict(
        self,
        mode: Mode,
        fc: Sequence[Evidence],
        ab: Sequence[Evidence],
        live: Sequence[Evidence],
        knn_sims: Sequence[float],
        knn_labels: Sequence[Optional[str]],
        live_error: Optional[str] = None,
    ) -> MetaPrediction:
        notes = []
        lengths = {}

        if mode is Mode.LIVE:
            if live:
                x = F.live_features(live)
                lengths["live"] = x.shape[0]
                return MetaPrediction(self._proba("live", x), mode, Mode.LIVE, lengths, notes, live_probs=None)
            notes.append(self._fallback_note(live_error))
            mode_eff = Mode.STATIC
        else:
            mode_eff = mode

        # static path (also the fallback)
        if not (fc or ab):
            raise PipelineError("No local evidence was retrieved; cannot build the static feature vector.")
        xs = F.static_features(fc, ab, knn_sims, knn_labels)
        lengths["static"] = xs.shape[0]
        p_static = self._proba("static", xs)

        if mode_eff is Mode.STATIC:
            return MetaPrediction(p_static, mode, Mode.STATIC, lengths, notes, static_probs=p_static)

        # hybrid
        if not live:
            notes.append(self._fallback_note(live_error))
            return MetaPrediction(p_static, mode, Mode.STATIC, lengths, notes, static_probs=p_static)
        xl = F.live_features(live)
        p_live = self._proba("live", xl)
        xh = F.hybrid_features(p_static, p_live, live)
        lengths.update(live=xl.shape[0], hybrid=xh.shape[0])
        return MetaPrediction(self._proba("hybrid_stack", xh), mode, Mode.HYBRID, lengths, notes,
                              static_probs=p_static, live_probs=p_live)

    @staticmethod
    def _fallback_note(err: Optional[str]) -> str:
        why = f" ({err})" if err else ""
        return f"PubMed returned no usable evidence{why}; fell back to the static knowledge-base result."


def abstain(confidence: float, threshold: float) -> bool:
    """True -> NEEDS_REVIEW (report 4.6: confidence < 0.5859)."""
    return confidence < threshold
