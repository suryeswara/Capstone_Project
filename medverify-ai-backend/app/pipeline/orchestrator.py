"""
MedVerify inference orchestrator.

    1  claim normalisation + safety screen
    2  medical-claim gate            -> REJECT / UNSURE / ACCEPT
    3  retrieval (by mode)           -> static FAISS and/or live PubMed
    4  MonoT5 reranking              -> keep 3 fact-checks + 3 abstracts (live: 5)
    5  pair classification           -> DeBERTa + BiomedBERT ensemble
    6  meta-model (by mode)          -> static(18) | live(6) | hybrid stacker(7)
    7  confidence / abstention       -> VERIFIED or NEEDS_REVIEW
    8  explanation + suggestions     -> Gemini or template, faithfulness audit

Models and indexes are loaded once (``load``) and reused across requests.
The pipeline never invents evidence or scores: if an artifact is missing it
raises ``PipelineError`` and the API reports the pipeline as unavailable.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from . import explain as X
from .features import N_HYBRID, N_LIVE, N_STATIC
from .gemini import GeminiClient
from .meta import MetaModels, abstain
from .models import MonoT5Reranker, PairEnsemble, QueryEncoder, resolve_device
from .retrieval import PubMedClient, StaticStore, dedupe
from .schema import (
    CLASSES, STANCE_BY_CLASS, VERDICT_DISPLAY, Evidence, GateResult, MetaPrediction, Mode, Pool, PipelineError,
)
from .settings import PipelineSettings, get_settings

logger = logging.getLogger(__name__)

MAX_CLAIM_CHARS = 1000
ProgressFn = Callable[[str, str], None]

STAGE_LABELS = {
    "validate": "Validating the claim...",
    "gate": "Checking that this is a medical claim...",
    "retrieve": "Retrieving evidence ({mode} mode)...",
    "rerank": "Reranking evidence with MonoT5...",
    "classify": "Classifying each claim–evidence pair (DeBERTa + BiomedBERT)...",
    "meta": "Combining evidence with the {meta} meta-model...",
    "explain": "Writing the explanation and checking it against the evidence...",
}


class MedVerifyPipeline:
    def __init__(self, settings: Optional[PipelineSettings] = None):
        self.s = settings or get_settings()
        self._lock = threading.Lock()
        self.loaded = False
        self.load_error: Optional[str] = None
        self.encoder = self.reranker = self.classifier = self.store = self.meta = self.gate = None
        self.pubmed: Optional[PubMedClient] = None
        self.gemini: Optional[GeminiClient] = None
        self.guardrail = None

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def load(self) -> None:
        """Load every artifact once. Raises PipelineError with a precise reason."""
        with self._lock:
            if self.loaded:
                return
            t0 = time.time()
            try:
                from .gate import MedicalGate, ZeroShotNLI
                from app.services.safety_guardrail import get_safety_guardrail

                s = self.s
                device = resolve_device(s.device)
                logger.info("[pipeline] loading on %s from %s", device, s.artifact_root)
                self.gemini = GeminiClient(s.gemini_api_key, s.gemini_models, s.http_timeout)
                self.encoder = QueryEncoder(s.query_encoder, s.query_max_tokens, device)
                self.store = StaticStore(s.vector_db_dir, expected_dim=self.encoder.dim)
                self.meta = MetaModels(s.meta_dir)
                self.classifier = PairEnsemble(s.deberta_dir, s.biomed_dir, s.pair_max_length, device)
                self.reranker = MonoT5Reranker(s.reranker, device)
                nli = ZeroShotNLI(s.gate_nli, s.gate_nli_fallback, device)
                self.gate = MedicalGate(s.gate_dir, s.gate_strictness, s.gate_min_words, nli, self.gemini)
                self.pubmed = PubMedClient(s.ncbi_api_key, s.ncbi_email, s.http_timeout, s.live_fetch,
                                           s.live_min_abstract_chars)
                self.guardrail = get_safety_guardrail()
            except Exception as exc:
                self.load_error = f"{type(exc).__name__}: {exc}"
                logger.error("[pipeline] load failed: %s", self.load_error)
                raise PipelineError(self.load_error) from exc
            self.loaded, self.load_error = True, None
            logger.info("[pipeline] ready in %.1fs", time.time() - t0)

    def status(self) -> Dict:
        return {"loaded": self.loaded, "error": self.load_error, "artifact_root": str(self.s.artifact_root),
                "gemini": bool(self.gemini and self.gemini.available) if self.loaded else bool(self.s.gemini_api_key)}

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def normalise(raw: str) -> str:
        text = " ".join((raw or "").split())
        if not text:
            raise ValueError("The claim is empty.")
        if len(text) > MAX_CLAIM_CHARS:
            raise ValueError(f"The claim is longer than {MAX_CLAIM_CHARS} characters.")
        return text

    def _static_candidates(self, claim: str, qvec: np.ndarray):
        return self.store.search(claim, qvec, self.s.k_fc, self.s.k_abs, self.s.evidence_max_words)

    def _rerank_and_keep(self, claim: str, items: List[Evidence], keep: int) -> List[Evidence]:
        if not items:
            return []
        scores = self.reranker.score(claim, [e.text for e in items])
        for e, sc in zip(items, scores):
            e.rerank = float(sc)
        return sorted(items, key=lambda e: e.rerank, reverse=True)[:keep]

    def _classify(self, claim: str, items: Sequence[Evidence]) -> None:
        if not items:
            return
        probs = self.classifier.predict(claim, [e.text for e in items])
        for e, p in zip(items, probs):
            e.probs = p

    # ------------------------------------------------------------------
    # evidence + prediction core (no gate, no explanation) -- reused to
    # validate suggested rewordings in static mode
    # ------------------------------------------------------------------
    def _predict(self, claim: str, mode: Mode, qvec: np.ndarray, cand, progress: Optional[ProgressFn],
                 timings: Dict[str, float]):
        fc_c, abs_c, _ = cand
        knn_sims = [e.dense_sim for e in fc_c]
        knn_labels = [e.record_label for e in fc_c]

        live: List[Evidence] = []
        live_error: Optional[str] = None
        if mode in (Mode.LIVE, Mode.HYBRID):
            if progress:
                progress("retrieve", STAGE_LABELS["retrieve"].format(mode=mode.value))
            t = time.time()
            try:
                found = self.pubmed.search(claim)
                live_cand = dedupe(found, against=abs_c if mode is Mode.HYBRID else ())
            except Exception as exc:  # network failure -> documented fallback to static
                live_cand, live_error = [], str(exc)[:160]
                logger.warning("[pipeline] PubMed unavailable: %s", exc)
            timings["live_retrieval"] = round(time.time() - t, 3)
            if progress:
                progress("rerank", STAGE_LABELS["rerank"])
            t = time.time()
            live = self._rerank_and_keep(claim, live_cand, self.s.keep_live)
            timings["live_rerank"] = round(time.time() - t, 3)

        need_static = mode in (Mode.STATIC, Mode.HYBRID) or not live
        fc: List[Evidence] = []
        ab: List[Evidence] = []
        if need_static:
            if progress and mode is Mode.STATIC:
                progress("rerank", STAGE_LABELS["rerank"])
            t = time.time()
            fc = self._rerank_and_keep(claim, list(fc_c), self.s.keep_fc)
            ab = self._rerank_and_keep(claim, list(abs_c), self.s.keep_abs)
            timings["static_rerank"] = round(time.time() - t, 3)

        if progress:
            progress("classify", STAGE_LABELS["classify"])
        t = time.time()
        self._classify(claim, [*fc, *ab, *live])
        timings["classify"] = round(time.time() - t, 3)

        if progress:
            progress("meta", STAGE_LABELS["meta"].format(
                meta={"static": "static", "live": "live", "hybrid": "hybrid stacker"}[mode.value]))
        t = time.time()
        pred = self.meta.predict(mode, fc, ab, live, knn_sims, knn_labels, live_error)
        timings["meta"] = round(time.time() - t, 3)
        return pred, fc, ab, live

    def _verify_static_text(self, text: str) -> Optional[Dict]:
        """Static-mode verdict for a suggested rewording (used to validate suggestions)."""
        q = self.encoder.encode([text])[0]
        cand = self._static_candidates(text, q)
        pred, *_ = self._predict(text, Mode.STATIC, q, cand, None, {})
        return {
            "verdict": VERDICT_DISPLAY[pred.label], "label": pred.label,
            "confidence": round(pred.confidence, 4),
            "needsReview": abstain(pred.confidence, self.s.abstain_threshold),
        }

    # ------------------------------------------------------------------
    # public entry point
    # ------------------------------------------------------------------
    def verify(self, raw_claim: str, mode: Mode = Mode.STATIC, progress: Optional[ProgressFn] = None) -> Dict:
        if not self.loaded:
            raise PipelineError(self.load_error or "The inference pipeline has not been loaded.")
        t_start = time.time()
        timings: Dict[str, float] = {}
        mode = Mode.parse(mode.value if isinstance(mode, Mode) else mode)

        if progress:
            progress("validate", STAGE_LABELS["validate"])
        claim = self.normalise(raw_claim)

        safety = self.guardrail.analyze(claim)
        if not safety.is_safe_to_verify:
            return {
                "status": "REFUSED_SAFETY", "mode": mode.value, "claim": claim,
                "reason": safety.emergency_message or safety.refusal_reason or "Refused by the safety guardrail.",
                "disclaimer": safety.disclaimer, "disclaimer_severity": safety.disclaimer_severity.value,
                "safety_action": safety.action.value,
            }

        # claim vector + local abstract similarities feed the gate
        t = time.time()
        qvec = self.encoder.encode([claim])[0]
        cand = self._static_candidates(claim, qvec)
        timings["embed_and_search"] = round(time.time() - t, 3)

        if progress:
            progress("gate", STAGE_LABELS["gate"])
        t = time.time()
        gate: GateResult = self.gate.evaluate(claim, qvec, cand[2])
        timings["gate"] = round(time.time() - t, 3)
        gate_dict = {
            "decision": gate.decision, "accepted": gate.accepted, "pMedical": round(gate.p_medical, 4),
            "reason": gate.reason, "strictness": gate.strictness, "tauLo": gate.tau_lo, "tauHi": gate.tau_hi,
            "tieBreaker": gate.tie_breaker, "signals": {k: round(v, 4) for k, v in gate.signals.items()},
        }
        if not gate.accepted:
            return {"status": "REJECTED_NON_MEDICAL", "mode": mode.value, "claim": claim, "gate": gate_dict,
                    "reason": gate.reason, "elapsed_seconds": round(time.time() - t_start, 2)}

        pred, fc, ab, live = self._predict(claim, mode, qvec, cand, progress, timings)
        label, conf = pred.label, pred.confidence
        needs_review = abstain(conf, self.s.abstain_threshold)

        # evidence shown to the user = evidence the effective mode actually used
        if pred.mode_effective is Mode.LIVE:
            kept = list(live)
        elif pred.mode_effective is Mode.HYBRID:
            kept = [*fc, *ab, *live]
        else:
            kept = [*fc, *ab]
        for e in kept:
            e.ev_id = f"ev-{uuid.uuid4().hex[:8]}"

        if progress:
            progress("explain", STAGE_LABELS["explain"])
        t = time.time()
        explanation = X.run_explanation(
            claim, label, pred.probs, needs_review, kept, self.gemini, self.s.faithfulness_threshold,
            self.s.faithfulness_audit,
            static_label=CLASSES[int(np.argmax(pred.static_probs))] if pred.static_probs is not None else None,
            live_label=CLASSES[int(np.argmax(pred.live_probs))] if pred.live_probs is not None else None,
        )
        suggestions = X.build_suggestions(claim, label, kept)
        if self.s.validate_suggestions:
            X.validate_suggestions(suggestions, self._verify_static_text)
        timings["explain"] = round(time.time() - t, 3)

        rer = np.array([e.rerank for e in kept]) if kept else np.zeros(1)
        pair_conf = np.array([float(np.max(e.probs)) for e in kept]) if kept else np.zeros(1)
        w = np.maximum(rer, 1e-6)
        credibility = {
            "overall": round(100 * conf, 1),
            "evidenceConfidence": round(100 * float((pair_conf * w).sum() / w.sum()), 1),
            "consensusConfidence": round(100 * explanation["strength"]["agreement"], 1),
            "sourceQuality": round(100 * float(rer.mean()), 1),  # mean MonoT5 relevance of the kept evidence
            "faithfulnessConfidence": explanation["faithfulness_confidence"],
        }

        return {
            "status": "COMPLETED",
            "claim": claim,
            "mode": mode.value,
            "mode_effective": pred.mode_effective.value,
            "notes": pred.notes,
            "gate": gate_dict,
            "label": label,
            "verdict": VERDICT_DISPLAY[label],
            "probabilities": {c: round(float(p), 4) for c, p in zip(CLASSES, pred.probs)},
            "confidence": round(conf, 4),
            "needs_review": needs_review,
            "review_threshold": self.s.abstain_threshold,
            "credibility_breakdown": credibility,
            "evidence": [self._evidence_dict(e) for e in kept],
            "explanation": explanation,
            "suggestions": suggestions,
            "safety": {"action": safety.action.value, "flags": [f.value for f in safety.vulnerability_flags]},
            "disclaimer": self.guardrail.get_verdict_disclaimer(
                verdict=VERDICT_DISPLAY[label],
                vulnerability_flags=safety.vulnerability_flags or None)[0],
            "trace": {
                "feature_lengths": pred.feature_lengths,
                "expected_feature_lengths": {"static": N_STATIC, "live": N_LIVE, "hybrid": N_HYBRID},
                "timings_s": timings,
                "static_probs": None if pred.static_probs is None else
                {c: round(float(p), 4) for c, p in zip(CLASSES, pred.static_probs)},
                "live_probs": None if pred.live_probs is None else
                {c: round(float(p), 4) for c, p in zip(CLASSES, pred.live_probs)},
            },
            "versions": self.versions(),
            "elapsed_seconds": round(time.time() - t_start, 2),
        }

    def versions(self) -> Dict[str, str]:
        s = self.s
        return {
            "modelVersion": "medverify-v4 (DeBERTa-v3-base + BiomedBERT ensemble)",
            "retriever": s.query_encoder, "reranker": s.reranker, "gateNli": s.gate_nli,
            "metaModels": "static.joblib / live.joblib / hybrid_stack.joblib",
            "abstainThreshold": str(s.abstain_threshold),
            "explanationWriter": "gemini" if (self.gemini and self.gemini.available) else "template",
        }

    @staticmethod
    def _evidence_dict(e: Evidence) -> Dict:
        return {
            "id": e.ev_id, "uid": e.uid, "pool": e.pool.value, "source": e.source, "title": e.title,
            "text": e.text, "url": e.url, "year": e.year, "pmid": e.pmid, "doi": e.doi,
            "dense_sim": e.dense_sim, "rerank": round(e.rerank, 4),
            "probs": {c: round(float(p), 4) for c, p in zip(CLASSES, e.probs)},
            "stance": STANCE_BY_CLASS[e.stance_class],
        }


# ---------------------------------------------------------------------------
# module-level singleton
# ---------------------------------------------------------------------------
_pipeline: Optional[MedVerifyPipeline] = None
_pipeline_lock = threading.Lock()


def get_pipeline() -> MedVerifyPipeline:
    global _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            _pipeline = MedVerifyPipeline()
        return _pipeline
