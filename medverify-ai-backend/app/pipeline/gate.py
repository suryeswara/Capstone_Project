"""
Medical-claim gate (report 4.1).

Nine signals -> logistic-regression stacker -> P(medical) -> three zones:

    P >= tau_hi   ACCEPT   (tau_hi set for <=1 % / 3 % / 8 % false accepts:
                            strict / balanced / lenient)
    P <  tau_lo   REJECT   (tau_lo keeps 95 % of medical claims)
    otherwise     UNSURE   -> Gemini tie-breaker if configured, else REJECT

The gate is fail-closed: missing artifacts, a model error, or a tie-breaker
that cannot decide all end in rejection, never in a silent accept.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .gemini import GeminiClient
from .schema import ArtifactError, GateResult

logger = logging.getLogger(__name__)

SIGNALS = (
    "probe_logit", "nli_medicine", "nli_topic_share", "nli_max_other_topic",
    "nli_only_medical_evidence", "lex_strong", "lex_weak", "abs_sim_max", "abs_sim_mean",
)

MEDICINE_HYPOTHESIS = "This text is about medicine, health, diseases, drugs, vaccines, nutrition or medical treatment."
OTHER_TOPICS = (
    "sports", "politics", "business", "technology", "entertainment",
    "geography or history", "science", "food", "everyday life",
)
ONLY_MEDICAL_HYPOTHESIS = "The truth of this claim can only be checked against medical or health-science evidence."

# Lexicon fallback. The stacker was trained on the notebook's lexicons, so the
# REAL lists should be shipped as gate/lexicon.json ({"strong": [...], "weak": [...]}).
# Matching is by word prefix, so "vaccin" covers vaccine / vaccination / vaccinated.
DEFAULT_STRONG = (
    "vaccin", "cancer", "insulin", "ivermectin", "diabet", "tumor", "tumour", "chemotherap", "antibiotic",
    "covid", "coronavirus", "hiv", "aids", "stroke", "cholesterol", "hypertension", "dementia", "alzheimer",
    "asthma", "pneumonia", "influenza", "measles", "autism", "antiviral", "immun", "pathogen", "symptom",
    "dosage", "clinical trial", "statin", "metformin", "opioid", "pharmac", "medic", "disease", "infect",
)
DEFAULT_WEAK = (
    "hospital", "doctor", "heart", "blood", "health", "diet", "pain", "sleep", "weight", "patient", "virus",
    "cure", "treat", "drug", "therap", "surgery", "clinic", "nurse", "pill", "supplement", "vitamin",
)


def count_terms(text: str, prefixes, cap: int = 3) -> int:
    low = (text or "").lower()
    n = sum(1 for p in prefixes if re.search(r"(?<![a-z])" + re.escape(p), low))
    return min(n, cap)


class ZeroShotNLI:
    """Premise = claim; returns P(hypothesis is entailed) per hypothesis."""

    def __init__(self, primary: str, fallback: str, device: str):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch, self.device = torch, device
        last: Optional[Exception] = None
        for name in (primary, fallback):
            try:
                self.name = name
                self.tok = AutoTokenizer.from_pretrained(name)
                self.model = AutoModelForSequenceClassification.from_pretrained(name).to(device).eval()
                break
            except Exception as exc:  # pragma: no cover - network / cache dependent
                last = exc
                logger.warning("Gate NLI model %s unavailable: %s", name, exc)
        else:
            raise ArtifactError(f"No gate NLI model could be loaded: {last}")
        l2i = {k.lower(): v for k, v in self.model.config.label2id.items()}
        if "entailment" not in l2i:
            raise ArtifactError(f"{self.name}: no 'entailment' label in {l2i}")
        self.ent = l2i["entailment"]
        # binary models (entailment / not_entailment) vs MNLI (entailment / neutral / contradiction)
        self.other = l2i.get("not_entailment", l2i.get("contradiction"))
        if self.other is None:
            raise ArtifactError(f"{self.name}: cannot find a contrast label in {l2i}")

    def entail_probs(self, premise: str, hypotheses: List[str]) -> np.ndarray:
        torch = self._torch
        enc = self.tok([premise] * len(hypotheses), hypotheses, truncation=True, padding=True,
                       max_length=256, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**enc).logits.float()
        two = torch.stack([logits[:, self.ent], logits[:, self.other]], dim=1)
        return torch.softmax(two, dim=1)[:, 0].cpu().numpy().astype(np.float64)


class MedicalGate:
    def __init__(self, gate_dir: Path, strictness: str, min_words: int, nli: ZeroShotNLI,
                 gemini: Optional[GeminiClient] = None):
        import joblib

        d = Path(gate_dir)
        needed = ("probe.joblib", "stacker.joblib", "gate_config.json")
        missing = [f for f in needed if not (d / f).exists()]
        if missing:
            raise ArtifactError(f"gate artifacts missing in {d}: {', '.join(missing)}")
        self.probe = joblib.load(d / "probe.joblib")
        self.stacker = joblib.load(d / "stacker.joblib")
        cfg = json.loads((d / "gate_config.json").read_text())
        self.tau_lo = float(cfg["tau_lo"])
        taus = cfg["tau_hi"]
        if strictness not in taus:
            raise ArtifactError(f"gate_config.json has no tau_hi for strictness {strictness!r} ({list(taus)})")
        self.tau_hi = float(taus[strictness])
        self.strictness = strictness
        self.order = tuple(cfg.get("feature_order", SIGNALS))
        if set(self.order) != set(SIGNALS):
            raise ArtifactError(f"gate feature_order must be a permutation of {SIGNALS}")
        n_in = getattr(self.stacker, "n_features_in_", len(self.order))
        if n_in != len(self.order):
            raise ArtifactError(f"gate stacker expects {n_in} features, config lists {len(self.order)}")
        lex = d / "lexicon.json"
        if lex.exists():
            data = json.loads(lex.read_text())
            self.strong, self.weak = tuple(data["strong"]), tuple(data["weak"])
        else:
            self.strong, self.weak = DEFAULT_STRONG, DEFAULT_WEAK
            logger.warning("gate/lexicon.json not found: using built-in lexicon; ship the notebook's lexicon "
                           "for exact parity with the trained stacker")
        self.min_words, self.nli, self.gemini = min_words, nli, gemini

    # ------------------------------------------------------------------
    def signals(self, claim: str, qvec: np.ndarray, abs_sims: List[float]) -> Dict[str, float]:
        probe_logit = float(self.probe.decision_function(qvec.reshape(1, -1))[0])
        topics = [MEDICINE_HYPOTHESIS] + [f"This text is about {t}." for t in OTHER_TOPICS] + [ONLY_MEDICAL_HYPOTHESIS]
        p = self.nli.entail_probs(claim, topics)
        p_med, p_other, p_only = float(p[0]), p[1:1 + len(OTHER_TOPICS)], float(p[-1])
        share = p_med / max(p_med + float(p_other.sum()), 1e-9)
        top = sorted(abs_sims, reverse=True)[:3]
        return {
            "probe_logit": probe_logit,
            "nli_medicine": p_med,
            "nli_topic_share": share,
            "nli_max_other_topic": float(p_other.max()),
            "nli_only_medical_evidence": p_only,
            "lex_strong": float(count_terms(claim, self.strong)),
            "lex_weak": float(count_terms(claim, self.weak)),
            "abs_sim_max": float(top[0]) if top else 0.0,
            "abs_sim_mean": float(np.mean(top)) if top else 0.0,
        }

    def p_medical(self, sig: Dict[str, float]) -> float:
        x = np.array([[sig[k] for k in self.order]], dtype=np.float64)
        proba = self.stacker.predict_proba(x)[0]
        classes = list(getattr(self.stacker, "classes_", [0, 1]))
        return float(proba[classes.index(1)] if 1 in classes else proba[-1])

    def decide(self, p: float) -> str:
        return decide_zone(p, self.tau_lo, self.tau_hi)

    def evaluate(self, claim: str, qvec: np.ndarray, abs_sims: List[float]) -> GateResult:
        if len(claim.split()) < self.min_words:
            return GateResult(False, 0.0, "REJECT", f"Claim has fewer than {self.min_words} words.", self.strictness)
        sig = self.signals(claim, qvec, abs_sims)
        p = self.p_medical(sig)
        zone = self.decide(p)
        base = dict(p_medical=p, strictness=self.strictness, tau_lo=self.tau_lo, tau_hi=self.tau_hi, signals=sig)
        if zone == "ACCEPT":
            return GateResult(True, decision="ACCEPT", reason="Recognised as a medical claim.", **base)
        if zone == "REJECT":
            return GateResult(False, decision="REJECT", reason="This does not look like a medical or health claim.", **base)
        # UNSURE band
        verdict = self._tie_break(claim)
        if verdict is True:
            return GateResult(True, decision="UNSURE_ACCEPT", tie_breaker="gemini",
                              reason="Borderline; accepted by the tie-breaker.", **base)
        if verdict is False:
            return GateResult(False, decision="UNSURE_REJECT", tie_breaker="gemini",
                              reason="Borderline; rejected by the tie-breaker.", **base)
        return GateResult(False, decision="UNSURE_REJECT",
                          reason="Borderline and no tie-breaker available (fail-closed policy).", **base)

    def _tie_break(self, claim: str) -> Optional[bool]:
        if not (self.gemini and self.gemini.available):
            return None
        out = self.gemini.generate_json(
            'Decide whether the text below is a health or medical claim whose truth can be checked against '
            'medical literature. Answer with JSON {"is_medical_claim": true|false}.\n\nText: ' + claim,
            temperature=0.0,
        )
        if isinstance(out, dict) and isinstance(out.get("is_medical_claim"), bool):
            return out["is_medical_claim"]
        return None


def decide_zone(p: float, tau_lo: float, tau_hi: float) -> str:
    """Pure zone logic, kept separate so it is trivially testable."""
    if p >= tau_hi:
        return "ACCEPT"
    if p < tau_lo:
        return "REJECT"
    return "UNSURE"
