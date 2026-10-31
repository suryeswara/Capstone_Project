"""
Explanation stage (report 4.7). Runs only for accepted claims and never
changes the verdict.

1. evidence strength  - heuristic: strong / moderate / weak
2. explanation writer - Gemini (grounded, JSON) or a rule-based template
3. better-claim suggestions - hedged / myth-check / source-backed / missing specifics
4. suggestion validation - each rewording goes back through the static verifier
5. faithfulness audit - every explanation sentence is checked against the
   evidence it cites: (a) the pair classifier's probability for the class the
   sentence asserts must be >= 0.70, (b) the sentence may not claim more
   certainty than its cited evidence (BioScope guard).
"""

from __future__ import annotations

import logging
import re
from typing import Callable, Dict, List, Optional, Sequence
from urllib.parse import quote_plus

import numpy as np

from .certainty import get_certainty_level
from .gemini import GeminiClient
from .retrieval import PubMedClient
from .schema import CLASSES, VERDICT_DISPLAY, Evidence

logger = logging.getLogger(__name__)

ABSOLUTE_WORDS = ("cures", "cure", "always", "never", "all", "every", "100%", "guarantee", "guarantees",
                  "proven", "eliminates", "permanently", "completely")
SOURCE_LABEL = {"PUBHEALTH": "Fact-check", "HEALTHVER": "Research abstract", "PUBMED": "Live PubMed abstract"}

HEDGE_RULES = (  # one substitution only (report 4.7)
    (r"\bcures\b", "may help treat"), (r"\bcure\b", "may help treat"),
    (r"\bprevents\b", "may help prevent"), (r"\bprevent\b", "may help prevent"),
    (r"\bcauses\b", "may be associated with"), (r"\bcause\b", "may be associated with"),
    (r"\breduces\b", "may reduce"), (r"\breduce\b", "may reduce"),
    (r"\bkills\b", "may inhibit"), (r"\beliminates\b", "may reduce"),
    (r"\bguarantees\b", "may improve"), (r"\bimproves\b", "may improve"),
    (r"\balways\b", "often"), (r"\bnever\b", "rarely"), (r"\bproven\b", "suggested"),
)


# ---------------------------------------------------------------------------
# 1. evidence strength
# ---------------------------------------------------------------------------

def evidence_strength(probs: np.ndarray, kept: Sequence[Evidence]) -> Dict:
    """Strong / moderate / weak from confidence, weighted agreement, dissent and source count."""
    label_idx = int(np.argmax(probs))
    conf = float(probs[label_idx])
    n = len(kept)
    if n == 0:
        return {"level": "weak", "agreement": 0.0, "opposing": 0, "sources": 0, "confidence": conf}
    w = np.array([max(e.rerank, 1e-6) for e in kept])
    agree = float(sum(wi for wi, e in zip(w, kept) if int(np.argmax(e.probs)) == label_idx) / w.sum())
    opposing = 0
    if label_idx != 2:  # SUPPORT / CONTRADICT: count the opposite polarity
        other = 1 - label_idx
        opposing = sum(1 for e in kept if int(np.argmax(e.probs)) == other)
    if conf >= 0.75 and agree >= 0.70 and opposing == 0 and n >= 4:
        level = "strong"
    elif conf < 0.60 or agree < 0.50 or n < 2:
        level = "weak"
    else:
        level = "moderate"
    return {"level": level, "agreement": round(agree, 3), "opposing": opposing, "sources": n,
            "confidence": round(conf, 4)}


# ---------------------------------------------------------------------------
# 2. explanation writer
# ---------------------------------------------------------------------------

def pubmed_search_link(claim: str) -> str:
    terms = " ".join(PubMedClient.key_terms(claim, 5))
    return ("https://pubmed.ncbi.nlm.nih.gov/?term=" + quote_plus(terms)
            + "+AND+(systematic+review%5Bpt%5D+OR+randomized+controlled+trial%5Bpt%5D)")


def template_caveats(claim: str, label: str, needs_review: bool, kept: Sequence[Evidence], strength: Dict,
                     static_label: Optional[str], live_label: Optional[str]) -> List[str]:
    c: List[str] = []
    if strength["opposing"] > 0:
        c.append("Some of the retrieved sources point the other way, so the evidence is conflicting.")
    if not any(e.source in ("HEALTHVER", "PUBMED") for e in kept):
        c.append("No research abstracts were retrieved; this rests on fact-checks alone.")
    if needs_review:
        c.append("The system's confidence is below its review threshold; please treat this verdict as provisional.")
    if static_label and live_label and static_label != live_label:
        c.append("The live PubMed check disagrees with the local knowledge base.")
    low = claim.lower()
    if any(re.search(r"(?<![a-z0-9])" + re.escape(w) + r"(?![a-z0-9])", low) for w in ABSOLUTE_WORDS):
        c.append("The claim uses absolute wording; real-world evidence rarely supports 'always', 'never' or 'cures'.")
    return c


def _cites(idx: Sequence[int]) -> str:
    return "".join(f"[{i}]" for i in idx)


def template_explanation(claim: str, label: str, kept: Sequence[Evidence], caveats: List[str]) -> Dict:
    pos = {c: [i for i, e in enumerate(kept, 1) if e.stance_class == c] for c in CLASSES}
    lead = pos[label][:3] or list(range(1, min(len(kept), 2) + 1))
    why = {
        "SUPPORT": "The retrieved evidence mostly supports this claim",
        "CONTRADICT": "The retrieved evidence mostly contradicts this claim",
        "NEI": "The retrieved evidence does not clearly support or contradict this claim",
    }[label] + (f" {_cites(lead)}." if lead else ".")

    def line(i: int, verb: str) -> str:
        e = kept[i - 1]
        yr = f" ({e.year})" if e.year else ""
        return f"{SOURCE_LABEL.get(e.source, 'Source')} “{e.title.strip()[:140]}”{yr} {verb} [{i}]."

    return {
        "why": why,
        "points_for": [line(i, "points toward the claim being true") for i in pos["SUPPORT"][:3]],
        "points_against": [line(i, "points toward the claim being false") for i in pos["CONTRADICT"][:3]],
        "caveats": caveats,
        "next_steps": [
            "Discuss any personal medical decision with a qualified clinician.",
            "Search systematic reviews and trials on this topic: " + pubmed_search_link(claim),
        ],
    }


GEMINI_SYSTEM = (
    "You explain the result of an automated health-claim verification. Use ONLY the numbered evidence supplied; "
    "no outside facts. Cite evidence as [1], [2]. Never soften a CONTRADICT verdict. Never give dosing, "
    "treatment or personal medical advice. Return JSON with keys: why (string), points_for (list of strings), "
    "points_against (list of strings), caveats (list of strings), better_claims (list of strings), "
    "next_steps (list of strings)."
)


def gemini_explanation(client: GeminiClient, claim: str, label: str, confidence: float,
                       kept: Sequence[Evidence]) -> Optional[Dict]:
    if not client.available:
        return None
    ev = "\n".join(
        f"[{i}] ({SOURCE_LABEL.get(e.source, e.source)}, {e.year or 'n.d.'}) {e.title.strip()}\n    {e.text.strip()[:700]}"
        for i, e in enumerate(kept, 1)
    )
    prompt = (f"Claim: {claim}\nSystem verdict: {label} (confidence {confidence:.2f})\n\nEvidence:\n{ev}\n\n"
              "Write the JSON explanation.")
    out = client.generate_json(prompt, temperature=0.2, system=GEMINI_SYSTEM)
    if not isinstance(out, dict) or not isinstance(out.get("why"), str):
        return None
    n = len(kept)
    for key in ("points_for", "points_against", "caveats", "better_claims", "next_steps"):
        v = out.get(key, [])
        out[key] = [str(x) for x in v if isinstance(x, str)] if isinstance(v, list) else []
    # reject answers that cite evidence that does not exist
    for text in [out["why"], *out["points_for"], *out["points_against"]]:
        for m in re.findall(r"\[(\d+)\]", text):
            if not 1 <= int(m) <= n:
                logger.warning("Gemini cited [%s] but only %d items exist; using template", m, n)
                return None
    return out


# ---------------------------------------------------------------------------
# 3 + 4. better-claim suggestions and their validation
# ---------------------------------------------------------------------------

def missing_specifics(claim: str) -> List[str]:
    low = claim.lower()
    checks = {
        "population": r"\b(adults?|children|child|infants?|women|men|elderly|patients?|pregnan\w+|people with|teen\w*)\b",
        "outcome": r"\b(risk|mortality|death|symptoms?|pain|blood pressure|infection|cancer|disease|recovery|outcome|"
                   r"severity|survival|hospitali[sz]ation)\b",
        "dose or amount": r"\b(\d+\s?(mg|g|ml|iu|mcg|%)|dose|daily|per day|a day)\b",
        "time frame": r"\b(days?|weeks?|months?|years?|long[- ]term|short[- ]term|within|hours?)\b",
    }
    return [name for name, rx in checks.items() if not re.search(rx, low)]


def hedge_claim(claim: str) -> Optional[str]:
    if get_certainty_level(claim) == 1:
        return None
    for rx, repl in HEDGE_RULES:
        new, n = re.subn(rx, repl, claim, count=1, flags=re.I)
        if n:
            return new
    return None


def build_suggestions(claim: str, label: str, kept: Sequence[Evidence]) -> List[Dict]:
    out: List[Dict] = []
    hedged = hedge_claim(claim)
    if hedged:
        out.append({"kind": "hedged", "text": hedged})
    if label == "CONTRADICT":
        out.append({"kind": "myth_check", "text": f"Myth check: {claim.strip().rstrip('.')}"})
    lead = next((e for e in kept if e.stance_class == label and e.source != "PUBHEALTH"), None) \
        or next((e for e in kept if e.stance_class == label), None)
    if lead:
        yr = f" ({lead.year})" if lead.year else ""
        out.append({"kind": "source_backed", "text": f"See: {lead.title.strip()[:160]}{yr}", "url": lead.url or None})
    miss = missing_specifics(claim)
    if miss:
        out.append({"kind": "missing_specifics", "text": "Add specifics: " + ", ".join(miss) + "."})
    return out


def validate_suggestions(suggestions: List[Dict], verify_static: Optional[Callable[[str], Optional[Dict]]]) -> List[Dict]:
    """Only rewordings are re-verified; each result is attached, never hidden."""
    for s in suggestions:
        if verify_static and s["kind"] in ("hedged",):
            try:
                s["validation"] = verify_static(s["text"])
            except Exception as exc:  # validation must never break the report
                logger.warning("Suggestion validation failed: %s", exc)
                s["validation"] = None
    return suggestions


# ---------------------------------------------------------------------------
# 5. faithfulness audit
# ---------------------------------------------------------------------------

def to_sentences(expl: Dict, label: str) -> List[Dict]:
    """Auditable sentences: the 'why' line and the points for / against."""
    out = [{"role": "why", "expect": label, "text": expl["why"]}]
    out += [{"role": "for", "expect": "SUPPORT", "text": t} for t in expl.get("points_for", [])]
    out += [{"role": "against", "expect": "CONTRADICT", "text": t} for t in expl.get("points_against", [])]
    for i, s in enumerate(out, 1):
        s["id"] = f"s-{i}"
        s["cites"] = sorted({int(m) for m in re.findall(r"\[(\d+)\]", s["text"])})
    return out


def audit(sentences: List[Dict], kept: Sequence[Evidence], threshold: float) -> List[Dict]:
    n = len(kept)
    for s in sentences:
        cites = [i for i in s["cites"] if 1 <= i <= n]
        s["certainty"] = get_certainty_level(re.sub(r"\[\d+\]", "", s["text"]))
        if not cites:
            s.update(status="PARTIALLY_SUPPORTED", confidence=0.0, evidence_ids=[])
            continue
        k = CLASSES.index(s["expect"])
        conf = max(float(kept[i - 1].probs[k]) for i in cites)
        ev_cert = max(get_certainty_level(kept[i - 1].text) for i in cites)
        if conf < threshold:
            status = "UNSUPPORTED"
        elif s["certainty"] > ev_cert:
            status = "CERTAINTY_ESCALATION"
        else:
            status = "SUPPORTED"
        s.update(status=status, confidence=round(conf, 4), evidence_ids=[kept[i - 1].ev_id for i in cites])
    return sentences


def faithfulness_score(sentences: Sequence[Dict]) -> Optional[float]:
    if not sentences:
        return None
    ok = sum(1 for s in sentences if s["status"] == "SUPPORTED")
    return round(100.0 * ok / len(sentences), 1)


# ---------------------------------------------------------------------------
# orchestration of the stage
# ---------------------------------------------------------------------------

def run_explanation(
    claim: str,
    label: str,
    probs: np.ndarray,
    needs_review: bool,
    kept: Sequence[Evidence],
    gemini: Optional[GeminiClient],
    faithfulness_threshold: float,
    do_audit: bool,
    static_label: Optional[str] = None,
    live_label: Optional[str] = None,
) -> Dict:
    strength = evidence_strength(probs, kept)
    caveats = template_caveats(claim, label, needs_review, kept, strength, static_label, live_label)
    expl = gemini_explanation(gemini, claim, label, float(np.max(probs)), kept) if gemini else None
    source = "gemini"
    if expl is None:
        expl, source = template_explanation(claim, label, kept, caveats), "template"
    else:  # always keep the deterministic caveats even when Gemini wrote its own
        expl["caveats"] = list(dict.fromkeys(expl.get("caveats", []) + caveats))
        expl.setdefault("next_steps", [])
        if not any("pubmed" in s.lower() for s in expl["next_steps"]):
            expl["next_steps"].append("Search systematic reviews and trials on this topic: " + pubmed_search_link(claim))
    sentences = to_sentences(expl, label)
    if do_audit:
        audit(sentences, kept, faithfulness_threshold)
    else:
        for s in sentences:
            s.update(status="NOT_AUDITED", confidence=0.0, certainty=get_certainty_level(s["text"]), evidence_ids=[])
    return {
        "source": source,
        "strength": strength,
        "why": expl["why"],
        "points_for": expl.get("points_for", []),
        "points_against": expl.get("points_against", []),
        "caveats": expl.get("caveats", []),
        "next_steps": expl.get("next_steps", []),
        "gemini_better_claims": expl.get("better_claims", []) if source == "gemini" else [],
        "sentences": sentences,
        "faithfulness_confidence": faithfulness_score(sentences) if do_audit else None,
        "verdict_display": VERDICT_DISPLAY[label],
    }
