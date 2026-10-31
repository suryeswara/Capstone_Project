"""
Evidence retrieval.

* StaticStore  -- 768-d FAISS fact-check + abstract indexes and records.jsonl
* PubMedClient -- live NCBI E-utilities (esearch + efetch)

Claim-aware evidence text is built exactly as described in the report
(4.5): PUBHEALTH = explanation + highest claim-overlap article sentences
(up to 150 words); HealthVer / PubMed = title + abstract.
"""

from __future__ import annotations

import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import requests

from .schema import RAW_LABEL_TO_CLASS, ArtifactError, Evidence, Pool

logger = logging.getLogger(__name__)

_STOP = frozenset(
    "a an the and or of to in on for with by from at as is are was were be been being it its this that these those "
    "can do does did will would should could may might not no than then so such into over under about after before "
    "up down out more most less very also has have had who whom which what when where why how your you they their".split()
)
_WORD = re.compile(r"[A-Za-z][A-Za-z\-']+")
_SENT = re.compile(r"(?<=[.!?])\s+")


def content_words(text: str) -> List[str]:
    return [w.lower() for w in _WORD.findall(text or "") if w.lower() not in _STOP and len(w) > 2]


def _first(rec: dict, keys: Sequence[str], default=None):
    for k in keys:
        v = rec.get(k)
        if v not in (None, "", []):
            return v
    return default


def overlap_sentences(claim: str, article: str, max_words: int) -> str:
    """Highest claim-overlap sentences of ``article``, kept in reading order."""
    claim_terms = set(content_words(claim))
    sents = [s.strip() for s in _SENT.split(article or "") if s.strip()]
    scored = sorted(
        ((len(claim_terms & set(content_words(s))), -i, s) for i, s in enumerate(sents)), reverse=True
    )
    picked, used = [], 0
    for score, neg_i, s in scored:
        if score == 0:
            break
        n = len(s.split())
        if used + n > max_words and picked:
            continue
        picked.append((-neg_i, s))
        used += n
        if used >= max_words:
            break
    return " ".join(s for _, s in sorted(picked))


def fact_check_text(claim: str, rec: dict, max_words: int) -> str:
    explanation = str(_first(rec, ("explanation",), "") or "").strip()
    article = str(_first(rec, ("main_text", "article", "text", "body"), "") or "")
    budget = max(max_words - len(explanation.split()), 0)
    body = overlap_sentences(claim, article, budget) if budget else ""
    text = " ".join(p for p in (explanation, body) if p).strip()
    return " ".join(text.split()[:max_words]) if text else str(_first(rec, ("title", "claim"), ""))


def abstract_text(rec: dict) -> Tuple[str, str]:
    title = str(_first(rec, ("title", "paper_title"), "") or "").strip()
    abstract = str(_first(rec, ("abstract", "text", "evidence", "body"), "") or "").strip()
    return title, (f"{title}. {abstract}".strip(". ") if title else abstract)


def _year(rec: dict) -> Optional[int]:
    v = _first(rec, ("year", "pub_year", "date", "publication_date", "pubdate"))
    m = re.search(r"(19|20)\d{2}", str(v)) if v is not None else None
    return int(m.group(0)) if m else None


# ---------------------------------------------------------------------------
# Static store
# ---------------------------------------------------------------------------

class StaticStore:
    """Loads the vector_db folder once; searches with a 768-d claim vector."""

    REQUIRED = ("fc_index.faiss", "abs_index.faiss", "fc_ids.npy", "abs_ids.npy", "records.jsonl")

    def __init__(self, directory: Path, expected_dim: int = 768):
        import faiss  # local import keeps the module importable without faiss

        d = Path(directory)
        missing = [f for f in self.REQUIRED if not (d / f).exists()]
        if missing:
            raise ArtifactError(f"vector_db at {d} is missing: {', '.join(missing)}")
        self.fc_index = faiss.read_index(str(d / "fc_index.faiss"))
        self.abs_index = faiss.read_index(str(d / "abs_index.faiss"))
        for name, idx in (("fc_index", self.fc_index), ("abs_index", self.abs_index)):
            if idx.d != expected_dim:
                raise ArtifactError(
                    f"{name}.faiss is {idx.d}-d but the MedCPT query encoder produces {expected_dim}-d vectors"
                )
        self.fc_ids = np.load(d / "fc_ids.npy", allow_pickle=True)
        self.abs_ids = np.load(d / "abs_ids.npy", allow_pickle=True)
        if len(self.fc_ids) != self.fc_index.ntotal or len(self.abs_ids) != self.abs_index.ntotal:
            raise ArtifactError("id arrays and FAISS indexes have different lengths")
        self.records: Dict[str, dict] = {}
        with open(d / "records.jsonl", "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rid = _first(rec, ("id", "record_id", "uid"))
                if rid is not None:
                    self.records[str(rid)] = rec
        logger.info(
            "StaticStore: %d fact-checks, %d abstracts, %d records",
            self.fc_index.ntotal, self.abs_index.ntotal, len(self.records),
        )

    def _lookup(self, raw_id) -> Optional[dict]:
        key = raw_id.item() if hasattr(raw_id, "item") else raw_id
        return self.records.get(str(key))

    def search(self, claim: str, qvec: np.ndarray, k_fc: int, k_abs: int, max_words: int):
        """Returns (fc_candidates, abs_candidates, abs_sims) -- sims are inner products."""
        q = np.ascontiguousarray(qvec.reshape(1, -1), dtype="float32")
        fc_s, fc_i = self.fc_index.search(q, k_fc)
        ab_s, ab_i = self.abs_index.search(q, k_abs)

        fc: List[Evidence] = []
        for sim, pos in zip(fc_s[0], fc_i[0]):
            if pos < 0:
                continue
            rec = self._lookup(self.fc_ids[pos])
            if rec is None:
                continue
            raw = str(_first(rec, ("label", "verdict", "gold_label"), "")).strip().lower()
            title = str(_first(rec, ("title", "claim"), "") or "")
            fc.append(Evidence(
                uid=f"fc-{self.fc_ids[pos]}", pool=Pool.FACT_CHECK, source="PUBHEALTH", title=title,
                text=fact_check_text(claim, rec, max_words), url=str(_first(rec, ("url", "source_url"), "") or ""),
                year=_year(rec), dense_sim=float(sim), record_label=RAW_LABEL_TO_CLASS.get(raw),
            ))
        ab: List[Evidence] = []
        for sim, pos in zip(ab_s[0], ab_i[0]):
            if pos < 0:
                continue
            rec = self._lookup(self.abs_ids[pos])
            if rec is None:
                continue
            title, text = abstract_text(rec)
            ab.append(Evidence(
                uid=f"abs-{self.abs_ids[pos]}", pool=Pool.ABSTRACT, source="HEALTHVER", title=title, text=text,
                url=str(_first(rec, ("url",), "") or ""), year=_year(rec), dense_sim=float(sim),
                pmid=str(rec["pmid"]) if rec.get("pmid") else None,
            ))
        return fc, ab, [float(s) for s, p in zip(ab_s[0], ab_i[0]) if p >= 0]


# ---------------------------------------------------------------------------
# Live PubMed
# ---------------------------------------------------------------------------

class PubMedClient:
    """NCBI E-utilities. Query relaxes 6 -> 4 -> 3 AND-terms, then OR (report 4.3)."""

    BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(self, api_key: str = "", email: str = "", timeout: float = 20.0, retmax: int = 25, min_chars: int = 150):
        self.api_key, self.email, self.timeout = api_key, email, timeout
        self.retmax, self.min_chars = retmax, min_chars
        self._last_call = 0.0
        self._min_gap = 0.11 if api_key else 0.34  # NCBI: 10 req/s with key, 3 without

    def _get(self, endpoint: str, params: dict) -> requests.Response:
        wait = self._min_gap - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        params = {**params, "tool": "medverify", **({"api_key": self.api_key} if self.api_key else {}),
                  **({"email": self.email} if self.email else {})}
        last: Optional[Exception] = None
        for attempt in range(3):
            try:
                self._last_call = time.monotonic()
                r = requests.get(f"{self.BASE}/{endpoint}", params=params, timeout=self.timeout)
                if r.status_code == 429:
                    time.sleep(1.0 + attempt)
                    continue
                r.raise_for_status()
                return r
            except requests.RequestException as exc:
                last = exc
                time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"PubMed {endpoint} failed: {last}")

    @staticmethod
    def key_terms(claim: str, limit: int = 6) -> List[str]:
        seen, terms = set(), []
        for w in content_words(claim):
            if len(w) > 3 and w not in seen:
                seen.add(w)
                terms.append(w)
        # longer words first: they are usually the more specific medical terms
        return sorted(terms, key=lambda w: (-len(w), terms.index(w)))[:limit]

    def search_ids(self, claim: str) -> List[str]:
        terms = self.key_terms(claim, 6)
        if not terms:
            return []
        attempts = []
        for n in (6, 4, 3):
            if len(terms) >= n:
                attempts.append(" AND ".join(terms[:n]))
        attempts.append(" OR ".join(terms[:6]))
        seen = set()
        for q in [a for a in attempts if not (a in seen or seen.add(a))]:
            r = self._get("esearch.fcgi", {"db": "pubmed", "term": q, "retmax": self.retmax,
                                           "retmode": "json", "sort": "relevance"})
            ids = r.json().get("esearchresult", {}).get("idlist", [])
            if ids:
                return ids
        return []

    def fetch(self, pmids: Sequence[str]) -> List[Evidence]:
        if not pmids:
            return []
        r = self._get("efetch.fcgi", {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"})
        root = ET.fromstring(r.content)
        out: List[Evidence] = []
        for art in root.findall(".//PubmedArticle"):
            pmid = (art.findtext(".//MedlineCitation/PMID") or "").strip()
            title = "".join(art.find(".//ArticleTitle").itertext()).strip() if art.find(".//ArticleTitle") is not None else ""
            parts = ["".join(a.itertext()).strip() for a in art.findall(".//Abstract/AbstractText")]
            abstract = " ".join(p for p in parts if p)
            if len(abstract) < self.min_chars:
                continue
            ym = art.findtext(".//PubDate/Year") or art.findtext(".//PubDate/MedlineDate") or ""
            m = re.search(r"(19|20)\d{2}", ym)
            doi = next((e.text for e in art.findall(".//ArticleId") if e.get("IdType") == "doi" and e.text), None)
            out.append(Evidence(
                uid=f"pm-{pmid}", pool=Pool.LIVE, source="PUBMED", title=title,
                text=f"{title}. {abstract}".strip(". "), url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                year=int(m.group(0)) if m else None, pmid=pmid, doi=doi,
            ))
        return out

    def search(self, claim: str) -> List[Evidence]:
        return self.fetch(self.search_ids(claim))


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def dedupe(candidates: Sequence[Evidence], against: Sequence[Evidence] = ()) -> List[Evidence]:
    """Drop duplicates by PMID or normalised title (first occurrence wins)."""
    seen_pmid = {e.pmid for e in against if e.pmid}
    seen_title = {_norm_title(e.title) for e in against if e.title}
    out: List[Evidence] = []
    for e in candidates:
        t = _norm_title(e.title)
        if (e.pmid and e.pmid in seen_pmid) or (t and t in seen_title):
            continue
        if e.pmid:
            seen_pmid.add(e.pmid)
        if t:
            seen_title.add(t)
        out.append(e)
    return out
