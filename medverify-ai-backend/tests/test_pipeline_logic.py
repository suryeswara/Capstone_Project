"""
Logic tests for the new inference pipeline. They use *stub* sklearn models and
fake retrieval / neural components purely to verify shapes, routing and
decision rules. They say nothing about model quality.
"""
import json
import types

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from app.pipeline import features as F
from app.pipeline.explain import (audit, build_suggestions, evidence_strength, faithfulness_score,
                                  hedge_claim, missing_specifics, run_explanation, to_sentences)
from app.pipeline.gate import count_terms, decide_zone
from app.pipeline.meta import MetaModels, abstain
from app.pipeline.models import class_permutation
from app.pipeline.retrieval import PubMedClient, dedupe, overlap_sentences
from app.pipeline.schema import CLASSES, ArtifactError, Evidence, Mode, Pool, PipelineError

RNG = np.random.default_rng(0)


def ev(pool, source="HEALTHVER", rerank=0.9, cls=0, title="t", text="x", pmid=None, sim=None, label=None, ev_id=None):
    p = np.full(3, 0.05)
    p[cls] = 0.9
    return Evidence(uid=f"{pool.value}-{title}", pool=pool, source=source, title=title, text=text, rerank=rerank,
                    probs=p / p.sum(), dense_sim=sim, pmid=pmid, record_label=label, ev_id=ev_id or f"ev-{title}")


# ---------------------------------------------------------------- features
def test_feature_shapes_exact():
    fc = [ev(Pool.FACT_CHECK, "PUBHEALTH", 0.9, 0, f"f{i}", sim=0.8, label="SUPPORT") for i in range(3)]
    ab = [ev(Pool.ABSTRACT, "HEALTHVER", 0.7, 1, f"a{i}") for i in range(3)]
    live = [ev(Pool.LIVE, "PUBMED", 0.8, 2, f"l{i}") for i in range(5)]
    xs = F.static_features(fc, ab, [0.8, 0.7], ["SUPPORT", "CONTRADICT"])
    xl = F.live_features(live)
    xh = F.hybrid_features(np.array([.6, .3, .1]), np.array([.2, .2, .6]), live)
    assert (xs.shape, xl.shape, xh.shape) == ((18,), (6,), (7,))
    assert np.isfinite(xs).all() and np.isfinite(xl).all() and np.isfinite(xh).all()
    assert xl[5] == 0.5            # 5 live abstracts / 10
    assert xh[6] == 0.8            # best live rerank score
    assert np.allclose(xs[3:6], fc[0].probs)   # fact-check pool mean == its (identical) members


def test_feature_block_layout_and_knn():
    fc = [ev(Pool.FACT_CHECK, "PUBHEALTH", 1.0, 0, "f")]
    ab = [ev(Pool.ABSTRACT, "HEALTHVER", 0.5, 1, "a")]
    xs = F.static_features(fc, ab, [0.9, 0.5], ["SUPPORT", "CONTRADICT"])
    assert xs[6] == 1.0 and xs[7] == 1.0       # fc max / mean rerank
    assert xs[11] == 0.5 and xs[12] == 0.5     # abs max / mean rerank
    w = np.array([0.9 ** 4, 0.5 ** 4])
    assert np.allclose(xs[13:16], [w[0] / w.sum(), w[1] / w.sum(), 0.0])
    assert xs[16] == pytest.approx(w.sum()) and xs[17] == 0.9


def test_empty_pools_are_zero_not_nan():
    xs = F.static_features([], [], [], [])
    assert xs.shape == (18,) and not xs.any()
    assert F.live_features([]).shape == (6,)


def test_knn_ignores_unlabelled_records():
    v = F.knn_vote([0.9, 0.8], [None, "NEI"])
    assert v[2] == 1.0 and v[4] == 0.8


# ---------------------------------------------------------------- meta models
def _fit(n_features, classes):
    X = RNG.normal(size=(120, n_features))
    y = np.array([classes[i % 3] for i in range(120)])
    return LogisticRegression(max_iter=200).fit(X, y)


@pytest.fixture()
def meta_dir(tmp_path):
    joblib.dump(_fit(18, [0, 1, 2]), tmp_path / "static.joblib")
    joblib.dump(_fit(6, [0, 1, 2]), tmp_path / "live.joblib")
    joblib.dump(_fit(7, [0, 1, 2]), tmp_path / "hybrid_stack.joblib")
    return tmp_path


def _pools():
    fc = [ev(Pool.FACT_CHECK, "PUBHEALTH", 0.9, 0, f"f{i}", sim=0.8, label="SUPPORT") for i in range(3)]
    ab = [ev(Pool.ABSTRACT, "HEALTHVER", 0.8, 0, f"a{i}") for i in range(3)]
    live = [ev(Pool.LIVE, "PUBMED", 0.9, 1, f"l{i}") for i in range(4)]
    return fc, ab, live


def test_mode_routing(meta_dir):
    m = MetaModels(meta_dir)
    fc, ab, live = _pools()
    ks, kl = [0.8, 0.7], ["SUPPORT", "NEI"]
    s = m.predict(Mode.STATIC, fc, ab, [], ks, kl)
    l = m.predict(Mode.LIVE, [], [], live, ks, kl)
    h = m.predict(Mode.HYBRID, fc, ab, live, ks, kl)
    assert (s.mode_effective, l.mode_effective, h.mode_effective) == (Mode.STATIC, Mode.LIVE, Mode.HYBRID)
    assert s.feature_lengths == {"static": 18}
    assert l.feature_lengths == {"live": 6}
    assert h.feature_lengths == {"static": 18, "live": 6, "hybrid": 7}
    for p in (s, l, h):
        assert p.probs.shape == (3,) and p.probs.sum() == pytest.approx(1.0)
    assert h.static_probs is not None and h.live_probs is not None


def test_live_and_hybrid_fall_back_to_static_with_note(meta_dir):
    m = MetaModels(meta_dir)
    fc, ab, _ = _pools()
    for mode in (Mode.LIVE, Mode.HYBRID):
        r = m.predict(mode, fc, ab, [], [0.8], ["SUPPORT"], live_error="timeout")
        assert r.mode_effective is Mode.STATIC and r.mode_requested is mode
        assert r.notes and "fell back" in r.notes[0] and "timeout" in r.notes[0]


def test_static_without_any_evidence_raises(meta_dir):
    with pytest.raises(PipelineError):
        MetaModels(meta_dir).predict(Mode.STATIC, [], [], [], [], [])


def test_wrong_feature_count_is_rejected(tmp_path):
    joblib.dump(_fit(17, [0, 1, 2]), tmp_path / "static.joblib")
    joblib.dump(_fit(6, [0, 1, 2]), tmp_path / "live.joblib")
    joblib.dump(_fit(7, [0, 1, 2]), tmp_path / "hybrid_stack.joblib")
    with pytest.raises(ArtifactError, match="17"):
        MetaModels(tmp_path)


def test_string_classes_are_reordered_to_canonical(tmp_path):
    # sklearn sorts classes alphabetically: CONTRADICT, NEI, SUPPORT -> must be remapped
    for name, n in (("static", 18), ("live", 6), ("hybrid_stack", 7)):
        joblib.dump(_fit(n, ["SUPPORT", "CONTRADICT", "NEI"]), tmp_path / f"{name}.joblib")
    m = MetaModels(tmp_path)
    assert m.order["static"] == [2, 0, 1]
    x = RNG.normal(size=18)
    raw = m.models["static"].predict_proba(x.reshape(1, -1))[0]
    got = m._proba("static", x)
    assert got[0] == pytest.approx(raw[2]) and got[1] == pytest.approx(raw[0]) and got[2] == pytest.approx(raw[1])


def test_abstain_threshold():
    assert abstain(0.5858, 0.5859) and not abstain(0.5859, 0.5859) and not abstain(0.9, 0.5859)


# ---------------------------------------------------------------- label maps / gate
def test_class_permutation():
    assert class_permutation({0: "SUPPORT", 1: "CONTRADICT", 2: "NEI"}, "x") == [0, 1, 2]
    assert class_permutation({0: "NEI", 1: "SUPPORT", 2: "CONTRADICT"}, "x") == [1, 2, 0]
    assert class_permutation({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}, "x") == [0, 1, 2]
    with pytest.raises(ArtifactError):
        class_permutation({0: "FOO", 1: "BAR", 2: "BAZ"}, "x")


def test_gate_zones():
    assert decide_zone(0.95, 0.3, 0.9) == "ACCEPT"
    assert decide_zone(0.9, 0.3, 0.9) == "ACCEPT"
    assert decide_zone(0.5, 0.3, 0.9) == "UNSURE"
    assert decide_zone(0.3, 0.3, 0.9) == "UNSURE"
    assert decide_zone(0.29, 0.3, 0.9) == "REJECT"


def test_lexicon_counts_are_capped_and_prefix_based():
    assert count_terms("vaccines and vaccination", ("vaccin",)) == 1
    assert count_terms("cancer insulin ivermectin vaccine", ("cancer", "insulin", "ivermectin", "vaccin")) == 3
    assert count_terms("The Eiffel Tower is in Paris", ("vaccin", "cancer")) == 0


# ---------------------------------------------------------------- retrieval helpers
def test_overlap_sentences_prefers_claim_terms():
    art = "Intro text here. Vitamin D reduces fracture risk in adults. Unrelated sports news. Fracture risk fell 20%."
    out = overlap_sentences("vitamin d fracture risk", art, 150)
    assert "Vitamin D reduces" in out and "sports" not in out


def test_dedupe_by_pmid_and_title():
    a = ev(Pool.ABSTRACT, title="Same Title", pmid=None)
    b = ev(Pool.LIVE, "PUBMED", title="same  title!", pmid="1")
    c = ev(Pool.LIVE, "PUBMED", title="Other", pmid="2")
    d = ev(Pool.LIVE, "PUBMED", title="Other again", pmid="2")
    out = dedupe([b, c, d], against=[a])
    assert [e.pmid for e in out] == ["2"]


PUBMED_XML = b"""<PubmedArticleSet>
<PubmedArticle><MedlineCitation><PMID>111</PMID><Article><ArticleTitle>Vitamin D and fractures</ArticleTitle>
<Abstract><AbstractText>%s</AbstractText></Abstract></Article></MedlineCitation>
<PubmedData><ArticleIdList><ArticleId IdType="doi">10.1/x</ArticleId></ArticleIdList></PubmedData></PubmedArticle>
<PubmedArticle><MedlineCitation><PMID>222</PMID><Article><ArticleTitle>Too short</ArticleTitle>
<Abstract><AbstractText>short</AbstractText></Abstract></Article></MedlineCitation></PubmedArticle>
</PubmedArticleSet>""" % (b"word " * 60)


def test_pubmed_fetch_parses_and_drops_short_abstracts(monkeypatch):
    c = PubMedClient()
    resp = types.SimpleNamespace(content=PUBMED_XML, status_code=200, raise_for_status=lambda: None)
    monkeypatch.setattr(c, "_get", lambda *a, **k: resp)
    out = c.fetch(["111", "222"])
    assert len(out) == 1 and out[0].pmid == "111" and out[0].doi == "10.1/x" and out[0].pool is Pool.LIVE


def test_pubmed_query_relaxation(monkeypatch):
    c = PubMedClient()
    queries = []

    def fake_get(endpoint, params):
        queries.append(params["term"])
        ids = ["9"] if " OR " in params["term"] else []
        return types.SimpleNamespace(json=lambda: {"esearchresult": {"idlist": ids}})

    monkeypatch.setattr(c, "_get", fake_get)
    ids = c.search_ids("Daily vitamin supplements reduce fracture risk among elderly women")
    assert ids == ["9"]
    assert queries[0].count(" AND ") == 5 and queries[1].count(" AND ") == 3 and queries[2].count(" AND ") == 2
    assert " OR " in queries[-1]


# ---------------------------------------------------------------- explanation
def test_hedge_and_specifics():
    assert hedge_claim("Garlic cures the common cold") == "Garlic may help treat the common cold"
    assert hedge_claim("Garlic may help colds") is None
    assert "population" in missing_specifics("Garlic cures colds")


def test_audit_statuses():
    kept = [ev(Pool.ABSTRACT, cls=0, title="a", text="Results show a benefit in adults.", ev_id="ev-1"),
            ev(Pool.ABSTRACT, cls=1, title="b", text="Trial found no benefit.", ev_id="ev-2")]
    sents = [
        {"id": "s-1", "text": "Evidence supports it [1].", "expect": "SUPPORT", "role": "for", "cites": [1]},
        {"id": "s-2", "text": "Another source supports it [2].", "expect": "SUPPORT", "role": "for", "cites": [2]},
        {"id": "s-3", "text": "No citation here.", "expect": "SUPPORT", "role": "for", "cites": []},
        {"id": "s-4", "text": "It definitively cures disease [1].", "expect": "SUPPORT", "role": "for", "cites": [1]},
        {"id": "s-5", "text": "Cites a missing item [7].", "expect": "SUPPORT", "role": "for", "cites": [7]},
    ]
    out = {s["id"]: s["status"] for s in audit(sents, kept, 0.70)}
    assert out == {"s-1": "SUPPORTED", "s-2": "UNSUPPORTED", "s-3": "PARTIALLY_SUPPORTED",
                   "s-4": "CERTAINTY_ESCALATION", "s-5": "PARTIALLY_SUPPORTED"}
    assert faithfulness_score(sents) == 20.0
    assert faithfulness_score([]) is None


def test_run_explanation_template_end_to_end():
    kept = [ev(Pool.FACT_CHECK, "PUBHEALTH", 0.9, 1, "Garlic myth", ev_id="ev-a"),
            ev(Pool.ABSTRACT, "HEALTHVER", 0.8, 1, "Garlic trial", ev_id="ev-b")]
    probs = np.array([0.1, 0.8, 0.1])
    r = run_explanation("Garlic cures all colds", "CONTRADICT", probs, False, kept, None, 0.70, True)
    assert r["source"] == "template" and r["strength"]["level"] in {"weak", "moderate"}
    assert r["points_against"] and not r["points_for"]
    assert any("absolute wording" in c for c in r["caveats"])
    assert all(s["status"] in {"SUPPORTED", "CERTAINTY_ESCALATION"} for s in r["sentences"])
    assert any("pubmed.ncbi.nlm.nih.gov" in s for s in r["next_steps"])


def test_evidence_strength_levels():
    strong = [ev(Pool.ABSTRACT, cls=0, title=str(i)) for i in range(5)]
    assert evidence_strength(np.array([.9, .05, .05]), strong)["level"] == "strong"
    assert evidence_strength(np.array([.5, .3, .2]), strong)["level"] == "weak"
    assert evidence_strength(np.array([.9, .05, .05]), [])["level"] == "weak"
