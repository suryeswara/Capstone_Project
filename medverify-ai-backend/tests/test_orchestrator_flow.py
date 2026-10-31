"""End-to-end orchestration with fake neural components and stub meta-models."""
import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from app.pipeline.meta import MetaModels
from app.pipeline.orchestrator import MedVerifyPipeline
from app.pipeline.schema import Evidence, GateResult, Mode, Pool, PipelineError
from app.pipeline.settings import PipelineSettings
from app.services.safety_guardrail import get_safety_guardrail


class FakeEncoder:
    dim = 768
    def encode(self, texts):
        return np.ones((len(texts), 768), dtype="float32") / np.sqrt(768)


class FakeStore:
    def search(self, claim, qvec, k_fc, k_abs, max_words):
        fc = [Evidence(uid=f"fc-{i}", pool=Pool.FACT_CHECK, source="PUBHEALTH", title=f"Fact check {i}",
                       text=f"Fact-check text {i} shows a result.", dense_sim=0.9 - i * 0.05, record_label="CONTRADICT",
                       year=2020) for i in range(k_fc)]
        ab = [Evidence(uid=f"abs-{i}", pool=Pool.ABSTRACT, source="HEALTHVER", title=f"Abstract {i}",
                       text=f"Abstract text {i} shows a result.", dense_sim=0.8 - i * 0.05, year=2021)
              for i in range(k_abs)]
        return fc, ab, [e.dense_sim for e in ab]


class FakeReranker:
    def score(self, claim, docs):
        return np.linspace(0.99, 0.5, len(docs))


class FakeClassifier:
    """Always leans CONTRADICT with 0.85 so the verdict is deterministic."""
    def predict(self, claim, texts):
        p = np.tile([0.07, 0.85, 0.08], (len(texts), 1))
        return p


class FakePubMed:
    def __init__(self, n=8, fail=False):
        self.n, self.fail = n, fail
    def search(self, claim):
        if self.fail:
            raise RuntimeError("boom")
        return [Evidence(uid=f"pm-{i}", pool=Pool.LIVE, source="PUBMED", title=f"PubMed {i}", pmid=str(i),
                         text=f"PubMed abstract {i} shows a result.", year=2023) for i in range(self.n)]


class FakeGate:
    def __init__(self, accept=True):
        self.accept = accept
    def evaluate(self, claim, qvec, abs_sims):
        return GateResult(self.accept, 0.97 if self.accept else 0.01, "ACCEPT" if self.accept else "REJECT",
                          "ok", "strict", 0.3, 0.9, {"probe_logit": 1.0})


def _fit_meta(tmp_path, favour):
    """Stub stackers whose training labels all equal `favour` -> predict_proba is ~one-hot-ish."""
    for name, n in (("static", 18), ("live", 6), ("hybrid_stack", 7)):
        X = np.random.default_rng(1).normal(size=(60, n))
        y = np.array([0, 1, 2] * 20)
        joblib.dump(LogisticRegression(max_iter=100, C=1e-4).fit(X, y), tmp_path / f"{name}.joblib")
    return MetaModels(tmp_path)


@pytest.fixture()
def pipe(tmp_path):
    p = MedVerifyPipeline(PipelineSettings())
    p.encoder, p.store, p.reranker, p.classifier = FakeEncoder(), FakeStore(), FakeReranker(), FakeClassifier()
    p.gate, p.pubmed, p.gemini = FakeGate(), FakePubMed(), None
    p.meta = _fit_meta(tmp_path, 1)
    p.guardrail = get_safety_guardrail()
    p.loaded = True
    return p


CLAIM = "Taking vitamin C every day prevents colds in adults"


def test_static_mode(pipe):
    r = pipe.verify(CLAIM, Mode.STATIC)
    assert r["status"] == "COMPLETED" and r["mode_effective"] == "static"
    assert r["trace"]["feature_lengths"] == {"static": 18}
    assert len(r["evidence"]) == 6 and {e["pool"] for e in r["evidence"]} == {"fc", "abs"}
    assert abs(sum(r["probabilities"].values()) - 1) < 1e-3
    assert r["needs_review"] is True   # near-uniform stub model -> below the 0.5859 threshold
    assert r["explanation"]["source"] == "template" and r["explanation"]["sentences"]


def test_live_mode_uses_only_live_evidence(pipe):
    r = pipe.verify(CLAIM, Mode.LIVE)
    assert r["mode_effective"] == "live" and r["trace"]["feature_lengths"] == {"live": 6}
    assert len(r["evidence"]) == 5 and {e["pool"] for e in r["evidence"]} == {"live"}


def test_hybrid_mode_uses_both_and_dedupes(pipe):
    r = pipe.verify(CLAIM, Mode.HYBRID)
    assert r["mode_effective"] == "hybrid"
    assert r["trace"]["feature_lengths"] == {"static": 18, "live": 6, "hybrid": 7}
    assert {e["pool"] for e in r["evidence"]} == {"fc", "abs", "live"} and len(r["evidence"]) == 11
    assert r["trace"]["static_probs"] and r["trace"]["live_probs"]


def test_pubmed_failure_falls_back_to_static(pipe):
    pipe.pubmed = FakePubMed(fail=True)
    for mode in (Mode.LIVE, Mode.HYBRID):
        r = pipe.verify(CLAIM, mode)
        assert r["status"] == "COMPLETED" and r["mode"] == mode.value and r["mode_effective"] == "static"
        assert r["notes"] and "fell back" in r["notes"][0]
        assert {e["pool"] for e in r["evidence"]} == {"fc", "abs"}


def test_gate_rejection_stops_before_retrieval_work(pipe):
    pipe.gate = FakeGate(accept=False)
    pipe.classifier = None  # would crash if reached
    r = pipe.verify("The Eiffel Tower is in Paris and is tall", Mode.HYBRID)
    assert r["status"] == "REJECTED_NON_MEDICAL" and r["gate"]["decision"] == "REJECT"
    assert "evidence" not in r


def test_unloaded_pipeline_never_fabricates():
    p = MedVerifyPipeline(PipelineSettings())
    with pytest.raises(PipelineError):
        p.verify(CLAIM, Mode.STATIC)


def test_input_validation(pipe):
    with pytest.raises(ValueError):
        pipe.verify("   ", Mode.STATIC)
    with pytest.raises(ValueError):
        pipe.verify("x " * 600, Mode.STATIC)
    with pytest.raises(ValueError):
        Mode.parse("turbo")


def test_evidence_ids_unique_and_referenced_by_explanation(pipe):
    r = pipe.verify(CLAIM, Mode.STATIC)
    ids = [e["id"] for e in r["evidence"]]
    assert len(ids) == len(set(ids)) and all(i.startswith("ev-") for i in ids)
    for s in r["explanation"]["sentences"]:
        assert set(s["evidence_ids"]) <= set(ids)


def test_suggestions_are_validated_in_static_mode(pipe):
    r = pipe.verify(CLAIM, Mode.STATIC)
    hedged = [s for s in r["suggestions"] if s["kind"] == "hedged"]
    assert hedged and hedged[0]["validation"]["verdict"] in {"Supported", "Contradicted", "Insufficient Evidence"}
