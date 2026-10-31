"""
Meta-model feature construction -- the single source of truth for the exact
vector shapes the three trained meta-models were fitted on (report 4.6):

    static  (static.joblib)       18 features
    live    (live.joblib)          6 features
    hybrid  (hybrid_stack.joblib)  7 features

Static, 18 = 3 + 5 + 5 + 5:
    [0:3]    relevance-weighted mean pair probs, all kept evidence
    [3:6]    same, PUBHEALTH fact-check pool only
    [6:8]    fact-check pool max / mean reranker score
    [8:11]   same weighted mean, HealthVer abstract pool only
    [11:13]  abstract pool max / mean reranker score
    [13:16]  kNN label vote over the dense-retrieved fact-checks (weight = sim**4)
    [16]     kNN total weight
    [17]     kNN top similarity

Live, 6 = 3 + 2 + 1:
    [0:3]    relevance-weighted mean pair probs, live PubMed evidence
    [3:5]    max / mean reranker score
    [5]      number of live abstracts / 10

Hybrid, 7 = 3 + 3 + 1:
    [0:3]    static meta-model probabilities
    [3:6]    live meta-model probabilities
    [6]      best live reranker score

Every probability triplet is in canonical CLASSES order (SUPPORT, CONTRADICT,
NEI). The order of the entries *inside* each block follows the report; if the
training notebook used a different order, change it here only -- and run
``scripts/check_artifacts.py`` which compares ``n_features_in_``.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .schema import CLASSES, Evidence, PipelineError

N_STATIC = 18
N_LIVE = 6
N_HYBRID = 7
KNN_POWER = 4


def weighted_mean_probs(probs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Relevance-weighted mean of pair probabilities (zeros when empty)."""
    if probs.shape[0] == 0:
        return np.zeros(len(CLASSES), dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    total = float(w.sum())
    if total <= 1e-12:  # all weights ~0: fall back to a plain mean
        return probs.mean(axis=0)
    return (probs * w[:, None]).sum(axis=0) / total


def _pool_block(items: Sequence[Evidence]) -> np.ndarray:
    """[mean-probs(3), max rerank, mean rerank] for one pool -> 5 values."""
    if not items:
        return np.zeros(5, dtype=np.float64)
    probs = np.stack([e.probs for e in items])
    rer = np.array([e.rerank for e in items], dtype=np.float64)
    return np.concatenate([weighted_mean_probs(probs, rer), [rer.max(), rer.mean()]])


def knn_vote(sims: Sequence[float], labels: Sequence[Optional[str]]) -> np.ndarray:
    """Similarity**4-weighted label vote -> [vote(3), total weight, top sim]."""
    pairs = [(float(s), l) for s, l in zip(sims, labels) if l in CLASSES]
    if not pairs:
        return np.zeros(len(CLASSES) + 2, dtype=np.float64)
    w = np.array([max(s, 0.0) ** KNN_POWER for s, _ in pairs], dtype=np.float64)
    vote = np.zeros(len(CLASSES), dtype=np.float64)
    for wi, (_, lab) in zip(w, pairs):
        vote[CLASSES.index(lab)] += wi
    total = float(w.sum())
    if total > 1e-12:
        vote = vote / total
    top = max(s for s, _ in pairs)
    return np.concatenate([vote, [total, top]])


def static_features(
    fc_items: Sequence[Evidence],
    abs_items: Sequence[Evidence],
    knn_sims: Sequence[float],
    knn_labels: Sequence[Optional[str]],
) -> np.ndarray:
    kept = list(fc_items) + list(abs_items)
    if kept:
        probs = np.stack([e.probs for e in kept])
        rer = np.array([e.rerank for e in kept], dtype=np.float64)
        overall = weighted_mean_probs(probs, rer)
    else:
        overall = np.zeros(len(CLASSES), dtype=np.float64)
    x = np.concatenate([overall, _pool_block(fc_items), _pool_block(abs_items), knn_vote(knn_sims, knn_labels)])
    if x.shape[0] != N_STATIC:
        raise PipelineError(f"static feature vector has {x.shape[0]} entries, expected {N_STATIC}")
    return x


def live_features(live_items: Sequence[Evidence]) -> np.ndarray:
    block = _pool_block(live_items)  # 5 values: probs(3), max, mean
    x = np.concatenate([block, [len(live_items) / 10.0]])
    if x.shape[0] != N_LIVE:
        raise PipelineError(f"live feature vector has {x.shape[0]} entries, expected {N_LIVE}")
    return x


def hybrid_features(static_probs: np.ndarray, live_probs: np.ndarray, live_items: Sequence[Evidence]) -> np.ndarray:
    best = max((e.rerank for e in live_items), default=0.0)
    x = np.concatenate([np.asarray(static_probs, dtype=np.float64), np.asarray(live_probs, dtype=np.float64), [best]])
    if x.shape[0] != N_HYBRID:
        raise PipelineError(f"hybrid feature vector has {x.shape[0]} entries, expected {N_HYBRID}")
    return x
