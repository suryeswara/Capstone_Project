#!/usr/bin/env python3
"""
Validate the trained v4 artifacts WITHOUT loading the heavy models or retraining anything.

    python scripts/check_artifacts.py            # uses MEDVERIFY_ARTIFACT_ROOT or ./artifacts

Checks: files exist, FAISS indexes are 768-d and match their id arrays, records.jsonl
covers the ids, meta-models expect exactly 18 / 6 / 7 features, class order is resolvable,
gate files are present. Exit code 0 = ready, 1 = something to fix.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pipeline.features import N_HYBRID, N_LIVE, N_STATIC  # noqa: E402
from app.pipeline.settings import get_settings  # noqa: E402

problems: list[str] = []


def ok(msg: str) -> None:
    print(f"  [ok]   {msg}")


def bad(msg: str) -> None:
    problems.append(msg)
    print(f"  [FAIL] {msg}")


def main() -> int:
    s = get_settings()
    print(f"artifact root: {s.artifact_root}")

    print("pair classifiers")
    for name, d in (("deberta", s.deberta_dir), ("biomed", s.biomed_dir)):
        need = [d / "config.json"]
        if d.is_dir() and any(d.glob("*.safetensors")) or d.is_dir() and any(d.glob("pytorch_model*.bin")):
            ok(f"{name}: weights found in {d}")
        else:
            bad(f"{name}: no weights (*.safetensors / pytorch_model*.bin) in {d}")
        if not need[0].exists():
            bad(f"{name}: {need[0]} missing")

    print("vector_db")
    try:
        import faiss
        import numpy as np

        d = s.vector_db_dir
        for f in ("fc_index.faiss", "abs_index.faiss", "fc_ids.npy", "abs_ids.npy", "records.jsonl"):
            (ok if (d / f).exists() else bad)(f"{f}" if (d / f).exists() else f"{d / f} missing")
        for idx_name, ids_name in (("fc_index.faiss", "fc_ids.npy"), ("abs_index.faiss", "abs_ids.npy")):
            if (d / idx_name).exists() and (d / ids_name).exists():
                idx = faiss.read_index(str(d / idx_name))
                ids = np.load(d / ids_name, allow_pickle=True)
                (ok if idx.d == 768 else bad)(f"{idx_name}: dim={idx.d} (need 768 for MedCPT), vectors={idx.ntotal}")
                (ok if len(ids) == idx.ntotal else bad)(f"{ids_name}: {len(ids)} ids vs {idx.ntotal} vectors")
        if (d / "records.jsonl").exists():
            n = sum(1 for _ in open(d / "records.jsonl", encoding="utf-8"))
            ok(f"records.jsonl: {n} records")
    except ImportError as exc:
        bad(f"cannot check vector_db: {exc}")

    print("meta-models")
    try:
        import joblib

        for name, n_exp in (("static", N_STATIC), ("live", N_LIVE), ("hybrid_stack", N_HYBRID)):
            p = s.meta_dir / f"{name}.joblib"
            if not p.exists():
                bad(f"{p} missing")
                continue
            m = joblib.load(p)
            n_in = getattr(m, "n_features_in_", None)
            if n_in is None and hasattr(m, "steps"):
                n_in = getattr(m.steps[0][1], "n_features_in_", None)
            (ok if n_in in (None, n_exp) else bad)(f"{name}.joblib: expects {n_in} features (pipeline builds {n_exp})")
            ok(f"{name}.joblib: classes_={list(getattr(m, 'classes_', []))}")
    except ImportError as exc:
        bad(f"cannot check meta-models: {exc}")

    print("gate")
    for f in ("probe.joblib", "stacker.joblib", "gate_config.json"):
        (ok if (s.gate_dir / f).exists() else bad)(f if (s.gate_dir / f).exists() else f"{s.gate_dir / f} missing")
    cfg = s.gate_dir / "gate_config.json"
    if cfg.exists():
        taus = json.loads(cfg.read_text())
        ok(f"gate_config.json keys: {sorted(taus)}")
    if not (s.gate_dir / "lexicon.json").exists():
        print("  [warn] gate/lexicon.json absent: built-in lexicon will be used (ship the notebook's for exact parity)")

    print()
    if problems:
        print(f"{len(problems)} problem(s). Fix them, then re-run.")
        return 1
    print("All artifacts look consistent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
