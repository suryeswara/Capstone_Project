"""
Neural model wrappers: MedCPT query encoder, MonoT5 reranker and the
DeBERTa + BiomedBERT pair-classifier ensemble.

torch / transformers are imported lazily so that the pure-logic parts of the
pipeline (features, meta routing, gate zones) can be imported and unit-tested
without them.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Sequence

import numpy as np

from .schema import CLASSES, ArtifactError

logger = logging.getLogger(__name__)


def resolve_device(pref: str = "auto") -> str:
    import torch

    if pref != "auto":
        return pref
    return "cuda" if torch.cuda.is_available() else "cpu"


# Spelling variants of the three classes found in HF id2label maps.
_LABEL_ALIASES = {
    "SUPPORT": "SUPPORT", "SUPPORTS": "SUPPORT", "SUPPORTED": "SUPPORT", "ENTAILMENT": "SUPPORT", "TRUE": "SUPPORT",
    "CONTRADICT": "CONTRADICT", "CONTRADICTS": "CONTRADICT", "CONTRADICTION": "CONTRADICT",
    "REFUTE": "CONTRADICT", "REFUTES": "CONTRADICT", "REFUTED": "CONTRADICT", "FALSE": "CONTRADICT",
    "NEI": "NEI", "NEUTRAL": "NEI", "NOT ENOUGH INFO": "NEI", "NOT_ENOUGH_INFO": "NEI",
}


def class_permutation(id2label: dict, where: str) -> List[int]:
    """Indices that re-order a model's logits into CLASSES order.

    ``id2label`` comes from the model's config.json. Generic names such as
    ``LABEL_0`` carry no information, so in that case the report's documented
    order (SUPPORT, CONTRADICT, NEI) is assumed and a warning is logged.
    """
    names = {int(i): str(n).strip().upper() for i, n in id2label.items()}
    if len(names) != len(CLASSES):
        raise ArtifactError(f"{where}: expected {len(CLASSES)} labels, found {len(names)}")
    if all(n.startswith("LABEL_") for n in names.values()):
        logger.warning("%s: id2label is generic (%s); assuming order %s", where, names, CLASSES)
        return [0, 1, 2]
    try:
        canon = {i: _LABEL_ALIASES[n] for i, n in names.items()}
    except KeyError as exc:
        raise ArtifactError(f"{where}: unrecognised label name {exc} in id2label {names}") from exc
    if sorted(canon.values()) != sorted(CLASSES):
        raise ArtifactError(f"{where}: id2label {names} does not cover {CLASSES}")
    inverse = {v: k for k, v in canon.items()}
    return [inverse[c] for c in CLASSES]


# ---------------------------------------------------------------------------
# MedCPT query encoder (claim -> 768-d vector)
# ---------------------------------------------------------------------------

class QueryEncoder:
    """ncbi/MedCPT-Query-Encoder: CLS vector, L2-normalised, <= 64 tokens."""

    def __init__(self, name: str, max_tokens: int, device: str):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.device = device
        self.max_tokens = max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(device).eval()
        self.dim = int(self.model.config.hidden_size)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        torch = self._torch
        enc = self.tokenizer(
            list(texts), truncation=True, padding=True, max_length=self.max_tokens, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            cls = self.model(**enc).last_hidden_state[:, 0, :]
            cls = torch.nn.functional.normalize(cls, p=2, dim=1)
        return cls.float().cpu().numpy().astype("float32")


# ---------------------------------------------------------------------------
# MonoT5 reranker
# ---------------------------------------------------------------------------

class MonoT5Reranker:
    """
    Scores (claim, document) as P("true") from one decoder step.

    Input format: ``Query: <claim> Document: <text> Relevant:``. fp32 on
    purpose -- T5 overflows in fp16 (report 4.4). The meta-models were fitted
    on exactly these scores, so do not rescale or calibrate them.
    """

    MAX_INPUT_TOKENS = 512
    DOC_WORD_LIMIT = 400

    def __init__(self, name: str, device: str, batch_size: int = 8):
        import torch
        from transformers import AutoTokenizer, T5ForConditionalGeneration

        self._torch = torch
        self.device = device
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = T5ForConditionalGeneration.from_pretrained(name, torch_dtype=torch.float32).to(device).eval()
        ids = self.tokenizer.convert_tokens_to_ids(["▁false", "▁true"])
        if any(i is None or i == self.tokenizer.unk_token_id for i in ids):
            raise ArtifactError("MonoT5 tokenizer is missing the '▁false'/'▁true' tokens")
        self.false_id, self.true_id = ids

    def score(self, claim: str, docs: Sequence[str]) -> np.ndarray:
        torch = self._torch
        out: List[float] = []
        for start in range(0, len(docs), self.batch_size):
            chunk = docs[start:start + self.batch_size]
            inputs = [
                f"Query: {claim} Document: {' '.join(d.split()[: self.DOC_WORD_LIMIT])} Relevant:" for d in chunk
            ]
            enc = self.tokenizer(
                inputs, truncation=True, padding=True, max_length=self.MAX_INPUT_TOKENS, return_tensors="pt"
            ).to(self.device)
            start_ids = torch.full(
                (enc["input_ids"].shape[0], 1), self.model.config.decoder_start_token_id, device=self.device
            )
            with torch.no_grad():
                logits = self.model(**enc, decoder_input_ids=start_ids).logits[:, 0, :]
                two = torch.stack([logits[:, self.false_id], logits[:, self.true_id]], dim=1)
                out.extend(torch.softmax(two, dim=1)[:, 1].float().cpu().tolist())
        return np.asarray(out, dtype=np.float64)


# ---------------------------------------------------------------------------
# Pair classifier ensemble: mean of DeBERTa-v3 and BiomedBERT softmax
# ---------------------------------------------------------------------------

class PairEnsemble:
    """Average of the two fine-tuned cross-encoders, 3-class, CLASSES order."""

    def __init__(self, deberta_dir: Path, biomed_dir: Path, max_length: int, device: str, batch_size: int = 8):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.members = []
        for label, d in (("deberta", deberta_dir), ("biomed", biomed_dir)):
            d = Path(d)
            if not (d / "config.json").exists():
                raise ArtifactError(f"{label} classifier not found at {d} (missing config.json)")
            cfg = json.loads((d / "config.json").read_text())
            perm = class_permutation(cfg.get("id2label", {}), f"{label} classifier")
            tok = AutoTokenizer.from_pretrained(str(d))
            model = AutoModelForSequenceClassification.from_pretrained(str(d))
            if model.config.num_labels != len(CLASSES):
                raise ArtifactError(f"{label} classifier has {model.config.num_labels} labels, expected 3")
            # Stored as fp16 (report section 7). fp16 on CPU is slow/unsupported -> fp32 there.
            model = (model.half() if device.startswith("cuda") else model.float()).to(device).eval()
            self.members.append((label, tok, model, perm))

    def predict(self, claim: str, evidence_texts: Sequence[str]) -> np.ndarray:
        """(n, 3) probabilities for the pairs (claim, evidence), CLASSES order."""
        torch = self._torch
        n = len(evidence_texts)
        if n == 0:
            return np.zeros((0, len(CLASSES)), dtype=np.float64)
        total = np.zeros((n, len(CLASSES)), dtype=np.float64)
        for _, tok, model, perm in self.members:
            rows = []
            for s in range(0, n, self.batch_size):
                chunk = list(evidence_texts[s:s + self.batch_size])
                enc = tok(
                    [claim] * len(chunk), chunk, truncation=True, padding=True,
                    max_length=self.max_length, return_tensors="pt",
                ).to(self.device)
                with torch.no_grad():
                    logits = model(**enc).logits.float()
                rows.append(torch.softmax(logits, dim=-1).cpu().numpy())
            probs = np.concatenate(rows, axis=0)[:, perm]
            total += probs
        return total / len(self.members)
