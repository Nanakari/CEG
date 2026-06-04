"""MNLI entailment scoring for counterfactual persistence."""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol


@dataclass
class NliResult:
    entailment_prob: float
    label: str
    latency_sec: float = 0.0
    backend: str = "mnli"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class NliScorer(Protocol):
    def score(self, premise: str, hypothesis: str) -> NliResult:
        ...


def build_nli_scorer(config: Mapping[str, Any]) -> NliScorer:
    nli = config.get("nli", {})
    backend = str(nli.get("backend", "mnli"))
    if backend == "fake":
        return FakeNliScorer(config)
    if backend == "mnli":
        return MnliScorer(config)
    raise ValueError(f"Unsupported nli backend: {backend}")


class FakeNliScorer:
    def __init__(self, config: Mapping[str, Any]) -> None:
        fake = config.get("nli", {}).get("fake", {})
        self.overrides = {str(key).lower(): float(value) for key, value in fake.get("scores", {}).items()}

    def score(self, premise: str, hypothesis: str) -> NliResult:
        claim = _claim_from_hypothesis(hypothesis)
        prob = self.overrides.get(claim.lower())
        if prob is None:
            normalized_claim = _normalize(claim)
            prob = 0.9 if normalized_claim and normalized_claim in _normalize(premise) else 0.1
        label = "entailment" if prob > 0.5 else "neutral"
        return NliResult(entailment_prob=prob, label=label, backend="fake")


class MnliScorer:
    def __init__(self, config: Mapping[str, Any]) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional model deps
            raise RuntimeError("MNLI scoring requires torch and transformers.") from exc

        runtime = config.get("runtime", {})
        nli = config.get("nli", {})
        self._torch = torch
        self.device = str(runtime.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.model_name_or_path = str(nli.get("model_name_or_path", "roberta-large-mnli"))
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path)
        self.model.to(self.device)
        self.model.eval()
        self.entailment_id = _entailment_label_id(self.model.config.label2id)

    def score(self, premise: str, hypothesis: str) -> NliResult:
        start = time.perf_counter()
        inputs = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, padding=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.inference_mode():
            outputs = self.model(**inputs)
            probs = self._torch.softmax(outputs.logits, dim=-1)[0]
        prob = float(probs[self.entailment_id].detach().cpu().item())
        return NliResult(
            entailment_prob=prob,
            label="entailment" if prob > 0.5 else "non_entailment",
            latency_sec=time.perf_counter() - start,
            backend="mnli",
        )


def _entailment_label_id(label2id: Mapping[str, int]) -> int:
    for label, index in label2id.items():
        if "entail" in label.lower():
            return int(index)
    return 2


def _claim_from_hypothesis(hypothesis: str) -> str:
    lowered = hypothesis.lower().strip()
    match = re.search(r"contains\s+(?:a|an|the)?\s*([^\.]+)", lowered)
    if match:
        return match.group(1).strip()
    match = re.search(r"the\s+([a-z ]+)\s+is\s+([a-z-]+)", lowered)
    if match:
        return f"{match.group(2)} {match.group(1)}".strip()
    return lowered.rstrip(".")


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()
