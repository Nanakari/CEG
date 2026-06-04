"""Deterministic fake generator for smoke tests."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

from ceg_reproduce.models.types import GenerationResult


class FakeGenerator:
    def __init__(self, config: Mapping[str, Any], *, method: str = "base") -> None:
        fake = config.get("generation", {}).get("fake", {})
        self.method = method
        self.outputs = {str(key): str(value) for key, value in fake.get("outputs", {}).items()}
        self.counterfactual_outputs = {
            str(key): str(value) for key, value in fake.get("counterfactual_outputs", {}).items()
        }
        self.default_caption = str(
            fake.get("default_caption", "A person is sitting beside a car.")
        )
        self.default_answer = str(fake.get("default_answer", self.default_caption))

    def generate(
        self,
        image_path: str | Path,
        prompt: str,
        *,
        sample_id: str | None = None,
        max_new_tokens: int | None = None,
    ) -> GenerationResult:
        start = time.perf_counter()
        text = self._select_text(sample_id or "", prompt)
        return GenerationResult(
            text=text,
            latency_sec=time.perf_counter() - start,
            metadata={
                "backend": "fake",
                "method": self.method,
                "max_new_tokens": max_new_tokens,
                "image_path": str(image_path),
            },
        )

    def _select_text(self, sample_id: str, prompt: str) -> str:
        if "::cf::" in sample_id:
            slug = sample_id.split("::cf::", 1)[1]
            return self.counterfactual_outputs.get(slug, self.default_answer)
        method_key = f"{sample_id}::{self.method}"
        if method_key in self.outputs:
            return self.outputs[method_key]
        if sample_id in self.outputs:
            return self.outputs[sample_id]
        lowered = prompt.lower()
        if "question" in lowered and "answer" in lowered:
            return self.default_answer
        return self.default_caption
