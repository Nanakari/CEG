"""Deterministic fake generator for smoke tests."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

from ceg_reproduce.models.types import GenerationResult


class FakeGenerator:
    def __init__(self, config: Mapping[str, Any], *, method: str = "base") -> None:
        fake = config.get("generation", {}).get("fake", {})
        generation = config.get("generation", {})
        self.method = method
        self.do_sample = bool(generation.get("do_sample", False))
        self.temperature = float(generation.get("temperature", 1.0))
        self.top_p = float(generation.get("top_p", 1.0))
        self.outputs = {str(key): str(value) for key, value in fake.get("outputs", {}).items()}
        self.counterfactual_outputs = {
            str(key): str(value) for key, value in fake.get("counterfactual_outputs", {}).items()
        }
        self.vqa_outputs = {
            str(key): str(value) for key, value in fake.get("vqa_outputs", {}).items()
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
        do_sample: bool | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
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
                "do_sample": self.do_sample if do_sample is None else bool(do_sample),
                "temperature": self.temperature if temperature is None else float(temperature),
                "top_p": self.top_p if top_p is None else float(top_p),
                "image_path": str(image_path),
            },
        )

    def _select_text(self, sample_id: str, prompt: str) -> str:
        if "::cf::" in sample_id:
            slug = sample_id.split("::cf::", 1)[1]
            return self.counterfactual_outputs.get(slug, self.default_answer)
        if "::vqa::" in sample_id:
            slug = sample_id.split("::vqa::", 1)[1]
            return self.vqa_outputs.get(f"original:{slug}", self.vqa_outputs.get(slug, "yes"))
        if "::cf_vqa::" in sample_id:
            slug = sample_id.split("::cf_vqa::", 1)[1]
            return self.vqa_outputs.get(
                f"counterfactual:{slug}",
                self.vqa_outputs.get(f"cf:{slug}", "no"),
            )
        method_key = f"{sample_id}::{self.method}"
        if method_key in self.outputs:
            return self.outputs[method_key]
        if sample_id in self.outputs:
            return self.outputs[sample_id]
        lowered = prompt.lower()
        if "question" in lowered and "answer" in lowered:
            return self.default_answer
        return self.default_caption
