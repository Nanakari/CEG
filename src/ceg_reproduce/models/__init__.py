"""Model factory."""

from __future__ import annotations

from typing import Any, Mapping

from ceg_reproduce.models.fake import FakeGenerator
from ceg_reproduce.models.types import ImageTextGenerator


def build_generator(config: Mapping[str, Any], *, method: str | None = None) -> ImageTextGenerator:
    generation = config.get("generation", {})
    backend = str(generation.get("backend", "llava_hf"))
    selected_method = method or str(generation.get("method", "base"))
    if backend == "fake":
        return FakeGenerator(config, method=selected_method)
    if backend in {"llava_hf", "llava"}:
        from ceg_reproduce.models.llava_hf import LlavaHfGenerator

        return LlavaHfGenerator(config)
    if backend in {"vcd_llava", "vcd"}:
        from ceg_reproduce.models.vcd_llava import VcdLlavaGenerator

        return VcdLlavaGenerator(config)
    raise ValueError(f"Unsupported generation backend: {backend}")
