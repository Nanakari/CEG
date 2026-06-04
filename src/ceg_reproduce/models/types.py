"""Generation type contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass
class GenerationResult:
    text: str
    latency_sec: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ImageTextGenerator(Protocol):
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
        ...
