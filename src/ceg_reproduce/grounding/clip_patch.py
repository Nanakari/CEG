"""CLIP patch-text localization for claim evidence regions."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol

from PIL import Image


@dataclass
class PatchEvidence:
    query: str
    boxes: list[list[int]]
    support_score: float
    latency_sec: float = 0.0
    backend: str = "clip"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Grounder(Protocol):
    def locate(self, image_path: str | Path, query: str) -> PatchEvidence:
        ...


def build_grounder(config: Mapping[str, Any]) -> Grounder:
    grounding = config.get("clip_grounding", {})
    backend = str(grounding.get("backend", "clip"))
    if backend == "fake":
        return FakePatchGrounder(config)
    if backend == "clip":
        return ClipPatchGrounder(config)
    raise ValueError(f"Unsupported clip_grounding backend: {backend}")


class FakePatchGrounder:
    def __init__(self, config: Mapping[str, Any]) -> None:
        fake = config.get("clip_grounding", {}).get("fake", {})
        self.default_score = float(fake.get("default_score", 0.5))
        self.scores = {str(key).lower(): float(value) for key, value in fake.get("scores", {}).items()}

    def locate(self, image_path: str | Path, query: str) -> PatchEvidence:
        key = str(query).lower()
        return PatchEvidence(
            query=query,
            boxes=[[0, 0, 16, 16]],
            support_score=self.scores.get(key, self.default_score),
            backend="fake",
            metadata={"image_path": str(image_path)},
        )


class ClipPatchGrounder:
    def __init__(self, config: Mapping[str, Any]) -> None:
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:  # pragma: no cover - optional model deps
            raise RuntimeError("CLIP patch grounding requires torch and transformers.") from exc

        runtime = config.get("runtime", {})
        grounding = config.get("clip_grounding", {})
        self._torch = torch
        self.model_name_or_path = str(
            grounding.get("model_name_or_path", "openai/clip-vit-base-patch16")
        )
        self.device = str(runtime.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.patch_size = int(grounding.get("patch_size", 16))
        self.top_m = int(grounding.get("top_m", 4))
        self.batch_size = int(grounding.get("batch_size", 64))
        self.processor = CLIPProcessor.from_pretrained(self.model_name_or_path)
        self.model = CLIPModel.from_pretrained(self.model_name_or_path)
        self.model.to(self.device)
        self.model.eval()

    def locate(self, image_path: str | Path, query: str) -> PatchEvidence:
        start = time.perf_counter()
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        image = Image.open(path).convert("RGB")
        patches = _make_patches(image, self.patch_size)
        patch_images = [item[0] for item in patches]
        boxes = [item[1] for item in patches]
        if not patch_images:
            return PatchEvidence(query=query, boxes=[], support_score=0.0)
        text_query = _query_text(query)
        scores: list[float] = []
        with self._torch.inference_mode():
            text_inputs = self.processor(text=[text_query], return_tensors="pt", padding=True)
            text_inputs = {key: value.to(self.device) for key, value in text_inputs.items()}
            text_features = self.model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            for batch_start in range(0, len(patch_images), self.batch_size):
                batch = patch_images[batch_start : batch_start + self.batch_size]
                image_inputs = self.processor(images=batch, return_tensors="pt")
                image_inputs = {key: value.to(self.device) for key, value in image_inputs.items()}
                image_features = self.model.get_image_features(**image_inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                batch_scores = (image_features @ text_features.T).squeeze(-1)
                scores.extend(float(item) for item in batch_scores.detach().cpu().tolist())
        ranked = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
        top_indices = ranked[: self.top_m]
        return PatchEvidence(
            query=query,
            boxes=[boxes[index] for index in top_indices],
            support_score=float(max((scores[index] for index in top_indices), default=0.0)),
            latency_sec=time.perf_counter() - start,
            backend="clip",
            metadata={"model_name_or_path": self.model_name_or_path, "patch_size": self.patch_size},
        )


def _make_patches(image: Image.Image, patch_size: int) -> list[tuple[Image.Image, list[int]]]:
    width, height = image.size
    patches: list[tuple[Image.Image, list[int]]] = []
    for top in range(0, height, patch_size):
        for left in range(0, width, patch_size):
            right = min(left + patch_size, width)
            bottom = min(top + patch_size, height)
            if right > left and bottom > top:
                patches.append((image.crop((left, top, right, bottom)), [left, top, right, bottom]))
    return patches


def _query_text(query: str) -> str:
    lowered = query.lower().strip()
    article = "an" if lowered[:1] in {"a", "e", "i", "o", "u"} else "a"
    return f"{article} {lowered}"
