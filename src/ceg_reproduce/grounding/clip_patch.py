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

    def locate_many(self, image_path: str | Path, texts: list[str]) -> list[PatchEvidence]:
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
        self.default_score_norm = float(fake.get("default_score_norm", self.default_score))
        self.score_norms = {
            str(key).lower(): float(value) for key, value in fake.get("score_norms", {}).items()
        }

    def locate(self, image_path: str | Path, query: str) -> PatchEvidence:
        return self.locate_many(image_path, [query])[0]

    def locate_many(self, image_path: str | Path, texts: list[str]) -> list[PatchEvidence]:
        evidences = []
        for query in texts:
            key = str(query).lower()
            score = self.scores.get(key, self.default_score)
            score_norm = self.score_norms.get(key, self.default_score_norm)
            evidences.append(
                PatchEvidence(
                    query=query,
                    boxes=[[0, 0, 16, 16]],
                    support_score=score,
                    backend="fake",
                    metadata={
                        "image_path": str(image_path),
                        "support_score_norm": score_norm,
                        "score_mean": 0.0,
                        "score_std": 1.0,
                        "num_patches": 1,
                    },
                )
            )
        return evidences


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
        return self.locate_many(image_path, [query])[0]

    def locate_many(self, image_path: str | Path, texts: list[str]) -> list[PatchEvidence]:
        if not texts:
            return []
        start = time.perf_counter()
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        image = Image.open(path).convert("RGB")
        patches = _make_patches(image, self.patch_size)
        patch_images = [item[0] for item in patches]
        boxes = [item[1] for item in patches]
        if not patch_images:
            return [
                PatchEvidence(
                    query=query,
                    boxes=[],
                    support_score=0.0,
                    metadata={"support_score_norm": 0.0, "score_mean": 0.0, "score_std": 0.0, "num_patches": 0},
                )
                for query in texts
            ]
        text_queries = [_query_text(query) for query in texts]
        image_feature_batches = []
        with self._torch.inference_mode():
            text_inputs = self.processor(text=text_queries, return_tensors="pt", padding=True)
            text_inputs = {key: value.to(self.device) for key, value in text_inputs.items()}
            text_features = self.model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            for batch_start in range(0, len(patch_images), self.batch_size):
                batch = patch_images[batch_start : batch_start + self.batch_size]
                image_inputs = self.processor(images=batch, return_tensors="pt")
                image_inputs = {key: value.to(self.device) for key, value in image_inputs.items()}
                image_features = self.model.get_image_features(**image_inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                image_feature_batches.append(image_features)
            patch_features = self._torch.cat(image_feature_batches, dim=0)
            score_matrix = text_features @ patch_features.T
            score_rows = score_matrix.detach().cpu().tolist()
        latency = time.perf_counter() - start
        latency_per_query = latency / len(texts) if texts else 0.0
        evidences = []
        for query, scores in zip(texts, score_rows):
            ranked = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
            top_indices = ranked[: self.top_m]
            score_mean = float(sum(scores) / len(scores)) if scores else 0.0
            score_std = _std(scores, score_mean)
            support_score = float(max((scores[index] for index in top_indices), default=0.0))
            support_score_norm = (support_score - score_mean) / (score_std + 1e-6)
            evidences.append(
                PatchEvidence(
                    query=query,
                    boxes=[boxes[index] for index in top_indices],
                    support_score=support_score,
                    latency_sec=latency_per_query,
                    backend="clip",
                    metadata={
                        "model_name_or_path": self.model_name_or_path,
                        "patch_size": self.patch_size,
                        "support_score_norm": support_score_norm,
                        "score_mean": score_mean,
                        "score_std": score_std,
                        "num_patches": len(scores),
                    },
                )
            )
        return evidences


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


def _std(values: list[float], mean: float) -> float:
    if not values:
        return 0.0
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return variance ** 0.5
