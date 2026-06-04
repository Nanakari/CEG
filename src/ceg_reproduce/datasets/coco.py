"""COCO Caption / CHAIR sample loading."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from ceg_reproduce.utils.config import resolve_path
from ceg_reproduce.utils.io import read_json, read_jsonl


@dataclass(frozen=True)
class CaptionSample:
    sample_id: str
    image_id: str
    image_path: str
    dataset: str
    gt_objects: list[str]

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def load_coco_caption_samples(
    dataset_config: Mapping[str, Any],
    project_root: str | Path,
    *,
    limit: int | None = None,
) -> list[CaptionSample]:
    sample_file = dataset_config.get("sample_file")
    if sample_file:
        records = list(read_jsonl(resolve_path(sample_file, project_root)))
        samples = [_sample_from_record(record, project_root) for record in records]
        return samples[:limit] if limit is not None else samples

    image_root = resolve_path(dataset_config.get("image_root"), project_root)
    instances_path = resolve_path(dataset_config.get("instances_file"), project_root)
    if not instances_path.exists():
        raise FileNotFoundError(f"COCO instances file not found: {instances_path}")
    if not image_root.exists():
        raise FileNotFoundError(f"COCO image root not found: {image_root}")

    instances = read_json(instances_path)
    categories = {
        int(category["id"]): str(category["name"]).lower()
        for category in instances.get("categories", [])
    }
    objects_by_image: dict[str, set[str]] = {}
    for annotation in instances.get("annotations", []):
        image_id = str(annotation.get("image_id"))
        name = categories.get(int(annotation.get("category_id")))
        if name:
            objects_by_image.setdefault(image_id, set()).add(name)

    samples: list[CaptionSample] = []
    for image in instances.get("images", []):
        image_id = str(image.get("id"))
        file_name = str(image.get("file_name"))
        samples.append(
            CaptionSample(
                sample_id=image_id,
                image_id=image_id,
                image_path=str(image_root / file_name),
                dataset="coco_chair",
                gt_objects=sorted(objects_by_image.get(image_id, set())),
            )
        )
        if limit is not None and len(samples) >= limit:
            break
    return samples


def _sample_from_record(record: Mapping[str, Any], project_root: str | Path) -> CaptionSample:
    image_path = str(record.get("image_path", ""))
    resolved = resolve_path(image_path, project_root) if image_path else Path("")
    return CaptionSample(
        sample_id=str(record.get("sample_id") or record.get("image_id")),
        image_id=str(record.get("image_id") or record.get("sample_id")),
        image_path=str(resolved),
        dataset=str(record.get("dataset", "coco_chair")),
        gt_objects=[str(item).lower() for item in record.get("gt_objects", [])],
    )
