"""Blur-mask local evidence regions."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFilter


def apply_blur_mask(
    image_path: str | Path,
    boxes: Iterable[Iterable[int | float]],
    output_path: str | Path,
    *,
    blur_radius: int = 12,
    expand_pixels: int = 8,
) -> Path:
    source = Path(image_path)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        return source

    image = Image.open(source).convert("RGB")
    blurred = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    width, height = image.size
    for raw_box in boxes:
        left, top, right, bottom = [int(round(float(item))) for item in raw_box]
        box = (
            max(0, left - expand_pixels),
            max(0, top - expand_pixels),
            min(width, right + expand_pixels),
            min(height, bottom + expand_pixels),
        )
        draw.rectangle(box, fill=255)
    image.paste(blurred, mask=mask)
    image.save(target)
    return target
