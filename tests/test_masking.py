from __future__ import annotations

from pathlib import Path

from PIL import Image

from ceg_reproduce.counterfactual.masking import apply_blur_mask


def test_blur_mask_writes_counterfactual_image(tmp_path: Path) -> None:
    source = tmp_path / "source.jpg"
    target = tmp_path / "masked.jpg"
    image = Image.new("RGB", (32, 32), "white")
    for x in range(16):
        for y in range(32):
            image.putpixel((x, y), (0, 0, 0))
    image.save(source)

    result = apply_blur_mask(source, [[12, 0, 20, 32]], target, blur_radius=4, expand_pixels=0)

    assert result == target
    assert target.exists()
    assert Image.open(target).size == (32, 32)
