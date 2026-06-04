from __future__ import annotations

from pathlib import Path

from PIL import Image

from ceg_reproduce.counterfactual.masking import apply_blur_mask
from ceg_reproduce.grounding.clip_patch import FakePatchGrounder


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


def test_fake_grounder_locate_many_matches_locate_shape(tmp_path: Path) -> None:
    grounder = FakePatchGrounder(
        {
            "clip_grounding": {
                "fake": {
                    "scores": {"car": 0.8, "dog": 0.2},
                    "score_norms": {"car": 1.4, "dog": 0.3},
                }
            }
        }
    )

    evidences = grounder.locate_many(tmp_path / "image.jpg", ["car", "dog"])
    single = grounder.locate(tmp_path / "image.jpg", "car")

    assert [item.query for item in evidences] == ["car", "dog"]
    assert evidences[0].boxes == single.boxes
    assert evidences[0].metadata["support_score_norm"] == 1.4
