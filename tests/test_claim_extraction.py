from __future__ import annotations

from pathlib import Path

from ceg_reproduce.extraction.claims import ClaimExtractor
from ceg_reproduce.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_extracts_object_and_attribute_claims() -> None:
    config = load_config(ROOT / "configs/smoke.yaml")
    extractor = ClaimExtractor.from_config(config, ROOT)

    claims = extractor.extract("A man is holding a tennis racket beside a red car.")

    by_text = {claim.text.lower(): claim for claim in claims}
    assert by_text["man"].normalized == "person"
    assert by_text["tennis racket"].claim_type == "object"
    assert by_text["red car"].claim_type == "attribute"
    assert by_text["red car"].attribute == "red"
    assert by_text["red car"].normalized == "car"


def test_orange_modifier_is_not_extracted_as_fruit_object() -> None:
    config = load_config(ROOT / "configs/smoke.yaml")
    extractor = ClaimExtractor.from_config(config, ROOT)

    claims = extractor.extract("A truck carries a tall orange load.")

    assert "orange" not in {claim.text.lower() for claim in claims}
