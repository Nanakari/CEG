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


def test_extracts_plural_coco_objects() -> None:
    config = load_config(ROOT / "configs/smoke.yaml")
    extractor = ClaimExtractor.from_config(config, ROOT)

    claims = extractor.extract("Two dogs sit on chairs beside traffic lights and wine glasses.")

    by_text = {claim.text.lower(): claim for claim in claims}
    assert by_text["dogs"].normalized == "dog"
    assert by_text["chairs"].normalized == "chair"
    assert by_text["traffic lights"].normalized == "traffic light"
    assert by_text["wine glasses"].normalized == "wine glass"


def test_extracts_chair_synonyms_and_protects_multiword_objects() -> None:
    config = load_config(ROOT / "configs/smoke.yaml")
    extractor = ClaimExtractor.from_config(config, ROOT)

    claims = extractor.extract(
        "People sit on sofas near tables with cellphones, bikes, knives, and hot dogs."
    )

    by_text = {claim.text.lower(): claim for claim in claims}
    assert by_text["people"].normalized == "person"
    assert by_text["sofas"].normalized == "couch"
    assert by_text["tables"].normalized == "dining table"
    assert by_text["cellphones"].normalized == "cell phone"
    assert by_text["bikes"].normalized == "bicycle"
    assert by_text["knives"].normalized == "knife"
    assert by_text["hot dogs"].normalized == "hot dog"
    assert "dogs" not in by_text


def test_extracts_additional_chair_aliases() -> None:
    config = load_config(ROOT / "configs/smoke.yaml")
    extractor = ClaimExtractor.from_config(config, ROOT)

    claims = extractor.extract("Cell phones sit beside a hair dryer, a ski, and snowboards.")

    by_text = {claim.text.lower(): claim for claim in claims}
    assert by_text["cell phones"].normalized == "cell phone"
    assert by_text["hair dryer"].normalized == "hair drier"
    assert by_text["ski"].normalized == "skis"
    assert by_text["snowboards"].normalized == "snowboard"


def test_extracts_size_age_attributes() -> None:
    config = load_config(ROOT / "configs/smoke.yaml")
    extractor = ClaimExtractor.from_config(config, ROOT)

    claims = extractor.extract("A young boy stands near a small car.")

    by_text = {claim.text.lower(): claim for claim in claims}
    assert by_text["young boy"].claim_type == "attribute"
    assert by_text["young boy"].attribute == "young"
    assert by_text["young boy"].object_text == "boy"
    assert by_text["small car"].claim_type == "attribute"
