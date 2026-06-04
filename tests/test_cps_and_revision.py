from __future__ import annotations

from pathlib import Path

from ceg_reproduce.extraction.claims import ClaimExtractor
from ceg_reproduce.grounding.clip_patch import PatchEvidence
from ceg_reproduce.grounding import build_grounder
from ceg_reproduce.models.fake import FakeGenerator
from ceg_reproduce.nli import build_nli_scorer
from ceg_reproduce.pipeline import run_ceg_record
from ceg_reproduce.utils.config import config_for_method, load_config


ROOT = Path(__file__).resolve().parents[1]


def test_counterfactual_persistence_generalizes_high_risk_object() -> None:
    config = load_config(ROOT / "configs/smoke.yaml")
    extractor = ClaimExtractor.from_config(config, ROOT)
    record = {
        "sample_id": "smoke_1",
        "image_id": "smoke_1",
        "image_path": str(ROOT / "examples/smoke/smoke_1.jpg"),
        "prompt": config["prompts"]["caption"],
        "caption": "A man is holding a tennis racket beside a red car.",
        "gt_objects": ["person", "car"],
        "latency_sec": 1.0,
    }

    revised = run_ceg_record(
        record,
        FakeGenerator(config, method="base"),
        build_grounder(config),
        build_nli_scorer(config),
        extractor,
        config,
        ROOT,
    )

    assert "tennis racket" not in revised["revised_caption"]
    assert "an object" in revised["revised_caption"]
    assert "red car" not in revised["revised_caption"]
    assert "car" in revised["revised_caption"]
    assert len(revised["verified_claims"]) == 3
    by_claim = {item["claim"]: item for item in revised["verified_claims"]}
    assert by_claim["tennis racket"]["original_state"] == "no"
    assert by_claim["tennis racket"]["counterfactual_state"] == "no"
    assert by_claim["tennis racket"]["risk"] is True
    assert by_claim["tennis racket"]["risk_reason"] == "no_and_low_support"
    assert by_claim["red car"]["risk_reason"] == "attribute_no"
    assert by_claim["red car"]["risk"] is True
    assert revised["risk_count"] == 2
    assert revised["revision_count"] == 2
    assert revised["external_lvlm_calls"] == 6


def test_original_yes_counterfactual_no_keeps_supported_object() -> None:
    config = load_config(ROOT / "configs/smoke.yaml")
    extractor = ClaimExtractor.from_config(config, ROOT)
    record = {
        "sample_id": "smoke_1",
        "image_id": "smoke_1",
        "image_path": str(ROOT / "examples/smoke/smoke_1.jpg"),
        "prompt": config["prompts"]["caption"],
        "caption": "A man is standing beside a car.",
        "gt_objects": ["person", "car"],
        "latency_sec": 1.0,
    }

    revised = run_ceg_record(
        record,
        FakeGenerator(config, method="base"),
        build_grounder(config),
        build_nli_scorer(config),
        extractor,
        config,
        ROOT,
    )

    assert revised["revised_caption"] == record["caption"]
    assert revised["risk_count"] == 0


def test_counterfactual_yes_is_diagnostic_not_direct_object_risk() -> None:
    config = load_config(ROOT / "configs/smoke.yaml")
    config["generation"]["fake"]["vqa_outputs"]["counterfactual:man"] = "Yes."
    extractor = ClaimExtractor.from_config(config, ROOT)
    record = {
        "sample_id": "smoke_1",
        "image_id": "smoke_1",
        "image_path": str(ROOT / "examples/smoke/smoke_1.jpg"),
        "prompt": config["prompts"]["caption"],
        "caption": "A man is standing beside a car.",
        "gt_objects": ["person", "car"],
        "latency_sec": 1.0,
    }

    revised = run_ceg_record(
        record,
        FakeGenerator(config, method="base"),
        build_grounder(config),
        build_nli_scorer(config),
        extractor,
        config,
        ROOT,
    )

    man = next(item for item in revised["verified_claims"] if item["claim"] == "man")
    assert man["counterfactual_state"] == "yes"
    assert man["risk"] is False
    assert man["risk_reason"] == "counterfactual_vqa_persistence_diagnostic"
    assert revised["revised_caption"] == record["caption"]


def test_support_guard_retains_object_when_vqa_no_but_local_support_is_high() -> None:
    config = load_config(ROOT / "configs/smoke.yaml")
    config["clip_grounding"]["fake"]["score_norms"]["tennis racket"] = 2.0
    extractor = ClaimExtractor.from_config(config, ROOT)
    record = {
        "sample_id": "smoke_1",
        "image_id": "smoke_1",
        "image_path": str(ROOT / "examples/smoke/smoke_1.jpg"),
        "prompt": config["prompts"]["caption"],
        "caption": "A man is holding a tennis racket.",
        "gt_objects": ["person", "tennis racket"],
        "latency_sec": 1.0,
    }

    revised = run_ceg_record(
        record,
        FakeGenerator(config, method="base"),
        build_grounder(config),
        build_nli_scorer(config),
        extractor,
        config,
        ROOT,
    )

    racket = next(item for item in revised["verified_claims"] if item["claim"] == "tennis racket")
    assert racket["original_state"] == "no"
    assert racket["support_low"] is False
    assert racket["risk"] is False
    assert racket["risk_reason"] == "no_but_local_support"
    assert revised["revised_caption"] == record["caption"]


def test_ceg_only_grounds_selected_topk_claims() -> None:
    config = load_config(ROOT / "configs/smoke.yaml")
    config["ceg"]["top_k"] = 2
    extractor = ClaimExtractor.from_config(config, ROOT)
    grounder = SpyGrounder()
    record = {
        "sample_id": "smoke_1",
        "image_id": "smoke_1",
        "image_path": str(ROOT / "examples/smoke/smoke_1.jpg"),
        "prompt": config["prompts"]["caption"],
        "caption": "A man is holding a tennis racket beside a red car.",
        "gt_objects": ["person", "car"],
        "latency_sec": 1.0,
    }

    run_ceg_record(
        record,
        FakeGenerator(config, method="base"),
        grounder,
        build_nli_scorer(config),
        extractor,
        config,
        ROOT,
    )

    assert grounder.texts == ["man", "tennis racket"]


def test_ceg_method_defaults_are_applied() -> None:
    config = load_config(ROOT / "configs/smoke.yaml")
    config["ceg"]["risk_mode"] = "targeted_vqa"

    method_config = config_for_method(config, "ceg")

    assert method_config["ceg"]["risk_mode"] == "targeted_vqa_support_guard"


class SpyGrounder:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def locate_many(self, image_path: str | Path, texts: list[str]) -> list[PatchEvidence]:
        self.texts = list(texts)
        return [
            PatchEvidence(
                query=text,
                boxes=[[0, 0, 16, 16]],
                support_score=0.9,
                metadata={
                    "support_score_norm": 2.0,
                    "score_mean": 0.0,
                    "score_std": 1.0,
                    "num_patches": 1,
                },
            )
            for text in texts
        ]

    def locate(self, image_path: str | Path, query: str) -> PatchEvidence:
        return self.locate_many(image_path, [query])[0]
