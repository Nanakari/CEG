from __future__ import annotations

from pathlib import Path

from ceg_reproduce.extraction.claims import ClaimExtractor
from ceg_reproduce.grounding.clip_patch import PatchEvidence
from ceg_reproduce.grounding import build_grounder
from ceg_reproduce.models.fake import FakeGenerator
from ceg_reproduce.nli import build_nli_scorer
from ceg_reproduce.pipeline import run_ceg_record
from ceg_reproduce.utils.config import load_config


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
    assert "red car" in revised["revised_caption"]
    assert len(revised["verified_claims"]) == 3
    by_claim = {item["claim"]: item for item in revised["verified_claims"]}
    assert by_claim["tennis racket"]["support_low"] is True
    assert by_claim["tennis racket"]["risk"] is True
    assert by_claim["red car"]["support_low"] is False
    assert by_claim["red car"]["risk"] is False


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

    assert grounder.texts == ["red car", "man"]


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
