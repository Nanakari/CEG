from __future__ import annotations

from pathlib import Path

from ceg_reproduce.extraction.claims import ClaimExtractor
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
    assert "red car" not in revised["revised_caption"]
    assert "car" in revised["revised_caption"]
    assert len(revised["verified_claims"]) == 2
