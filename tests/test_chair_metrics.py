from __future__ import annotations

from pathlib import Path

from ceg_reproduce.evaluation.chair import compute_chair_metrics
from ceg_reproduce.extraction.claims import ClaimExtractor
from ceg_reproduce.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_chair_metrics_count_hallucinated_objects_and_recall() -> None:
    config = load_config(ROOT / "configs/smoke.yaml")
    extractor = ClaimExtractor.from_config(config, ROOT)
    records = [
        {
            "caption": "A person is beside a car and a tennis racket.",
            "gt_objects": ["person", "car"],
            "latency_sec": 2.0,
        }
    ]

    metrics = compute_chair_metrics(records, extractor, base_latency_sec=1.0)

    assert metrics["chairs"] == 1.0
    assert metrics["chairi"] == 1 / 3
    assert metrics["recall"] == 1.0
    assert metrics["relative_time"] == 2.0
