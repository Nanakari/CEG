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
            "risk_count": 2,
            "revision_count": 1,
            "latency_sec": 2.0,
        }
    ]

    metrics = compute_chair_metrics(records, extractor, base_latency_sec=1.0)

    assert metrics["chairs"] == 1.0
    assert metrics["chairi"] == 1 / 3
    assert metrics["recall"] == 1.0
    assert metrics["risk_count"] == 2.0
    assert metrics["revision_count"] == 1.0
    assert metrics["relative_time"] == 2.0


def test_chair_metrics_track_rejection_and_removal_rates() -> None:
    config = load_config(ROOT / "configs/smoke.yaml")
    extractor = ClaimExtractor.from_config(config, ROOT)
    records = [
        {
            "original_caption": "A person is beside a car and a tennis racket.",
            "revised_caption": "A person is beside an object.",
            "gt_objects": ["person", "car"],
            "latency_sec": 2.0,
        }
    ]

    metrics = compute_chair_metrics(records, extractor, base_latency_sec=1.0)

    assert metrics["false_rejection_rate"] == 0.5
    assert metrics["correct_retention_rate"] == 0.5
    assert metrics["hallucinated_removal_rate"] == 1.0


def test_chair_metrics_count_plural_objects_for_recall() -> None:
    config = load_config(ROOT / "configs/smoke.yaml")
    extractor = ClaimExtractor.from_config(config, ROOT)
    records = [
        {
            "caption": "Two dogs sit on chairs beside cars.",
            "gt_objects": ["dog", "chair", "car"],
        }
    ]

    metrics = compute_chair_metrics(records, extractor)

    assert metrics["recall"] == 1.0
    assert metrics["chairi"] == 0.0


def test_chair_metrics_use_chair_style_synonyms_and_multiword_matching() -> None:
    config = load_config(ROOT / "configs/smoke.yaml")
    extractor = ClaimExtractor.from_config(config, ROOT)
    records = [
        {
            "caption": "People sit at tables with cellphones and hot dogs.",
            "gt_objects": ["person", "dining table", "cell phone", "hot dog"],
        }
    ]

    metrics = compute_chair_metrics(records, extractor)

    assert metrics["recall"] == 1.0
    assert metrics["chairi"] == 0.0


def test_chair_metrics_preserve_empty_revised_caption() -> None:
    config = load_config(ROOT / "configs/smoke.yaml")
    extractor = ClaimExtractor.from_config(config, ROOT)
    records = [
        {
            "original_caption": "A tennis racket.",
            "caption": "A tennis racket.",
            "revised_caption": "",
            "gt_objects": [],
        }
    ]

    metrics = compute_chair_metrics(records, extractor)

    assert metrics["chairs"] == 0.0
    assert metrics["chairi"] == 0.0
    assert metrics["average_length"] == 0.0
