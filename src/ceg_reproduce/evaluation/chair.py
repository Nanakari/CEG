"""COCO-object-compatible CHAIR-style metrics."""

from __future__ import annotations

from typing import Any, Mapping

from ceg_reproduce.extraction.claims import ClaimExtractor


def compute_chair_metrics(
    records: list[Mapping[str, Any]],
    extractor: ClaimExtractor,
    *,
    base_latency_sec: float | None = None,
) -> dict[str, float]:
    sentence_total = 0
    hallucinated_sentence_total = 0
    object_total = 0
    hallucinated_object_total = 0
    gt_total = 0
    recalled_total = 0
    lengths: list[int] = []
    verified_claim_counts: list[int] = []
    latencies: list[float] = []

    for record in records:
        caption = str(record.get("revised_caption") or record.get("caption") or record.get("text") or "")
        claims = extractor.extract(caption)
        predicted = [claim.normalized for claim in claims]
        predicted_set = set(predicted)
        gt_objects = {str(item).lower() for item in record.get("gt_objects", [])}
        hallucinated = [obj for obj in predicted if obj and gt_objects and obj not in gt_objects]
        sentence_total += 1
        object_total += len(predicted)
        hallucinated_object_total += len(hallucinated)
        hallucinated_sentence_total += int(bool(hallucinated))
        gt_total += len(gt_objects)
        recalled_total += len(predicted_set & gt_objects)
        lengths.append(len(caption.split()))
        verified_claim_counts.append(len(record.get("verified_claims", []) or []))
        latencies.append(float(record.get("latency_sec", 0.0)))

    avg_latency = _mean(latencies)
    relative_time = avg_latency / base_latency_sec if base_latency_sec and base_latency_sec > 0 else 1.0
    return {
        "chairs": hallucinated_sentence_total / sentence_total if sentence_total else 0.0,
        "chairi": hallucinated_object_total / object_total if object_total else 0.0,
        "recall": recalled_total / gt_total if gt_total else 0.0,
        "average_length": _mean(lengths),
        "verified_claims": _mean(verified_claim_counts),
        "relative_time": relative_time,
        "num_samples": float(sentence_total),
    }


def _mean(values: list[int] | list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0
