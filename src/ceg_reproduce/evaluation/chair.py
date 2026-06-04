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
    risk_counts: list[int] = []
    revision_counts: list[int] = []
    correct_original_total = 0
    correct_retained_total = 0
    hallucinated_original_total = 0
    hallucinated_removed_total = 0

    for record in records:
        caption = _caption_text(record, prefer_revised=True)
        claims = extractor.extract(caption)
        predicted = [claim.normalized for claim in claims]
        predicted_set = set(predicted)
        gt_objects = {str(item).lower() for item in record.get("gt_objects", [])}
        original_caption = _caption_text(record, prefer_revised=False)
        original_predicted = [claim.normalized for claim in extractor.extract(original_caption)]
        original_predicted_set = set(original_predicted)
        correct_original = {obj for obj in original_predicted_set if obj in gt_objects}
        hallucinated_original = {obj for obj in original_predicted_set if gt_objects and obj not in gt_objects}
        correct_original_total += len(correct_original)
        correct_retained_total += len(correct_original & predicted_set)
        hallucinated_original_total += len(hallucinated_original)
        hallucinated_removed_total += len(hallucinated_original - predicted_set)
        hallucinated = [obj for obj in predicted if obj and gt_objects and obj not in gt_objects]
        sentence_total += 1
        object_total += len(predicted)
        hallucinated_object_total += len(hallucinated)
        hallucinated_sentence_total += int(bool(hallucinated))
        gt_total += len(gt_objects)
        recalled_total += len(predicted_set & gt_objects)
        lengths.append(len(caption.split()))
        verified_claim_counts.append(len(record.get("verified_claims", []) or []))
        risk_counts.append(int(record.get("risk_count", 0) or 0))
        revision_counts.append(int(record.get("revision_count", len(record.get("revision_actions", []) or [])) or 0))
        latencies.append(float(record.get("latency_sec", 0.0)))

    avg_latency = _mean(latencies)
    relative_time = avg_latency / base_latency_sec if base_latency_sec and base_latency_sec > 0 else 1.0
    return {
        "chairs": hallucinated_sentence_total / sentence_total if sentence_total else 0.0,
        "chairi": hallucinated_object_total / object_total if object_total else 0.0,
        "recall": recalled_total / gt_total if gt_total else 0.0,
        "average_length": _mean(lengths),
        "verified_claims": _mean(verified_claim_counts),
        "risk_count": _mean(risk_counts),
        "revision_count": _mean(revision_counts),
        "relative_time": relative_time,
        "false_rejection_rate": (
            (correct_original_total - correct_retained_total) / correct_original_total
            if correct_original_total
            else 0.0
        ),
        "correct_retention_rate": (
            correct_retained_total / correct_original_total if correct_original_total else 0.0
        ),
        "hallucinated_removal_rate": (
            hallucinated_removed_total / hallucinated_original_total
            if hallucinated_original_total
            else 0.0
        ),
        "num_samples": float(sentence_total),
    }


def _mean(values: list[int] | list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _caption_text(record: Mapping[str, Any], *, prefer_revised: bool) -> str:
    keys = ("revised_caption", "caption", "text") if prefer_revised else ("original_caption", "caption", "text")
    for key in keys:
        if key in record and record[key] is not None:
            return str(record[key])
    return ""
