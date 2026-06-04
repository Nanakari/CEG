"""Efficiency helpers."""

from __future__ import annotations

from typing import Any, Mapping


def average_latency(records: list[Mapping[str, Any]]) -> float:
    values = [float(record.get("latency_sec", 0.0)) for record in records]
    return float(sum(values) / len(values)) if values else 0.0
