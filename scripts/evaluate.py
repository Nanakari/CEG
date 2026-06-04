"""Evaluate COCO/CHAIR outputs for Base, VCD, and CEG."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ceg_reproduce.evaluation.chair import compute_chair_metrics
from ceg_reproduce.evaluation.efficiency import average_latency
from ceg_reproduce.extraction.claims import ClaimExtractor
from ceg_reproduce.utils.config import load_config, output_root
from ceg_reproduce.utils.io import dump_json, read_jsonl


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    out_root = output_root(config, ROOT)
    extractor = ClaimExtractor.from_config(config, ROOT)
    report: dict[str, object] = {"dataset": "coco_chair", "methods": {}}

    base_records = _read_if_exists(out_root / "coco_base_captions.jsonl")
    base_latency = average_latency(base_records) if base_records else None
    for method in ["base", "vcd"]:
        path = out_root / f"coco_{method}_captions.jsonl"
        records = _read_if_exists(path)
        if records:
            report["methods"][method] = compute_chair_metrics(
                records, extractor, base_latency_sec=base_latency
            )
    ceg_records = _read_if_exists(out_root / "coco_ceg_revisions.jsonl")
    if ceg_records:
        report["methods"]["ceg"] = compute_chair_metrics(
            ceg_records, extractor, base_latency_sec=base_latency
        )
    dump_json(out_root / "metrics_coco_chair.json", report)


def _read_if_exists(path: Path) -> list[dict]:
    return list(read_jsonl(path)) if path.exists() else []


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    main()
