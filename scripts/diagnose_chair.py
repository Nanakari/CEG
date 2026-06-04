"""Print CHAIR object matching diagnostics for generated captions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ceg_reproduce.extraction.claims import ClaimExtractor
from ceg_reproduce.utils.config import load_config, output_root
from ceg_reproduce.utils.io import read_jsonl


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    out_root = output_root(config, ROOT)
    extractor = ClaimExtractor.from_config(config, ROOT)
    path = Path(args.input) if args.input else _path_for_method(out_root, args.method)
    if not path.exists():
        raise FileNotFoundError(f"Caption file not found: {path}")

    for index, record in enumerate(read_jsonl(path)):
        if args.limit is not None and index >= args.limit:
            break
        caption = _caption_text(record)
        caption_objects = sorted({claim.normalized for claim in extractor.extract(caption)})
        gt_objects = sorted({str(item).lower() for item in record.get("gt_objects", [])})
        matched = sorted(set(caption_objects) & set(gt_objects))
        missed = sorted(set(gt_objects) - set(caption_objects))
        hallucinated = sorted(set(caption_objects) - set(gt_objects)) if gt_objects else []
        print(
            json.dumps(
                {
                    "sample_id": record.get("sample_id"),
                    "image_id": record.get("image_id"),
                    "gt_objects": gt_objects,
                    "caption_objects": caption_objects,
                    "matched": matched,
                    "missed": missed,
                    "hallucinated": hallucinated,
                    "caption": caption,
                },
                ensure_ascii=False,
            )
        )


def _path_for_method(out_root: Path, method: str) -> Path:
    if method == "ceg":
        return out_root / "coco_ceg_revisions.jsonl"
    return out_root / f"coco_{method}_captions.jsonl"


def _caption_text(record: dict) -> str:
    for key in ("revised_caption", "caption", "text"):
        if key in record and record[key] is not None:
            return str(record[key])
    return ""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--method", choices=["base", "vcd", "ceg"], default="base")
    parser.add_argument("--input", help="Optional explicit JSONL caption/revision file.")
    parser.add_argument("--limit", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    main()
