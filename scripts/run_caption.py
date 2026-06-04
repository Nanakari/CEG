"""Run COCO Caption Base or VCD inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ceg_reproduce.datasets.coco import load_coco_caption_samples
from ceg_reproduce.models import build_generator
from ceg_reproduce.pipeline import generate_for_sample
from ceg_reproduce.utils.config import config_for_method, load_config, output_root
from ceg_reproduce.utils.io import write_jsonl
from ceg_reproduce.utils.reproducibility import seed_everything


def main() -> None:
    args = _parse_args()
    base_config = load_config(args.config)
    config = config_for_method(base_config, args.method)
    seed_everything(config)
    out_root = output_root(config, ROOT)
    generator = build_generator(config, method=args.method)
    samples = load_coco_caption_samples(config["datasets"]["coco_chair"], ROOT, limit=args.limit)
    prompt = str(config.get("prompts", {}).get("caption", "Please describe the image in detail."))
    max_new_tokens = int(config.get("generation", {}).get("caption_max_new_tokens", 512))
    records = []
    for sample in samples:
        result = generate_for_sample(
            generator,
            sample.image_path,
            prompt,
            sample_id=sample.sample_id,
            max_new_tokens=max_new_tokens,
        )
        records.append(
            {
                **sample.to_record(),
                "method": args.method,
                "prompt": prompt,
                "caption": result["text"],
                "text": result["text"],
                "generation": result.get("metadata", {}),
                "latency_sec": float(result.get("latency_sec", 0.0)),
            }
        )
    write_jsonl(out_root / f"coco_{args.method}_captions.jsonl", records)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--method", choices=["base", "vcd"], required=True)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    main()
