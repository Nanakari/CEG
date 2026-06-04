"""Check local assets referenced by a config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ceg_reproduce.utils.config import load_config, resolve_path


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    dataset = config.get("datasets", {}).get("coco_chair", {})
    checks = []
    for key in ["image_root", "instances_file", "captions_file", "sample_file"]:
        value = dataset.get(key)
        if value:
            checks.append((key, resolve_path(value, ROOT)))
    missing = [(name, path) for name, path in checks if not path.exists()]
    for name, path in checks:
        status = "OK" if path.exists() else "MISSING"
        print(f"{status}: {name}: {path}")
    if missing and args.strict:
        raise SystemExit(1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
