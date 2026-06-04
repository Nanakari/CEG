"""Run the complete CEG main-experiment sequence."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    args = _parse_args()
    commands = [
        [sys.executable, "scripts/run_caption.py", "--config", args.config, "--method", "base"],
        [sys.executable, "scripts/run_caption.py", "--config", args.config, "--method", "vcd"],
        [sys.executable, "scripts/run_ceg.py", "--config", args.config],
        [sys.executable, "scripts/evaluate.py", "--config", args.config],
        [sys.executable, "scripts/export_results.py", "--config", args.config],
    ]
    if args.limit is not None:
        for command in commands[:3]:
            command.extend(["--limit", str(args.limit)])
    for command in commands:
        print("+" + " ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
