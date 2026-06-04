"""Run the no-model smoke pipeline."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    smoke_out = ROOT / "outputs" / "smoke"
    if smoke_out.exists():
        shutil.rmtree(smoke_out)
    subprocess.run(
        [sys.executable, "scripts/run_all.py", "--config", "configs/smoke.yaml"],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
