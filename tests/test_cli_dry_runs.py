from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_run_all_dry_run_prints_main_commands() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_all.py",
            "--config",
            "configs/smoke.yaml",
            "--limit",
            "1",
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "scripts/run_caption.py" in result.stdout
    assert "scripts/run_ceg.py" in result.stdout
    assert "--limit 1" in result.stdout
