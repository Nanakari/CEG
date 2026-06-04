from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts.run_ceg import _uses_targeted_vqa


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


def test_run_ceg_skips_mnli_for_targeted_vqa_modes() -> None:
    assert _uses_targeted_vqa({"verification": {"mode": "targeted_vqa"}, "ceg": {}})
    assert _uses_targeted_vqa({"verification": {"mode": "caption"}, "ceg": {"risk_mode": "targeted_vqa_support_guard"}})
    assert not _uses_targeted_vqa({"verification": {"mode": "caption"}, "ceg": {"risk_mode": "balanced"}})
