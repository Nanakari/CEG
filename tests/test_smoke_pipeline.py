from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from ceg_reproduce.utils.io import read_json, read_jsonl


ROOT = Path(__file__).resolve().parents[1]


def test_smoke_pipeline_runs_end_to_end() -> None:
    subprocess.run([sys.executable, "scripts/smoke_test.py"], cwd=ROOT, check=True)

    revisions = list(read_jsonl(ROOT / "outputs/smoke/coco_ceg_revisions.jsonl"))
    metrics = read_json(ROOT / "outputs/smoke/metrics_coco_chair.json")
    table = ROOT / "outputs/smoke/tables/main_results.md"

    assert revisions[0]["revised_caption"] == "A man is holding an object beside a car."
    assert "ceg" in metrics["methods"]
    assert table.exists()
