"""Export main result tables from metric JSON."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ceg_reproduce.utils.config import load_config, output_root
from ceg_reproduce.utils.io import dump_json, dump_text, read_json


COLUMNS = [
    ("Method", None),
    ("CHAIRs", "chairs"),
    ("CHAIRi", "chairi"),
    ("Recall", "recall"),
    ("Average Length", "average_length"),
    ("Verified Claims", "verified_claims"),
    ("Relative Time", "relative_time"),
]


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    out_root = output_root(config, ROOT)
    metrics_path = out_root / "metrics_coco_chair.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    metrics = read_json(metrics_path)
    methods = metrics.get("methods", {})
    rows = []
    for method, label in [("base", "Base"), ("vcd", "VCD"), ("ceg", "CEG")]:
        row = {"Method": label}
        values = methods.get(method, {})
        for column, key in COLUMNS[1:]:
            row[column] = _format_value(values.get(key))
        rows.append(row)
    table_dir = out_root / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    _write_markdown(table_dir / "main_results.md", rows)
    _write_csv(table_dir / "main_results.csv", rows)
    dump_json(table_dir / "main_results.json", {"columns": [item[0] for item in COLUMNS], "rows": rows})


def _format_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    headers = [item[0] for item in COLUMNS]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row.get(header, "") for header in headers) + " |")
    dump_text(path, "\n".join(lines) + "\n")


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    headers = [item[0] for item in COLUMNS]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    main()
