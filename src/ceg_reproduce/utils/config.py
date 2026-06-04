"""Configuration loading and path resolution."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

from ceg_reproduce.utils.io import load_yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = resolve_path(path, PROJECT_ROOT)
    config = load_yaml(config_path)
    config["_config_path"] = str(config_path)
    return config


def config_for_method(config: Mapping[str, Any], method: str) -> dict[str, Any]:
    cloned = copy.deepcopy(dict(config))
    generation = cloned.setdefault("generation", {})
    generation["method"] = method
    if method == "vcd":
        generation["backend"] = "vcd_llava" if generation.get("backend") != "fake" else "fake"
    elif method == "base" and generation.get("backend") == "vcd_llava":
        generation["backend"] = "llava_hf"
    return cloned


def output_root(config: Mapping[str, Any], project_root: str | Path = PROJECT_ROOT) -> Path:
    value = config.get("outputs", {}).get("root", "outputs/main")
    return resolve_path(value, Path(project_root))


def resolve_path(path: str | Path | None, project_root: str | Path = PROJECT_ROOT) -> Path:
    if path in {None, ""}:
        return Path("")
    raw = Path(str(path))
    if raw.is_absolute():
        return raw
    return Path(project_root) / raw
