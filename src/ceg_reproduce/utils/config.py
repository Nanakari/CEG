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
    _apply_method_defaults(cloned, method)
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


def _apply_method_defaults(config: dict[str, Any], method: str) -> None:
    method_path = PROJECT_ROOT / "configs" / "methods" / f"{method}.yaml"
    if not method_path.exists():
        return
    method_config = load_yaml(method_path)
    defaults = method_config.get("defaults", {}) if isinstance(method_config, Mapping) else {}
    for key, value in defaults.items():
        section = _method_default_section(key, method)
        config.setdefault(section, {}).setdefault(key, value)


def _method_default_section(key: str, method: str) -> str:
    if key in {"cd_alpha", "cd_beta", "noise_step"}:
        return "vcd"
    if key in {"top_k", "risk_mode", "object_revision"}:
        return "ceg"
    if key in {"entailment_threshold"}:
        return "nli"
    return method
