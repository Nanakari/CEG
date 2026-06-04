"""Seed helpers."""

from __future__ import annotations

import random
from typing import Any, Mapping

import numpy as np


def seed_everything(config: Mapping[str, Any]) -> None:
    seed = int(config.get("runtime", {}).get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        if hasattr(torch, "manual_seed"):
            torch.manual_seed(seed)
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except (ImportError, AttributeError):
        return
