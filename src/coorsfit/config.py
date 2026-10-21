"""Load config.yaml and hand out seeded random generators, so every stage shares one setup."""

from __future__ import annotations

import random
from functools import cache
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config.yaml"


@cache
def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def data_path(key: str) -> Path:
    """Absolute path for a configured directory such as "raw" or "processed"."""
    return ROOT / load_config()["paths"][key]


def rng(stream: int = 0) -> np.random.Generator:
    """A generator seeded from config. Use a distinct `stream` per independent use."""
    return np.random.default_rng([load_config()["seed"], stream])


def set_global_seeds() -> int:
    """Seed the global `random` and legacy NumPy state for libraries that read them."""
    seed = load_config()["seed"]
    random.seed(seed)
    np.random.seed(seed)
    return seed
