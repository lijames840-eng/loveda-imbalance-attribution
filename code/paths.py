"""Repository-wide path resolution.

Every script derives its inputs and outputs from this module instead of
hard-coding an absolute path, so the repository can be cloned and run
anywhere.

Layout assumed (see README)::

    <repo>/code/           scripts and models
    <repo>/data/           dataset root, split list, cached statistics
    <repo>/experiments/    one directory per run
    <repo>/results/        figures and tables produced by the analysis scripts

The repository root is inferred from this file's location. Set the
``RS_ROOT`` environment variable to point at a different root, e.g. when the
dataset lives on another disk::

    RS_ROOT=/mnt/data/rs_course python code/train.py --config code/configs/v2.yaml
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("RS_ROOT") or Path(__file__).resolve().parents[1]).resolve()

CODE = ROOT / "code"
CONFIGS = CODE / "configs"

DATA = ROOT / "data"
DATA_RAW = DATA / "raw" / "LoveDA"
DATA_STATS = DATA / "stats"
SPLITS = DATA / "splits.json"
TREE_CFG = DATA / "tree.cfg"

EXPERIMENTS = ROOT / "experiments"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
LOGS = ROOT / "logs"


def resolve(path) -> Path:
    """Return an absolute path. Relative paths are taken w.r.t. the repo root."""
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    for base in (ROOT, CODE):
        if (base / p).exists():
            return base / p
    return ROOT / p


def load_cfg(path):
    """Load a YAML config and expand its path entries to absolute paths."""
    import yaml

    cfg = yaml.safe_load(resolve(path).read_text(encoding="utf-8"))

    paths = cfg.setdefault("paths", {})
    paths["data"] = str(resolve(paths.get("data", DATA_RAW.relative_to(ROOT))))
    paths["experiments"] = str(
        resolve(paths.get("experiments", EXPERIMENTS.relative_to(ROOT)))
    )

    loss = (cfg.get("train") or {}).get("loss") or {}
    for key in ("share_scale", "splits", "init_ckpt"):
        if loss.get(key):
            loss[key] = str(resolve(loss[key]))

    return cfg


def ensure(*dirs):
    """Create output directories if they do not exist yet."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
