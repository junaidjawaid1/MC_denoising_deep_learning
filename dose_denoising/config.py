"""YAML configuration loading."""
from pathlib import Path

import yaml


def load_config(path, data_root=None):
    """Load the YAML config; ``data_root`` (e.g. from the command line) overrides ``data.root``."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if data_root is not None:
        cfg["data"]["root"] = str(data_root)
    if not cfg["data"].get("root"):
        raise ValueError("Set data.root in the config or pass --data-root")
    return cfg


def split_dir(cfg, split):
    """Directory of a split ('train', 'validation' or 'test')."""
    return Path(cfg["data"]["root"]) / split


def voxel_size(cfg):
    return float(cfg["data"]["voxel_size_mm"])
