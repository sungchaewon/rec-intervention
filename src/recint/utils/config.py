"""YAML config loading.

An experiment config references dataset / model / intervention configs by path
(relative to the experiment config file). They are inlined on load so the
resolved config is a single self-contained dict that can be saved with results.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

REFERENCED_SECTIONS = ("dataset", "model", "intervention")


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        content = yaml.safe_load(f)
    if not isinstance(content, dict):
        raise ValueError(f"Config {path} must be a YAML mapping, got {type(content).__name__}")
    return content


def load_experiment_config(path: str | Path, overrides: Iterable[str] = ()) -> dict[str, Any]:
    """Load and resolve an experiment config.

    Overrides of a whole referenced section (e.g. `dataset=../dataset/other.yaml`)
    are applied before resolution; all other overrides after it.
    """
    path = Path(path)
    config = load_yaml(path)
    overrides = list(overrides)
    is_section_swap = [o.partition("=")[0] in REFERENCED_SECTIONS for o in overrides]
    apply_overrides(config, [o for o, swap in zip(overrides, is_section_swap) if swap])
    for section in REFERENCED_SECTIONS:
        reference = config.get(section)
        if isinstance(reference, str):
            config[section] = load_yaml((path.parent / reference).resolve())
    apply_overrides(config, [o for o, swap in zip(overrides, is_section_swap) if not swap])
    return config


def apply_overrides(config: dict[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Apply `dotted.key=value` overrides in place; values are parsed as YAML."""
    for override in overrides:
        key, sep, raw_value = override.partition("=")
        if not sep or not key:
            raise ValueError(f"Override must look like 'a.b.c=value', got {override!r}")
        *parents, leaf = key.split(".")
        node = config
        for part in parents:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"Cannot override {key!r}: {part!r} is not a mapping")
        node[leaf] = yaml.safe_load(raw_value)
    return config
