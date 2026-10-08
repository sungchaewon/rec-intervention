"""Prepare a dataset as CSVs: interactions (user_id, item_id, timestamp, ...) and,
when available, item categories (item_id, category) plus stats.json.

Real datasets: the dataset config's `preparation` section describes download +
filtering; outputs go to the config's `path` and `item_categories_path`.
Synthetic datasets: materialized to `--output` (categories alongside) with `--seed`.

Examples:
    python scripts/prepare_data.py --config configs/dataset/ml-1m.yaml
    python scripts/prepare_data.py --config configs/dataset/toy.yaml --output data/toy/interactions.csv --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from recint.data import generate_synthetic_dataset
from recint.data.preprocess import interaction_stats, load_raw_dataset, prepare_interactions
from recint.experiment import RNG_STREAMS
from recint.utils import load_yaml, make_rngs

STATS_FILENAME = "stats.json"
CATEGORIES_FILENAME = "item_categories.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="Dataset YAML config")
    parser.add_argument("--output", type=Path, help="Output CSV path (synthetic only)")
    parser.add_argument("--seed", type=int, help="Generation seed (synthetic only)")
    return parser.parse_args()


def prepare_real(config: dict) -> tuple[pd.DataFrame, pd.DataFrame | None, Path, Path | None, dict]:
    preparation = config["preparation"]
    raw, categories = load_raw_dataset(preparation)
    frame = prepare_interactions(raw, preparation)
    if categories is not None:
        categories = categories[categories["item_id"].isin(frame["item_id"])]
    categories_path = config.get("item_categories_path")
    stats = {"raw": interaction_stats(raw), "processed": interaction_stats(frame)}
    return frame, categories, Path(config["path"]), Path(categories_path) if categories_path else None, stats


def prepare_synthetic(
    config: dict, output: Path | None, seed: int | None
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path, dict]:
    if output is None or seed is None:
        raise SystemExit("Synthetic datasets require --output and --seed")
    # Same stream as run_intervention_experiment, so CSV and in-memory runs match.
    rng = make_rngs(seed, RNG_STREAMS)["data"]
    frame, categories = generate_synthetic_dataset(rng=rng, **config.get("params", {}))
    return frame, categories, output, output.parent / CATEGORIES_FILENAME, {"processed": interaction_stats(frame)}


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if "preparation" in config:
        frame, categories, output, categories_output, stats = prepare_real(config)
    elif config["source"] == "synthetic":
        frame, categories, output, categories_output, stats = prepare_synthetic(config, args.output, args.seed)
    else:
        raise SystemExit(f"{args.config} has no `preparation` section and is not synthetic")

    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    if categories is not None and categories_output is not None:
        categories_output.parent.mkdir(parents=True, exist_ok=True)
        categories.to_csv(categories_output, index=False)
        stats["n_categories"] = int(categories["category"].nunique())
        print(f"Wrote {len(categories)} item-category rows to {categories_output}")
    with open(output.parent / STATS_FILENAME, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
    print(f"Wrote {len(frame)} interactions to {output}")


if __name__ == "__main__":
    main()
