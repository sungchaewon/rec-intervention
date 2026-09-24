"""Prepare a dataset as a (user_id, item_id, timestamp, ...) CSV.

Real datasets: the dataset config's `preparation` section describes download +
filtering, and the result is written to the config's `path` (with stats.json).
Synthetic datasets: materialized to `--output` with `--seed`.

Examples:
    python scripts/prepare_data.py --config configs/dataset/ml-1m.yaml
    python scripts/prepare_data.py --config configs/dataset/toy.yaml --output data/toy/interactions.csv --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from recint.data import generate_synthetic_interactions
from recint.data.preprocess import interaction_stats, load_raw_interactions, prepare_interactions
from recint.experiment import RNG_STREAMS
from recint.utils import load_yaml, make_rngs

STATS_FILENAME = "stats.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="Dataset YAML config")
    parser.add_argument("--output", type=Path, help="Output CSV path (synthetic only)")
    parser.add_argument("--seed", type=int, help="Generation seed (synthetic only)")
    return parser.parse_args()


def prepare_real(config: dict) -> tuple[pd.DataFrame, Path, dict]:
    preparation = config["preparation"]
    raw = load_raw_interactions(preparation)
    frame = prepare_interactions(raw, preparation)
    stats = {"raw": interaction_stats(raw), "processed": interaction_stats(frame)}
    return frame, Path(config["path"]), stats


def prepare_synthetic(config: dict, output: Path | None, seed: int | None) -> tuple[pd.DataFrame, Path, dict]:
    if output is None or seed is None:
        raise SystemExit("Synthetic datasets require --output and --seed")
    # Same stream as run_intervention_experiment, so CSV and in-memory runs match.
    rng = make_rngs(seed, RNG_STREAMS)["data"]
    frame = generate_synthetic_interactions(rng=rng, **config.get("params", {}))
    return frame, output, {"processed": interaction_stats(frame)}


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if "preparation" in config:
        frame, output, stats = prepare_real(config)
    elif config["source"] == "synthetic":
        frame, output, stats = prepare_synthetic(config, args.output, args.seed)
    else:
        raise SystemExit(f"{args.config} has no `preparation` section and is not synthetic")

    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    with open(output.parent / STATS_FILENAME, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
    print(f"Wrote {len(frame)} interactions to {output}")


if __name__ == "__main__":
    main()
