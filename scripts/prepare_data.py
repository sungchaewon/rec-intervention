"""Materialize a dataset config as a (user_id, item_id, timestamp) CSV.

Currently supports `source: synthetic`. Real-dataset preprocessing (e.g.
MovieLens) is planned.

Example:
    python scripts/prepare_data.py --config configs/dataset/toy.yaml --output data/toy/interactions.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from recint.data import generate_synthetic_interactions
from recint.experiment import RNG_STREAMS
from recint.utils import load_yaml, make_rngs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="Dataset YAML config")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path")
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if config["source"] != "synthetic":
        raise NotImplementedError(f"prepare_data does not support source {config['source']!r} yet")
    # Same stream as run_intervention_experiment, so CSV and in-memory runs match.
    rng = make_rngs(args.seed, RNG_STREAMS)["data"]
    frame = generate_synthetic_interactions(rng=rng, **config.get("params", {}))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(
        f"Wrote {len(frame)} interactions ({frame['user_id'].nunique()} users, "
        f"{frame['item_id'].nunique()} items) to {args.output}"
    )


if __name__ == "__main__":
    main()
