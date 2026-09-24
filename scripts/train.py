"""Train the experiment's base recommender and save it to `model_checkpoint`.

Uses the same data, split, and seed as analyze_intervention.py, which then loads
the checkpoint.

Example:
    python scripts/train.py --config configs/experiment/ml-1m_sasrec_popularity_penalty.yaml
    python scripts/train.py --config ... --override seed=7 --override model_checkpoint=outputs/checkpoints/ml-1m_sasrec_seed7/model.pt
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
import yaml

from recint.experiment import prepare_run
from recint.models import build_model
from recint.utils import load_experiment_config

HISTORY_FILENAME = "train_history.csv"
CONFIG_FILENAME = "train_config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="Experiment YAML config")
    parser.add_argument(
        "--override", action="append", default=[], metavar="KEY=VALUE",
        help="Dotted config override, e.g. model.params.device=cuda (repeatable)",
    )  # fmt: skip
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    config = load_experiment_config(args.config, args.override)
    if not config.get("model_checkpoint"):
        raise SystemExit("Config has no `model_checkpoint`; nothing to train")
    checkpoint = Path(config["model_checkpoint"])

    _, split, _ = prepare_run(config)
    model = build_model(config["model"])
    logging.info(
        "training %s on %d users / %d items (device: %s)",
        config["model"]["name"], split.train.n_users, split.train.n_items,
        getattr(model, "device", "n/a"),
    )  # fmt: skip
    started = time.perf_counter()
    model.fit(split.train)
    elapsed = time.perf_counter() - started

    model.save(checkpoint)
    history = pd.DataFrame(getattr(model, "history", []))
    history.to_csv(checkpoint.parent / HISTORY_FILENAME, index=False)
    with open(checkpoint.parent / CONFIG_FILENAME, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    if not history.empty:
        metric = next(c for c in history.columns if c.startswith("valid_"))
        best = history.loc[history[metric].idxmax()]
        print(f"best epoch {int(best['epoch'])}: {metric} = {best[metric]:.4f}")
    print(f"trained in {elapsed / 60:.1f} min; saved checkpoint to {checkpoint}")


if __name__ == "__main__":
    main()
