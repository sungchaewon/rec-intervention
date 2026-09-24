"""Recommender backbones. Implemented: item co-occurrence, SASRec. Planned: LightGCN."""

from pathlib import Path
from typing import Any

from recint.data.interactions import InteractionData
from recint.models.base import Recommender, mask_seen_items, top_k_items
from recint.models.cooccurrence import ItemCooccurrenceRecommender
from recint.models.sasrec import SASRecRecommender

MODEL_REGISTRY: dict[str, type[Recommender]] = {
    "item_cooccurrence": ItemCooccurrenceRecommender,
    "sasrec": SASRecRecommender,
}


def _model_class(config: dict[str, Any]) -> type[Recommender]:
    name = config["name"]
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {name!r}; available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name]


def build_model(config: dict[str, Any]) -> Recommender:
    return _model_class(config)(**config.get("params", {}))


def load_model(config: dict[str, Any], checkpoint: Path, train: InteractionData) -> Recommender:
    return _model_class(config).load(checkpoint, train, **config.get("params", {}))


__all__ = [
    "MODEL_REGISTRY",
    "ItemCooccurrenceRecommender",
    "Recommender",
    "SASRecRecommender",
    "build_model",
    "load_model",
    "mask_seen_items",
    "top_k_items",
]
