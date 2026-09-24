"""Recommender backbones. Planned: SASRec, LightGCN."""

from typing import Any

from recint.models.base import Recommender, mask_seen_items, top_k_items
from recint.models.cooccurrence import ItemCooccurrenceRecommender

MODEL_REGISTRY: dict[str, type[Recommender]] = {
    "item_cooccurrence": ItemCooccurrenceRecommender,
}


def build_model(config: dict[str, Any]) -> Recommender:
    name = config["name"]
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {name!r}; available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**config.get("params", {}))


__all__ = [
    "MODEL_REGISTRY",
    "ItemCooccurrenceRecommender",
    "Recommender",
    "build_model",
    "mask_seen_items",
    "top_k_items",
]
