from pathlib import Path

import pytest

from recint.utils import apply_overrides, load_experiment_config

TOY_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "experiment" / "toy_popularity_penalty.yaml"


def test_referenced_sections_are_inlined():
    config = load_experiment_config(TOY_CONFIG)
    assert config["dataset"]["source"] == "synthetic"
    assert config["model"]["name"] == "item_cooccurrence"
    assert all(item["name"] == "popularity_penalty" for item in config["interventions"])


def test_intervention_list_resolves_paths_and_inline_items():
    config = load_experiment_config(
        TOY_CONFIG, ["interventions=[../intervention/identity.yaml, {name: identity, label: control}]"]
    )
    assert config["interventions"] == [
        {"name": "identity", "params": {}},
        {"name": "identity", "label": "control"},
    ]


def test_section_swap_and_dotted_overrides():
    config = load_experiment_config(
        TOY_CONFIG,
        ["interventions.1.params.penalty_weight=1.5", "dataset=../dataset/toy_csv.yaml"],
    )
    assert config["dataset"]["source"] == "csv"
    assert config["interventions"][1]["params"]["penalty_weight"] == 1.5


def test_malformed_override_raises():
    with pytest.raises(ValueError):
        apply_overrides({}, ["no_equals_sign"])
