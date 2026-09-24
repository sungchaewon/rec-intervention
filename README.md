# rec-intervention

State-aware intervention analysis (and, later, routing) for recommender systems.

## Research Question

> How does the effect of an intervention vary across different user states in recommender systems?

An *intervention* is any transformation applied on top of a frozen base
recommender's output (e.g. reranking, candidate filtering). For each user we
measure

```
intervention_gain = metric_after_intervention - metric_before_intervention
```

and classify it as **beneficial**, **neutral** (|gain| <= threshold), or **harmful**.
We then compare these effects across user states (history sparsity, preference
drift, uncertainty, ...).

## Current Goal: WWW 2027 Short Paper

State-dependent intervention **effect analysis**: is an intervention helpful on
average, how much does its effect vary with user state, and where does it hurt?
No routing/policy model is built at this stage.

## Long-term Goal: RecSys 2027 Full Paper

State-conditioned intervention **routing**: learn `user state -> intervention action`.
The routing module will consume the per-user effect table produced by the
analysis pipeline (`recint.analysis.build_user_effect_table`), so the two stages
stay separate in code.

## Status

| Component | Status |
|---|---|
| Interaction data abstraction (user, item, optional timestamp), leave-last-out split | Implemented |
| Synthetic toy dataset; CSV loader | Implemented |
| Metrics: HR@K, NDCG@K (per user) | Implemented |
| Intervention gain + beneficial / neutral / harmful classification | Implemented |
| Base model: item co-occurrence (training-free, for pipeline development only) | Implemented |
| Interventions: popularity-penalty reranking, identity (control) | Implemented |
| User state: history length, grouped into short / medium / long | Implemented |
| State-conditioned summary: bootstrap CI of mean gain, Wilcoxon signed-rank p | Implemented |
| SASRec, LightGCN backbones; `scripts/train.py` | Planned |
| Real datasets (e.g. MovieLens) preprocessing | Planned |
| States: preference drift/stability, recommendation uncertainty, popularity exposure | Planned |
| More interventions; multi-intervention comparison | Planned |
| Plotting utilities | Planned |
| Routing extension (RecSys) | Planned |

No results on real data exist yet. The toy output below is a pipeline check,
not a finding.

## Repository Layout

```
configs/
  dataset/         # data source + split  (toy.yaml, toy_csv.yaml)
  model/           # base recommender     (item_cooccurrence.yaml)
  intervention/    # intervention         (popularity_penalty.yaml, identity.yaml)
  experiment/      # references the above + state, evaluation, analysis settings
src/recint/
  data/            # InteractionData, leave_last_out, synthetic generator
  models/          # Recommender base class, top-k / seen-item masking, backbones
  interventions/   # Intervention base class and implementations
  states/          # user-state extractors and grouping
  metrics/         # HR@K, NDCG@K, intervention gain, effect labels
  analysis/        # per-user effect table, state-conditioned summary, bootstrap
  utils/           # config loading/overrides, seeding
  experiment.py    # end-to-end pipeline used by scripts and tests
scripts/
  prepare_data.py          # materialize a dataset config as CSV
  analyze_intervention.py  # run an experiment, print + save the state table
tests/
outputs/           # experiment results (git-ignored)
data/              # prepared data (git-ignored)
```

New components are added to the registry dict in the relevant package
`__init__.py` (`MODEL_REGISTRY`, `INTERVENTION_REGISTRY`, `STATE_REGISTRY`) and
selected by `name` in YAML.

## Setup

Requires Miniconda/Anaconda.

```bash
conda env create -f environment.yml
conda activate rec-intervention
pytest
```

The environment uses Python 3.11 and conda-forge only. The package is installed
in editable mode (`pip install -e .`), so `import recint` works anywhere.

### GPU (optional)

The default environment has the CPU build of PyTorch. For a CUDA build, which
conda-forge pairs with the matching MKL/OpenMP:

```bash
conda install -n rec-intervention -c conda-forge "pytorch=2.13.*=cuda*"
```

Do not `pip install torch` into this environment on Windows: the PyPI wheel
conflicts with conda-forge's OpenMP runtime.

## Quick Start

Run the toy experiment (synthetic data -> item co-occurrence base -> popularity
penalty reranking -> effect by history-length group):

```bash
python scripts/analyze_intervention.py --config configs/experiment/toy_popularity_penalty.yaml
```

This prints one row per (state group, metric) with `n_users`, `base`,
`intervention`, `mean_gain` and its bootstrap CI, `beneficial_rate`,
`neutral_rate`, `harm_rate`, and `wilcoxon_p`. It writes `config.yaml`
(fully resolved), `user_effects.csv` (per user), and `state_summary.csv` to
`outputs/toy_popularity_penalty/`.

Override any config value from the CLI:

```bash
python scripts/analyze_intervention.py --config configs/experiment/toy_popularity_penalty.yaml \
  --override intervention.params.penalty_weight=1.0 \
  --override analysis.neutral_threshold=0.01 \
  --override "state.grouping={method: fixed, cut_points: [5, 20], labels: [short, medium, long]}"
```

Run from a CSV instead of in-memory synthetic data (same seed gives an identical result):

```bash
python scripts/prepare_data.py --config configs/dataset/toy.yaml --output data/toy/interactions.csv --seed 42
python scripts/analyze_intervention.py --config configs/experiment/toy_popularity_penalty.yaml \
  --override dataset=../dataset/toy_csv.yaml
```

## Reproducibility

- A single `seed` in the experiment config drives independent RNG streams per
  stage (`data`, `analysis`) via `numpy.random.SeedSequence`.
- The resolved config is saved next to every result.
- Ties in rankings are broken by item index.

## License

MIT
