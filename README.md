# rec-intervention

State-dependent intervention effect analysis for recommender systems.

## Setup

```bash
conda env create -f environment.cuda.yml   # GPU server; use environment.yml for CPU-only
conda activate rec-intervention
python scripts/check_env.py                # add --require-cuda on GPU servers
pytest
```

## Quick Start

Toy experiment (synthetic data, no training):

```bash
python scripts/analyze_intervention.py --config configs/experiment/toy_popularity_penalty.yaml
```

MovieLens-1M with SASRec:

```bash
python scripts/prepare_data.py --config configs/dataset/ml-1m.yaml
python scripts/train.py --config configs/experiment/ml-1m_sasrec_popularity_penalty.yaml
python scripts/analyze_intervention.py --config configs/experiment/ml-1m_sasrec_popularity_penalty.yaml
```

Results are written to `outputs/`. Any config value can be overridden with
`--override key.subkey=value`.
