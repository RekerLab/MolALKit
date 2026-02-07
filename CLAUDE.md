# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MolALKit is a Python toolkit for active learning on molecular data. It supports molecular property prediction for drug discovery with classification and regression tasks on single-task datasets.

**Requirements**: Python 3.9+ (3.12 recommended), GPU/CUDA required for graph kernel functionality (mgktools dependency).

## Common Commands

### Installation
```bash
# Minimum installation (requires GPU)
pip install git+https://gitlab.com/Xiangyan93/graphdot.git@v0.8.2 molalkit

# With Chemprop support
pip install git+https://github.com/Xiangyan93/chemprop4molalkit.git

# With GraphGPS support
pip install torch-scatter torch-sparse torch-geometric pytorch-lightning yacs torchmetrics performer-pytorch ogb git+https://github.com/Xiangyan93/graphgps4molalkit.git
```

### Running Active Learning
```bash
# Example with BACE dataset
molalkit_run --data_public bace --metrics roc_auc mcc accuracy precision recall f1_score \
  --model_configs RandomForest_Morgan_Config --split_type scaffold_order \
  --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir ./results \
  --init_size 2 --select_method explorative --s_batch_size 1 --max_iter 100

# Custom dataset
molalkit_run --data_path dataset.csv --smiles_columns smiles --targets_columns target \
  --task_type classification --model_configs RandomForest_Morgan_Config ...
```

### Running Tests
```bash
# Run all tests (uses pytest with heavy parametrization, tests sample 1% of combinations)
pytest tests/

# Run specific test file
pytest tests/test_model.py
pytest tests/test_learning.py
```

## Architecture

### Core Components

**Active Learning (`molalkit/active_learning/`)**
- `ActiveLearner` - Main orchestrator for AL workflows. Manages train/pool sets, selection, forgetting, and evaluation cycles
- `BaseSelector` subclasses - Selection strategies: `RandomSelector`, `ExplorativeSelector` (uncertainty-based), `ExploitiveSelector`, cluster variants, partial query variants
- `BaseForgetter` subclasses - Forgetting strategies: `FirstForgetter`, `RandomForgetter`, OOB-based (RandomForest), LOO-based (GaussianProcess)
- `ActiveLearningTrajectory` - Tracks performance metrics across iterations

**Models (`molalkit/models/`)**
- `BaseModel` - Abstract interface defining `fit_molalkit(data, iteration)`, `predict_value(data)`, `predict_uncertainty(data)`
- `BaseSklearnModel` - Wrapper for sklearn models
- 93 pre-built model configurations in `molalkit/models/configs/`
- Model types: RandomForest, XGBoost, GaussianProcess, SVM, NaiveBayes, MLP, MPNN/DMPNN (Chemprop), GraphGPS, MolFormer
- Uncertainty methods: MVE, Ensemble, Evidential, Dropout

**Data (`molalkit/data/`)**
- Supports three data formats: mgktools (graph kernels), Chemprop, GraphGPS (PyTorch Geometric)
- Split types: random, scaffold_order (deterministic), scaffold_random
- 20+ embedded public datasets in `molalkit/data/datasets/`

**CLI (`molalkit/exe/`)**
- Entry point: `molalkit_run` command
- `LearningArgs` - Typed argument parser (TAP)
- Supports checkpoint recovery for resuming runs

### Data Flow

1. Arguments parsed → datasets loaded/split → models initialized
2. Main loop: select samples → forget samples → evaluate (periodic) → save checkpoints
3. Output: `al_traj.csv` (metrics trajectory), final train/pool datasets

### Key Patterns

- Factory functions: `get_model()`, `get_kernel()` for model instantiation
- JSON config files define model architectures for reproducibility
- Pickle-based checkpointing for AL state recovery
- Pluggable selector/forgetter/model components

## Available Resources

```python
# List available public datasets
from molalkit.data.datasets import AVAILABLE_DATASETS
print(AVAILABLE_DATASETS)

# List available model configs
from molalkit.models.configs import AVAILABLE_MODELS
print(AVAILABLE_MODELS)
```

## Testing Notes

Tests use extensive pytest parametrization. `test_learning.py` randomly samples ~1% of parameter combinations (see `np.random.uniform(0, 1) > 0.01` early return). Tests validate:
- Datasets: freesolv (regression), dili (classification)
- Split types: random, scaffold_random, scaffold_order
- Selection methods: random, explorative, exploitive
- Forget methods: first, random, OOB variants (RandomForest only), LOO variants (GaussianProcess only)
- Batch modes: naive, cluster
