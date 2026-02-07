#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for ML model configs using a small fake dataset.

Tests model instantiation, fit, predict_value, and predict_uncertainty
via molalkit_run with --max_iter 2.

Each model is tested with explorative (uncertainty-based) and/or exploitive
(prediction-based) selection. Models that support predict_uncertainty are
tested with both; others use exploitive only.
"""
import json
import os
import pytest
import pandas as pd
import tempfile

from molalkit.exe.run import molalkit_run

# ============================================================================
# Fake dataset
# ============================================================================

_SMILES = [
    "CCO", "CCCO", "CC(=O)O", "c1ccccc1", "CC(C)O",
    "CCN", "CC=O", "CCCC", "CC(=O)N", "c1ccc(O)cc1",
]
_BINARY_TARGETS = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
_REGRESSION_TARGETS = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]


@pytest.fixture
def classification_csv(tmp_path):
    path = tmp_path / "cls.csv"
    df = pd.DataFrame({"smiles": _SMILES, "target": _BINARY_TARGETS})
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def regression_csv(tmp_path):
    path = tmp_path / "reg.csv"
    df = pd.DataFrame({"smiles": _SMILES, "target": _REGRESSION_TARGETS})
    df.to_csv(path, index=False)
    return str(path)


# ============================================================================
# Helper
# ============================================================================

def _run_model_test(csv_path, task_type, metrics, model_config, temp_dir,
                    select_method="exploitive", s_exploitive_target="max"):
    args = [
        "--data_path", csv_path,
        "--smiles_columns", "smiles",
        "--targets_columns", "target",
        "--task_type", task_type,
        "--metrics", *metrics,
        "--model_configs", model_config,
        "--split_type", "random",
        "--split_sizes", "0.5", "0.5",
        "--init_size", "2",
        "--s_batch_size", "1",
        "--max_iter", "2",
        "--evaluate_stride", "1",
        "--save_dir", temp_dir,
        "--seed", "0",
        "--select_method", select_method,
    ]
    if select_method == "exploitive":
        args.extend(["--s_exploitive_target", s_exploitive_target])
    molalkit_run(args)

    traj_path = os.path.join(temp_dir, "al_traj.csv")
    assert os.path.exists(traj_path)
    df = pd.read_csv(traj_path)
    assert len(df) > 0
    return df


# ============================================================================
# Classification model tests (all support uncertainty → both methods)
# ============================================================================

CLASSIFICATION_CONFIGS = [
    "RandomForest_Morgan_Config",
    "XGBoost_Morgan_Config",
    "AdaBoost_Morgan_Config",
    "GaussianNB_Morgan_Config",
    "BernoulliNB_Morgan_Config",
    "MultinomialNB_Morgan_Config",
    "LogisticRegression_Morgan_Config",
    "GaussianProcessClassification_RBFKernelRDKitNorm_Config",
    "SupportVectorMachine_RBFKernelRDKitNorm_Config",
    "GaussianProcessClassification_DotProductKernelMorgan_Config",
]


class TestClassificationModels:

    @pytest.mark.unit
    @pytest.mark.parametrize("model_config", CLASSIFICATION_CONFIGS)
    @pytest.mark.parametrize("select_method", ["explorative", "exploitive"])
    def test_classification_model(self, model_config, select_method,
                                  classification_csv, temp_dir):
        df = _run_model_test(
            classification_csv,
            "binary",
            ["roc_auc", "accuracy"],
            model_config,
            temp_dir,
            select_method=select_method,
        )
        assert "roc_auc-model_0" in df.columns


# ============================================================================
# Regression model tests
# ============================================================================

REGRESSION_CONFIGS_WITH_UNCERTAINTY = [
    "RandomForest_Morgan_Config",
    "GaussianProcessRegressionPosteriorUncertainty_RBFKernelRDKitNorm_Config",
    "GaussianProcessRegressionDecisionBoundaryUncertainty_RBFKernelRDKitNorm_Config",
]

REGRESSION_CONFIGS_WITHOUT_UNCERTAINTY = [
    "AdaBoost_Morgan_Config",
    "SupportVectorMachine_RBFKernelRDKitNorm_Config",
]


class TestRegressionModels:

    @pytest.mark.unit
    @pytest.mark.parametrize("model_config", REGRESSION_CONFIGS_WITH_UNCERTAINTY)
    @pytest.mark.parametrize("select_method", ["explorative", "exploitive"])
    def test_regression_model_with_uncertainty(self, model_config, select_method,
                                               regression_csv, temp_dir):
        df = _run_model_test(
            regression_csv,
            "regression",
            ["rmse", "r2"],
            model_config,
            temp_dir,
            select_method=select_method,
        )
        assert "rmse-model_0" in df.columns

    @pytest.mark.unit
    @pytest.mark.parametrize("model_config", REGRESSION_CONFIGS_WITHOUT_UNCERTAINTY)
    def test_regression_model_exploitive(self, model_config, regression_csv, temp_dir):
        df = _run_model_test(
            regression_csv,
            "regression",
            ["rmse", "r2"],
            model_config,
            temp_dir,
            select_method="exploitive",
        )
        assert "rmse-model_0" in df.columns


# ============================================================================
# Marginalized Graph Kernel model tests (require GPU, slow)
# ============================================================================

MGK_CLASSIFICATION_CONFIGS = [
    "GaussianProcessClassification_MarginalizedGraphKernel_Config",
    "SupportVectorMachine_MarginalizedGraphKernel_Config",
]

MGK_REGRESSION_CONFIGS_WITH_UNCERTAINTY = [
    "GaussianProcessRegressionPosteriorUncertainty_MarginalizedGraphKernel_Config",
]

MGK_REGRESSION_CONFIGS_WITHOUT_UNCERTAINTY = [
    "SupportVectorMachine_MarginalizedGraphKernel_Config",
]


class TestMGKModels:

    @pytest.mark.unit
    @pytest.mark.slow
    @pytest.mark.parametrize("model_config", MGK_CLASSIFICATION_CONFIGS)
    @pytest.mark.parametrize("select_method", ["explorative", "exploitive"])
    def test_mgk_classification(self, model_config, select_method,
                                classification_csv, temp_dir):
        df = _run_model_test(
            classification_csv,
            "binary",
            ["roc_auc", "accuracy"],
            model_config,
            temp_dir,
            select_method=select_method,
        )
        assert "roc_auc-model_0" in df.columns

    @pytest.mark.unit
    @pytest.mark.slow
    @pytest.mark.parametrize("model_config", MGK_REGRESSION_CONFIGS_WITH_UNCERTAINTY)
    @pytest.mark.parametrize("select_method", ["explorative", "exploitive"])
    def test_mgk_regression_with_uncertainty(self, model_config, select_method,
                                             regression_csv, temp_dir):
        df = _run_model_test(
            regression_csv,
            "regression",
            ["rmse", "r2"],
            model_config,
            temp_dir,
            select_method=select_method,
        )
        assert "rmse-model_0" in df.columns

    @pytest.mark.unit
    @pytest.mark.slow
    @pytest.mark.parametrize("model_config", MGK_REGRESSION_CONFIGS_WITHOUT_UNCERTAINTY)
    def test_mgk_regression_exploitive(self, model_config, regression_csv, temp_dir):
        df = _run_model_test(
            regression_csv,
            "regression",
            ["rmse", "r2"],
            model_config,
            temp_dir,
            select_method="exploitive",
        )
        assert "rmse-model_0" in df.columns


# ============================================================================
# Chemprop model tests (skip if unavailable)
# ============================================================================

try:
    import chemprop
    HAS_CHEMPROP = True
except ImportError:
    HAS_CHEMPROP = False

try:
    import graphgps
    HAS_GRAPHGPS = True
except ImportError:
    HAS_GRAPHGPS = False

try:
    from molformer.checkpoints import AVAILABLE_CHECKPOINTS
    HAS_MOLFORMER = len(AVAILABLE_CHECKPOINTS) > 0
except ImportError:
    HAS_MOLFORMER = False
    AVAILABLE_CHECKPOINTS = []

CHEMPROP_CLASSIFICATION_CONFIGS = [
    "DMPNN_BinaryClassification_Config",
    "DMPNN_BinaryClassification_CBP_Config",
    "MLP_Morgan_BinaryClassification_Config",
    "MLP_Morgan_BinaryClassification_CBP_Config",
]

CHEMPROP_REGRESSION_CONFIGS = [
    "DMPNN_Regression_Config",
    "DMPNN_Regression_CBP_Config",
    "MLP_Morgan_Regression_Config",
    "MLP_Morgan_Regression_CBP_Config",
]


class TestChempropModels:

    @pytest.mark.unit
    @pytest.mark.slow
    @pytest.mark.skipif(not HAS_CHEMPROP, reason="chemprop not installed")
    @pytest.mark.parametrize("model_config", CHEMPROP_CLASSIFICATION_CONFIGS)
    @pytest.mark.parametrize("select_method", ["explorative", "exploitive"])
    def test_chemprop_classification(self, model_config, select_method,
                                     classification_csv, temp_dir):
        _run_model_test(
            classification_csv,
            "binary",
            ["roc_auc"],
            model_config,
            temp_dir,
            select_method=select_method,
        )

    @pytest.mark.unit
    @pytest.mark.slow
    @pytest.mark.skipif(not HAS_CHEMPROP, reason="chemprop not installed")
    @pytest.mark.parametrize("model_config", CHEMPROP_REGRESSION_CONFIGS)
    def test_chemprop_regression(self, model_config, regression_csv, temp_dir):
        _run_model_test(
            regression_csv,
            "regression",
            ["rmse"],
            model_config,
            temp_dir,
            select_method="exploitive",
        )


# ============================================================================
# GraphGPS model tests (skip if unavailable)
# ============================================================================

GRAPHGPS_CLASSIFICATION_CONFIGS = [
    "GraphGPS_BinaryClassification_Config",
]

GRAPHGPS_REGRESSION_CONFIGS = [
    "GraphGPS_Regression_Config",
]


class TestGraphGPSModels:

    @pytest.mark.unit
    @pytest.mark.slow
    @pytest.mark.skipif(not HAS_GRAPHGPS, reason="graphgps not installed")
    @pytest.mark.parametrize("model_config", GRAPHGPS_CLASSIFICATION_CONFIGS)
    @pytest.mark.parametrize("select_method", ["explorative", "exploitive"])
    def test_graphgps_classification(self, model_config, select_method,
                                     classification_csv, temp_dir):
        _run_model_test(
            classification_csv,
            "binary",
            ["roc_auc"],
            model_config,
            temp_dir,
            select_method=select_method,
        )

    @pytest.mark.unit
    @pytest.mark.slow
    @pytest.mark.skipif(not HAS_GRAPHGPS, reason="graphgps not installed")
    @pytest.mark.parametrize("model_config", GRAPHGPS_REGRESSION_CONFIGS)
    def test_graphgps_regression(self, model_config, regression_csv, temp_dir):
        _run_model_test(
            regression_csv,
            "regression",
            ["rmse"],
            model_config,
            temp_dir,
            select_method="exploitive",
        )


# ============================================================================
# MolFormer model tests (skip if unavailable)
# ============================================================================

@pytest.fixture
def molformer_config(tmp_path):
    """Write a MolFormer config with the absolute checkpoint path."""
    config = {
        "data_format": "mgktools",
        "model": "MolFormer",
        "pretrained_path": AVAILABLE_CHECKPOINTS[0] if AVAILABLE_CHECKPOINTS else "",
        "n_head": 12,
        "n_layer": 12,
        "n_embd": 768,
        "d_dropout": 0.1,
        "dropout": 0.1,
        "learning_rate": 3e-5,
        "num_feats": 32,
        "batch_size": 128,
        "epochs": 2,
        "ensemble_size": 1,
    }
    config_path = tmp_path / "MolFormer_Test_Config"
    config_path.write_text(json.dumps(config))
    return str(config_path)


class TestMolFormerModels:

    @pytest.mark.unit
    @pytest.mark.slow
    @pytest.mark.skipif(not HAS_MOLFORMER, reason="molformer not installed or no checkpoints")
    @pytest.mark.parametrize("select_method", ["explorative", "exploitive"])
    def test_molformer_classification(self, select_method, molformer_config,
                                      classification_csv, temp_dir):
        df = _run_model_test(
            classification_csv,
            "binary",
            ["roc_auc"],
            molformer_config,
            temp_dir,
            select_method=select_method,
        )
        assert "roc_auc-model_0" in df.columns

    @pytest.mark.unit
    @pytest.mark.slow
    @pytest.mark.skipif(not HAS_MOLFORMER, reason="molformer not installed or no checkpoints")
    def test_molformer_regression(self, molformer_config, regression_csv, temp_dir):
        df = _run_model_test(
            regression_csv,
            "regression",
            ["rmse"],
            molformer_config,
            temp_dir,
            select_method="exploitive",
        )
        assert "rmse-model_0" in df.columns
