#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
End-to-end tests for full MolALKit pipeline.

These tests use explicit, representative scenarios instead of
combinatorial parameter explosion.
"""
import pytest
import os
import pandas as pd
from molalkit.exe.run import molalkit_run


# ============================================================================
# Test Scenarios - Explicit, representative combinations
# ============================================================================

CLASSIFICATION_SCENARIOS = [
    {
        "id": "rf_classification_random_selection",
        "description": "RandomForest with random selection",
        "data_public": "dili",
        "model_config": "RandomForest_RDKitNorm_Config",
        "select_method": "random",
        "forget_method": None,
        "split_type": "random",
    },
    {
        "id": "rf_classification_explorative",
        "description": "RandomForest with uncertainty-based selection",
        "data_public": "dili",
        "model_config": "RandomForest_RDKitNorm_Config",
        "select_method": "explorative",
        "forget_method": None,
        "split_type": "scaffold_order",
    },
    {
        "id": "rf_classification_exploitive_max",
        "description": "RandomForest with exploitive selection (max)",
        "data_public": "dili",
        "model_config": "RandomForest_RDKitNorm_Config",
        "select_method": "exploitive",
        "s_exploitive_target": "max",
        "forget_method": None,
        "split_type": "random",
    },
    {
        "id": "rf_classification_with_oob_forget",
        "description": "RandomForest with OOB-based forgetting",
        "data_public": "dili",
        "model_config": "RandomForest_RDKitNorm_Config",
        "select_method": "explorative",
        "forget_method": "min_oob_uncertainty",
        "split_type": "random",
    },
    {
        "id": "xgboost_classification",
        "description": "XGBoost classification",
        "data_public": "dili",
        "model_config": "XGBoost_Morgan_Config",
        "select_method": "explorative",
        "forget_method": None,
        "split_type": "random",
    },
    {
        "id": "gp_classification",
        "description": "Gaussian Process classification",
        "data_public": "dili",
        "model_config": "GaussianProcessClassification_RBFKernelRDKitNorm_Config",
        "select_method": "explorative",
        "forget_method": None,
        "split_type": "random",
    },
    {
        "id": "naive_bayes_classification",
        "description": "Naive Bayes classification",
        "data_public": "dili",
        "model_config": "GaussianNB_Morgan_Config",
        "select_method": "explorative",
        "forget_method": None,
        "split_type": "random",
    },
    {
        "id": "logistic_regression_classification",
        "description": "Logistic Regression classification",
        "data_public": "dili",
        "model_config": "LogisticRegression_Morgan_Config",
        "select_method": "explorative",
        "forget_method": None,
        "split_type": "random",
    },
]

REGRESSION_SCENARIOS = [
    {
        "id": "rf_regression_random_selection",
        "description": "RandomForest regression with random selection",
        "data_public": "freesolv",
        "model_config": "RandomForest_RDKitNorm_Config",
        "select_method": "random",
        "forget_method": None,
        "split_type": "random",
    },
    {
        "id": "rf_regression_explorative",
        "description": "RandomForest regression with uncertainty-based selection",
        "data_public": "freesolv",
        "model_config": "RandomForest_RDKitNorm_Config",
        "select_method": "explorative",
        "forget_method": None,
        "split_type": "scaffold_order",
    },
    {
        "id": "gp_regression_explorative",
        "description": "Gaussian Process regression with explorative selection",
        "data_public": "freesolv",
        "model_config": "GaussianProcessRegressionPosteriorUncertainty_RBFKernelRDKitNorm_Config",
        "select_method": "explorative",
        "forget_method": None,
        "split_type": "random",
    },
    {
        "id": "svm_regression",
        "description": "Support Vector Machine regression with random selection",
        "data_public": "freesolv",
        "model_config": "SupportVectorMachine_RBFKernelRDKitNorm_Config",
        "select_method": "random",
        "forget_method": None,
        "split_type": "random",
    },
]

ADVANCED_SCENARIOS = [
    {
        "id": "multiple_models",
        "description": "Multiple models (RandomForest + XGBoost)",
        "data_public": "dili",
        "model_configs": ["RandomForest_RDKitNorm_Config", "XGBoost_Morgan_Config"],
        "select_method": "explorative",
        "forget_method": None,
        "split_type": "random",
    },
    {
        "id": "batch_selection",
        "description": "Batch selection (multiple samples per iteration)",
        "data_public": "dili",
        "model_config": "RandomForest_RDKitNorm_Config",
        "select_method": "explorative",
        "s_batch_size": "5",
        "forget_method": None,
        "split_type": "random",
    },
    {
        "id": "batch_forget",
        "description": "Batch forgetting",
        "data_public": "dili",
        "model_config": "RandomForest_RDKitNorm_Config",
        "select_method": "explorative",
        "forget_method": "first",
        "f_batch_size": "3",
        "split_type": "random",
        "init_size": "30",
    },
]


def get_scenario_ids(scenarios):
    """Extract scenario IDs for pytest parametrization."""
    return [s["id"] for s in scenarios]


# ============================================================================
# Test Functions
# ============================================================================

class TestClassificationPipelines:
    """E2E tests for classification pipelines."""

    @pytest.mark.e2e
    @pytest.mark.parametrize("scenario", CLASSIFICATION_SCENARIOS, ids=get_scenario_ids(CLASSIFICATION_SCENARIOS))
    def test_classification_scenario(self, scenario, temp_dir):
        """Test classification scenario."""
        args = [
            "--data_public", scenario["data_public"],
            "--metrics", "roc_auc", "mcc",
            "--model_configs", scenario["model_config"],
            "--split_type", scenario["split_type"],
            "--split_sizes", "0.8", "0.2",
            "--init_size", "10",
            "--s_batch_size", scenario.get("s_batch_size", "1"),
            "--max_iter", "5",
            "--evaluate_stride", "1",
            "--save_dir", temp_dir,
            "--seed", "42",
        ]

        # Add selection method
        if scenario["select_method"]:
            args.extend(["--select_method", scenario["select_method"]])
            if scenario["select_method"] == "exploitive":
                args.extend(["--s_exploitive_target", scenario.get("s_exploitive_target", "max")])

        # Add forget method
        if scenario.get("forget_method"):
            args.extend([
                "--forget_method", scenario["forget_method"],
                "--f_batch_size", scenario.get("f_batch_size", "1"),
                "--n_forget", "1",
            ])

        molalkit_run(args)

        # Verify outputs
        assert os.path.exists(os.path.join(temp_dir, "al_traj.csv"))
        df = pd.read_csv(os.path.join(temp_dir, "al_traj.csv"))
        assert len(df) > 0
        assert "roc_auc-model_0" in df.columns


class TestRegressionPipelines:
    """E2E tests for regression pipelines."""

    @pytest.mark.e2e
    @pytest.mark.parametrize("scenario", REGRESSION_SCENARIOS, ids=get_scenario_ids(REGRESSION_SCENARIOS))
    def test_regression_scenario(self, scenario, temp_dir):
        """Test regression scenario."""
        args = [
            "--data_public", scenario["data_public"],
            "--metrics", "rmse", "r2",
            "--model_configs", scenario["model_config"],
            "--split_type", scenario["split_type"],
            "--split_sizes", "0.8", "0.2",
            "--init_size", "10",
            "--s_batch_size", scenario.get("s_batch_size", "1"),
            "--max_iter", "5",
            "--evaluate_stride", "1",
            "--save_dir", temp_dir,
            "--seed", "42",
        ]

        # Add selection method
        if scenario["select_method"]:
            args.extend(["--select_method", scenario["select_method"]])
            if scenario["select_method"] == "exploitive":
                args.extend(["--s_exploitive_target", scenario.get("s_exploitive_target", "min")])

        # Add forget method
        if scenario.get("forget_method"):
            args.extend([
                "--forget_method", scenario["forget_method"],
                "--f_batch_size", scenario.get("f_batch_size", "1"),
                "--n_forget", "1",
            ])

        molalkit_run(args)

        # Verify outputs
        assert os.path.exists(os.path.join(temp_dir, "al_traj.csv"))
        df = pd.read_csv(os.path.join(temp_dir, "al_traj.csv"))
        assert len(df) > 0
        assert "rmse-model_0" in df.columns


class TestAdvancedPipelines:
    """E2E tests for advanced pipeline configurations."""

    @pytest.mark.e2e
    @pytest.mark.parametrize("scenario", ADVANCED_SCENARIOS, ids=get_scenario_ids(ADVANCED_SCENARIOS))
    def test_advanced_scenario(self, scenario, temp_dir):
        """Test advanced scenario."""
        # Handle multiple models
        model_configs = scenario.get("model_configs", [scenario.get("model_config")])

        args = [
            "--data_public", scenario["data_public"],
            "--metrics", "roc_auc", "mcc",
            "--model_configs", *model_configs,
            "--split_type", scenario["split_type"],
            "--split_sizes", *scenario.get("split_sizes", ["0.8", "0.2"]),
            "--init_size", scenario.get("init_size", "10"),
            "--s_batch_size", scenario.get("s_batch_size", "1"),
            "--max_iter", "5",
            "--evaluate_stride", "1",
            "--save_dir", temp_dir,
            "--seed", "42",
        ]

        # Add selection method
        if scenario["select_method"]:
            args.extend(["--select_method", scenario["select_method"]])

        # Add forget method
        if scenario.get("forget_method"):
            args.extend([
                "--forget_method", scenario["forget_method"],
                "--f_batch_size", scenario.get("f_batch_size", "1"),
                "--n_forget", "1",
            ])

        molalkit_run(args)

        # Verify outputs
        assert os.path.exists(os.path.join(temp_dir, "al_traj.csv"))


class TestREADMEExample:
    """Test the exact example from README.md"""

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_readme_example(self, temp_dir):
        """Test the first example from README.md (shortened for testing)."""
        molalkit_run([
            "--data_public", "bace",
            "--metrics", "roc_auc", "mcc", "accuracy", "precision", "recall", "f1_score",
            "--model_configs", "RandomForest_Morgan_Config",
            "--split_type", "scaffold_order",
            "--split_sizes", "0.5", "0.5",
            "--evaluate_stride", "5",
            "--seed", "0",
            "--save_dir", temp_dir,
            "--init_size", "2",
            "--select_method", "explorative",
            "--s_batch_size", "1",
            "--max_iter", "20",  # Shortened from 100 for testing
        ])

        # Verify outputs
        assert os.path.exists(os.path.join(temp_dir, "al_traj.csv"))
        df = pd.read_csv(os.path.join(temp_dir, "al_traj.csv"))

        # Check all metrics are present
        assert "roc_auc-model_0" in df.columns
        assert "mcc-model_0" in df.columns
        assert "accuracy-model_0" in df.columns
        assert "precision-model_0" in df.columns
        assert "recall-model_0" in df.columns
        assert "f1_score-model_0" in df.columns
