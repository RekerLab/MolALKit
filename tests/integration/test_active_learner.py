#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Integration tests for ActiveLearner class.
"""
import pytest
import os
import pandas as pd
from molalkit.exe.run import molalkit_run


class TestActiveLearnerBasicWorkflow:
    """Tests for basic ActiveLearner workflow."""

    @pytest.mark.integration
    def test_creates_output_files(self, temp_dir):
        """Test that all expected output files are created."""
        molalkit_run([
            "--data_public", "dili",
            "--metrics", "roc_auc", "mcc",
            "--model_configs", "RandomForest_RDKitNorm_Config",
            "--split_type", "random",
            "--split_sizes", "0.8", "0.2",
            "--init_size", "5",
            "--select_method", "random",
            "--s_batch_size", "1",
            "--max_iter", "5",
            "--evaluate_stride", "1",
            "--save_dir", temp_dir,
            "--seed", "42",
        ])

        # Check all expected files exist
        assert os.path.exists(os.path.join(temp_dir, "al_traj.csv"))
        assert os.path.exists(os.path.join(temp_dir, "full.csv"))
        assert os.path.exists(os.path.join(temp_dir, "train_init.csv"))
        assert os.path.exists(os.path.join(temp_dir, "pool_init.csv"))
        assert os.path.exists(os.path.join(temp_dir, "train_end.csv"))
        assert os.path.exists(os.path.join(temp_dir, "pool_end.csv"))
        assert os.path.exists(os.path.join(temp_dir, "val.csv"))

    @pytest.mark.integration
    def test_trajectory_has_correct_structure(self, temp_dir):
        """Test that trajectory CSV has correct columns."""
        molalkit_run([
            "--data_public", "dili",
            "--metrics", "roc_auc", "mcc",
            "--model_configs", "RandomForest_RDKitNorm_Config",
            "--split_type", "random",
            "--split_sizes", "0.8", "0.2",
            "--init_size", "5",
            "--select_method", "random",
            "--s_batch_size", "1",
            "--max_iter", "5",
            "--evaluate_stride", "1",
            "--save_dir", temp_dir,
            "--seed", "42",
        ])

        df = pd.read_csv(os.path.join(temp_dir, "al_traj.csv"))

        # Check required columns
        assert "n_iter" in df.columns
        assert "roc_auc-model_0" in df.columns
        assert "mcc-model_0" in df.columns
        assert "uidx_before" in df.columns
        assert "uidx_select" in df.columns
        assert "uidx_after" in df.columns

    @pytest.mark.integration
    def test_training_set_grows_over_iterations(self, temp_dir):
        """Test that training set grows as samples are selected."""
        molalkit_run([
            "--data_public", "dili",
            "--metrics", "roc_auc",
            "--model_configs", "RandomForest_RDKitNorm_Config",
            "--split_type", "random",
            "--split_sizes", "0.8", "0.2",
            "--init_size", "5",
            "--select_method", "random",
            "--s_batch_size", "2",
            "--max_iter", "10",
            "--save_dir", temp_dir,
            "--seed", "42",
        ])

        train_init = pd.read_csv(os.path.join(temp_dir, "train_init.csv"))
        train_end = pd.read_csv(os.path.join(temp_dir, "train_end.csv"))

        # Training set should have grown
        # init_size=5, max_iter=10, s_batch_size=2 -> should add 20 samples
        assert len(train_end) > len(train_init)
        assert len(train_end) == len(train_init) + 20  # 10 iterations * 2 samples

    @pytest.mark.integration
    def test_pool_set_shrinks_over_iterations(self, temp_dir):
        """Test that pool set shrinks as samples are selected."""
        molalkit_run([
            "--data_public", "dili",
            "--metrics", "roc_auc",
            "--model_configs", "RandomForest_RDKitNorm_Config",
            "--split_type", "random",
            "--split_sizes", "0.8", "0.2",
            "--init_size", "5",
            "--select_method", "random",
            "--s_batch_size", "2",
            "--max_iter", "10",
            "--save_dir", temp_dir,
            "--seed", "42",
        ])

        pool_init = pd.read_csv(os.path.join(temp_dir, "pool_init.csv"))
        pool_end = pd.read_csv(os.path.join(temp_dir, "pool_end.csv"))

        # Pool set should have shrunk
        assert len(pool_end) < len(pool_init)
        assert len(pool_end) == len(pool_init) - 20


class TestActiveLearnerSelectionMethods:
    """Tests for different selection methods."""

    @pytest.mark.integration
    def test_random_selection(self, temp_dir):
        """Test random selection method."""
        molalkit_run([
            "--data_public", "dili",
            "--metrics", "roc_auc",
            "--model_configs", "RandomForest_RDKitNorm_Config",
            "--split_type", "random",
            "--split_sizes", "0.8", "0.2",
            "--init_size", "5",
            "--select_method", "random",
            "--s_batch_size", "1",
            "--max_iter", "5",
            "--save_dir", temp_dir,
            "--seed", "42",
        ])

        assert os.path.exists(os.path.join(temp_dir, "al_traj.csv"))

    @pytest.mark.integration
    def test_explorative_selection(self, temp_dir):
        """Test explorative (uncertainty-based) selection method."""
        molalkit_run([
            "--data_public", "dili",
            "--metrics", "roc_auc",
            "--model_configs", "RandomForest_RDKitNorm_Config",
            "--split_type", "random",
            "--split_sizes", "0.8", "0.2",
            "--init_size", "5",
            "--select_method", "explorative",
            "--s_batch_size", "1",
            "--max_iter", "5",
            "--save_dir", temp_dir,
            "--seed", "42",
        ])

        df = pd.read_csv(os.path.join(temp_dir, "al_traj.csv"))
        # Check that acquisition values are recorded
        assert "acquisition_select" in df.columns

    @pytest.mark.integration
    def test_exploitive_selection(self, temp_dir):
        """Test exploitive (prediction-based) selection method."""
        molalkit_run([
            "--data_public", "dili",
            "--metrics", "roc_auc",
            "--model_configs", "RandomForest_RDKitNorm_Config",
            "--split_type", "random",
            "--split_sizes", "0.8", "0.2",
            "--init_size", "5",
            "--select_method", "exploitive",
            "--s_exploitive_target", "max",
            "--s_batch_size", "1",
            "--max_iter", "5",
            "--save_dir", temp_dir,
            "--seed", "42",
        ])

        assert os.path.exists(os.path.join(temp_dir, "al_traj.csv"))


class TestActiveLearnerForgetMethods:
    """Tests for different forget methods."""

    @pytest.mark.integration
    def test_first_forgetter(self, temp_dir):
        """Test first (FIFO) forget method."""
        molalkit_run([
            "--data_public", "dili",
            "--metrics", "roc_auc",
            "--model_configs", "RandomForest_RDKitNorm_Config",
            "--split_type", "random",
            "--split_sizes", "0.8", "0.2",
            "--init_size", "10",
            "--select_method", "random",
            "--s_batch_size", "1",
            "--forget_method", "first",
            "--f_batch_size", "1",
            "--n_forget", "1",
            "--max_iter", "5",
            "--save_dir", temp_dir,
            "--seed", "42",
        ])

        # With forget, training set should stay roughly the same size
        train_end = pd.read_csv(os.path.join(temp_dir, "train_end.csv"))
        # init_size=10, 5 iterations: +5 select, -5 forget = 10 final
        assert len(train_end) == 10

    @pytest.mark.integration
    def test_random_forgetter(self, temp_dir):
        """Test random forget method."""
        molalkit_run([
            "--data_public", "dili",
            "--metrics", "roc_auc",
            "--model_configs", "RandomForest_RDKitNorm_Config",
            "--split_type", "random",
            "--split_sizes", "0.8", "0.2",
            "--init_size", "10",
            "--select_method", "random",
            "--s_batch_size", "1",
            "--forget_method", "random",
            "--f_batch_size", "1",
            "--n_forget", "1",
            "--max_iter", "5",
            "--save_dir", temp_dir,
            "--seed", "42",
        ])

        assert os.path.exists(os.path.join(temp_dir, "al_traj.csv"))

    @pytest.mark.integration
    def test_oob_forgetter_with_random_forest(self, temp_dir):
        """Test OOB-based forget method with RandomForest."""
        molalkit_run([
            "--data_public", "dili",
            "--metrics", "roc_auc",
            "--model_configs", "RandomForest_RDKitNorm_Config",
            "--split_type", "random",
            "--split_sizes", "0.8", "0.2",
            "--init_size", "20",
            "--select_method", "explorative",
            "--s_batch_size", "1",
            "--forget_method", "min_oob_uncertainty",
            "--f_batch_size", "1",
            "--n_forget", "1",
            "--max_iter", "5",
            "--save_dir", temp_dir,
            "--seed", "42",
        ])

        assert os.path.exists(os.path.join(temp_dir, "al_traj.csv"))


class TestActiveLearnerSplitTypes:
    """Tests for different data split types."""

    @pytest.mark.integration
    def test_random_split(self, temp_dir):
        """Test random data split."""
        molalkit_run([
            "--data_public", "dili",
            "--metrics", "roc_auc",
            "--model_configs", "RandomForest_RDKitNorm_Config",
            "--split_type", "random",
            "--split_sizes", "0.8", "0.2",
            "--init_size", "5",
            "--select_method", "random",
            "--s_batch_size", "1",
            "--max_iter", "3",
            "--save_dir", temp_dir,
            "--seed", "42",
        ])

        assert os.path.exists(os.path.join(temp_dir, "al_traj.csv"))

    @pytest.mark.integration
    def test_scaffold_order_split(self, temp_dir):
        """Test scaffold order data split."""
        molalkit_run([
            "--data_public", "dili",
            "--metrics", "roc_auc",
            "--model_configs", "RandomForest_RDKitNorm_Config",
            "--split_type", "scaffold_order",
            "--split_sizes", "0.8", "0.2",
            "--init_size", "5",
            "--select_method", "random",
            "--s_batch_size", "1",
            "--max_iter", "3",
            "--save_dir", temp_dir,
            "--seed", "42",
        ])

        assert os.path.exists(os.path.join(temp_dir, "al_traj.csv"))

    @pytest.mark.integration
    def test_scaffold_random_split(self, temp_dir):
        """Test scaffold random data split."""
        molalkit_run([
            "--data_public", "dili",
            "--metrics", "roc_auc",
            "--model_configs", "RandomForest_RDKitNorm_Config",
            "--split_type", "scaffold_random",
            "--split_sizes", "0.8", "0.2",
            "--init_size", "5",
            "--select_method", "random",
            "--s_batch_size", "1",
            "--max_iter", "3",
            "--save_dir", temp_dir,
            "--seed", "42",
        ])

        assert os.path.exists(os.path.join(temp_dir, "al_traj.csv"))
