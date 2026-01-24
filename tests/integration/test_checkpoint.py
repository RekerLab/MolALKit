#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Integration tests for checkpoint save/load functionality.
"""
import pytest
import os
import pandas as pd
from molalkit.exe.run import molalkit_run


class TestCheckpointSaveLoad:
    """Tests for checkpoint save and load functionality."""

    @pytest.mark.integration
    def test_checkpoint_save_creates_file(self, temp_dir):
        """Test that checkpoint file is created when save_cpt_stride is set."""
        molalkit_run([
            "--data_public", "dili",
            "--metrics", "roc_auc",
            "--model_configs", "RandomForest_RDKitNorm_Config",
            "--split_type", "random",
            "--split_sizes", "0.8", "0.2",
            "--init_size", "5",
            "--select_method", "random",
            "--s_batch_size", "1",
            "--max_iter", "10",
            "--save_cpt_stride", "5",
            "--save_dir", temp_dir,
            "--seed", "42",
        ])

        # Check that checkpoint file was created
        assert os.path.exists(os.path.join(temp_dir, "al.pkl"))

    @pytest.mark.integration
    def test_checkpoint_resume_continues_from_correct_iteration(self, temp_dir):
        """Test that resuming from checkpoint continues from correct iteration."""
        # Run first 10 iterations
        molalkit_run([
            "--data_public", "dili",
            "--metrics", "roc_auc",
            "--model_configs", "RandomForest_RDKitNorm_Config",
            "--split_type", "random",
            "--split_sizes", "0.8", "0.2",
            "--init_size", "5",
            "--select_method", "random",
            "--s_batch_size", "1",
            "--max_iter", "10",
            "--save_cpt_stride", "5",
            "--save_dir", temp_dir,
            "--seed", "42",
        ])

        # Check initial trajectory
        df1 = pd.read_csv(os.path.join(temp_dir, "al_traj.csv"))
        initial_rows = len(df1)

        # Resume and run to 20 iterations
        molalkit_run([
            "--data_public", "dili",
            "--metrics", "roc_auc",
            "--model_configs", "RandomForest_RDKitNorm_Config",
            "--split_type", "random",
            "--split_sizes", "0.8", "0.2",
            "--init_size", "5",
            "--select_method", "random",
            "--s_batch_size", "1",
            "--max_iter", "20",
            "--save_cpt_stride", "5",
            "--save_dir", temp_dir,
            "--seed", "42",
            "--load_checkpoint",
        ])

        # Check that trajectory continued
        df2 = pd.read_csv(os.path.join(temp_dir, "al_traj.csv"))
        assert len(df2) > initial_rows

    @pytest.mark.integration
    def test_checkpoint_preserves_training_data(self, temp_dir):
        """Test that checkpoint preserves training data state."""
        # Run first phase
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
            "--save_cpt_stride", "5",
            "--save_dir", temp_dir,
            "--seed", "42",
        ])

        # Load checkpoint and verify state
        import pickle
        with open(os.path.join(temp_dir, "al.pkl"), "rb") as f:
            checkpoint = pickle.load(f)

        # Verify essential components are saved
        assert "datasets_train" in checkpoint
        assert "datasets_pool" in checkpoint
        assert "current_iter" in checkpoint


class TestCheckpointEdgeCases:
    """Tests for checkpoint edge cases."""

    @pytest.mark.integration
    def test_no_checkpoint_when_stride_not_set(self, temp_dir):
        """Test that no checkpoint is created when save_cpt_stride is not set."""
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

        # Checkpoint should not exist
        assert not os.path.exists(os.path.join(temp_dir, "al.pkl"))

    @pytest.mark.integration
    def test_load_checkpoint_without_file_starts_fresh(self, temp_dir):
        """Test that load_checkpoint without existing file starts fresh."""
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
            "--load_checkpoint",  # No checkpoint exists
        ])

        # Should still create output files
        assert os.path.exists(os.path.join(temp_dir, "al_traj.csv"))
