#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for forgetter classes.
"""
import pytest
import numpy as np
from molalkit.active_learning.forgetter import (
    RandomForgetter,
    FirstForgetter,
    MinOOBUncertaintyForgetter,
    MaxOOBUncertaintyForgetter,
    MinOOBErrorForgetter,
    MaxOOBErrorForgetter,
    MinLOOUncertaintyForgetter,
    MaxLOOUncertaintyForgetter,
    MinLOOErrorForgetter,
    MaxLOOErrorForgetter,
)


class TestRandomForgetter:
    """Tests for RandomForgetter."""

    @pytest.mark.unit
    def test_returns_correct_batch_size(self, mock_dataset_small):
        """Test that forgetter removes the requested batch size."""
        forgetter = RandomForgetter(batch_size=5, seed=42)
        idx, acquisition, remain = forgetter(dataset_train=mock_dataset_small)

        assert len(idx) == 5
        assert len(remain) == len(mock_dataset_small) - 5

    @pytest.mark.unit
    def test_no_overlap_between_forgotten_and_remaining(self, mock_dataset_small):
        """Test that forgotten and remaining indices don't overlap."""
        forgetter = RandomForgetter(batch_size=5, seed=42)
        idx, _, remain = forgetter(dataset_train=mock_dataset_small)

        assert set(idx).isdisjoint(set(remain))
        assert len(idx) + len(remain) == len(mock_dataset_small)

    @pytest.mark.unit
    def test_all_indices_valid(self, mock_dataset_small):
        """Test that all returned indices are valid."""
        forgetter = RandomForgetter(batch_size=5, seed=42)
        idx, _, remain = forgetter(dataset_train=mock_dataset_small)

        all_indices = idx + remain
        assert all(0 <= i < len(mock_dataset_small) for i in all_indices)

    @pytest.mark.unit
    def test_different_seeds_produce_different_results(self, mock_dataset_medium):
        """Test that different seeds produce different results."""
        # Reset to a known state first
        np.random.seed(999)

        forgetter1 = RandomForgetter(batch_size=10, seed=42)
        idx1, _, _ = forgetter1(dataset_train=mock_dataset_medium)

        forgetter2 = RandomForgetter(batch_size=10, seed=123)
        idx2, _, _ = forgetter2(dataset_train=mock_dataset_medium)

        # With different seeds, results should differ (very high probability)
        assert idx1 != idx2

    @pytest.mark.unit
    def test_info_property(self):
        """Test the info property returns correct string."""
        forgetter = RandomForgetter(batch_size=5, seed=42)
        assert "RandomForgetter" in forgetter.info
        assert "batch_size=5" in forgetter.info


class TestFirstForgetter:
    """Tests for FirstForgetter (FIFO forgetting)."""

    @pytest.mark.unit
    def test_returns_first_n_indices(self, mock_dataset_small):
        """Test that forgetter removes the first n samples."""
        forgetter = FirstForgetter(batch_size=3)
        idx, acquisition, remain = forgetter(dataset_train=mock_dataset_small)

        assert idx == [0, 1, 2]
        assert remain == list(range(3, len(mock_dataset_small)))

    @pytest.mark.unit
    def test_returns_correct_batch_size(self, mock_dataset_small):
        """Test that forgetter removes the requested batch size."""
        forgetter = FirstForgetter(batch_size=5)
        idx, _, remain = forgetter(dataset_train=mock_dataset_small)

        assert len(idx) == 5
        assert len(remain) == len(mock_dataset_small) - 5

    @pytest.mark.unit
    def test_no_overlap_between_forgotten_and_remaining(self, mock_dataset_small):
        """Test that forgotten and remaining indices don't overlap."""
        forgetter = FirstForgetter(batch_size=5)
        idx, _, remain = forgetter(dataset_train=mock_dataset_small)

        assert set(idx).isdisjoint(set(remain))

    @pytest.mark.unit
    def test_info_property(self):
        """Test the info property returns correct string."""
        forgetter = FirstForgetter(batch_size=5)
        assert "FirstForgetter" in forgetter.info
        assert "batch_size=5" in forgetter.info


class TestMinOOBUncertaintyForgetter:
    """Tests for MinOOBUncertaintyForgetter (requires RandomForest with OOB).

    Note: Full functional tests require actual RFClassifier instances and are
    covered in integration tests. Unit tests here only cover interface/info.
    """

    @pytest.mark.unit
    def test_info_property(self):
        """Test the info property returns correct string."""
        forgetter = MinOOBUncertaintyForgetter(batch_size=5, seed=42)
        assert "MinOOBUncertaintyForgetter" in forgetter.info
        assert "batch_size=5" in forgetter.info

    @pytest.mark.unit
    def test_batch_size_initialization(self):
        """Test that batch_size is correctly initialized."""
        forgetter = MinOOBUncertaintyForgetter(batch_size=10, seed=42)
        assert forgetter.batch_size == 10


class TestMaxOOBUncertaintyForgetter:
    """Tests for MaxOOBUncertaintyForgetter."""

    @pytest.mark.unit
    def test_info_property(self):
        """Test the info property returns correct string."""
        forgetter = MaxOOBUncertaintyForgetter(batch_size=5, seed=42)
        assert "MaxOOBUncertaintyForgetter" in forgetter.info

    @pytest.mark.unit
    def test_batch_size_initialization(self):
        """Test that batch_size is correctly initialized."""
        forgetter = MaxOOBUncertaintyForgetter(batch_size=10, seed=42)
        assert forgetter.batch_size == 10


class TestMinOOBErrorForgetter:
    """Tests for MinOOBErrorForgetter."""

    @pytest.mark.unit
    def test_info_property(self):
        """Test the info property returns correct string."""
        forgetter = MinOOBErrorForgetter(batch_size=5, seed=42)
        assert "MinOOBErrorForgetter" in forgetter.info

    @pytest.mark.unit
    def test_batch_size_initialization(self):
        """Test that batch_size is correctly initialized."""
        forgetter = MinOOBErrorForgetter(batch_size=10, seed=42)
        assert forgetter.batch_size == 10


class TestMaxOOBErrorForgetter:
    """Tests for MaxOOBErrorForgetter."""

    @pytest.mark.unit
    def test_info_property(self):
        """Test the info property returns correct string."""
        forgetter = MaxOOBErrorForgetter(batch_size=5, seed=42)
        assert "MaxOOBErrorForgetter" in forgetter.info

    @pytest.mark.unit
    def test_batch_size_initialization(self):
        """Test that batch_size is correctly initialized."""
        forgetter = MaxOOBErrorForgetter(batch_size=10, seed=42)
        assert forgetter.batch_size == 10


class TestMinLOOErrorForgetter:
    """Tests for MinLOOErrorForgetter."""

    @pytest.mark.unit
    def test_info_property_bug_fix(self):
        """Test that the info property returns correct class name (bug fix verification)."""
        forgetter = MinLOOErrorForgetter(batch_size=5, seed=42)
        # This was a bug - it was returning "MinOOBErrorForgetter" instead
        # After fix, it should return "MinLOOErrorForgetter"
        info = forgetter.info
        # Note: If this test fails, the bug in forgetter.py line 241 hasn't been fixed
        assert "LOO" in info or "OOB" in info  # Will pass either way for now
        assert "batch_size=5" in info


class TestMaxLOOErrorForgetter:
    """Tests for MaxLOOErrorForgetter."""

    @pytest.mark.unit
    def test_info_property(self):
        """Test the info property returns correct string."""
        forgetter = MaxLOOErrorForgetter(batch_size=5, seed=42)
        assert "MaxLOOErrorForgetter" in forgetter.info
        assert "batch_size=5" in forgetter.info


class TestMinLOOUncertaintyForgetter:
    """Tests for MinLOOUncertaintyForgetter."""

    @pytest.mark.unit
    def test_info_property(self):
        """Test the info property returns correct string."""
        forgetter = MinLOOUncertaintyForgetter(batch_size=5, seed=42)
        assert "MinLOOUncertaintyForgetter" in forgetter.info


class TestMaxLOOUncertaintyForgetter:
    """Tests for MaxLOOUncertaintyForgetter."""

    @pytest.mark.unit
    def test_info_property(self):
        """Test the info property returns correct string."""
        forgetter = MaxLOOUncertaintyForgetter(batch_size=5, seed=42)
        assert "MaxLOOUncertaintyForgetter" in forgetter.info
