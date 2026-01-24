#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for utility functions.
"""
import pytest
import numpy as np
from molalkit.active_learning.utils import random_choice, get_topn_idx


class TestRandomChoice:
    """Tests for random_choice utility function."""

    @pytest.mark.unit
    def test_returns_correct_size(self):
        """Test that function returns correct number of indices."""
        np.random.seed(42)
        idx, remain = random_choice(N=100, size=10)

        assert len(idx) == 10
        assert len(remain) == 90

    @pytest.mark.unit
    def test_no_overlap(self):
        """Test that selected and remaining indices don't overlap."""
        np.random.seed(42)
        idx, remain = random_choice(N=100, size=10)

        assert set(idx).isdisjoint(set(remain))

    @pytest.mark.unit
    def test_all_indices_covered(self):
        """Test that all indices from 0 to N-1 are covered."""
        np.random.seed(42)
        idx, remain = random_choice(N=100, size=10)

        all_indices = set(idx) | set(remain)
        assert all_indices == set(range(100))

    @pytest.mark.unit
    def test_size_equals_n(self):
        """Test behavior when size equals N."""
        np.random.seed(42)
        idx, remain = random_choice(N=10, size=10)

        assert len(idx) == 10
        assert len(remain) == 0

    @pytest.mark.unit
    def test_size_greater_than_n(self):
        """Test behavior when size exceeds N."""
        np.random.seed(42)
        idx, remain = random_choice(N=10, size=20)

        # Should return all available indices
        assert len(idx) == 10
        assert len(remain) == 0

    @pytest.mark.unit
    def test_size_zero(self):
        """Test behavior when size is zero."""
        np.random.seed(42)
        idx, remain = random_choice(N=10, size=0)

        assert len(idx) == 0
        assert len(remain) == 10


class TestGetTopnIdx:
    """Tests for get_topn_idx utility function."""

    @pytest.mark.unit
    def test_max_target_returns_highest_values(self):
        """Test that max target returns indices of highest values."""
        values = np.array([1.0, 5.0, 3.0, 2.0, 4.0])
        idx = get_topn_idx(values, n=2, target="max")

        # Should return indices of 5.0 and 4.0
        assert set(idx) == {1, 4}

    @pytest.mark.unit
    def test_min_target_returns_lowest_values(self):
        """Test that min target returns indices of lowest values."""
        values = np.array([1.0, 5.0, 3.0, 2.0, 4.0])
        idx = get_topn_idx(values, n=2, target="min")

        # Should return indices of 1.0 and 2.0
        assert set(idx) == {0, 3}

    @pytest.mark.unit
    def test_float_target_returns_closest_values(self):
        """Test that float target returns indices closest to target value."""
        values = np.array([1.0, 5.0, 3.0, 2.0, 4.0])
        idx = get_topn_idx(values, n=2, target=3.5)

        # Should return indices of 3.0 and 4.0 (closest to 3.5)
        assert set(idx) == {2, 4}

    @pytest.mark.unit
    def test_returns_correct_n(self):
        """Test that function returns exactly n indices."""
        values = np.array([1.0, 5.0, 3.0, 2.0, 4.0])
        idx = get_topn_idx(values, n=3, target="max")

        assert len(idx) == 3

    @pytest.mark.unit
    def test_n_larger_than_array(self):
        """Test behavior when n exceeds array length."""
        values = np.array([1.0, 5.0, 3.0])
        idx = get_topn_idx(values, n=10, target="max")

        # Should return all indices
        assert len(idx) == 3

    @pytest.mark.unit
    def test_cutoff_filters_values_max(self):
        """Test that cutoff filters values when target is max."""
        values = np.array([1.0, 5.0, 3.0, 2.0, 4.0])
        idx = get_topn_idx(values, n=10, target="max", cutoff=3.5)

        # Only values >= 3.5 should be considered: 5.0, 4.0
        assert len(idx) == 2
        assert set(idx) == {1, 4}

    @pytest.mark.unit
    def test_cutoff_filters_values_min(self):
        """Test that cutoff filters values when target is min."""
        values = np.array([1.0, 5.0, 3.0, 2.0, 4.0])
        idx = get_topn_idx(values, n=10, target="min", cutoff=2.5)

        # For min, cutoff=-2.5, so -values >= -2.5 means values <= 2.5
        # Values <= 2.5: 1.0, 2.0
        assert len(idx) == 2
        assert set(idx) == {0, 3}

    @pytest.mark.unit
    def test_cutoff_no_values_pass(self):
        """Test behavior when no values pass the cutoff."""
        values = np.array([1.0, 2.0, 3.0])
        idx = get_topn_idx(values, n=10, target="max", cutoff=5.0)

        # No values >= 5.0
        assert len(idx) == 0

    @pytest.mark.unit
    def test_accepts_list_input(self):
        """Test that function accepts list input."""
        values = [1.0, 5.0, 3.0, 2.0, 4.0]
        idx = get_topn_idx(values, n=2, target="max")

        assert len(idx) == 2

    @pytest.mark.unit
    def test_handles_duplicate_values(self):
        """Test that function handles duplicate values."""
        values = np.array([1.0, 3.0, 3.0, 3.0, 5.0])
        idx = get_topn_idx(values, n=3, target="max")

        # Should return 3 indices (5.0 and two of the 3.0s)
        assert len(idx) == 3
        assert 4 in idx  # Index of 5.0 should definitely be included

    @pytest.mark.unit
    def test_single_element_array(self):
        """Test behavior with single element array."""
        values = np.array([3.0])
        idx = get_topn_idx(values, n=1, target="max")

        assert idx == [0]

    @pytest.mark.unit
    def test_n_equals_one(self):
        """Test behavior when n=1."""
        values = np.array([1.0, 5.0, 3.0, 2.0, 4.0])
        idx = get_topn_idx(values, n=1, target="max")

        assert len(idx) == 1
        assert idx[0] == 1  # Index of max value (5.0)
