#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for selector classes.
"""
import pytest
import numpy as np

from molalkit.active_learning.selector import (
    RandomSelector,
    ExplorativeSelector,
    ExploitiveSelector,
    PartialQueryExplorativeSelector,
    PartialQueryExploitiveSelector,
)


class TestRandomSelector:
    """Tests for RandomSelector."""

    @pytest.mark.unit
    def test_returns_correct_batch_size(self, mock_dataset_small):
        """Test that selector returns the requested batch size."""
        selector = RandomSelector(batch_size=5, seed=42)
        idx, acquisition, remain = selector(dataset_pool=mock_dataset_small)

        assert len(idx) == 5
        assert len(remain) == len(mock_dataset_small) - 5

    @pytest.mark.unit
    def test_no_overlap_between_selected_and_remaining(self, mock_dataset_small):
        """Test that selected and remaining indices don't overlap."""
        selector = RandomSelector(batch_size=5, seed=42)
        idx, _, remain = selector(dataset_pool=mock_dataset_small)

        assert set(idx).isdisjoint(set(remain))
        assert len(idx) + len(remain) == len(mock_dataset_small)

    @pytest.mark.unit
    def test_all_indices_valid(self, mock_dataset_small):
        """Test that all returned indices are valid."""
        selector = RandomSelector(batch_size=5, seed=42)
        idx, _, remain = selector(dataset_pool=mock_dataset_small)

        all_indices = idx + remain
        assert all(0 <= i < len(mock_dataset_small) for i in all_indices)

    @pytest.mark.unit
    def test_batch_size_larger_than_pool(self, mock_dataset_small):
        """Test behavior when batch size exceeds pool size."""
        selector = RandomSelector(batch_size=100, seed=42)
        idx, _, remain = selector(dataset_pool=mock_dataset_small)

        # Should return all available samples
        assert len(idx) == len(mock_dataset_small)
        assert len(remain) == 0

    @pytest.mark.unit
    def test_different_seeds_produce_different_results(self, mock_dataset_medium):
        """Test that different seeds produce different results."""
        # Use the same dataset but different selectors with different seeds
        # The selectors set numpy random seed in __init__, so we need to be careful
        np.random.seed(999)  # Reset to a known state first

        selector1 = RandomSelector(batch_size=10, seed=42)
        idx1, _, _ = selector1(dataset_pool=mock_dataset_medium)

        selector2 = RandomSelector(batch_size=10, seed=123)
        idx2, _, _ = selector2(dataset_pool=mock_dataset_medium)

        # With different seeds, results should differ (very high probability)
        assert idx1 != idx2

    @pytest.mark.unit
    def test_info_property(self):
        """Test the info property returns correct string."""
        selector = RandomSelector(batch_size=5, seed=42)
        assert "RandomSelector" in selector.info
        assert "batch_size=5" in selector.info


class TestExplorativeSelector:
    """Tests for ExplorativeSelector (uncertainty-based selection)."""

    @pytest.mark.unit
    def test_returns_correct_batch_size(self, mock_dataset_small, mock_model):
        """Test that selector returns the requested batch size."""
        mock_model.fit_molalkit(mock_dataset_small)
        selector = ExplorativeSelector(batch_size=5, seed=42)
        idx, acquisition, remain = selector(model=mock_model, dataset_pool=mock_dataset_small)

        assert len(idx) == 5
        assert len(acquisition) == 5
        assert len(remain) == len(mock_dataset_small) - 5

    @pytest.mark.unit
    def test_acquisition_values_are_returned(self, mock_dataset_small, mock_model):
        """Test that acquisition values (uncertainties) are returned."""
        mock_model.fit_molalkit(mock_dataset_small)
        selector = ExplorativeSelector(batch_size=5, seed=42)
        idx, acquisition, _ = selector(model=mock_model, dataset_pool=mock_dataset_small)

        assert len(acquisition) == len(idx)
        # All uncertainty values should be non-negative (using abs in mock)
        assert all(a >= 0 for a in acquisition)

    @pytest.mark.unit
    def test_no_overlap_between_selected_and_remaining(self, mock_dataset_small, mock_model):
        """Test that selected and remaining indices don't overlap."""
        mock_model.fit_molalkit(mock_dataset_small)
        selector = ExplorativeSelector(batch_size=5, seed=42)
        idx, _, remain = selector(model=mock_model, dataset_pool=mock_dataset_small)

        assert set(idx).isdisjoint(set(remain))

    @pytest.mark.unit
    def test_batch_size_larger_than_pool(self, mock_dataset_small, mock_model):
        """Test behavior when batch size exceeds pool size."""
        mock_model.fit_molalkit(mock_dataset_small)
        selector = ExplorativeSelector(batch_size=100, seed=42)
        idx, _, remain = selector(model=mock_model, dataset_pool=mock_dataset_small)

        assert len(idx) == len(mock_dataset_small)
        assert len(remain) == 0

    @pytest.mark.unit
    def test_info_property(self):
        """Test the info property returns correct string."""
        selector = ExplorativeSelector(batch_size=5, seed=42)
        assert "ExplorativeSelector" in selector.info
        assert "batch_size=5" in selector.info


class _FixedUncertaintyModel:
    """Model that returns fixed uncertainty values for deterministic testing."""

    def __init__(self, uncertainties):
        self._uncertainties = np.array(uncertainties)

    def predict_uncertainty(self, data):
        return self._uncertainties[:len(data)]


class TestExplorativeThresholdSelector:
    """Tests for ExplorativeSelector with target (threshold-based selection)."""

    @pytest.mark.unit
    def test_target_selects_lowest_above_cutoff(self):
        """When target is set, select the point with lowest uncertainty above the cutoff."""
        uncertainties = [0.1, 0.3, 0.6, 0.8, 0.9]
        model = _FixedUncertaintyModel(uncertainties)
        dataset = type("D", (), {"__len__": lambda self: 5})()

        selector = ExplorativeSelector(batch_size=1, seed=42, target=0.5)
        idx, acquisition, remain = selector(model=model, dataset_pool=dataset)

        assert len(idx) == 1
        assert idx[0] == 2  # uncertainty 0.6 is the lowest above 0.5
        assert acquisition[0] == pytest.approx(0.6)

    @pytest.mark.unit
    def test_target_selects_multiple_lowest_above_cutoff(self):
        """When batch_size > 1, select the N lowest uncertainties above cutoff."""
        uncertainties = [0.1, 0.3, 0.6, 0.8, 0.9]
        model = _FixedUncertaintyModel(uncertainties)
        dataset = type("D", (), {"__len__": lambda self: 5})()

        selector = ExplorativeSelector(batch_size=2, seed=42, target=0.5)
        idx, acquisition, remain = selector(model=model, dataset_pool=dataset)

        assert len(idx) == 2
        assert set(idx) == {2, 3}  # uncertainties 0.6 and 0.8

    @pytest.mark.unit
    def test_target_fallback_when_none_above_cutoff(self):
        """When no points exceed cutoff, fall back to max uncertainty selection."""
        uncertainties = [0.1, 0.2, 0.3, 0.4, 0.5]
        model = _FixedUncertaintyModel(uncertainties)
        dataset = type("D", (), {"__len__": lambda self: 5})()

        selector = ExplorativeSelector(batch_size=1, seed=42, target=0.9)
        idx, acquisition, remain = selector(model=model, dataset_pool=dataset)

        assert len(idx) == 1
        assert idx[0] == 4  # highest uncertainty (0.5) is selected as fallback

    @pytest.mark.unit
    def test_target_none_preserves_original_behavior(self, mock_dataset_small, mock_model):
        """When target is None, behavior is identical to standard ExplorativeSelector."""
        mock_model.fit_molalkit(mock_dataset_small)

        selector_default = ExplorativeSelector(batch_size=5, seed=42, target=None)
        selector_original = ExplorativeSelector(batch_size=5, seed=42)

        np.random.seed(100)
        idx1, acq1, _ = selector_default(model=mock_model, dataset_pool=mock_dataset_small)
        np.random.seed(100)
        idx2, acq2, _ = selector_original(model=mock_model, dataset_pool=mock_dataset_small)

        assert idx1 == idx2
        assert acq1 == acq2

    @pytest.mark.unit
    def test_target_batch_larger_than_above_cutoff_count(self):
        """When batch_size exceeds number of points above cutoff, select all above cutoff."""
        uncertainties = [0.1, 0.3, 0.6, 0.8, 0.9]
        model = _FixedUncertaintyModel(uncertainties)
        dataset = type("D", (), {"__len__": lambda self: 5})()

        selector = ExplorativeSelector(batch_size=5, seed=42, target=0.5)
        idx, acquisition, remain = selector(model=model, dataset_pool=dataset)

        assert set(idx) == {2, 3, 4}  # only 3 points above 0.5

    @pytest.mark.unit
    def test_no_overlap_with_target(self):
        """Selected and remaining indices should not overlap when target is set."""
        uncertainties = [0.1, 0.3, 0.6, 0.8, 0.9]
        model = _FixedUncertaintyModel(uncertainties)
        dataset = type("D", (), {"__len__": lambda self: 5})()

        selector = ExplorativeSelector(batch_size=2, seed=42, target=0.5)
        idx, _, remain = selector(model=model, dataset_pool=dataset)

        assert set(idx).isdisjoint(set(remain))
        assert len(idx) + len(remain) == 5

    @pytest.mark.unit
    def test_info_property_with_target(self):
        """Test info property includes target when set."""
        selector = ExplorativeSelector(batch_size=5, seed=42, target=0.5)
        assert "ExplorativeSelector" in selector.info
        assert "target=0.5" in selector.info

    @pytest.mark.unit
    def test_info_property_without_target(self):
        """Test info property omits target when None."""
        selector = ExplorativeSelector(batch_size=5, seed=42)
        assert "target" not in selector.info


class TestExploitiveSelector:
    """Tests for ExploitiveSelector (prediction-based selection)."""

    @pytest.mark.unit
    def test_returns_correct_batch_size_max_target(self, mock_dataset_small, mock_model):
        """Test selector with max target."""
        mock_model.fit_molalkit(mock_dataset_small)
        selector = ExploitiveSelector(target="max", batch_size=5, seed=42)
        idx, acquisition, remain = selector(model=mock_model, dataset_pool=mock_dataset_small)

        assert len(idx) == 5
        assert len(acquisition) == 5
        assert len(remain) == len(mock_dataset_small) - 5

    @pytest.mark.unit
    def test_returns_correct_batch_size_min_target(self, mock_dataset_small, mock_model):
        """Test selector with min target."""
        mock_model.fit_molalkit(mock_dataset_small)
        selector = ExploitiveSelector(target="min", batch_size=5, seed=42)
        idx, acquisition, remain = selector(model=mock_model, dataset_pool=mock_dataset_small)

        assert len(idx) == 5
        assert len(acquisition) == 5

    @pytest.mark.unit
    def test_returns_correct_batch_size_float_target(self, mock_dataset_small, mock_model):
        """Test selector with float target (find closest to value)."""
        mock_model.fit_molalkit(mock_dataset_small)
        selector = ExploitiveSelector(target=0.5, batch_size=5, seed=42)
        idx, acquisition, remain = selector(model=mock_model, dataset_pool=mock_dataset_small)

        assert len(idx) == 5

    @pytest.mark.unit
    def test_no_overlap_between_selected_and_remaining(self, mock_dataset_small, mock_model):
        """Test that selected and remaining indices don't overlap."""
        mock_model.fit_molalkit(mock_dataset_small)
        selector = ExploitiveSelector(target="max", batch_size=5, seed=42)
        idx, _, remain = selector(model=mock_model, dataset_pool=mock_dataset_small)

        assert set(idx).isdisjoint(set(remain))

    @pytest.mark.unit
    def test_info_property(self):
        """Test the info property returns correct string."""
        selector = ExploitiveSelector(target="max", batch_size=5, seed=42)
        assert "ExploitiveSelector" in selector.info
        assert "batch_size=5" in selector.info
        assert "target=max" in selector.info


class TestPartialQueryExplorativeSelector:
    """Tests for PartialQueryExplorativeSelector.

    Note: Full functional tests with subset data are covered in integration tests.
    Unit tests here focus on initialization constraints and interface.
    """

    @pytest.mark.unit
    def test_query_size_constraint(self):
        """Test that batch_size must be <= query_size."""
        with pytest.raises(AssertionError):
            PartialQueryExplorativeSelector(query_size=5, batch_size=10, seed=42)

    @pytest.mark.unit
    def test_valid_initialization(self):
        """Test valid initialization."""
        selector = PartialQueryExplorativeSelector(query_size=50, batch_size=5, seed=42)
        assert selector.batch_size == 5
        assert selector.query_size == 50

    @pytest.mark.unit
    def test_info_property(self):
        """Test the info property returns correct string."""
        selector = PartialQueryExplorativeSelector(query_size=50, batch_size=5, seed=42)
        assert "PartialQueryExplorativeSelector" in selector.info
        assert "batch_size=5" in selector.info
        assert "query_size=50" in selector.info


class TestPartialQueryExploitiveSelector:
    """Tests for PartialQueryExploitiveSelector.

    Note: Full functional tests with subset data are covered in integration tests.
    Unit tests here focus on initialization constraints and interface.
    """

    @pytest.mark.unit
    def test_query_size_constraint(self):
        """Test that batch_size must be <= query_size."""
        with pytest.raises(AssertionError):
            PartialQueryExploitiveSelector(target="max", query_size=5, batch_size=10, seed=42)

    @pytest.mark.unit
    def test_valid_initialization(self):
        """Test valid initialization."""
        selector = PartialQueryExploitiveSelector(target="max", query_size=50, batch_size=5, seed=42)
        assert selector.batch_size == 5
        assert selector.query_size == 50
        assert selector.target == "max"

    @pytest.mark.unit
    def test_info_property(self):
        """Test the info property returns correct string."""
        selector = PartialQueryExploitiveSelector(target="min", query_size=50, batch_size=5, seed=42)
        assert "PartialQueryExploitiveSelector" in selector.info
        assert "target=min" in selector.info
