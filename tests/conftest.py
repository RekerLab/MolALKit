#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Shared fixtures for MolALKit tests.
"""
import pytest
import numpy as np
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import List


# ============================================================================
# Mock Data Classes
# ============================================================================

@dataclass
class MockDataPoint:
    """Mock data point for testing."""
    uidx: int
    targets: List[float]


class MockDataset:
    """Mock dataset for testing selectors and forgetters without real molecular data."""

    def __init__(self, size: int, n_features: int = 10, seed: int = 42):
        np.random.seed(seed)
        self.size = size
        self.n_features = n_features
        self._X = np.random.randn(size, n_features)
        self._y = np.random.randn(size, 1)
        self.data = [MockDataPoint(uidx=i, targets=[self._y[i, 0]]) for i in range(size)]

    def __len__(self):
        return self.size

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y

    def __copy__(self):
        """Support copy operations for get_subset_from_idx."""
        new_ds = MockDataset.__new__(MockDataset)
        new_ds.size = self.size
        new_ds.n_features = self.n_features
        new_ds._X = self._X
        new_ds._y = self._y
        new_ds.data = self.data.copy()
        return new_ds


class MockModel:
    """Mock model for testing selectors."""

    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.fitted = False

    def fit_molalkit(self, data, iteration: int = 0):
        self.fitted = True
        self._n_samples = len(data)
        return self

    def predict_value(self, data):
        """Return random predictions."""
        return np.random.randn(len(data))

    def predict_uncertainty(self, data):
        """Return random uncertainties (positive values)."""
        return np.abs(np.random.randn(len(data)))


class MockModelWithOOB(MockModel):
    """
    Mock model with OOB functionality (like RandomForest).

    Note: This mock cannot fully replace RFClassifier for OOB forgetter tests
    because the forgetters use isinstance() checks. Use this only for basic
    interface testing. For full OOB forgetter testing, use integration tests
    with real RandomForest models.
    """

    def __init__(self, seed: int = 42):
        super().__init__(seed)
        self.oob_score = True
        self.oob_decision_function_ = None

    def fit_molalkit(self, data, iteration: int = 0):
        super().fit_molalkit(data, iteration)
        # Generate OOB predictions for classification (2 classes)
        n_samples = len(data)
        np.random.seed(42)  # Fixed seed for reproducibility
        probs = np.random.rand(n_samples, 2)
        probs = probs / probs.sum(axis=1, keepdims=True)  # Normalize to sum to 1
        self.oob_decision_function_ = probs
        return self


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_dataset_small():
    """Small dataset with 20 samples for quick tests."""
    return MockDataset(size=20, n_features=10, seed=42)


@pytest.fixture
def mock_dataset_medium():
    """Medium dataset with 100 samples."""
    return MockDataset(size=100, n_features=10, seed=42)


@pytest.fixture
def mock_dataset_large():
    """Large dataset with 500 samples."""
    return MockDataset(size=500, n_features=10, seed=42)


@pytest.fixture
def mock_model():
    """Mock model for testing."""
    return MockModel(seed=42)


@pytest.fixture
def mock_model_with_oob():
    """Mock model with OOB functionality."""
    return MockModelWithOOB(seed=42)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    dirpath = tempfile.mkdtemp()
    yield dirpath
    # Cleanup after test
    if os.path.exists(dirpath):
        shutil.rmtree(dirpath)


@pytest.fixture
def sample_smiles():
    """Sample SMILES strings for testing."""
    return [
        "CCO",  # Ethanol
        "CC(=O)O",  # Acetic acid
        "c1ccccc1",  # Benzene
        "CC(C)O",  # Isopropanol
        "CCCC",  # Butane
        "CC=O",  # Acetaldehyde
        "CCN",  # Ethylamine
        "CCC",  # Propane
        "CO",  # Methanol
        "C",  # Methane
    ]


# ============================================================================
# Test Configuration
# ============================================================================

def pytest_configure(config):
    """Add custom markers."""
    config.addinivalue_line("markers", "unit: Fast unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "slow: Slow tests")
