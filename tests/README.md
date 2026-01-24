# MolALKit Test Suite

This directory contains tests organized by test level for efficient development and CI workflows.

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and mock classes
├── unit/                    # Fast, isolated unit tests
│   ├── test_selectors.py    # Tests for selector classes
│   ├── test_forgetters.py   # Tests for forgetter classes
│   └── test_utils.py        # Tests for utility functions
├── integration/             # Component interaction tests
│   ├── test_active_learner.py   # ActiveLearner workflow tests
│   └── test_checkpoint.py       # Checkpoint save/load tests
├── e2e/                     # End-to-end pipeline tests
│   └── test_full_pipeline.py    # Full pipeline with explicit scenarios
└── (legacy files)           # Old test files (test_learning.py, test_model.py)
```

## Running Tests

### Run all tests
```bash
pytest tests/
```

### Run by test level
```bash
# Fast unit tests only (no ML training)
pytest tests/ -m unit

# Integration tests (may train small models)
pytest tests/ -m integration

# End-to-end tests (full pipeline, slower)
pytest tests/ -m e2e

# Skip slow tests
pytest tests/ -m "not slow"
```

### Run specific test file
```bash
pytest tests/unit/test_selectors.py
pytest tests/integration/test_checkpoint.py
```

### Run specific test class or function
```bash
pytest tests/unit/test_selectors.py::TestRandomSelector
pytest tests/unit/test_selectors.py::TestRandomSelector::test_returns_correct_batch_size
```

### Run with verbose output
```bash
pytest tests/ -v
```

### Run with coverage
```bash
pytest tests/ --cov=molalkit --cov-report=html
```

## Test Markers

- `@pytest.mark.unit` - Fast unit tests that don't train ML models
- `@pytest.mark.integration` - Tests that may train small models
- `@pytest.mark.e2e` - End-to-end pipeline tests
- `@pytest.mark.slow` - Tests that take a long time to run

## Writing New Tests

### Unit Tests
- Test individual functions/classes in isolation
- Use mock objects from `conftest.py` (MockDataset, MockModel)
- Should run in milliseconds
- Place in `tests/unit/`

### Integration Tests
- Test component interactions
- May use real (but small) datasets
- Should run in seconds
- Place in `tests/integration/`

### E2E Tests
- Test full pipeline with real data
- Use explicit, representative scenarios (not combinatorial explosion)
- May take longer to run
- Place in `tests/e2e/`

## Fixtures

Common fixtures are defined in `conftest.py`:

- `mock_dataset_small` - 20 samples
- `mock_dataset_medium` - 100 samples
- `mock_dataset_large` - 500 samples
- `mock_model` - Basic mock model
- `mock_model_with_oob` - Mock model with OOB functionality
- `temp_dir` - Temporary directory for test outputs

## Legacy Tests

The files `test_learning.py` and `test_model.py` in the root tests directory
are legacy tests with combinatorial parameter explosion. They use random
sampling to reduce test count, which makes them non-deterministic.

For new tests, prefer the structured approach in `unit/`, `integration/`, and `e2e/`.
