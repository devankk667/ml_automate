# AutoML Pipeline Tests

This directory contains tests for the AutoML pipeline. The tests are written using `pytest` and cover various aspects of the pipeline including data preprocessing, model training, and model evaluation.

## Running the Tests

### Prerequisites

1. Install the test dependencies:

```bash
pip install -r requirements-test.txt
```

### Running All Tests

To run all tests:

```bash
pytest -v
```

### Running Specific Tests

To run a specific test file:

```bash
pytest -v test_pipeline.py
```

To run a specific test function:

```bash
pytest -v test_pipeline.py::test_classification_pipeline
```

### Running Tests with Coverage

To run tests with coverage reporting:

```bash
pytest --cov=src --cov-report=term-missing -v
```

To generate an HTML coverage report:

```bash
pytest --cov=src --cov-report=html -v
```

### Running Tests in Parallel

To run tests in parallel (using all available CPU cores):

```bash
pytest -n auto -v
```

## Test Organization

- `test_pipeline.py`: Contains tests for the main AutoML pipeline functionality
  - `test_classification_pipeline`: Tests the pipeline with a synthetic classification dataset
  - `test_regression_pipeline`: Tests the pipeline with a synthetic regression dataset
  - `test_pipeline_save_load`: Tests saving and loading of the pipeline
  - `test_missing_values_handling`: Tests the pipeline's ability to handle missing values

## Adding New Tests

When adding new features to the AutoML pipeline, please add corresponding tests to ensure the functionality works as expected. Follow these guidelines:

1. Create a new test file or add to an existing one if appropriate
2. Use descriptive test function names that start with `test_`
3. Use fixtures for common test setup
4. Include assertions to verify the expected behavior
5. Add docstrings to explain what each test is checking

## Debugging Tests

To drop into the Python debugger on test failures:

```bash
pytest --pdb -v
```

To run tests with detailed output:

```bash
pytest -v -s
```

## Continuous Integration

These tests should be run as part of your CI/CD pipeline to ensure that changes don't introduce regressions. A sample GitHub Actions workflow (`.github/workflows/tests.yml`) is provided in the root of the repository.

## Code Coverage

We aim to maintain high test coverage for the codebase. The current coverage can be viewed by running:

```bash
pytest --cov=src --cov-report=html -v
```

Then open `htmlcov/index.html` in your browser to view the coverage report.
