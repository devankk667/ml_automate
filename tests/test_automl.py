import pytest
import pandas as pd
import numpy as np
import yaml
import os
from unittest.mock import patch, MagicMock

from src.auto_ml import AutoML, get_class

# Create a dummy dataset for testing
@pytest.fixture
def dummy_classification_data():
    X = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'category': ['A', 'B'] * 50
    })
    y = pd.Series(np.random.randint(0, 2, 100))
    return X, y

@pytest.fixture
def dummy_regression_data():
    X = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100)
    })
    y = pd.Series(np.random.rand(100) * 100)
    return X, y

# Create a dummy models.yaml for testing
@pytest.fixture(scope="function")
def dummy_model_config(tmpdir):
    config_content = """
classification:
  logistic_regression:
    model: "sklearn.linear_model.LogisticRegression"
    params:
      C: [0.1, 1]
      penalty: ['l2']

regression:
  linear_regression:
    model: "sklearn.linear_model.LinearRegression"
    params: {}
"""
    config_file = tmpdir.join("models.yaml")
    config_file.write(config_content)
    return str(config_file)

def test_get_class():
    """Test the dynamic class importer."""
    cls = get_class("pandas.DataFrame")
    assert cls == pd.DataFrame

    with pytest.raises(ImportError):
        get_class("non_existent_module.NonExistentClass")

def test_automl_initialization(dummy_model_config):
    """Test if AutoML initializes correctly."""
    automl = AutoML(task='classification', models_config_path=dummy_model_config)
    assert automl.task == 'classification'
    assert 'logistic_regression' in automl.models
    assert 'model' in automl.models['logistic_regression']
    assert 'params' in automl.models['logistic_regression']

def test_automl_init_raises_error_for_bad_task(dummy_model_config):
    """Test that AutoML raises an error for an unsupported task in the config."""
    with pytest.raises(ValueError):
        AutoML(task='unsupported_task', models_config_path=dummy_model_config)

@patch('src.auto_ml.AutoML._train_sklearn_model')
def test_automl_fit_classification(mock_train, dummy_classification_data, dummy_model_config):
    """Test the fit method for a classification task."""
    # Configure the mock model and its predict method
    mock_model = MagicMock()
    mock_model.predict.return_value = np.zeros(20) # Return a valid prediction array
    mock_model.predict_proba.return_value = np.zeros((20, 2)) # Return a valid probability array
    mock_train.return_value = (mock_model, 0.95) # (mock_model, mock_score)

    X, y = dummy_classification_data

    automl = AutoML(task='classification', models_config_path=dummy_model_config, verbose=0)

    # Run fit
    summary = automl.fit(X, y, test_size=0.2, cv=2, n_iter=1)

    # Assertions
    assert mock_train.called
    assert automl.best_model is not None
    assert automl.best_model_name == 'logistic_regression'
    assert 'test_metrics' in summary
    assert 'accuracy' in summary['test_metrics']
    assert hasattr(automl, 'X_test') and hasattr(automl, 'y_test')
    assert len(automl.X_test) == 20 # 100 * 0.2

@patch('src.auto_ml.AutoML._train_sklearn_model')
def test_automl_fit_regression(mock_train, dummy_regression_data, dummy_model_config):
    """Test the fit method for a regression task."""
    # Configure the mock model and its predict method
    mock_model = MagicMock()
    mock_model.predict.return_value = np.zeros(30) # Return a valid prediction array
    mock_train.return_value = (mock_model, -10.0) # (mock_model, mock_score)

    X, y = dummy_regression_data

    automl = AutoML(task='regression', scoring='neg_mean_squared_error', models_config_path=dummy_model_config, verbose=0)

    # Run fit
    summary = automl.fit(X, y, test_size=0.3, cv=2, n_iter=1)

    # Assertions
    assert mock_train.called
    assert automl.best_model is not None
    assert automl.best_model_name == 'linear_regression'
    assert 'test_metrics' in summary
    assert 'mse' in summary['test_metrics']
    assert hasattr(automl, 'X_test') and hasattr(automl, 'y_test')
    assert len(automl.X_test) == 30 # 100 * 0.3

def test_automl_evaluate(dummy_classification_data, dummy_model_config):
    """Test the evaluation method."""
    X, y = dummy_classification_data
    automl = AutoML(task='classification', models_config_path=dummy_model_config, verbose=0)

    # Manually set a mock model and test data
    from sklearn.linear_model import LogisticRegression
    mock_model = LogisticRegression()
    mock_model.fit(X.drop('category', axis=1), y) # Drop categorical for simplicity

    automl.best_model = mock_model
    automl.X_test = X.drop('category', axis=1)
    automl.y_test = y

    metrics = automl.evaluate()

    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    assert metrics['accuracy'] >= 0.0
    assert metrics['accuracy'] <= 1.0

def test_automl_predict(dummy_classification_data, dummy_model_config):
    """Test the predict method."""
    X, y = dummy_classification_data
    automl = AutoML(task='classification', models_config_path=dummy_model_config, verbose=0)

    from sklearn.linear_model import LogisticRegression
    mock_model = LogisticRegression()
    mock_model.fit(X.drop('category', axis=1), y)

    automl.best_model = mock_model

    predictions = automl.predict(X.drop('category', axis=1))

    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(y)
