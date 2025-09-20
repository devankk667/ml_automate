import pytest
import pandas as pd
import numpy as np
import os
import yaml

from src.data_handling import AutoDataPreprocessor

# Dummy data for testing
@pytest.fixture
def dummy_data():
    data = {
        'numeric_full': np.random.rand(20) * 10,
        'numeric_missing': np.concatenate([np.random.rand(15) * 10, [np.nan] * 5]),
        'categorical_full': ['A', 'B', 'C', 'A', 'B'] * 4,
        'categorical_missing': ['X', 'Y', 'Z', 'X', np.nan] * 4,
        'boolean_col': [True, False] * 10,
        'target': np.random.randint(0, 2, 20)
    }
    return pd.DataFrame(data)

# Dummy config file
@pytest.fixture
def dummy_config(tmpdir):
    config_content = """
missing_values:
  strategy: 'median'
categorical:
  strategy: 'onehot'
scaling:
  strategy: 'standard'
feature_selection:
  strategy: 'none'
outliers:
  strategy: 'none'
"""
    config_file = tmpdir.join("config.yaml")
    config_file.write(config_content)
    return str(config_file)

def test_initialization(dummy_config):
    """Test if the preprocessor initializes correctly with a config file."""
    preprocessor = AutoDataPreprocessor(config_path=dummy_config)
    assert preprocessor.config['missing_values']['strategy'] == 'median'
    assert preprocessor.config['categorical']['strategy'] == 'onehot'

def test_initialization_no_config():
    """Test if the preprocessor initializes with default settings."""
    preprocessor = AutoDataPreprocessor()
    assert preprocessor.config['missing_values']['strategy'] == 'auto'
    assert preprocessor.config['scaling']['strategy'] == 'standard'

def test_handle_missing_values(dummy_data):
    """Test the missing value handling."""
    preprocessor = AutoDataPreprocessor()
    X = dummy_data.drop('target', axis=1)

    X_processed, missing_info = preprocessor._handle_missing_values(X.copy())

    # Check that missing values in 'numeric_missing' and 'categorical_missing' are filled
    assert X_processed['numeric_missing'].isnull().sum() == 0
    assert X_processed['categorical_missing'].isnull().sum() == 0

    # Check if the info dictionary is populated correctly
    assert 'numeric_missing' in missing_info
    assert 'categorical_missing' in missing_info
    assert missing_info['numeric_missing']['missing_count'] == 5
    assert missing_info['categorical_missing']['missing_count'] == 4

def test_fit_transform(dummy_data, dummy_config):
    """Test the main fit_transform method."""
    preprocessor = AutoDataPreprocessor(config_path=dummy_config)
    X = dummy_data.drop('target', axis=1)
    y = dummy_data['target']

    X_transformed, missing_info = preprocessor.fit_transform(X, y)

    # Check output shape and type
    assert isinstance(X_transformed, pd.DataFrame)
    assert X_transformed.shape[0] == X.shape[0]

    # Check that no NaNs are left
    assert X_transformed.isnull().sum().sum() == 0

    # Check if one-hot encoding was applied
    # The number of columns should be greater than the original number of columns
    # because of one-hot encoding.
    assert X_transformed.shape[1] > X.shape[1]

    # Check that boolean column is present
    assert 'bool__boolean_col' in X_transformed.columns

def test_transform_on_new_data(dummy_data, dummy_config):
    """Test applying a fitted preprocessor to new data."""
    preprocessor = AutoDataPreprocessor(config_path=dummy_config)
    X = dummy_data.drop('target', axis=1)
    y = dummy_data['target']

    # Fit on the original data
    preprocessor.fit_transform(X, y)

    # Create new data with the same structure
    new_data = {
        'numeric_full': np.random.rand(10) * 10,
        'numeric_missing': np.concatenate([np.random.rand(5) * 10, [np.nan] * 5]),
        'categorical_full': ['A', 'B', 'C', 'A', 'B'] * 2,
        'categorical_missing': ['X', 'Y', 'Z', 'X', np.nan] * 2,
        'boolean_col': [True, False] * 5,
    }
    X_new = pd.DataFrame(new_data)

    # Transform the new data
    X_new_transformed = preprocessor.transform(X_new)

    # Check shape and for NaNs
    assert X_new_transformed.shape[0] == X_new.shape[0]
    assert X_new_transformed.isnull().sum().sum() == 0

    # Check that column names are consistent
    assert all(X_new_transformed.columns == preprocessor.preprocessor.get_feature_names_out())

def test_outlier_handling(dummy_data):
    """Test the outlier handling functionality."""
    # Add an obvious outlier
    dummy_data.loc[0, 'numeric_full'] = 1000

    config = {
        'outliers': {'strategy': 'zscore', 'method': 'clip', 'threshold': 2.0}
    }

    # Create a temporary config file
    with open('temp_config.yaml', 'w') as f:
        yaml.dump(config, f)

    preprocessor = AutoDataPreprocessor(config_path='temp_config.yaml')
    X = dummy_data[['numeric_full']]

    X_transformed, _ = preprocessor.fit_transform(X)

    # The clipped value should be much lower than the original outlier
    assert X_transformed.loc[0, 'num__numeric_full'] < 1000

    # Clean up the temporary file
    os.remove('temp_config.yaml')
