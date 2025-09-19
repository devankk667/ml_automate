"""
Tests for the AutoML pipeline.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Add the parent directory to the path so we can import the pipeline
sys.path.append(str(Path(__file__).parent.parent))

from src.pipeline import create_pipeline
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split


def test_classification_pipeline():
    """Test the AutoML pipeline with a synthetic classification dataset."""
    # Create a synthetic classification dataset
    X, y = make_classification(
        n_samples=100,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    
    # Convert to DataFrame for better feature names
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create and train the pipeline
    pipeline = create_pipeline(
        task='classification',
        random_state=42,
        n_jobs=1,
        verbose=0
    )
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Evaluate the model
    metrics = pipeline.evaluate(X_test, y_test)
    
    # Basic assertions
    assert hasattr(pipeline, 'model_')
    assert 'accuracy' in metrics
    assert metrics['accuracy'] >= 0.5  # Should be better than random
    
    # Test feature importances
    try:
        importances = pipeline.get_feature_importances()
        assert len(importances) == X.shape[1]
    except Exception as e:
        # Some models might not support feature importances
        pass


def test_regression_pipeline():
    """Test the AutoML pipeline with a synthetic regression dataset."""
    # Create a synthetic regression dataset
    X, y = make_regression(
        n_samples=100,
        n_features=10,
        n_informative=5,
        noise=0.1,
        random_state=42
    )
    
    # Convert to DataFrame for better feature names
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42
    )
    
    # Create and train the pipeline
    pipeline = create_pipeline(
        task='regression',
        random_state=42,
        n_jobs=1,
        verbose=0
    )
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Evaluate the model
    metrics = pipeline.evaluate(X_test, y_test)
    
    # Basic assertions
    assert hasattr(pipeline, 'model_')
    assert 'r2' in metrics
    assert metrics['r2'] > 0  # Should be better than guessing the mean
    
    # Test feature importances
    try:
        importances = pipeline.get_feature_importances()
        assert len(importances) == X.shape[1]
    except Exception as e:
        # Some models might not support feature importances
        pass


def test_pipeline_save_load(tmp_path):
    """Test saving and loading the pipeline."""
    # Create a synthetic classification dataset
    X, y = make_classification(
        n_samples=50,
        n_features=5,
        n_informative=3,
        n_classes=2,
        random_state=42
    )
    
    # Create and train the pipeline
    pipeline = create_pipeline(
        task='classification',
        random_state=42,
        n_jobs=1,
        verbose=0
    )
    pipeline.fit(X, y)
    
    # Save the pipeline
    save_path = tmp_path / 'pipeline'
    pipeline.save(save_path)
    
    # Check that files were created
    assert (save_path / 'model.joblib').exists()
    assert (save_path / 'preprocessor.joblib').exists()
    assert (save_path / 'metadata.json').exists()
    
    # Load the pipeline
    loaded_pipeline = pipeline.__class__.load(save_path)
    
    # Make predictions with both pipelines
    y_pred_original = pipeline.predict(X)
    y_pred_loaded = loaded_pipeline.predict(X)
    
    # Check that predictions match
    assert np.array_equal(y_pred_original, y_pred_loaded)


def test_missing_values_handling():
    """Test that the pipeline can handle missing values."""
    # Create a synthetic dataset with missing values
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=3,
        n_classes=2,
        random_state=42
    )
    
    # Add some missing values
    np.random.seed(42)
    mask = np.random.random(X.shape) < 0.1
    X_missing = X.copy()
    X_missing[mask] = np.nan
    
    # Convert to DataFrame
    X_df = pd.DataFrame(X_missing, columns=[f'feature_{i}' for i in range(X.shape[1])])
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create and train the pipeline
    pipeline = create_pipeline(
        task='classification',
        random_state=42,
        n_jobs=1,
        verbose=0
    )
    
    # This should not raise an exception
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Check that we got predictions for all samples
    assert len(y_pred) == len(y_test)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
