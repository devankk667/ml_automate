"""
AutoML Pipeline
--------------
This module provides a complete pipeline for automated machine learning,
integrating data preprocessing, model training, and evaluation.
"""

import os
import json
import yaml
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Union, Any, Optional

# Import our modules
from .auto_ml import AutoML, auto_train
from .data_handling import AutoDataPreprocessor
from .utils import (
    load_config, save_config, create_directory, 
    get_classification_metrics, get_regression_metrics,
    plot_confusion_matrix, plot_roc_curve, plot_feature_importance
)


class AutoMLPipeline:
    """
    End-to-end AutoML pipeline that handles data preprocessing, model training,
    evaluation, and deployment.
    """
    
    def __init__(
        self,
        task: str = 'classification',
        config_path: Optional[str] = None,
        random_state: int = 42,
        n_jobs: int = -1,
        verbose: int = 1
    ):
        """
        Initialize the AutoML pipeline.
        
        Args:
            task: Type of task - 'classification' or 'regression'
            config_path: Path to a YAML configuration file
            random_state: Random seed for reproducibility
            n_jobs: Number of jobs to run in parallel (-1 uses all available cores)
            verbose: Verbosity level (0: silent, 1: info, 2: debug)
        """
        self.task = task
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        # Load and validate configuration
        self.config = load_config(config_path)
        self._validate_config()
        
        # Initialize components
        self.preprocessor = AutoDataPreprocessor(
            config_path=os.path.join(os.path.dirname(config_path), 'preprocessing_config.yaml')
            if config_path else None
        )
        
        self.automl = AutoML(
            task=task,
            scoring=self.config['general']['scoring'][task],
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose
        )
        
        # Initialize attributes
        self.feature_names_ = None
        self.target_name_ = None
        self.classes_ = None
        self.metrics_ = {}
        self.model_ = None
        self.history_ = None
    
    def _validate_config(self) -> None:
        """Validate the configuration dictionary."""
        required_sections = ['general', 'preprocessing', 'models', 'hyperparameter_search']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")
    
    def preprocess_data(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        fit: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Preprocess the input data.
        
        Args:
            X: Feature matrix
            y: Target vector (optional)
            fit: Whether to fit the preprocessor (True for training, False for inference)
            
        Returns:
            Tuple of (X_processed, y_processed)
        """
        if fit:
            X_processed, _ = self.preprocessor.fit_transform(X, y)
            self.feature_names_ = X_processed.columns.tolist()
            
            if y is not None:
                if hasattr(y, 'name'):
                    self.target_name_ = y.name
                
                if self.task == 'classification':
                    self.classes_ = np.unique(y)
                
                return X_processed, y
            return X_processed, None
        else:
            return self.preprocessor.transform(X), y
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        **kwargs
    ) -> 'AutoMLPipeline':
        """
        Fit the AutoML pipeline on the training data.
        
        Args:
            X: Training feature matrix
            y: Training target vector
            X_val: Validation feature matrix (optional)
            y_val: Validation target vector (optional)
            **kwargs: Additional arguments to pass to the AutoML fit method
            
        Returns:
            self: Fitted pipeline
        """
        # Preprocess the training data
        X_processed, y_processed = self.preprocess_data(X, y, fit=True)
        
        # Preprocess validation data if provided
        if X_val is not None and y_val is not None:
            X_val_processed, y_val_processed = self.preprocess_data(X_val, y_val, fit=False)
            kwargs['X_val'] = X_val_processed
            kwargs['y_val'] = y_val_processed
        
        # Train the model
        self.history_ = self.automl.fit(
            X_processed, 
            y_processed,
            **kwargs
        )
        
        # Store the best model and metrics
        self.model_ = self.automl.best_model
        self.metrics_ = self.history_.get('metrics', {})
        
        return self
    
    def predict(
        self, 
        X: Union[pd.DataFrame, np.ndarray],
        return_proba: bool = False
    ) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Feature matrix
            return_proba: Whether to return class probabilities (for classification)
            
        Returns:
            Predicted values or probabilities
        """
        if self.model_ is None:
            raise RuntimeError("The model has not been trained yet. Call 'fit' first.")
        
        X_processed = self.preprocess_data(X, fit=False)[0]
        
        if return_proba and hasattr(self.model_, 'predict_proba'):
            return self.model_.predict_proba(X_processed)
        return self.model_.predict(X_processed)
    
    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate the model on the given data.
        
        Args:
            X: Feature matrix
            y: True target values
            metrics: List of metric names to compute (default: task-specific metrics)
            
        Returns:
            Dictionary of metric scores
        """
        if self.model_ is None:
            raise RuntimeError("The model has not been trained yet. Call 'fit' first.")
        
        X_processed, y_processed = self.preprocess_data(X, y, fit=False)
        y_pred = self.predict(X_processed)
        
        if self.task == 'classification':
            y_score = self.predict(X_processed, return_proba=True) if hasattr(self.model_, 'predict_proba') else None
            return get_classification_metrics(y_processed, y_pred, y_score)
        else:
            return get_regression_metrics(y_processed, y_pred)
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save the pipeline to disk.
        
        Args:
            path: Directory path to save the pipeline
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save the model
        joblib.dump(self.model_, path / 'model.joblib')
        
        # Save the preprocessor
        joblib.dump(self.preprocessor, path / 'preprocessor.joblib')
        
        # Save metadata
        metadata = {
            'task': self.task,
            'feature_names': self.feature_names_,
            'target_name': self.target_name_,
            'classes': self.classes_.tolist() if self.classes_ is not None else None,
            'metrics': self.metrics_,
            'config': self.config
        }
        
        with open(path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        if self.verbose > 0:
            print(f"Pipeline saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'AutoMLPipeline':
        """
        Load a saved pipeline from disk.
        
        Args:
            path: Directory path containing the saved pipeline
            
        Returns:
            Loaded AutoMLPipeline instance
        """
        path = Path(path)
        
        # Load metadata
        with open(path / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Create a new pipeline instance
        pipeline = cls(
            task=metadata['task'],
            random_state=metadata['config'].get('random_state', 42)
        )
        
        # Load the model and preprocessor
        pipeline.model_ = joblib.load(path / 'model.joblib')
        pipeline.preprocessor = joblib.load(path / 'preprocessor.joblib')
        
        # Set attributes from metadata
        pipeline.feature_names_ = metadata['feature_names']
        pipeline.target_name_ = metadata['target_name']
        pipeline.classes_ = np.array(metadata['classes']) if metadata['classes'] else None
        pipeline.metrics_ = metadata['metrics']
        pipeline.config = metadata['config']
        
        return pipeline
    
    def get_feature_importances(self) -> pd.Series:
        """
        Get feature importances from the trained model.
        
        Returns:
            Series with feature importances
        """
        if self.model_ is None:
            raise RuntimeError("The model has not been trained yet. Call 'fit' first.")
        
        if hasattr(self.model_, 'feature_importances_'):
            importances = self.model_.feature_importances_
        elif hasattr(self.model_, 'coef_'):
            importances = np.abs(self.model_.coef_).mean(axis=0)
        else:
            raise AttributeError("Model does not have feature importances or coefficients")
        
        return pd.Series(importances, index=self.feature_names_).sort_values(ascending=False)
    
    def plot_learning_curves(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        cv: int = 5,
        train_sizes: Optional[np.ndarray] = None,
        scoring: Optional[str] = None,
        title: str = 'Learning Curves',
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot learning curves for the model.
        
        Args:
            X: Feature matrix
            y: Target vector
            cv: Number of cross-validation folds
            train_sizes: Array of training set sizes to use
            scoring: Scoring metric (default: task-specific)
            title: Plot title
            figsize: Figure size (width, height)
            save_path: Path to save the plot (if None, display the plot)
        """
        from sklearn.model_selection import learning_curve
        
        if scoring is None:
            scoring = self.config['general']['scoring'][self.task]
        
        X_processed, y_processed = self.preprocess_data(X, y, fit=False)
        
        train_sizes, train_scores, test_scores = learning_curve(
            self.model_,
            X_processed,
            y_processed,
            cv=cv,
            scoring=scoring,
            n_jobs=self.n_jobs,
            train_sizes=train_sizes,
            verbose=self.verbose
        )
        
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        
        plt.figure(figsize=figsize)
        plt.title(title)
        plt.xlabel("Training examples")
        plt.ylabel(scoring)
        
        plt.grid()
        plt.fill_between(
            train_sizes,
            train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std,
            alpha=0.1,
            color="r"
        )
        plt.fill_between(
            train_sizes,
            test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std,
            alpha=0.1,
            color="g"
        )
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
        
        plt.legend(loc="best")
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def create_pipeline(
    task: str = 'classification',
    config_path: Optional[str] = None,
    **kwargs
) -> AutoMLPipeline:
    """
    Create a new AutoML pipeline.
    
    Args:
        task: Type of task - 'classification' or 'regression'
        config_path: Path to a YAML configuration file
        **kwargs: Additional arguments to pass to AutoMLPipeline
        
    Returns:
        Configured AutoMLPipeline instance
    """
    return AutoMLPipeline(task=task, config_path=config_path, **kwargs)
