"""
Automated Machine Learning Pipeline
----------------------------------
This module provides automated model selection, training, and evaluation.
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Any, Optional
from pathlib import Path

# Model imports
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
    RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
)
from sklearn.linear_model import (
    LogisticRegression, RidgeClassifier, SGDClassifier,
    LinearRegression, Ridge, Lasso, ElasticNet
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

# Model selection and evaluation
from sklearn.model_selection import (
    cross_val_score, GridSearchCV, RandomizedSearchCV,
    train_test_split, StratifiedKFold, KFold
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, mean_absolute_error, r2_score,
    make_scorer, confusion_matrix, classification_report
)

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)

# Type aliases
ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series]
ModelType = Union[
    RandomForestClassifier, GradientBoostingClassifier, LogisticRegression,
    SVC, XGBClassifier, LGBMClassifier, CatBoostClassifier, Sequential
]

class AutoML:
    """Automated Machine Learning pipeline for model selection and training."""
    
    def __init__(
        self,
        task: str = 'auto',
        scoring: str = None,
        n_jobs: int = -1,
        random_state: int = 42,
        verbose: int = 1
    ):
        """
        Initialize the AutoML pipeline.
        
        Args:
            task: Type of task - 'classification', 'regression', or 'auto' to infer from data
            scoring: Scoring metric to optimize
            n_jobs: Number of jobs to run in parallel
            random_state: Random seed for reproducibility
            verbose: Verbosity level
        """
        self.task = task
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.models = {}
        self.best_model = None
        self.best_score = -np.inf
        self.best_model_name = None
        self.feature_importances_ = None
        self.scoring_history_ = {}
        
        # Set random seeds for reproducibility
        np.random.seed(random_state)
        tf.random.set_seed(random_state)

    def _infer_task(self, y: pd.Series) -> str:
        """Infers the task type from the target variable."""
        # Heuristic for task detection
        if pd.api.types.is_float_dtype(y.dtype):
            # Float target is almost always regression
            return 'regression'

        # Check for object or categorical types
        if y.dtype == 'object' or pd.api.types.is_categorical_dtype(y.dtype):
            return 'classification'

        # Check for integer types
        if pd.api.types.is_integer_dtype(y.dtype):
            # Low cardinality integer is classification
            unique_values = y.nunique()
            if unique_values / len(y) < 0.05 or unique_values <= 20:
                return 'classification'
            else:
                return 'regression'
        
        # Default fallback
        return 'regression'

    def _validate_input(self, X: ArrayLike, y: ArrayLike):
        """Performs validation checks on the input data X and y."""

        # Check for empty inputs
        if X is None or y is None:
            raise ValueError("Input data X and y cannot be None.")

        # Convert to pandas for robust checks
        if not isinstance(X, pd.DataFrame):
            X_df = pd.DataFrame(X)
        else:
            X_df = X

        if not isinstance(y, pd.Series):
            y_s = pd.Series(y)
        else:
            y_s = y

        if X_df.empty or y_s.empty:
            raise ValueError("Input data X and y cannot be empty.")

        # Check for consistent lengths
        if len(X_df) != len(y_s):
            raise ValueError(f"Inconsistent number of samples. X has {len(X_df)} and y has {len(y_s)} samples.")

        # Check X dimensions
        if len(X_df.shape) != 2:
            raise ValueError(f"Input X must be 2-dimensional. Got shape: {X_df.shape}")

        # Check for NaNs in y
        if y_s.isnull().any():
            raise ValueError("Target y contains NaN values. Please handle them before training.")

        # Warn about NaNs in X
        if X_df.isnull().values.any():
            if self.verbose > 0:
                print("Warning: Input X contains NaN values. These will be handled by the preprocessor.")

        # Task-specific validation
        if self.task == 'classification':
            if y_s.nunique() < 2:
                raise ValueError(f"Classification task requires at least 2 classes, but found only {y_s.nunique()} in the target variable y.")
            if y_s.nunique() > 50 and self.verbose > 0: # Heuristic threshold
                 print(f"Warning: The target variable has a high number of unique values ({y_s.nunique()}). "
                       "This might be a regression problem misidentified as classification.")
    
    def _init_models(self) -> None:
        """Initialize the models to evaluate."""
        if self.task == 'classification':
            self.models = {
                'random_forest': {
                    'model': RandomForestClassifier(random_state=self.random_state),
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [None, 10, 20, 30],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    }
                },
                'xgboost': {
                    'model': XGBClassifier(random_state=self.random_state, n_jobs=self.n_jobs),
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [3, 6, 9],
                        'learning_rate': [0.01, 0.1, 0.3],
                        'subsample': [0.8, 1.0],
                        'colsample_bytree': [0.8, 1.0]
                    }
                },
                'logistic_regression': {
                    'model': LogisticRegression(random_state=self.random_state, max_iter=1000, n_jobs=self.n_jobs),
                    'params': {
                        'C': [0.001, 0.01, 0.1, 1, 10, 100],
                        'penalty': ['l2'],
                        'solver': ['lbfgs', 'saga']
                    }
                },
                'gradient_boosting': {
                    'model': GradientBoostingClassifier(random_state=self.random_state),
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.1, 0.3],
                        'max_depth': [3, 5, 7],
                        'subsample': [0.8, 1.0]
                    }
                },
                'svm': {
                    'model': SVC(random_state=self.random_state, probability=True),
                    'params': {
                        'C': [0.1, 1, 10],
                        'kernel': ['linear', 'rbf', 'poly'],
                        'gamma': ['scale', 'auto']
                    }
                },
                'dnn': {
                    'model': 'neural_network',
                    'params': {
                        'layers': [
                            {'units': 64, 'activation': 'relu', 'dropout': 0.2},
                            {'units': 32, 'activation': 'relu', 'dropout': 0.2},
                            {'units': 16, 'activation': 'relu', 'dropout': 0.1}
                        ],
                        'learning_rate': [0.001, 0.0001],
                        'batch_size': [32, 64],
                        'epochs': [50, 100],
                        'optimizer': ['adam', 'rmsprop']
                    }
                }
            }
        else:  # regression
            self.models = {
                'random_forest': {
                    'model': RandomForestRegressor(random_state=self.random_state, n_jobs=self.n_jobs),
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [None, 10, 20, 30],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    }
                },
                'xgboost': {
                    'model': XGBRegressor(random_state=self.random_state, n_jobs=self.n_jobs),
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [3, 6, 9],
                        'learning_rate': [0.01, 0.1, 0.3],
                        'subsample': [0.8, 1.0],
                        'colsample_bytree': [0.8, 1.0]
                    }
                },
                'linear_regression': {
                    'model': LinearRegression(n_jobs=self.n_jobs),
                    'params': {}
                },
                'ridge': {
                    'model': Ridge(random_state=self.random_state),
                    'params': {
                        'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
                        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
                    }
                },
                'lasso': {
                    'model': Lasso(random_state=self.random_state),
                    'params': {
                        'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
                        'selection': ['cyclic', 'random']
                    }
                },
                'gradient_boosting': {
                    'model': GradientBoostingRegressor(random_state=self.random_state),
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.1, 0.3],
                        'max_depth': [3, 5, 7],
                        'subsample': [0.8, 1.0]
                    }
                },
                'dnn': {
                    'model': 'neural_network',
                    'params': {
                        'layers': [
                            {'units': 64, 'activation': 'relu', 'dropout': 0.2},
                            {'units': 32, 'activation': 'relu', 'dropout': 0.2},
                            {'units': 16, 'activation': 'relu', 'dropout': 0.1}
                        ],
                        'learning_rate': [0.001, 0.0001],
                        'batch_size': [32, 64],
                        'epochs': [50, 100],
                        'optimizer': ['adam', 'rmsprop']
                    }
                }
            }
    
    def _build_neural_network(
        self,
        input_shape: Tuple[int],
        output_units: int,
        params: Dict[str, Any]
    ) -> tf.keras.Model:
        """Build a neural network model."""
        model = Sequential()
        model.add(Input(shape=input_shape))
        
        # Add hidden layers
        for layer_config in params.get('layers', []):
            model.add(Dense(
                units=layer_config['units'],
                activation=layer_config['activation']
            ))
            if 'dropout' in layer_config and layer_config['dropout'] > 0:
                model.add(Dropout(layer_config['dropout']))
            model.add(BatchNormalization())
        
        # Add output layer
        output_activation = 'softmax' if self.task == 'classification' and output_units > 1 else 'sigmoid' if self.task == 'classification' else 'linear'
        model.add(Dense(output_units, activation=output_activation))
        
        # Compile model
        optimizer_name = params.get('optimizer', 'adam').lower()
        if optimizer_name == 'adam':
            optimizer = Adam(learning_rate=params.get('learning_rate', 0.001))
        elif optimizer_name == 'rmsprop':
            optimizer = RMSprop(learning_rate=params.get('learning_rate', 0.001))
        else:
            optimizer = SGD(learning_rate=params.get('learning_rate', 0.01))
        
        loss = 'sparse_categorical_crossentropy' if self.task == 'classification' and output_units > 1 else \
               'binary_crossentropy' if self.task == 'classification' else 'mse'
        
        metrics = ['accuracy'] if self.task == 'classification' else ['mae', 'mse']
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        return model
    
    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        test_size: float = 0.2,
        cv: int = 5,
        n_iter: int = 10,
        search_method: str = 'random',
        early_stopping_rounds: int = 10,
        tensorboard_logdir: str = None
    ) -> Dict[str, Any]:
        """
        Find and train the best model for the given data.
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Fraction of data to use for testing
            cv: Number of cross-validation folds
            n_iter: Number of parameter settings sampled (for random search)
            search_method: 'grid' for GridSearchCV, 'random' for RandomizedSearchCV
            early_stopping_rounds: Number of rounds for early stopping (for neural networks)
            tensorboard_logdir: Directory to save TensorBoard logs
            
        Returns:
            Dictionary containing the best model and evaluation metrics
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        # Infer task if not specified
        if self.task == 'auto':
            self.task = self._infer_task(y)
            if self.verbose > 0:
                print(f"Automatically detected task: {self.task}")

        # Validate input data
        self._validate_input(X, y)

        # Set default scoring metric if not provided
        if self.scoring is None:
            self.scoring = 'accuracy' if self.task == 'classification' else 'neg_mean_squared_error'

        # Initialize models now that task is known
        self._init_models()
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state,
            stratify=y if self.task == 'classification' and y.nunique() > 1 else None
        )
        
        # Store test data for final evaluation
        self.X_test = X_test
        self.y_test = y_test
        
        # Dictionary to store results
        results = {}
        
        # Iterate through each model and find the best one
        for model_name, model_info in self.models.items():
            if self.verbose > 0:
                print(f"\n{'='*50}")
                print(f"Training {model_name}...")
                print(f"{'='*50}")
            
            start_time = time.time()
            
            try:
                if model_name == 'dnn':
                    # Special handling for neural networks
                    model, score = self._train_neural_network(
                        X_train, y_train, X_test, y_test,
                        model_info['params'],
                        early_stopping_rounds,
                        tensorboard_logdir
                    )
                else:
                    # Standard scikit-learn models
                    model, score = self._train_sklearn_model(
                        model_name, model_info, X_train, y_train, cv, n_iter, search_method
                    )
                
                # Store results
                results[model_name] = {
                    'model': model,
                    'score': score,
                    'training_time': time.time() - start_time
                }
                
                # Update best model if current model is better
                if score > self.best_score:
                    self.best_score = score
                    self.best_model = model
                    self.best_model_name = model_name
                
                if self.verbose > 0:
                    print(f"{model_name} - Score: {score:.4f}, Time: {time.time() - start_time:.2f}s")
            
            except Exception as e:
                print(f"Error training {model_name}: {str(e)}")
                continue
        
        # Store all results
        self.results_ = results
        
        # Get feature importances if available
        self._get_feature_importances()
        
        return self._get_summary()
    
    def _train_sklearn_model(
        self,
        model_name: str,
        model_info: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        cv: int = 5,
        n_iter: int = 10,
        search_method: str = 'random'
    ) -> Tuple[Any, float]:
        """Train a scikit-learn model with hyperparameter tuning."""
        model = model_info['model']
        params = model_info.get('params', {})
        
        # Skip if no parameters to tune
        if not params:
            model.fit(X_train, y_train)
            score = cross_val_score(
                model, X_train, y_train, cv=cv, scoring=self.scoring, n_jobs=self.n_jobs
            ).mean()
            return model, score
        
        # Choose search method
        if search_method == 'grid':
            search = GridSearchCV(
                model, params, cv=cv, scoring=self.scoring, n_jobs=self.n_jobs, verbose=self.verbose
            )
        else:  # random search
            search = RandomizedSearchCV(
                model, params, n_iter=n_iter, cv=cv, scoring=self.scoring,
                n_jobs=self.n_jobs, random_state=self.random_state, verbose=self.verbose
            )
        
        # Perform the search
        search.fit(X_train, y_train)
        
        return search.best_estimator_, search.best_score_
    
    def _train_neural_network(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        params: Dict[str, Any],
        early_stopping_rounds: int = 10,
        tensorboard_logdir: str = None
    ) -> Tuple[tf.keras.Model, float]:
        """Train a neural network model with hyperparameter tuning."""
        # Set up callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_rounds,
                restore_best_weights=True,
                verbose=self.verbose
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=early_stopping_rounds // 2,
                min_lr=1e-6,
                verbose=self.verbose
            )
        ]
        
        if tensorboard_logdir:
            log_dir = os.path.join(tensorboard_logdir, f"dnn_{int(time.time())}")
            os.makedirs(log_dir, exist_ok=True)
            callbacks.append(TensorBoard(log_dir=log_dir))
        
        # Get output units
        output_units = len(np.unique(y_train)) if self.task == 'classification' else 1
        
        # Build and compile the model
        model = self._build_neural_network(
            input_shape=(X_train.shape[1],),
            output_units=output_units,
            params=params
        )
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=params.get('epochs', 100),
            batch_size=params.get('batch_size', 32),
            callbacks=callbacks,
            verbose=self.verbose
        )
        
        # Get the best validation score
        val_metric = 'val_accuracy' if self.task == 'classification' else 'val_loss'
        best_score = max(history.history[val_metric]) if self.task == 'classification' else -min(history.history[val_metric])
        
        return model, best_score
    
    def _get_feature_importances(self) -> None:
        """Extract feature importances from the best model if available."""
        if self.best_model is None:
            return
        
        # For tree-based models
        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importances_ = self.best_model.feature_importances_
        # For linear models
        elif hasattr(self.best_model, 'coef_'):
            self.feature_importances_ = np.abs(self.best_model.coef_.flatten())
        # For neural networks (simplified)
        elif hasattr(self.best_model, 'layers'):
            # This is a very simplified approach
            weights = self.best_model.layers[0].get_weights()[0]
            self.feature_importances_ = np.mean(np.abs(weights), axis=1)
    
    def predict(self, X: ArrayLike) -> np.ndarray:
        """Make predictions using the best model."""
        if self.best_model is None:
            raise ValueError("No model has been trained yet. Call fit() first.")
        
        if hasattr(X, 'values'):
            X = X.values
        
        # For scikit-learn models
        if hasattr(self.best_model, 'predict'):
            return self.best_model.predict(X)
        # For Keras models
        elif hasattr(self.best_model, 'predict_classes'):
            return self.best_model.predict_classes(X)
        else:
            predictions = self.best_model.predict(X)
            if self.task == 'classification' and len(predictions.shape) > 1 and predictions.shape[1] > 1:
                return np.argmax(predictions, axis=1)
            return predictions
    
    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Predict class probabilities (for classification)."""
        if self.task != 'classification':
            raise ValueError("Probability predictions are only available for classification tasks.")
        
        if hasattr(X, 'values'):
            X = X.values
        
        # For scikit-learn models
        if hasattr(self.best_model, 'predict_proba'):
            return self.best_model.predict_proba(X)
        # For Keras models
        elif hasattr(self.best_model, 'predict'):
            return self.best_model.predict(X)
        else:
            raise ValueError("Probability predictions not available for this model.")
    
    def evaluate(self, X: ArrayLike = None, y: ArrayLike = None) -> Dict[str, float]:
        """Evaluate the best model on the test set."""
        if X is None or y is None:
            if not hasattr(self, 'X_test') or not hasattr(self, 'y_test'):
                raise ValueError("No test data provided. Please provide X and y or call fit() with test data.")
            X = self.X_test
            y = self.y_test
        
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        
        y_pred = self.predict(X)
        
        metrics = {}
        
        if self.task == 'classification':
            metrics.update({
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y, y_pred, average='weighted', zero_division=0)
            })
            
            # For binary classification, add ROC AUC if possible
            if len(np.unique(y)) == 2 and hasattr(self.best_model, 'predict_proba'):
                y_proba = self.predict_proba(X)[:, 1]
                metrics['roc_auc'] = roc_auc_score(y, y_proba)
            
            # For multi-class, try to calculate ROC AUC if possible
            elif len(np.unique(y)) > 2 and hasattr(self.best_model, 'predict_proba'):
                try:
                    y_proba = self.predict_proba(X)
                    metrics['roc_auc_ovr'] = roc_auc_score(y, y_proba, multi_class='ovr')
                    metrics['roc_auc_ovo'] = roc_auc_score(y, y_proba, multi_class='ovo')
                except Exception as e:
                    if self.verbose > 0:
                        print(f"Could not calculate ROC AUC: {str(e)}")
        
        else:  # regression
            metrics.update({
                'mse': mean_squared_error(y, y_pred),
                'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                'mae': mean_absolute_error(y, y_pred),
                'r2': r2_score(y, y_pred)
            })
        
        return metrics
    
    def _get_summary(self) -> Dict[str, Any]:
        """Get a summary of the model training results."""
        if not hasattr(self, 'results_'):
            raise ValueError("No models have been trained yet. Call fit() first.")
        
        summary = {
            'best_model': self.best_model_name,
            'best_score': self.best_score,
            'models_trained': len(self.results_),
            'model_scores': {name: result['score'] for name, result in self.results_.items()},
            'training_times': {name: result['training_time'] for name, result in self.results_.items()}
        }
        
        # Add evaluation metrics if test data is available
        if hasattr(self, 'X_test') and hasattr(self, 'y_test'):
            test_metrics = self.evaluate()
            summary['test_metrics'] = test_metrics
        
        return summary
    
    def save_model(self, filepath: str) -> None:
        """Save the best model to a file."""
        if self.best_model is None:
            raise ValueError("No model has been trained yet. Call fit() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # For scikit-learn models
        if hasattr(self.best_model, 'predict'):
            joblib.dump(self.best_model, filepath)
        # For Keras models
        elif hasattr(self.best_model, 'save'):
            self.best_model.save(filepath)
        else:
            raise ValueError("Model type not supported for saving.")
    
    @classmethod
    def load_model(cls, filepath: str) -> Any:
        """Load a saved model from a file."""
        # Try to load as a scikit-learn model first
        try:
            return joblib.load(filepath)
        except:
            pass
        
        # Try to load as a Keras model
        try:
            return tf.keras.models.load_model(filepath)
        except:
            pass
        
        raise ValueError("Could not load the model. Unsupported file format or corrupted file.")


def auto_train(
    X: ArrayLike,
    y: ArrayLike,
    task: str = 'classification',
    test_size: float = 0.2,
    cv: int = 5,
    n_iter: int = 10,
    search_method: str = 'random',
    early_stopping_rounds: int = 10,
    tensorboard_logdir: str = None,
    verbose: int = 1
) -> Tuple[Any, Dict[str, Any]]:
    """
    Automatically train and evaluate multiple models on the given data.
    
    This is a convenience function that creates an AutoML instance, fits it to the data,
    and returns the best model and evaluation results.
    
    Args:
        X: Feature matrix
        y: Target vector
        task: Type of task - 'classification' or 'regression'
        test_size: Fraction of data to use for testing
        cv: Number of cross-validation folds
        n_iter: Number of parameter settings sampled (for random search)
        search_method: 'grid' for GridSearchCV, 'random' for RandomizedSearchCV
        early_stopping_rounds: Number of rounds for early stopping (for neural networks)
        tensorboard_logdir: Directory to save TensorBoard logs
        verbose: Verbosity level
        
    Returns:
        A tuple containing the best model and a dictionary of results
    """
    automl = AutoML(task=task, verbose=verbose)
    results = automl.fit(
        X, y,
        test_size=test_size,
        cv=cv,
        n_iter=n_iter,
        search_method=search_method,
        early_stopping_rounds=early_stopping_rounds,
        tensorboard_logdir=tensorboard_logdir
    )
    
    return automl.best_model, results
