"""
Enhanced Automated Machine Learning Pipeline
------------------------------------------
This module provides advanced automated model selection, training, and evaluation
with intelligent task detection, ensemble methods, and comprehensive experiment tracking.
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Tuple, Union, Any, Optional, Callable
from pathlib import Path
from collections import defaultdict
import logging

# Core ML imports
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
    RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor,
    VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor
)
from sklearn.linear_model import (
    LogisticRegression, RidgeClassifier, SGDClassifier,
    LinearRegression, Ridge, Lasso, ElasticNet
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# External libraries
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    import optuna
    from optuna.integration import OptunaSearchCV
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Model selection and evaluation
from sklearn.model_selection import (
    cross_val_score, GridSearchCV, RandomizedSearchCV,
    train_test_split, StratifiedKFold, KFold, cross_validate
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, mean_absolute_error, r2_score,
    make_scorer, confusion_matrix, classification_report, log_loss
)
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

# Deep Learning
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
    from tensorflow.keras.optimizers import Adam, RMSprop, SGD
    from tensorflow.keras.callbacks import (
        EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
    )
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Type aliases
ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series]
ModelType = Union[
    RandomForestClassifier, GradientBoostingClassifier, LogisticRegression,
    SVC, XGBClassifier, LGBMClassifier, CatBoostClassifier
]

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskDetector:
    """Enhanced task detection with sophisticated heuristics."""
    
    @staticmethod
    def detect_task(y: ArrayLike, threshold_ratio: float = 0.05) -> str:
        """
        Detect whether the task is classification or regression using multiple heuristics.
        
        Args:
            y: Target variable
            threshold_ratio: Ratio threshold for classification detection
            
        Returns:
            str: 'classification' or 'regression'
        """
        if hasattr(y, 'values'):
            y = y.values
        
        y = np.asarray(y)
        
        # Handle string/object types - always classification
        if y.dtype == 'object' or np.issubdtype(y.dtype, np.str_):
            return 'classification'
        
        # Handle boolean types - always classification
        if np.issubdtype(y.dtype, np.bool_):
            return 'classification'
        
        # For numeric types, use multiple heuristics
        unique_values = np.unique(y[~np.isnan(y)])
        n_unique = len(unique_values)
        n_samples = len(y)
        
        # Heuristic 1: Very few unique values relative to sample size
        if n_unique / n_samples < threshold_ratio:
            return 'classification'
        
        # Heuristic 2: All values are integers and reasonable number of classes
        if np.all(unique_values == unique_values.astype(int)) and n_unique <= 50:
            return 'classification'
        
        # Heuristic 3: Check if values look like class labels (0, 1, 2, ...)
        if (n_unique <= 20 and 
            np.all(unique_values >= 0) and 
            np.all(unique_values == np.arange(n_unique))):
            return 'classification'
        
        # Heuristic 4: Binary classification check
        if n_unique == 2:
            return 'classification'
        
        # Default to regression for continuous numeric data
        return 'regression'


class DatasetAnalyzer:
    """Analyze dataset characteristics for intelligent model selection."""
    
    @staticmethod
    def analyze_dataset(X: ArrayLike, y: ArrayLike) -> Dict[str, Any]:
        """
        Analyze dataset characteristics.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Dict containing dataset characteristics
        """
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
            
        n_samples, n_features = X.shape
        
        # Basic statistics
        characteristics = {
            'n_samples': n_samples,
            'n_features': n_features,
            'feature_to_sample_ratio': n_features / n_samples,
            'is_high_dimensional': n_features > n_samples,
            'is_small_dataset': n_samples < 1000,
            'is_large_dataset': n_samples > 100000,
        }
        
        # Feature analysis
        if isinstance(X, np.ndarray):
            # Numeric features analysis
            characteristics.update({
                'has_missing_values': np.isnan(X).any(),
                'missing_ratio': np.isnan(X).sum() / (n_samples * n_features),
                'feature_variance': np.var(X, axis=0).mean() if not np.isnan(X).all() else 0,
                'feature_skewness': np.mean([abs(np.mean((col - np.mean(col))**3) / np.std(col)**3) 
                                           for col in X.T if not np.isnan(col).all()]) if X.size > 0 else 0,
            })
        
        # Target analysis
        task = TaskDetector.detect_task(y)
        characteristics['task'] = task
        
        if task == 'classification':
            unique_classes = np.unique(y[~np.isnan(y)])
            characteristics.update({
                'n_classes': len(unique_classes),
                'is_binary': len(unique_classes) == 2,
                'is_multiclass': len(unique_classes) > 2,
                'class_imbalance_ratio': np.max(np.bincount(y.astype(int))) / np.min(np.bincount(y.astype(int)))
                                       if len(unique_classes) > 1 else 1.0
            })
        else:
            characteristics.update({
                'target_range': np.max(y) - np.min(y),
                'target_std': np.std(y),
                'target_skewness': abs(np.mean((y - np.mean(y))**3) / np.std(y)**3) if np.std(y) > 0 else 0
            })
        
        return characteristics


class IntelligentModelSelector:
    """Select models based on dataset characteristics."""
    
    @staticmethod
    def select_models(characteristics: Dict[str, Any]) -> List[str]:
        """
        Select appropriate models based on dataset characteristics.
        
        Args:
            characteristics: Dataset characteristics from DatasetAnalyzer
            
        Returns:
            List of recommended model names
        """
        task = characteristics['task']
        n_samples = characteristics['n_samples']
        n_features = characteristics['n_features']
        is_high_dim = characteristics['is_high_dimensional']
        is_small = characteristics['is_small_dataset']
        is_large = characteristics['is_large_dataset']
        
        models = []
        
        if task == 'classification':
            # Always include these robust models
            models.extend(['random_forest', 'logistic_regression'])
            
            # For small datasets
            if is_small:
                models.extend(['naive_bayes', 'knn'])
            
            # For high-dimensional data
            if is_high_dim:
                models.extend(['ridge_classifier', 'lda'])
            
            # For large datasets
            if not is_small:
                if XGBOOST_AVAILABLE:
                    models.append('xgboost')
                if LIGHTGBM_AVAILABLE:
                    models.append('lightgbm')
            
            # For binary classification
            if characteristics.get('is_binary', False):
                models.append('svm')
            
            # For imbalanced datasets
            if characteristics.get('class_imbalance_ratio', 1) > 3:
                models.append('gradient_boosting')
                
        else:  # regression
            # Always include these robust models
            models.extend(['random_forest', 'linear_regression'])
            
            # For small datasets
            if is_small:
                models.extend(['knn', 'ridge'])
            
            # For high-dimensional data
            if is_high_dim:
                models.extend(['lasso', 'elastic_net'])
            
            # For large datasets
            if not is_small:
                if XGBOOST_AVAILABLE:
                    models.append('xgboost')
                if LIGHTGBM_AVAILABLE:
                    models.append('lightgbm')
            
            # For non-linear relationships
            if n_features < 50:  # Avoid SVM for high-dimensional data
                models.append('svr')
        
        return list(set(models))  # Remove duplicates


class EnsembleBuilder:
    """Build ensemble models using stacking and voting."""
    
    @staticmethod
    def create_voting_ensemble(models: Dict[str, Any], task: str) -> Any:
        """Create a voting ensemble."""
        if task == 'classification':
            return VotingClassifier(
                estimators=[(name, model) for name, model in models.items()],
                voting='soft'
            )
        else:
            return VotingRegressor(
                estimators=[(name, model) for name, model in models.items()]
            )
    
    @staticmethod
    def create_stacking_ensemble(models: Dict[str, Any], task: str, meta_learner: Any = None) -> Any:
        """Create a stacking ensemble."""
        if meta_learner is None:
            meta_learner = LogisticRegression() if task == 'classification' else LinearRegression()
        
        if task == 'classification':
            return StackingClassifier(
                estimators=[(name, model) for name, model in models.items()],
                final_estimator=meta_learner,
                cv=5
            )
        else:
            return StackingRegressor(
                estimators=[(name, model) for name, model in models.items()],
                final_estimator=meta_learner,
                cv=5
            )


class InputValidator:
    """Comprehensive input validation."""
    
    @staticmethod
    def validate_input(X: ArrayLike, y: ArrayLike = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Validate and convert input data.
        
        Args:
            X: Feature matrix
            y: Target variable (optional)
            
        Returns:
            Tuple of validated (X, y)
        """
        # Convert to numpy arrays
        if hasattr(X, 'values'):
            X = X.values
        if y is not None and hasattr(y, 'values'):
            y = y.values
        
        X = np.asarray(X)
        if y is not None:
            y = np.asarray(y)
        
        # Basic shape validation
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got {X.ndim} dimensions")
        
        if y is not None:
            if y.ndim not in [1, 2]:
                raise ValueError(f"y must be 1 or 2-dimensional, got {y.ndim} dimensions")
            
            if y.ndim == 2 and y.shape[1] != 1:
                raise ValueError(f"y must have 1 column if 2-dimensional, got {y.shape[1]} columns")
            
            if y.ndim == 2:
                y = y.ravel()
            
            if len(X) != len(y):
                raise ValueError(f"X and y must have same number of samples, got {len(X)} and {len(y)}")
        
        # Check for empty data
        if X.size == 0:
            raise ValueError("X cannot be empty")
        
        if y is not None and y.size == 0:
            raise ValueError("y cannot be empty")
        
        # Check for infinite values
        if np.isinf(X).any():
            logger.warning("X contains infinite values, consider preprocessing")
        
        if y is not None and np.isinf(y).any():
            logger.warning("y contains infinite values, consider preprocessing")
        
        return X, y


class AutoML:
    """Enhanced Automated Machine Learning pipeline."""
    
    def __init__(
        self,
        task: str = 'auto',
        scoring: str = None,
        n_jobs: int = -1,
        random_state: int = 42,
        verbose: int = 1,
        use_ensembles: bool = True,
        use_optuna: bool = True,
        optuna_trials: int = 100,
        time_budget: Optional[int] = None,
        memory_limit: Optional[str] = None
    ):
        """
        Initialize the enhanced AutoML pipeline.
        
        Args:
            task: Type of task - 'classification', 'regression', or 'auto'
            scoring: Scoring metric to optimize
            n_jobs: Number of jobs to run in parallel
            random_state: Random seed for reproducibility
            verbose: Verbosity level
            use_ensembles: Whether to create ensemble models
            use_optuna: Whether to use Optuna for hyperparameter optimization
            optuna_trials: Number of Optuna trials
            time_budget: Time budget in seconds (optional)
            memory_limit: Memory limit (e.g., '4GB', optional)
        """
        self.task = task
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.use_ensembles = use_ensembles
        self.use_optuna = use_optuna and OPTUNA_AVAILABLE
        self.optuna_trials = optuna_trials
        self.time_budget = time_budget
        self.memory_limit = memory_limit
        
        # Initialize attributes
        self.models = {}
        self.best_model = None
        self.best_score = -np.inf
        self.best_model_name = None
        self.feature_importances_ = None
        self.scoring_history_ = {}
        self.dataset_characteristics_ = {}
        self.ensemble_models_ = {}
        self.shap_explainer_ = None
        self.shap_values_ = None
        
        # Set random seeds for reproducibility
        np.random.seed(random_state)
        if TENSORFLOW_AVAILABLE:
            tf.random.set_seed(random_state)
        
        # Initialize MLflow if available
        self.mlflow_run = None
        if MLFLOW_AVAILABLE:
            self._setup_mlflow()
    
    def _setup_mlflow(self):
        """Setup MLflow experiment tracking."""
        try:
            experiment_name = f"automl_experiment_{int(time.time())}"
            mlflow.set_experiment(experiment_name)
            self.mlflow_run = mlflow.start_run()
            
            # Log initial parameters
            mlflow.log_params({
                'task': self.task,
                'use_ensembles': self.use_ensembles,
                'use_optuna': self.use_optuna,
                'optuna_trials': self.optuna_trials,
                'random_state': self.random_state
            })
        except Exception as e:
            logger.warning(f"Failed to setup MLflow: {e}")
            self.mlflow_run = None
    
    def _init_models(self, task: str, characteristics: Dict[str, Any]) -> None:
        """Initialize models based on task and dataset characteristics."""
        # Get recommended models
        recommended_models = IntelligentModelSelector.select_models(characteristics)
        
        if task == 'classification':
            model_configs = {
                'random_forest': {
                    'model': RandomForestClassifier(random_state=self.random_state, n_jobs=self.n_jobs),
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [None, 10, 20, 30],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4],
                        'max_features': ['sqrt', 'log2', None]
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
                        'kernel': ['linear', 'rbf'],
                        'gamma': ['scale', 'auto']
                    }
                },
                'naive_bayes': {
                    'model': GaussianNB(),
                    'params': {
                        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
                    }
                },
                'knn': {
                    'model': KNeighborsClassifier(n_jobs=self.n_jobs),
                    'params': {
                        'n_neighbors': [3, 5, 7, 9],
                        'weights': ['uniform', 'distance'],
                        'metric': ['euclidean', 'manhattan']
                    }
                },
                'ridge_classifier': {
                    'model': RidgeClassifier(random_state=self.random_state),
                    'params': {
                        'alpha': [0.1, 1.0, 10.0, 100.0]
                    }
                },
                'lda': {
                    'model': LinearDiscriminantAnalysis(),
                    'params': {
                        'solver': ['svd', 'lsqr', 'eigen']
                    }
                }
            }
            
            # Add external library models if available
            if XGBOOST_AVAILABLE and 'xgboost' in recommended_models:
                model_configs['xgboost'] = {
                    'model': XGBClassifier(random_state=self.random_state, n_jobs=self.n_jobs),
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [3, 6, 9],
                        'learning_rate': [0.01, 0.1, 0.3],
                        'subsample': [0.8, 1.0],
                        'colsample_bytree': [0.8, 1.0]
                    }
                }
            
            if LIGHTGBM_AVAILABLE and 'lightgbm' in recommended_models:
                model_configs['lightgbm'] = {
                    'model': LGBMClassifier(random_state=self.random_state, n_jobs=self.n_jobs, verbose=-1),
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [3, 6, 9],
                        'learning_rate': [0.01, 0.1, 0.3],
                        'subsample': [0.8, 1.0],
                        'colsample_bytree': [0.8, 1.0]
                    }
                }
        
        else:  # regression
            model_configs = {
                'random_forest': {
                    'model': RandomForestRegressor(random_state=self.random_state, n_jobs=self.n_jobs),
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [None, 10, 20, 30],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4],
                        'max_features': ['sqrt', 'log2', None]
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
                'elastic_net': {
                    'model': ElasticNet(random_state=self.random_state),
                    'params': {
                        'alpha': [0.001, 0.01, 0.1, 1, 10],
                        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
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
                'svr': {
                    'model': SVR(),
                    'params': {
                        'C': [0.1, 1, 10],
                        'kernel': ['linear', 'rbf'],
                        'gamma': ['scale', 'auto']
                    }
                },
                'knn': {
                    'model': KNeighborsRegressor(n_jobs=self.n_jobs),
                    'params': {
                        'n_neighbors': [3, 5, 7, 9],
                        'weights': ['uniform', 'distance'],
                        'metric': ['euclidean', 'manhattan']
                    }
                }
            }
            
            # Add external library models if available
            if XGBOOST_AVAILABLE and 'xgboost' in recommended_models:
                model_configs['xgboost'] = {
                    'model': XGBRegressor(random_state=self.random_state, n_jobs=self.n_jobs),
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [3, 6, 9],
                        'learning_rate': [0.01, 0.1, 0.3],
                        'subsample': [0.8, 1.0],
                        'colsample_bytree': [0.8, 1.0]
                    }
                }
            
            if LIGHTGBM_AVAILABLE and 'lightgbm' in recommended_models:
                model_configs['lightgbm'] = {
                    'model': LGBMRegressor(random_state=self.random_state, n_jobs=self.n_jobs, verbose=-1),
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [3, 6, 9],
                        'learning_rate': [0.01, 0.1, 0.3],
                        'subsample': [0.8, 1.0],
                        'colsample_bytree': [0.8, 1.0]
                    }
                }
        
        # Filter models based on recommendations
        self.models = {name: config for name, config in model_configs.items() 
                      if name in recommended_models}
        
        if self.verbose > 0:
            logger.info(f"Selected models based on dataset characteristics: {list(self.models.keys())}")
    
    def _optimize_hyperparameters_optuna(
        self,
        model_name: str,
        model_info: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        cv: int = 5
    ) -> Tuple[Any, float]:
        """Optimize hyperparameters using Optuna."""
        if not self.use_optuna or not OPTUNA_AVAILABLE:
            return self._train_sklearn_model(model_name, model_info, X_train, y_train, cv)
        
        def objective(trial):
            # Sample hyperparameters
            params = {}
            for param_name, param_values in model_info.get('params', {}).items():
                if isinstance(param_values, list):
                    if all(isinstance(v, (int, float)) for v in param_values):
                        if all(isinstance(v, int) for v in param_values):
                            params[param_name] = trial.suggest_int(param_name, min(param_values), max(param_values))
                        else:
                            params[param_name] = trial.suggest_float(param_name, min(param_values), max(param_values))
                    else:
                        params[param_name] = trial.suggest_categorical(param_name, param_values)
            
            # Create model with sampled parameters
            model = model_info['model'].__class__(**{**model.get_params(), **params})
            
            # Cross-validation
            scores = cross_val_score(
                model, X_train, y_train, cv=cv, scoring=self.scoring, n_jobs=1
            )
            return scores.mean()
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        # Optimize
        study.optimize(objective, n_trials=self.optuna_trials, timeout=self.time_budget)
        
        # Get best model
        best_params = study.best_params
        best_model = model_info['model'].__class__(**{**model_info['model'].get_params(), **best_params})
        best_model.fit(X_train, y_train)
        
        return best_model, study.best_value
    
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
                model, params, cv=cv, scoring=self.scoring, n_jobs=self.n_jobs, verbose=0
            )
        else:  # random search
            search = RandomizedSearchCV(
                model, params, n_iter=n_iter, cv=cv, scoring=self.scoring,
                n_jobs=self.n_jobs, random_state=self.random_state, verbose=0
            )
        
        # Perform the search
        search.fit(X_train, y_train)
        
        return search.best_estimator_, search.best_score_
    
    def _create_ensembles(self, trained_models: Dict[str, Any]) -> Dict[str, Any]:
        """Create ensemble models from trained base models."""
        if not self.use_ensembles or len(trained_models) < 2:
            return {}
        
        ensembles = {}
        
        try:
            # Voting ensemble
            voting_ensemble = EnsembleBuilder.create_voting_ensemble(trained_models, self.task)
            ensembles['voting'] = voting_ensemble
            
            # Stacking ensemble
            stacking_ensemble = EnsembleBuilder.create_stacking_ensemble(trained_models, self.task)
            ensembles['stacking'] = stacking_ensemble
            
        except Exception as e:
            logger.warning(f"Failed to create ensembles: {e}")
        
        return ensembles
    
    def _calculate_shap_values(self, model: Any, X_sample: np.ndarray) -> None:
        """Calculate SHAP values for model interpretability."""
        if not SHAP_AVAILABLE:
            return
        
        try:
            # Select appropriate explainer based on model type
            if hasattr(model, 'predict_proba'):
                # Tree-based models
                if hasattr(model, 'estimators_'):
                    self.shap_explainer_ = shap.TreeExplainer(model)
                else:
                    # Use KernelExplainer for other models
                    self.shap_explainer_ = shap.KernelExplainer(model.predict_proba, X_sample[:100])
            else:
                # Regression models
                if hasattr(model, 'estimators_'):
                    self.shap_explainer_ = shap.TreeExplainer(model)
                else:
                    self.shap_explainer_ = shap.KernelExplainer(model.predict, X_sample[:100])
            
            # Calculate SHAP values for a sample
            sample_size = min(100, len(X_sample))
            self.shap_values_ = self.shap_explainer_.shap_values(X_sample[:sample_size])
            
        except Exception as e:
            logger.warning(f"Failed to calculate SHAP values: {e}")
    
    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        test_size: float = 0.2,
        cv: int = 5,
        n_iter: int = 10,
        search_method: str = 'random'
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
            
        Returns:
            Dictionary containing the best model and evaluation metrics
        """
        start_time = time.time()
        
        # Validate input
        X, y = InputValidator.validate_input(X, y)
        
        # Detect task if auto
        if self.task == 'auto':
            self.task = TaskDetector.detect_task(y)
            if self.verbose > 0:
                logger.info(f"Auto-detected task: {self.task}")
        
        # Set default scoring if not provided
        if self.scoring is None:
            self.scoring = 'accuracy' if self.task == 'classification' else 'neg_mean_squared_error'
        
        # Analyze dataset characteristics
        self.dataset_characteristics_ = DatasetAnalyzer.analyze_dataset(X, y)
        
        if self.verbose > 0:
            logger.info(f"Dataset characteristics: {self.dataset_characteristics_}")
        
        # Initialize models based on characteristics
        self._init_models(self.task, self.dataset_characteristics_)
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state,
            stratify=y if self.task == 'classification' and len(np.unique(y)) > 1 else None
        )
        
        # Store test data for final evaluation
        self.X_test = X_test
        self.y_test = y_test
        
        # Dictionary to store results
        results = {}
        trained_models = {}
        
        # Train individual models
        for model_name, model_info in self.models.items():
            if self.verbose > 0:
                logger.info(f"Training {model_name}...")
            
            model_start_time = time.time()
            
            try:
                if self.use_optuna and OPTUNA_AVAILABLE:
                    model, score = self._optimize_hyperparameters_optuna(
                        model_name, model_info, X_train, y_train, cv
                    )
                else:
                    model, score = self._train_sklearn_model(
                        model_name, model_info, X_train, y_train, cv, n_iter, search_method
                    )
                
                # Store results
                training_time = time.time() - model_start_time
                results[model_name] = {
                    'model': model,
                    'score': score,
                    'training_time': training_time
                }
                
                trained_models[model_name] = model
                
                # Update best model if current model is better
                if score > self.best_score:
                    self.best_score = score
                    self.best_model = model
                    self.best_model_name = model_name
                
                if self.verbose > 0:
                    logger.info(f"{model_name} - Score: {score:.4f}, Time: {training_time:.2f}s")
                
                # Log to MLflow
                if self.mlflow_run:
                    mlflow.log_metrics({
                        f'{model_name}_score': score,
                        f'{model_name}_training_time': training_time
                    })
            
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        # Create ensemble models
        if self.use_ensembles:
            if self.verbose > 0:
                logger.info("Creating ensemble models...")
            
            ensemble_models = self._create_ensembles(trained_models)
            
            # Train and evaluate ensembles
            for ensemble_name, ensemble_model in ensemble_models.items():
                try:
                    ensemble_start_time = time.time()
                    ensemble_model.fit(X_train, y_train)
                    
                    # Evaluate ensemble
                    ensemble_scores = cross_val_score(
                        ensemble_model, X_train, y_train, cv=cv, scoring=self.scoring, n_jobs=self.n_jobs
                    )
                    ensemble_score = ensemble_scores.mean()
                    ensemble_training_time = time.time() - ensemble_start_time
                    
                    results[ensemble_name] = {
                        'model': ensemble_model,
                        'score': ensemble_score,
                        'training_time': ensemble_training_time
                    }
                    
                    # Update best model if ensemble is better
                    if ensemble_score > self.best_score:
                        self.best_score = ensemble_score
                        self.best_model = ensemble_model
                        self.best_model_name = ensemble_name
                    
                    if self.verbose > 0:
                        logger.info(f"{ensemble_name} - Score: {ensemble_score:.4f}, Time: {ensemble_training_time:.2f}s")
                    
                    # Log to MLflow
                    if self.mlflow_run:
                        mlflow.log_metrics({
                            f'{ensemble_name}_score': ensemble_score,
                            f'{ensemble_name}_training_time': ensemble_training_time
                        })
                
                except Exception as e:
                    logger.error(f"Error training {ensemble_name}: {str(e)}")
                    continue
        
        # Store all results
        self.results_ = results
        
        # Get feature importances if available
        self._get_feature_importances()
        
        # Calculate SHAP values for interpretability
        if SHAP_AVAILABLE and self.best_model:
            self._calculate_shap_values(self.best_model, X_train)
        
        # Log final results to MLflow
        if self.mlflow_run:
            mlflow.log_metrics({
                'best_score': self.best_score,
                'total_training_time': time.time() - start_time
            })
            mlflow.log_params({
                'best_model': self.best_model_name,
                'n_models_trained': len(results)
            })
        
        return self._get_summary()
    
    def _get_feature_importances(self) -> None:
        """Extract feature importances from the best model if available."""
        if self.best_model is None:
            return
        
        # For ensemble models, try to get feature importances from base estimators
        if hasattr(self.best_model, 'estimators_') and hasattr(self.best_model, 'feature_importances_'):
            self.feature_importances_ = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'feature_importances_'):
            self.feature_importances_ = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            self.feature_importances_ = np.abs(self.best_model.coef_.flatten())
        elif hasattr(self.best_model, 'estimators_'):
            # For voting/stacking ensembles, average feature importances
            importances = []
            for estimator in self.best_model.estimators_:
                if hasattr(estimator, 'feature_importances_'):
                    importances.append(estimator.feature_importances_)
                elif hasattr(estimator, 'coef_'):
                    importances.append(np.abs(estimator.coef_.flatten()))
            
            if importances:
                self.feature_importances_ = np.mean(importances, axis=0)
    
    def predict(self, X: ArrayLike) -> np.ndarray:
        """Make predictions using the best model."""
        if self.best_model is None:
            raise ValueError("No model has been trained yet. Call fit() first.")
        
        X, _ = InputValidator.validate_input(X)
        return self.best_model.predict(X)
    
    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Predict class probabilities (for classification)."""
        if self.task != 'classification':
            raise ValueError("Probability predictions are only available for classification tasks.")
        
        if self.best_model is None:
            raise ValueError("No model has been trained yet. Call fit() first.")
        
        X, _ = InputValidator.validate_input(X)
        
        if hasattr(self.best_model, 'predict_proba'):
            return self.best_model.predict_proba(X)
        else:
            raise ValueError("Best model does not support probability predictions.")
    
    def evaluate(self, X: ArrayLike = None, y: ArrayLike = None) -> Dict[str, float]:
        """Evaluate the best model on the test set."""
        if X is None or y is None:
            if not hasattr(self, 'X_test') or not hasattr(self, 'y_test'):
                raise ValueError("No test data provided. Please provide X and y or call fit() with test data.")
            X = self.X_test
            y = self.y_test
        
        X, y = InputValidator.validate_input(X, y)
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
                try:
                    y_proba = self.predict_proba(X)[:, 1]
                    metrics['roc_auc'] = roc_auc_score(y, y_proba)
                except Exception as e:
                    logger.warning(f"Could not calculate ROC AUC: {e}")
        
        else:  # regression
            metrics.update({
                'mse': mean_squared_error(y, y_pred),
                'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                'mae': mean_absolute_error(y, y_pred),
                'r2': r2_score(y, y_pred)
            })
        
        # Log metrics to MLflow
        if self.mlflow_run:
            mlflow.log_metrics({f'test_{k}': v for k, v in metrics.items()})
        
        return metrics
    
    def get_shap_values(self, X: ArrayLike = None) -> Optional[np.ndarray]:
        """Get SHAP values for model interpretability."""
        if not SHAP_AVAILABLE or self.shap_explainer_ is None:
            logger.warning("SHAP is not available or explainer not initialized")
            return None
        
        if X is None:
            return self.shap_values_
        
        X, _ = InputValidator.validate_input(X)
        try:
            return self.shap_explainer_.shap_values(X)
        except Exception as e:
            logger.error(f"Error calculating SHAP values: {e}")
            return None
    
    def _get_summary(self) -> Dict[str, Any]:
        """Get a summary of the model training results."""
        if not hasattr(self, 'results_'):
            raise ValueError("No models have been trained yet. Call fit() first.")
        
        summary = {
            'best_model': self.best_model_name,
            'best_score': self.best_score,
            'models_trained': len(self.results_),
            'model_scores': {name: result['score'] for name, result in self.results_.items()},
            'training_times': {name: result['training_time'] for name, result in self.results_.items()},
            'dataset_characteristics': self.dataset_characteristics_,
            'task': self.task
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
        
        # Save the model
        joblib.dump(self.best_model, filepath)
        
        # Save additional metadata
        metadata = {
            'model_name': self.best_model_name,
            'task': self.task,
            'best_score': self.best_score,
            'dataset_characteristics': self.dataset_characteristics_,
            'feature_importances': self.feature_importances_.tolist() if self.feature_importances_ is not None else None
        }
        
        metadata_path = filepath.replace('.joblib', '_metadata.json')
        with open(metadata_path, 'w') as f:
            import json
            json.dump(metadata, f, indent=2)
    
    @classmethod
    def load_model(cls, filepath: str) -> Any:
        """Load a saved model from a file."""
        return joblib.load(filepath)
    
    def __del__(self):
        """Cleanup MLflow run on deletion."""
        if self.mlflow_run and MLFLOW_AVAILABLE:
            try:
                mlflow.end_run()
            except:
                pass


def auto_train(
    X: ArrayLike,
    y: ArrayLike,
    task: str = 'auto',
    test_size: float = 0.2,
    cv: int = 5,
    n_iter: int = 10,
    search_method: str = 'random',
    use_ensembles: bool = True,
    use_optuna: bool = True,
    verbose: int = 1
) -> Tuple[Any, Dict[str, Any]]:
    """
    Automatically train and evaluate multiple models on the given data.
    
    This is a convenience function that creates an AutoML instance, fits it to the data,
    and returns the best model and evaluation results.
    
    Args:
        X: Feature matrix
        y: Target vector
        task: Type of task - 'classification', 'regression', or 'auto'
        test_size: Fraction of data to use for testing
        cv: Number of cross-validation folds
        n_iter: Number of parameter settings sampled (for random search)
        search_method: 'grid' for GridSearchCV, 'random' for RandomizedSearchCV
        use_ensembles: Whether to create ensemble models
        use_optuna: Whether to use Optuna for hyperparameter optimization
        verbose: Verbosity level
        
    Returns:
        A tuple containing the best model and a dictionary of results
    """
    automl = AutoML(
        task=task,
        use_ensembles=use_ensembles,
        use_optuna=use_optuna,
        verbose=verbose
    )
    
    results = automl.fit(
        X, y,
        test_size=test_size,
        cv=cv,
        n_iter=n_iter,
        search_method=search_method
    )
    
    return automl.best_model, results