import os
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union, Any, Optional
import logging

# Preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, OrdinalEncoder,
    MinMaxScaler, RobustScaler, PowerTransformer,
    LabelEncoder, LabelBinarizer, TargetEncoder
)

# Feature selection
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    f_regression, mutual_info_regression, RFE
)

# Models for feature selection
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.base import BaseEstimator, TransformerMixin

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    from category_encoders import TargetEncoder as CatTargetEncoder
    TARGET_ENCODER_AVAILABLE = True
except ImportError:
    TARGET_ENCODER_AVAILABLE = False

class AutoDataPreprocessor:
    """
    Enhanced automated data preprocessing pipeline that handles:
    - Missing value imputation
    - Categorical encoding
    - Feature scaling
    - Feature selection
    - Outlier detection and handling
    - Advanced imputation methods
    - Target encoding
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the AutoDataPreprocessor.
        
        Args:
            config_path: Path to a YAML configuration file with preprocessing settings
        """
        self.config = self._load_config(config_path)
        self.preprocessor = None
        self.feature_names = None
        self.target_encoder = None
        self.scaler = None
        self.imputer = None
        self.encoder = None
        self.selector = None
        
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load preprocessing configuration from a YAML file."""
        default_config = {
            'missing_values': {
                'strategy': 'auto',  # 'auto', 'mean', 'median', 'most_frequent', 'constant', 'drop', 'iterative', 'knn'
                'fill_value': None,
                'drop_columns_with_high_missing': 0.8,  # Drop columns with >80% missing values
                'iterative_max_iter': 10,  # Max iterations for iterative imputer
                'knn_neighbors': 5  # Number of neighbors for KNN imputer
            },
            'categorical': {
                'strategy': 'onehot',  # 'onehot', 'ordinal', 'target', 'drop', 'binary'
                'handle_unknown': 'ignore',  # 'error', 'ignore'
                'min_frequency': 0.01,  # Minimum frequency for categories to be kept
                'target_smoothing': 1.0,  # Smoothing parameter for target encoding
                'target_min_samples_leaf': 1  # Min samples per leaf for target encoding
            },
            'scaling': {
                'strategy': 'standard',  # 'standard', 'minmax', 'robust', 'power', 'none'
                'with_mean': True,
                'with_std': True,
                'quantile_range': (25.0, 75.0)  # For robust scaling
            },
            'feature_selection': {
                'strategy': 'none',  # 'variance', 'correlation', 'model', 'mutual_info', 'none'
                'n_features': 'auto',  # Number of features to select, 'auto' for sqrt(n_features)
                'threshold': 0.0,  # Threshold for selection
                'correlation_threshold': 0.95  # Threshold for correlation-based selection
            },
            'outliers': {
                'strategy': 'none',  # 'zscore', 'iqr', 'isolation_forest', 'none'
                'threshold': 3.0,  # Z-score threshold
                'method': 'clip',  # 'clip', 'remove', 'impute'
                'contamination': 0.1  # Contamination parameter for isolation forest
            },
            'advanced': {
                'memory_efficient': True,  # Use memory-efficient data types
                'chunk_size': 10000,  # Chunk size for large datasets
                'parallel_jobs': -1  # Number of parallel jobs
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f) or {}
                return self._merge_dicts(default_config, user_config)
        
        return default_config
    
    @staticmethod
    def _merge_dicts(default: Dict, user: Dict) -> Dict:
        """Recursively merge two dictionaries."""
        result = default.copy()
        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = AutoDataPreprocessor._merge_dicts(result[key], value)
            else:
                result[key] = value
        return result
    
    def _detect_column_types(self, X: pd.DataFrame) -> Dict[str, List[str]]:
        """Detect column types (numerical, categorical, datetime, etc.)."""
        result = {
            'numerical': [],
            'categorical': [],
            'datetime': [],
            'boolean': [],
            'text': []
        }
        
        for col in X.columns:
            # Check datetime columns
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                result['datetime'].append(col)
            # Check boolean columns
            elif pd.api.types.is_bool_dtype(X[col]):
                result['boolean'].append(col)
            # Check categorical columns (low cardinality and object/string type)
            elif (X[col].dtype == 'object' or pd.api.types.is_categorical_dtype(X[col])) and X[col].nunique() < 0.5 * len(X):
                result['categorical'].append(col)
            # Check text columns (high cardinality object/string type)
            elif X[col].dtype == 'object' and X[col].nunique() >= 0.5 * len(X):
                result['text'].append(col)
            # Everything else is numerical
            else:
                result['numerical'].append(col)
                
        return result
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types for memory efficiency."""
        if not self.config['advanced']['memory_efficient']:
            return df
        
        df_optimized = df.copy()
        
        for col in df_optimized.columns:
            col_type = df_optimized[col].dtype
            
            if col_type != 'object':
                c_min = df_optimized[col].min()
                c_max = df_optimized[col].max()
                
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df_optimized[col] = df_optimized[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df_optimized[col] = df_optimized[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df_optimized[col] = df_optimized[col].astype(np.int32)
                else:
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df_optimized[col] = df_optimized[col].astype(np.float32)
        
        memory_reduction = (df.memory_usage(deep=True).sum() - df_optimized.memory_usage(deep=True).sum()) / df.memory_usage(deep=True).sum() * 100
        logger.info(f"Memory usage reduced by {memory_reduction:.1f}%")
        
        return df_optimized
    
    def _handle_missing_values(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Handle missing values in the dataset."""
        missing_info = {}
        missing_cols = X.isnull().sum()
        missing_cols = missing_cols[missing_cols > 0]
        
        if len(missing_cols) == 0:
            return X, missing_info
        
        strategy = self.config['missing_values']['strategy']
        drop_threshold = self.config['missing_values']['drop_columns_with_high_missing']
        fill_value = self.config['missing_values']['fill_value']
        
        
        # Drop columns with too many missing values
        if drop_threshold < 1.0:  # If 1.0, don't drop any columns
            cols_to_drop = missing_cols[missing_cols / len(X) > drop_threshold].index.tolist()
            if cols_to_drop:
                X = X.drop(columns=cols_to_drop)
                missing_info['dropped_columns'] = cols_to_drop
                missing_cols = missing_cols.drop(cols_to_drop, errors='ignore')
        
        # Handle remaining missing values
        # Handle remaining missing values
        if strategy == 'auto':
            # Use different strategies based on column type
            col_types = self._detect_column_types(X)
            
            for col in missing_cols.index:
                if col in col_types['numerical']:
                    # For numerical columns, use median for skewed data, mean otherwise
                    if X[col].skew() > 1.0 or X[col].skew() < -1.0:
                        fill_val = X[col].median()
                        strategy_used = 'median'
                    else:
                        fill_val = X[col].mean()
                        strategy_used = 'mean'
                elif col in col_types['categorical'] or col in col_types['boolean']:
                    # For categorical/boolean, use mode
                    fill_val = X[col].mode()[0] if not X[col].mode().empty else None
                    strategy_used = 'most_frequent'
                else:
                    # For other types, use a constant or drop
                    fill_val = fill_value if fill_value is not None else 'missing'
                    strategy_used = 'constant'
                
                X[col] = X[col].fillna(fill_val)
                missing_info[col] = {
                    'strategy': strategy_used,
                    'fill_value': fill_val,
                    'missing_count': missing_cols[col],
                    'missing_ratio': missing_cols[col] / len(X)
                }
        elif strategy == 'iterative':
            # Use iterative imputation
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                imputer = IterativeImputer(
                    max_iter=self.config['missing_values']['iterative_max_iter'],
                    random_state=42
                )
                X[numeric_cols] = imputer.fit_transform(X[numeric_cols])
                
                for col in numeric_cols:
                    if col in missing_cols.index:
                        missing_info[col] = {
                            'strategy': 'iterative',
                            'missing_count': missing_cols[col],
                            'missing_ratio': missing_cols[col] / len(X)
                        }
        
        elif strategy == 'knn':
            # Use KNN imputation
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                imputer = KNNImputer(
                    n_neighbors=self.config['missing_values']['knn_neighbors']
                )
                X[numeric_cols] = imputer.fit_transform(X[numeric_cols])
                
                for col in numeric_cols:
                    if col in missing_cols.index:
                        missing_info[col] = {
                            'strategy': 'knn',
                            'missing_count': missing_cols[col],
                            'missing_ratio': missing_cols[col] / len(X)
                        }
            
            # Handle categorical columns with mode
            cat_cols = X.select_dtypes(include=['object', 'category']).columns
            for col in cat_cols:
                if col in missing_cols.index:
                    X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'missing')
        else:
            # Use the specified strategy for all columns
            if strategy == 'drop':
                X = X.dropna()
            else:
                if strategy == 'constant' and fill_value is None:
                    raise ValueError("fill_value must be specified when strategy='constant'")
                
                for col in missing_cols.index:
                    if strategy in ['mean', 'median', 'most_frequent']:
                        if strategy == 'mean':
                            fill_val = X[col].mean()
                        elif strategy == 'median':
                            fill_val = X[col].median()
                        else:  # most_frequent
                            fill_val = X[col].mode()[0] if not X[col].mode().empty else None
                    else:  # constant
                        fill_val = fill_value
                    
                    X[col] = X[col].fillna(fill_val)
                    missing_info[col] = {
                        'strategy': strategy,
                        'fill_value': fill_val,
                        'missing_count': missing_cols[col],
                        'missing_ratio': missing_cols[col] / len(X)
                    }
        
        return X, missing_info
    
    def _create_preprocessing_pipeline(self, X: pd.DataFrame, y: pd.Series = None) -> Tuple[ColumnTransformer, List[str]]:
        """Create a preprocessing pipeline based on the data types and configuration."""
        col_types = self._detect_column_types(X)
        
        # Define transformers for different column types
        numeric_features = col_types['numerical']
        categorical_features = col_types['categorical']
        boolean_features = col_types['boolean']
        
        # Numeric pipeline
        imputer_strategy = 'median' if self.config['missing_values']['strategy'] in ['auto', 'median'] else 'mean'
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=imputer_strategy)),
            ('scaler', self._get_scaler())
        ])
        
        # Categorical pipeline
        if self.config['categorical']['strategy'] == 'onehot':
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
        elif self.config['categorical']['strategy'] == 'ordinal':
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
            ])
        elif self.config['categorical']['strategy'] == 'target' and y is not None:
            # Use target encoding
            if TARGET_ENCODER_AVAILABLE:
                categorical_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                    ('target', CatTargetEncoder(
                        smoothing=self.config['categorical']['target_smoothing'],
                        min_samples_leaf=self.config['categorical']['target_min_samples_leaf']
                    ))
                ])
            else:
                logger.warning("Target encoding not available, falling back to ordinal encoding")
                categorical_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
                ])
        else:  # 'drop' or unsupported strategy
            categorical_transformer = 'drop'
        
        # Boolean pipeline (convert to int)
        boolean_transformer = 'passthrough'  # Will be converted to int later
        
        # Combine all transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features),
                ('bool', boolean_transformer, boolean_features)
            ],
            remainder='drop'  # Drop columns that don't match any transformer
        )
        
        # Get feature names after transformation
        feature_names = numeric_features.copy()
        
        if categorical_features and self.config['categorical']['strategy'] == 'onehot':
            # For one-hot encoding, we'll need to get the feature names after fitting
            feature_names.extend([f"cat_{i}" for i in range(len(categorical_features) * 5)])  # Approximate
        elif categorical_features and self.config['categorical']['strategy'] == 'ordinal':
            feature_names.extend([f"{col}_encoded" for col in categorical_features])
        
        feature_names.extend(boolean_features)
        
        return preprocessor, feature_names
    
    def _get_scaler(self):
        """Get the appropriate scaler based on configuration."""
        scaling_config = self.config['scaling']
        strategy = scaling_config['strategy']
        
        if strategy == 'standard':
            return StandardScaler(
                with_mean=scaling_config.get('with_mean', True),
                with_std=scaling_config.get('with_std', True)
            )
        elif strategy == 'minmax':
            return MinMaxScaler()
        elif strategy == 'robust':
            return RobustScaler(
                quantile_range=scaling_config.get('quantile_range', (25.0, 75.0))
            )
        elif strategy == 'power':
            return PowerTransformer(method='yeo-johnson')
        else:  # 'none'
            return 'passthrough'
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Fit the preprocessor to the data and transform it.
        
        Args:
            X: Input feature matrix
            y: Target vector (optional, needed for some feature selection methods)
            
        Returns:
            Tuple of (transformed_X, missing_info)
        """
        # Store original column names
        self.original_columns = X.columns.tolist()
        
        # Optimize data types for memory efficiency
        X = self._optimize_dtypes(X)
        
        # Handle missing values
        X_processed, missing_info = self._handle_missing_values(X)
        
        # Create and fit the preprocessing pipeline
        self.preprocessor, self.feature_names = self._create_preprocessing_pipeline(X_processed)
        
        # Fit and transform the data
        X_transformed = self.preprocessor.fit_transform(X_processed)
        
        # Convert to DataFrame with appropriate column names
        if hasattr(self.preprocessor, 'get_feature_names_out'):
            # For newer scikit-learn versions
            feature_names_out = self.preprocessor.get_feature_names_out()
            X_transformed = pd.DataFrame(X_transformed, columns=feature_names_out, index=X.index)
        else:
            # Fallback for older versions
            X_transformed = pd.DataFrame(X_transformed, columns=self.feature_names, index=X.index)
        
        # Convert boolean columns to int
        bool_cols = X_processed.select_dtypes(include=['bool']).columns
        for col in bool_cols:
            if col in X_transformed.columns:
                X_transformed[col] = X_transformed[col].astype(int)
        
        # Apply feature selection if enabled
        if self.config['feature_selection']['strategy'] != 'none':
            X_transformed = self._apply_feature_selection(X_transformed, y)
        
        # Handle outliers if enabled
        if self.config['outliers']['strategy'] != 'none':
            X_transformed = self._handle_outliers(X_transformed)
        
        return X_transformed, missing_info
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using the fitted preprocessor."""
        if self.preprocessor is None:
            raise ValueError("The preprocessor has not been fitted yet. Call fit_transform first.")
        
        # Optimize data types for memory efficiency
        X = self._optimize_dtypes(X)
        
        # Handle missing values
        X_processed, _ = self._handle_missing_values(X)
        
        # Apply the fitted preprocessing
        X_transformed = self.preprocessor.transform(X_processed)
        
        # Convert to DataFrame
        if hasattr(self.preprocessor, 'get_feature_names_out'):
            # For newer scikit-learn versions
            feature_names_out = self.preprocessor.get_feature_names_out()
            X_transformed = pd.DataFrame(X_transformed, columns=feature_names_out, index=X.index)
        else:
            # Fallback for older versions
            X_transformed = pd.DataFrame(X_transformed, columns=self.feature_names, index=X.index)
        
        # Convert boolean columns to int
        bool_cols = X_processed.select_dtypes(include=['bool']).columns
        for col in bool_cols:
            if col in X_transformed.columns:
                X_transformed[col] = X_transformed[col].astype(int)
        
        # Apply feature selection if enabled
        if hasattr(self, 'feature_selector') and self.feature_selector is not None:
            X_transformed = self.feature_selector.transform(X_transformed)
        
        # Handle outliers if enabled
        if hasattr(self, 'outlier_handler') and self.outlier_handler is not None:
            X_transformed = self.outlier_handler.transform(X_transformed)
        
        return X_transformed
    
    def _apply_feature_selection(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Apply feature selection to the data."""
        strategy = self.config['feature_selection']['strategy']
        n_features = self.config['feature_selection']['n_features']
        threshold = self.config['feature_selection']['threshold']
        correlation_threshold = self.config['feature_selection']['correlation_threshold']
        
        if strategy == 'variance':
            # Remove low-variance features
            from sklearn.feature_selection import VarianceThreshold
            selector = VarianceThreshold(threshold=threshold)
            X_selected = selector.fit_transform(X)
            selected_features = X.columns[selector.get_support()].tolist()
            
        elif strategy == 'correlation':
            # Remove highly correlated features
            corr_matrix = X.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
            X_selected = X.drop(columns=to_drop)
            selected_features = X_selected.columns.tolist()
            
        elif strategy == 'model':
            # Use model-based feature selection
            if y is None:
                raise ValueError("Target variable y is required for model-based feature selection.")
                
            if n_features == 'auto':
                n_features = 'sqrt'
                
            if isinstance(n_features, str) and n_features.startswith('sqrt'):
                n_features = int(np.sqrt(X.shape[1]))
            elif isinstance(n_features, float) and 0 < n_features < 1:
                n_features = int(X.shape[1] * n_features)
            
            # Use RandomForest for feature importance
            if len(y.unique()) <= 2:  # Binary classification
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            elif len(y.unique()) > 2 and len(y.unique()) < 20:  # Multi-class classification
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:  # Regression
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            model.fit(X, y)
            
            # Get feature importances
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Select top n features
            n_features = min(n_features, X.shape[1])
            selected_indices = indices[:n_features]
            X_selected = X.iloc[:, selected_indices]
            selected_features = X.columns[selected_indices].tolist()
        
        elif strategy == 'mutual_info':
            # Use mutual information for feature selection
            if y is None:
                raise ValueError("Target variable y is required for mutual information feature selection.")
            
            if n_features == 'auto':
                n_features = int(np.sqrt(X.shape[1]))
            elif isinstance(n_features, float) and 0 < n_features < 1:
                n_features = int(X.shape[1] * n_features)
            
            # Determine if classification or regression
            is_classification = len(np.unique(y)) < 0.1 * len(y)
            score_func = mutual_info_classif if is_classification else mutual_info_regression
            
            selector = SelectKBest(score_func=score_func, k=n_features)
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            X_selected = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
            
        else:  # 'none' or unsupported strategy
            return X
        
        # Store the selected features
        self.selected_features_ = selected_features
        
        # Update feature names
        self.feature_names = selected_features
        
        return X_selected
    
    def _handle_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers in the data."""
        strategy = self.config['outliers']['strategy']
        threshold = self.config['outliers']['threshold']
        method = self.config['outliers']['method']
        contamination = self.config['outliers']['contamination']
        
        if strategy == 'zscore':
            z_scores = (X - X.mean()) / X.std()
            outliers = np.abs(z_scores) > threshold
            
            if method == 'clip':
                # Clip values to threshold
                X_clipped = X.copy()
                lower_bound = X.mean() - threshold * X.std()
                upper_bound = X.mean() + threshold * X.std()
                
                for col in X.columns:
                    X_clipped[col] = X[col].clip(lower_bound[col], upper_bound[col])
                
                return X_clipped
                
            elif method == 'remove':
                # Remove rows with any outliers
                return X[~outliers.any(axis=1)]
                
            elif method == 'impute':
                # Impute outliers with median
                X_imputed = X.copy()
                for col in X.columns:
                    col_outliers = outliers[col]
                    if col_outliers.any():
                        median_val = X[col].median()
                        X_imputed.loc[col_outliers, col] = median_val
                return X_imputed
        
        elif strategy == 'iqr':
            # IQR method for outlier detection
            Q1 = X.quantile(0.25)
            Q3 = X.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers = (X < lower_bound) | (X > upper_bound)
            
            if method == 'clip':
                # Clip values to bounds
                X_clipped = X.copy()
                for col in X.columns:
                    X_clipped[col] = X[col].clip(lower_bound[col], upper_bound[col])
                return X_clipped
                
            elif method == 'remove':
                # Remove rows with any outliers
                return X[~outliers.any(axis=1)]
                
            elif method == 'impute':
                # Impute outliers with median
                X_imputed = X.copy()
                for col in X.columns:
                    col_outliers = outliers[col]
                    if col_outliers.any():
                        median_val = X[col].median()
                        X_imputed.loc[col_outliers, col] = median_val
                return X_imputed
        
        elif strategy == 'isolation_forest':
            # Use Isolation Forest for outlier detection
            from sklearn.ensemble import IsolationForest
            
            iso_forest = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_jobs=self.config['advanced']['parallel_jobs']
            )
            
            outliers = iso_forest.fit_predict(X) == -1
            
            if method == 'remove':
                return X[~outliers]
            elif method == 'impute':
                X_imputed = X.copy()
                for col in X.columns:
                    col_outliers = outliers
                    if col_outliers.any():
                        median_val = X[col].median()
                        X_imputed.loc[col_outliers, col] = median_val
                return X_imputed
        
        # If no valid strategy or method, return original data
        return X

import json

def prepare_data(input_path: str, output_path: str, dataset_type: str):
    """
    Loads and prepares raw data based on the dataset type.
    
    Args:
        input_path: Path to the input data file
        output_path: Path to save the processed data
        dataset_type: Type of dataset ('iris', 'wine', or 'custom')
    """
    print(f"Loading {dataset_type} data from {input_path}...")

    if dataset_type == 'iris':
        column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
        df = pd.read_csv(input_path, header=None, names=column_names)
        print("Data loaded successfully. Processing...")

        # Encode the categorical species column
        label_encoder = LabelEncoder()
        df['species'] = label_encoder.fit_transform(df['species'])
        print("Species column encoded.")

    elif dataset_type == 'wine':
        column_names = [
            'class', 'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash',
            'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
            'proanthocyanins', 'color_intensity', 'hue',
            'od280_od315_of_diluted_wines', 'proline'
        ]
        df = pd.read_csv(input_path, header=None, names=column_names)
        print("Data loaded successfully. Processing...")

        # The target 'class' is 1-indexed. Make it 0-indexed.
        df['class'] = df['class'] - 1
        print("Class column adjusted to be 0-indexed.")
        
        # Set target column
        target_col = 'class'
        
    elif dataset_type == 'custom':
        # For custom datasets, assume CSV with header
        df = pd.read_csv(input_path)
        print("Custom dataset loaded successfully.")
        
        # For custom datasets, the target column should be specified separately
        # or we can try to infer it (e.g., last column named 'target' or 'class')
        possible_targets = ['target', 'class', 'label', 'y', 'target_variable']
        target_col = None
        
        for col in possible_targets:
            if col in df.columns:
                target_col = col
                break
        
        if target_col is None:
            # If no obvious target, assume last column is the target
            target_col = df.columns[-1]
            print(f"No obvious target column found. Using '{target_col}' as the target.")
    
    else:
        raise ValueError("Unsupported dataset type. Please choose 'iris', 'wine', or 'custom'.")
    
    # Initialize the preprocessor
    preprocessor = AutoDataPreprocessor()
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Preprocess the data
    print("Preprocessing data...")
    X_processed, missing_info = preprocessor.fit_transform(X, y)
    
    # Add the target column back
    processed_df = pd.concat([X_processed, y.reset_index(drop=True)], axis=1)
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save the processed data
    processed_df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")
    
    # Save preprocessing info
    if missing_info:
        info_path = os.path.join(output_dir, 'preprocessing_info.json')
        with open(info_path, 'w') as f:
            json.dump(missing_info, f, indent=2)
        print(f"Preprocessing information saved to {info_path}")
    
    return processed_df, target_col


def main():
    """Main function to handle command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare raw data for modeling')
    parser.add_argument('--input_path', type=str, required=True, 
                        help='Path to the raw data file')
    parser.add_argument('--output_path', type=str, required=True, 
                        help='Path to save the processed data')
    parser.add_argument('--dataset_type', type=str, required=True, 
                        choices=['iris', 'wine', 'custom'],
                        help='Type of dataset')
    
    args = parser.parse_args()
    prepare_data(args.input_path, args.output_path, args.dataset_type)


if __name__ == "__main__":
    main()
