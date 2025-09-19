import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import json
import yaml
from typing import Dict, List, Tuple, Union, Any, Optional

# Import the AutoML class
import sys
sys.path.append('src')
from auto_ml import AutoML
from data_handling import AutoDataPreprocessor  # Assuming this exists in your project

class DataCleaner:
    """Handles data cleaning and preprocessing."""
    
    def __init__(self, config_path: str = None):
        """Initialize with optional config."""
        self.config = self._load_config(config_path)
        self.preprocessor = AutoDataPreprocessor()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return {
            'missing_threshold': 0.7,  # Drop columns with >70% missing
            'correlation_threshold': 0.95,  # Drop highly correlated features
            'outlier_method': 'zscore',  # 'zscore' or 'iqr'
            'outlier_threshold': 3.0,
            'categorical_threshold': 0.05  # Max cardinality for categorical
        }
    
    def clean(self, df: pd.DataFrame, target_col: str = None) -> Tuple[pd.DataFrame, Dict]:
        """Main cleaning pipeline."""
        df = df.copy()
        cleaning_report = {}
        
        # 1. Handle missing values
        missing_report = self._handle_missing_values(df, target_col)
        cleaning_report['missing_values'] = missing_report
        
        # 2. Handle duplicates
        dupes = df.duplicated().sum()
        if dupes > 0:
            df = df.drop_duplicates()
            cleaning_report['duplicates_removed'] = dupes
        
        # 3. Handle outliers (only on numerical columns)
        if target_col:
            X = df.drop(columns=[target_col])
            y = df[target_col]
            X_clean = self._handle_outliers(X, self.config['outlier_method'], 
                                          self.config['outlier_threshold'])
            df = pd.concat([X_clean, y], axis=1)
        
        # 4. Feature engineering
        df = self._basic_feature_engineering(df, target_col)
        
        # 5. Advanced preprocessing
        if target_col:
            X = df.drop(columns=[target_col])
            y = df[target_col]
            X_processed, _ = self.preprocessor.fit_transform(X, y)
            df = pd.concat([X_processed, y.reset_index(drop=True)], axis=1)
        
        return df, cleaning_report
    
    def _handle_missing_values(self, df: pd.DataFrame, target_col: str = None) -> Dict:
        """Handle missing values with reporting."""
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        missing_report = {
            'total_missing': int(missing.sum()),
            'columns_removed': [],
            'columns_imputed': []
        }
        
        # Drop columns with too many missing values
        high_missing = missing_pct[missing_pct > self.config['missing_threshold'] * 100].index
        if not high_missing.empty:
            df.drop(columns=high_missing, inplace=True)
            missing_report['columns_removed'] = high_missing.tolist()
        
        # Impute remaining missing values
        for col in df.columns[df.isnull().any()]:
            if df[col].dtype in ['int64', 'float64']:
                df[col].fillna(df[col].median(), inplace=True)
                method = 'median'
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
                method = 'mode'
            missing_report['columns_imputed'].append({'column': col, 'method': method})
        
        return missing_report
    
    def _handle_outliers(self, X: pd.DataFrame, method: str = 'zscore', 
                        threshold: float = 3.0) -> pd.DataFrame:
        """Handle outliers in numerical columns."""
        X = X.copy()
        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
        
        for col in numeric_cols:
            if method == 'zscore':
                z_scores = (X[col] - X[col].mean()) / X[col].std()
                X = X[abs(z_scores) <= threshold]
            elif method == 'iqr':
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                X = X[~((X[col] < (Q1 - 1.5 * IQR)) | (X[col] > (Q3 + 1.5 * IQR)))]
        
        return X
    
    def _basic_feature_engineering(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        """Apply basic feature engineering."""
        df = df.copy()
        
        # Date features
        date_cols = df.select_dtypes(include=['datetime64']).columns
        for col in date_cols:
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_day'] = df[col].dt.day
        
        # Categorical encoding (simple version, AutoDataPreprocessor handles the rest)
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        if target_col and target_col in cat_cols:
            cat_cols = cat_cols.drop(target_col)
        
        return df

def detect_task(y: pd.Series) -> str:
    """Detect if the task is classification or regression."""
    unique_values = y.nunique()
    if unique_values < 0.1 * len(y) or y.dtype == 'object' or y.dtype.name == 'category':
        return 'classification'
    return 'regression'

def evaluate_model(y_true, y_pred, task: str) -> Dict:
    """Evaluate model performance."""
    metrics = {}
    
    if task == 'classification':
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['classification_report'] = classification_report(y_true, y_pred, output_dict=True)
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
    else:  # regression
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['r2'] = r2_score(y_true, y_pred)
    
    return metrics

def save_plots(df: pd.DataFrame, target_col: str, output_dir: Path, task: str):
    """Generate and save EDA and model evaluation plots."""
    # Create plots directory
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # 1. Target distribution
    plt.figure(figsize=(10, 6))
    if task == 'classification':
        sns.countplot(x=target_col, data=df)
    else:
        sns.histplot(df[target_col], kde=True)
    plt.title(f'Distribution of {target_col}')
    plt.tight_layout()
    plt.savefig(plots_dir / 'target_distribution.png')
    plt.close()
    
    # 2. Correlation heatmap (for numerical features)
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(num_cols) > 1:  # Need at least 2 numerical columns
        plt.figure(figsize=(12, 10))
        corr = df[num_cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(plots_dir / 'correlation_heatmap.png')
        plt.close()
    
    # 3. Pairplot for small datasets
    if len(df) <= 1000 and len(df.columns) <= 10:
        sns.pairplot(df.sample(min(100, len(df))), hue=target_col if task == 'classification' else None)
        plt.suptitle('Pair Plot of Features', y=1.02)
        plt.savefig(plots_dir / 'pair_plot.png', bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='AutoML Pipeline with Advanced Data Cleaning')
    parser.add_argument('--input_path', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--target_column', type=str, help='Name of the target column')
    parser.add_argument('--output_dir', type=str, default='output/automl_results', 
                       help='Directory to save results')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size (0.0 to 1.0)')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed')
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    
    args = parser.parse_args()
    
    # Setup output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and clean data
    print(f"Loading data from {args.input_path}...")
    df = pd.read_csv(args.input_path)
    print(f"Original data shape: {df.shape}")
    
    # Identify target column if not provided
    if not args.target_column:
        possible_targets = ['target', 'class', 'label', 'y', 'target_variable', 'price', 'value']
        for col in possible_targets:
            if col in df.columns:
                args.target_column = col
                print(f"Using '{col}' as the target column.")
                break
        
        if not args.target_column:
            args.target_column = df.columns[-1]
            print(f"No target column specified. Using the last column '{args.target_column}' as the target.")
    
    # Clean and preprocess data
    print("\nCleaning and preprocessing data...")
    cleaner = DataCleaner(args.config)
    df_clean, cleaning_report = cleaner.clean(df, args.target_column)
    
    # Save cleaning report
    with open(output_dir / 'cleaning_report.json', 'w') as f:
        json.dump(cleaning_report, f, indent=2)
    
    # Detect task type
    task = detect_task(df_clean[args.target_column])
    print(f"\nDetected task type: {task.upper()}")
    
    # Split data
    X = df_clean.drop(columns=[args.target_column])
    y = df_clean[args.target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, 
        stratify=y if task == 'classification' else None
    )
    
    # Initialize and run AutoML
    print("\nStarting AutoML...")
    automl = AutoML(task=task)
    automl.fit(X_train, y_train)
    
    # Evaluate
    print("\nEvaluating on test set...")
    y_pred = automl.predict(X_test)
    metrics = evaluate_model(y_test, y_pred, task)
    
    # Save results
    results = {
        'best_model': automl.best_model_name,
        'best_score': float(automl.best_score),
        'task': task,
        'input_file': args.input_path,
        'target_column': args.target_column,
        'test_size': args.test_size,
        'random_state': args.random_state,
        'timestamp': timestamp,
        'metrics': metrics,
        'cleaning_summary': {
            'original_shape': df.shape,
            'cleaned_shape': df_clean.shape,
            'columns_removed': list(set(df.columns) - set(df_clean.columns))
        }
    }
    
    # Save results
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save predictions
    predictions = X_test.copy()
    predictions['true'] = y_test
    predictions['predicted'] = y_pred
    predictions.to_csv(output_dir / 'predictions.csv', index=False)
    
    # Generate and save plots
    print("\nGenerating visualizations...")
    save_plots(df_clean, args.target_column, output_dir, task)
    
    # Print summary
    print(f"\n{'='*50}")
    print("AutoML Pipeline Complete!")
    print(f"Results saved to: {output_dir}")
    print(f"\nBest Model: {results['best_model']}")
    print(f"Best Score: {results['best_score']:.4f}")
    
    if task == 'classification':
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
    else:
        print(f"\nMSE: {metrics['mse']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"R²: {metrics['r2']:.4f}")
    
    print("\nCleaning Summary:")
    print(f"- Original shape: {df.shape}")
    print(f"- Cleaned shape: {df_clean.shape}")
    removed_cols = results['cleaning_summary']['columns_removed']
    if removed_cols:
        print(f"- Columns removed: {', '.join(removed_cols)}")
    else:
        print("- No columns removed during cleaning")

if __name__ == "__main__":
    main()