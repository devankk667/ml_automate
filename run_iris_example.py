import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from src.pipeline import AutoMLPipeline
from src.utils import save_config as save_yaml

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

def load_iris_data():
    """Load and prepare the Iris dataset."""
    print("Loading Iris dataset...")
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='target')
    
    # Add some random noise to make it more interesting
    np.random.seed(42)
    for col in X.columns:
        X[col] = X[col] + np.random.normal(0, 0.1, size=len(X))
    
    # Add some missing values
    for col in X.columns:
        mask = np.random.random(len(X)) < 0.05  # 5% missing values
        X.loc[mask, col] = np.nan
    
    # Add a categorical column
    X['petal_ratio'] = X['petal length (cm)'] / (X['petal width (cm)'] + 1e-8)
    X['petal_ratio_category'] = pd.cut(
        X['petal_ratio'],
        bins=[-np.inf, 1, 2, 3, np.inf],
        labels=['very_small', 'small', 'medium', 'large']
    )
    
    return X, y

def main():
    # Load and prepare data
    X, y = load_iris_data()
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Define the configuration for the pipeline
    config = {
        'preprocessing': {
            'missing_values': 'impute',
            'categorical_encoding': 'onehot',
            'scaling': 'standard',
            'feature_selection': True,
            'feature_selection_params': {
                'n_features': 0.8,  # Keep 80% of features
                'scoring': 'f1_weighted',
                'cv': 5
            }
        },
        'model': {
            'task': 'classification',
            'models': ['random_forest', 'xgboost', 'logistic_regression'],
            'hyperparameter_tuning': {
                'n_iter': 10,  # Number of iterations for random search
                'cv': 5,       # Number of cross-validation folds
                'scoring': 'f1_weighted'
            }
        },
        'evaluation': {
            'metrics': ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted'],
            'confusion_matrix': True,
            'roc_curve': True,
            'feature_importance': True
        },
        'output': {
            'save_model': True,
            'output_dir': 'output/iris_example',
            'save_predictions': True
        }
    }
    
    # Save the config for reference
    save_yaml(config, 'output/iris_example/config.yaml')
    
    # Initialize the pipeline
    print("Initializing AutoML pipeline...")
    pipeline = AutoMLPipeline(config=config)
    
    # Train the pipeline
    print("Training models...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    results = pipeline.evaluate(X_test, y_test)
    
    # Print evaluation results
    print("\nTest set results:")
    for metric, value in results['metrics'].items():
        print(f"{metric}: {value:.4f}")
    
    # Print best model info
    print(f"\nBest model: {pipeline.best_model_name}")
    print(f"Best parameters: {pipeline.best_params}")
    
    # Save the pipeline
    pipeline.save('output/iris_example/pipeline.joblib')
    print("\nPipeline saved to 'output/iris_example/'")
