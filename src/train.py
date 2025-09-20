import os
import argparse
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Import our custom modules
from auto_ml import AutoML
from data_handling import AutoDataPreprocessor

# MLflow for experiment tracking
import mlflow
import mlflow.sklearn
import mlflow.keras

# Scikit-learn metrics
from sklearn.metrics import confusion_matrix

# Set styles
plt.style.use('seaborn')
sns.set_style('whitegrid')

# Create output directories
os.makedirs('models', exist_ok=True)
os.makedirs('reports/figures', exist_ok=True)
os.makedirs('reports/metrics', exist_ok=True)
os.makedirs('tensorboard_logs', exist_ok=True)
os.makedirs('config', exist_ok=True)

def _generate_visualizations(
    automl: AutoML,
    X: np.ndarray,
    y: np.ndarray,
    output_dir: Path
) -> None:
    """Generate visualizations for model evaluation."""
    try:
        # Create visualizations directory
        vis_dir = output_dir / 'reports' / 'figures'
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # Feature importance plot
        if hasattr(automl, 'feature_importances_') and automl.feature_importances_ is not None:
            # This assumes feature_names are stored in the preprocessor object after fitting
            # and that the preprocessor is accessible here.
            # A better approach would be to pass feature_names to this function.
            # For now, let's assume automl object has feature_names attribute.
            if hasattr(automl, 'feature_names'):
                importances = pd.DataFrame({
                    'feature': automl.feature_names,
                    'importance': automl.feature_importances_
                }).sort_values('importance', ascending=False)
                
                plt.figure(figsize=(12, 8))
                sns.barplot(x='importance', y='feature', data=importances.head(20))
                plt.title('Top 20 Most Important Features')
                plt.tight_layout()
                plt.savefig(str(vis_dir / 'feature_importance.png'))
                plt.close()

        # Confusion matrix for classification
        if automl.task == 'classification':
            y_pred = automl.predict(automl.X_test)
            cm = confusion_matrix(automl.y_test, y_pred)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.tight_layout()
            plt.savefig(str(vis_dir / 'confusion_matrix.png'))
            plt.close()
        
        # Actual vs Predicted for regression
        else:
            y_pred = automl.predict(automl.X_test)
            
            plt.figure(figsize=(10, 8))
            plt.scatter(automl.y_test, y_pred, alpha=0.5)
            plt.plot([min(automl.y_test), max(automl.y_test)],
                     [min(automl.y_test), max(automl.y_test)], 'r--')
            plt.title('Actual vs Predicted Values')
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.tight_layout()
            plt.savefig(str(vis_dir / 'actual_vs_predicted.png'))
            plt.close()

    except Exception as e:
        print(f"Warning: Could not generate some visualizations: {str(e)}")


def train_model(
    input_path: str,
    target_column: str,
    experiment_name: str,
    task_type: str = 'classification',
    test_size: float = 0.2,
    random_state: int = 42,
    use_mlflow: bool = True,
    config_path: str = None,
    tensorboard_logdir: str = None
) -> Dict[str, Any]:
    """
    Automated machine learning pipeline that handles data preprocessing, model selection,
    training, and evaluation.
    
    Args:
        input_path: Path to the input CSV file
        target_column: Name of the target column
        experiment_name: Name for the MLflow experiment
        task_type: Type of task - 'classification' or 'regression'
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        use_mlflow: Whether to log to MLflow
        config_path: Path to custom configuration file for preprocessing
        tensorboard_logdir: Directory to save TensorBoard logs
        
    Returns:
        Dictionary containing training results and model information
    """
    # Set up directories and paths
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path(f'runs/{experiment_name}_{timestamp}')
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize MLflow if enabled
    if use_mlflow:
        mlflow.set_experiment(experiment_name)
        mlflow.start_run(run_name=f"automl_{timestamp}")
        mlflow.log_param("task_type", task_type)
        mlflow.log_param("target_column", target_column)
    
    try:
        # Load and preprocess data
        print(f"\n{'='*50}")
        print("Loading and preprocessing data...")
        print(f"{'='*50}")
        
        df = pd.read_csv(input_path)
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Initialize auto preprocessor
        preprocessor = AutoDataPreprocessor(config_path=config_path)
        
        # Preprocess the data
        X_processed, missing_info = preprocessor.fit_transform(X, y)
        
        # Log preprocessing info
        if use_mlflow and missing_info:
            mlflow.log_params({"missing_values_handled": bool(missing_info)})
            mlflow.log_dict(missing_info, "preprocessing/missing_values_handled.json")
        
        # Convert to numpy arrays for training
        if hasattr(X_processed, 'values'):
            X_processed = X_processed.values
        if hasattr(y, 'values'):
            y = y.values
            
        # Train and evaluate models
        print(f"\n{'='*50}")
        print("Training and evaluating models...")
        print(f"{'='*50}")
        
        automl = AutoML(task=task_type, random_state=random_state)
        
        # Set up tensorboard directory
        if tensorboard_logdir is None:
            tensorboard_logdir = str(run_dir / 'tensorboard_logs')
        
        # Train models
        automl.fit(
            X_processed, y,
            test_size=test_size,
            cv=5,
            n_iter=10,
            search_method='random',
            early_stopping_rounds=10,
            tensorboard_logdir=tensorboard_logdir
        )
        
        # Get the best model
        best_model = automl.best_model
        best_model_name = automl.best_model_name
        
        # Evaluate on test set
        test_metrics = automl.evaluate()
        
        # Save model
        model_path = str(run_dir / 'model' / 'best_model')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save based on model type
        if hasattr(best_model, 'save'):  # Keras model
            best_model.save(model_path)
            model_type = 'keras'
        else:  # Scikit-learn model
            joblib.dump(best_model, f"{model_path}.joblib")
            model_type = 'sklearn'
        
        # Pass feature names to visualization function
        if hasattr(preprocessor, 'get_feature_names_out'):
            automl.feature_names = preprocessor.get_feature_names_out()
        else:
            automl.feature_names = preprocessor.feature_names

        # Generate visualizations
        _generate_visualizations(automl, X_processed, y, run_dir)
        
        # Log to MLflow
        if use_mlflow:
            mlflow.log_params({
                'best_model': best_model_name,
                'model_type': model_type,
                'test_size': test_size,
                'random_state': random_state
            })
            mlflow.log_metrics(test_metrics)
            
            if model_type == 'keras':
                mlflow.keras.log_model(best_model, "model")
            else:
                mlflow.sklearn.log_model(best_model, "model")
            
            mlflow.log_artifacts(str(run_dir / 'reports'), "reports")
            
            if hasattr(automl, 'feature_importances_') and automl.feature_importances_ is not None:
                importances_df = pd.DataFrame({
                    'feature': automl.feature_names,
                    'importance': automl.feature_importances_
                }).sort_values('importance', ascending=False)
                
                importance_path = str(run_dir / 'feature_importances.csv')
                importances_df.to_csv(importance_path, index=False)
                mlflow.log_artifact(importance_path)
        
        # Prepare results
        final_results = {
            'best_model': best_model_name,
            'model_path': model_path,
            'model_type': model_type,
            'metrics': test_metrics,
            'feature_names': list(automl.feature_names) if hasattr(automl, 'feature_names') else None,
            'run_dir': str(run_dir)
        }
        
        # Save results to JSON
        results_path = str(run_dir / 'training_results.json')
        with open(results_path, 'w') as f:
            # Convert numpy types to native python types for JSON serialization
            sanitized_results = json.loads(pd.Series(final_results).to_json())
            json.dump(sanitized_results, f, indent=4)
        
        print(f"\n{'='*50}")
        print("Training completed successfully!")
        print(f"Best model: {best_model_name}")
        print(f"Test metrics: {test_metrics}")
        print(f"Results saved to: {run_dir}")
        print(f"{'='*50}")
        
        return final_results
    
    except Exception as e:
        print(f"\n{'='*50}")
        print(f"Error during training: {str(e)}")
        print(f"{'='*50}")
        raise
    
    finally:
        if use_mlflow:
            mlflow.end_run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automated Machine Learning Pipeline. "
                    "Trains and evaluates multiple models on the given dataset."
    )
    
    # Required arguments
    parser.add_argument(
        "--input_path", 
        type=str, 
        required=True,
        help="Path to the input CSV file containing the dataset."
    )
    
    parser.add_argument(
        "--target_column",
        type=str,
        required=True,
        help="Name of the target column in the dataset."
    )
    
    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        help="Name of the MLflow experiment to track the run."
    )
    
    # Optional arguments
    parser.add_argument(
        "--task_type",
        type=str,
        default="classification",
        choices=["classification", "regression"],
        help="Type of machine learning task. Options: 'classification' or 'regression' (default: classification)"
    )
    
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Fraction of the data to use for testing (default: 0.2)"
    )
    
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    parser.add_argument(
        "--no_mlflow",
        action="store_true",
        help="Disable MLflow experiment tracking"
    )
    
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to a custom configuration file for data preprocessing"
    )
    
    parser.add_argument(
        "--tensorboard_logdir",
        type=str,
        default=None,
        help="Directory to save TensorBoard logs (default: runs/<experiment_name>_<timestamp>/tensorboard_logs)"
    )
    
    args = parser.parse_args()
    
    # Run the training pipeline
    train_model(
        input_path=args.input_path,
        target_column=args.target_column,
        experiment_name=args.experiment_name,
        task_type=args.task_type,
        test_size=args.test_size,
        random_state=args.random_state,
        use_mlflow=not args.no_mlflow,
        config_path=args.config_path,
        tensorboard_logdir=args.tensorboard_logdir
    )
