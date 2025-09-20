import os
import sys
import argparse
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union

# Import our custom modules
from auto_ml import AutoML, auto_train
from data_handling import AutoDataPreprocessor

# MLflow for experiment tracking
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Set styles
plt.style.use('seaborn')
sns.set_style('whitegrid')

# Create output directories
os.makedirs('models', exist_ok=True)
os.makedirs('reports/figures', exist_ok=True)
os.makedirs('reports/metrics', exist_ok=True)
os.makedirs('tensorboard_logs', exist_ok=True)
os.makedirs('config', exist_ok=True)


def train_model(
    input_path: str,
    target_column: str,
    experiment_name: str,
    task_type: str = 'auto',
    test_size: float = 0.2,
    random_state: int = 42,
    use_mlflow: bool = True,
    config_path: str = None,
    use_ensembles: bool = True,
    use_optuna: bool = True,
    optuna_trials: int = 100
) -> Dict[str, Any]:
    """
    Enhanced automated machine learning pipeline with advanced features.
    
    Args:
        input_path: Path to the input CSV file
        target_column: Name of the target column
        experiment_name: Name for the MLflow experiment
        task_type: Type of task - 'classification', 'regression', or 'auto'
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        use_mlflow: Whether to log to MLflow
        config_path: Path to custom configuration file
        use_ensembles: Whether to create ensemble models
        use_optuna: Whether to use Optuna for hyperparameter optimization
        optuna_trials: Number of Optuna trials
        
    Returns:
        Dictionary containing training results and model information
    """
    # Set up directories and paths
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path(f'runs/{experiment_name}_{timestamp}')
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize MLflow if enabled
    if use_mlflow and MLFLOW_AVAILABLE:
        mlflow.set_experiment(experiment_name)
        mlflow.start_run(run_name=f"automl_{timestamp}")
        mlflow.log_param("task_type", task_type)
        mlflow.log_param("target_column", target_column)
        mlflow.log_param("use_ensembles", use_ensembles)
        mlflow.log_param("use_optuna", use_optuna)
    
    try:
        # Load and preprocess data
        print(f"\n{'='*50}")
        print("Loading and preprocessing data...")
        print(f"{'='*50}")
        
        df = pd.read_csv(input_path)
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        print(f"Dataset shape: {X.shape}")
        print(f"Target column: {target_column}")
        
        # Initialize enhanced AutoML
        print(f"\n{'='*50}")
        print("Initializing Enhanced AutoML Pipeline...")
        print(f"{'='*50}")
        
        automl = AutoML(
            task=task_type,
            random_state=random_state,
            use_ensembles=use_ensembles,
            use_optuna=use_optuna,
            optuna_trials=optuna_trials,
            verbose=1
        )
        
        # Train the models
        results = automl.fit(
            X, y,
            test_size=test_size,
            cv=5,
            n_iter=20,
            search_method='random'
        )
        
        # Evaluate on test set
        test_metrics = automl.evaluate()
        
        # Save model
        model_path = str(run_dir / 'model' / 'best_model.joblib')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        automl.save_model(model_path)
        
        # Generate visualizations
        _generate_visualizations(automl, run_dir)
        
        # Log to MLflow
        if use_mlflow and MLFLOW_AVAILABLE:
            # Log parameters
            mlflow.log_params({
                'best_model': automl.best_model_name,
                'test_size': test_size,
                'random_state': random_state,
                'models_trained': results['models_trained']
            })
            
            # Log metrics
            mlflow.log_metrics(test_metrics)
            mlflow.log_metrics({f'cv_{k}': v for k, v in results['model_scores'].items()})
            
            # Log model
            mlflow.sklearn.log_model(automl.best_model, "model")
            
            # Log artifacts
            mlflow.log_artifacts(str(run_dir / 'reports'), "reports")
            
            # Log feature importances if available
            if automl.feature_importances_ is not None:
                try:
                    importances = pd.DataFrame({
                        'feature': [f'feature_{i}' for i in range(len(automl.feature_importances_))],
                        'importance': automl.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    importance_path = str(run_dir / 'feature_importances.csv')
                    importances.to_csv(importance_path, index=False)
                    mlflow.log_artifact(importance_path)
                except Exception as e:
                    print(f"Warning: Could not save feature importances: {e}")
        
        # Save dataset characteristics
        if hasattr(automl, 'dataset_characteristics_'):
            char_path = str(run_dir / 'dataset_characteristics.json')
            with open(char_path, 'w') as f:
                json.dump(automl.dataset_characteristics_, f, indent=2)
        
        # Save SHAP values if available
        shap_values = automl.get_shap_values()
        if shap_values is not None:
            shap_path = str(run_dir / 'shap_values.npy')
            np.save(shap_path, shap_values)
            print("SHAP values saved for model interpretability")
        
        # Prepare final results
        final_results = {
            'best_model': automl.best_model_name,
            'best_score': automl.best_score,
            'task_detected': automl.task,
            'models_trained': results['models_trained'],
            'test_metrics': test_metrics,
            'model_scores': results['model_scores'],
            'training_times': results['training_times'],
            'dataset_characteristics': getattr(automl, 'dataset_characteristics_', {}),
            'feature_importances_available': automl.feature_importances_ is not None,
            'shap_available': shap_values is not None,
            'run_directory': str(run_dir)
        }
        
        # Save final results
        results_path = str(run_dir / 'final_results.json')
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"\n{'='*50}")
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print(f"{'='*50}")
        print(f"Task detected: {automl.task}")
        print(f"Best model: {automl.best_model_name}")
        print(f"Best CV score: {automl.best_score:.4f}")
        print(f"Models trained: {results['models_trained']}")
        print(f"\nTest metrics:")
        for metric, value in test_metrics.items():
            print(f"  {metric}: {value:.4f}")
        print(f"\nResults saved to: {run_dir}")
        print(f"{'='*50}")
        
        return final_results
    
    except Exception as e:
        print(f"\n{'='*50}")
        print(f"ERROR DURING TRAINING: {str(e)}")
        print(f"{'='*50}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        if use_mlflow and MLFLOW_AVAILABLE:
            try:
                mlflow.end_run()
            except:
                pass
                

def _generate_visualizations(automl: AutoML, output_dir: Path) -> None:
    """Generate visualizations for model evaluation."""
    try:
        # Create visualizations directory
        vis_dir = output_dir / 'reports' / 'figures'
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # Feature importance plot
        if hasattr(automl, 'feature_importances_') and automl.feature_importances_ is not None:
            try:
                n_features = len(automl.feature_importances_)
                feature_names = [f'feature_{i}' for i in range(n_features)]
                
                importances = pd.DataFrame({
                    'feature': feature_names,
                    'importance': automl.feature_importances_
                }).sort_values('importance', ascending=False)
                
                plt.figure(figsize=(12, 8))
                top_features = importances.head(20)  # Show top 20 features
                sns.barplot(x='importance', y='feature', data=top_features)
                plt.title('Top 20 Most Important Features')
                plt.tight_layout()
                plt.savefig(str(vis_dir / 'feature_importance.png'))
                plt.close()
                print("✓ Feature importance plot saved")
            except Exception as e:
                print(f"Warning: Could not create feature importance plot: {e}")
        
        # Model comparison plot
        if hasattr(automl, 'results_'):
            try:
                model_names = list(automl.results_.keys())
                scores = [automl.results_[name]['score'] for name in model_names]
                
                plt.figure(figsize=(12, 6))
                bars = plt.bar(model_names, scores)
                plt.xlabel('Models')
                plt.ylabel('Cross-Validation Score')
                plt.title('Model Performance Comparison')
                plt.xticks(rotation=45, ha='right')
                
                # Highlight best model
                if scores:
                    best_idx = scores.index(max(scores))
                    bars[best_idx].set_color('red')
                    bars[best_idx].set_alpha(0.8)
                
                plt.tight_layout()
                plt.savefig(str(vis_dir / 'model_comparison.png'))
                plt.close()
                print("✓ Model comparison plot saved")
            except Exception as e:
                print(f"Warning: Could not create model comparison plot: {e}")
        
        # SHAP summary plot (if available)
        try:
            import shap
            shap_values = automl.get_shap_values()
            if shap_values is not None and hasattr(automl, 'X_test'):
                plt.figure(figsize=(12, 8))
                if isinstance(shap_values, list):
                    # Multi-class classification
                    shap.summary_plot(shap_values[0], automl.X_test, show=False)
                else:
                    shap.summary_plot(shap_values, automl.X_test, show=False)
                plt.savefig(str(vis_dir / 'shap_summary.png'), dpi=300, bbox_inches='tight')
                plt.close()
                print("✓ SHAP summary plot saved")
        except Exception as e:
            print(f"Could not create SHAP plot: {e}")
        
        # Confusion matrix for classification
        if automl.task == 'classification' and hasattr(automl, 'X_test') and hasattr(automl, 'y_test'):
            try:
                from sklearn.metrics import confusion_matrix
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
                print("✓ Confusion matrix saved")
            except Exception as e:
                print(f"Warning: Could not create confusion matrix: {e}")
        
        # Actual vs Predicted for regression
        elif automl.task == 'regression' and hasattr(automl, 'X_test') and hasattr(automl, 'y_test'):
            try:
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
                print("✓ Actual vs Predicted plot saved")
            except Exception as e:
                print(f"Warning: Could not create actual vs predicted plot: {e}")
            
    except Exception as e:
        print(f"Warning: Could not generate some visualizations: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enhanced Automated Machine Learning Pipeline with advanced features."
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
        default="auto",
        choices=["classification", "regression", "auto"],
        help="Type of machine learning task. Options: 'classification', 'regression', or 'auto' (default: auto)"
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
        "--use_ensembles",
        action="store_true",
        default=True,
        help="Enable ensemble methods (voting, stacking)"
    )
    
    parser.add_argument(
        "--use_optuna",
        action="store_true", 
        default=True,
        help="Enable Optuna hyperparameter optimization"
    )
    
    parser.add_argument(
        "--optuna_trials",
        type=int,
        default=100,
        help="Number of Optuna trials for hyperparameter optimization (default: 100)"
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
        use_ensembles=args.use_ensembles,
        use_optuna=args.use_optuna,
        optuna_trials=args.optuna_trials
    )
