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
import mlflow
import mlflow.sklearn
import mlflow.keras
from mlflow.models.signature import infer_signature

# Set styles
plt.style.use('seaborn')
sns.set_style('whitegrid')

# Create output directories
os.makedirs('models', exist_ok=True)
os.makedirs('reports/figures', exist_ok=True)
os.makedirs('reports/metrics', exist_ok=True)
os.makedirs('tensorboard_logs', exist_ok=True)
os.makedirs('config', exist_ok=True)

# Model configuration
MODEL_CONFIG = {
    # Classification models
    'logistic_regression': {
        'class': LogisticRegression,
        'params': {
            'max_iter': [100, 200, 500],
            'C': [0.1, 1, 10],
            'solver': ['lbfgs', 'saga']
        }
    },
    'random_forest': {
        'class': RandomForestClassifier,
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
    },
    'xgboost': {
        'class': XGBClassifier,
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.3]
        }
    },
    # Add more models as needed...
    
    # Neural Networks
    'dnn': {
        'class': 'neural_network',
        'params': {
            'layers': [
                {'units': 64, 'activation': 'relu', 'dropout': 0.2},
                {'units': 32, 'activation': 'relu', 'dropout': 0.2},
                {'units': 16, 'activation': 'relu', 'dropout': 0.1}
            ],
            'learning_rate': [0.001, 0.0001],
            'batch_size': [32, 64],
            'epochs': [50, 100]
        }
    }
}

# Data preprocessing
def preprocess_data(X, y=None, task_type='classification'):
    """Preprocess the input data."""
    # Handle missing values
    X = X.fillna(X.median())
    
    # Encode categorical variables
    if y is not None and y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

# Model building
def build_model(model_name: str, input_shape: tuple, num_classes: int = None):
    """Build a model based on the given name and parameters."""
    config = MODEL_CONFIG.get(model_name, MODEL_CONFIG['logistic_regression'])
    
    if model_name == 'dnn':
        return build_neural_network(input_shape, num_classes, config['params'])
    
    # For scikit-learn models
    model_class = config['class']
    return model_class()

def build_neural_network(input_shape: tuple, num_classes: int, params: dict):
    """Build a neural network model."""
    model = Sequential()
    
    # Input layer
    model.add(Input(shape=input_shape))
    
    # Hidden layers
    for layer_config in params.get('layers', []):
        model.add(Dense(layer_config['units'], activation=layer_config['activation']))
        if 'dropout' in layer_config:
            model.add(Dropout(layer_config['dropout']))
        model.add(BatchNormalization())
    
    # Output layer
    output_activation = 'softmax' if num_classes > 2 else 'sigmoid'
    output_units = num_classes if num_classes > 2 else 1
    model.add(Dense(output_units, activation=output_activation))
    
    # Compile model
    optimizer = Adam(learning_rate=params.get('learning_rate', 0.001))
    loss = 'sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy'
    metrics = ['accuracy']
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model

def plot_learning_curve(estimator, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 5), model_type='sklearn'):
    """Generate a learning curve plot for the model."""
    if model_type == 'sklearn':
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=-1, 
            train_sizes=train_sizes, random_state=42
        )
        
        plt.figure(figsize=(12, 6))
        plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training score')
        plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label='Cross-validation score')
        plt.xlabel('Training Examples')
        plt.ylabel('Score')
        plt.title('Learning Curve')
        plt.legend(loc='best')
        return plt
    
    elif model_type == 'keras':
        # For Keras models, we'll use a custom learning curve implementation
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        train_scores = []
        val_scores = []
        
        for train_size in train_sizes:
            train_size_abs = int(train_size * len(X))
            fold_train_scores = []
            fold_val_scores = []
            
            for train_idx, val_idx in kf.split(X):
                if train_size_abs < len(train_idx):
                    train_idx = np.random.choice(train_idx, train_size_abs, replace=False)
                
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Clone and train the model
                model_clone = tf.keras.models.clone_model(estimator)
                model_clone.compile(
                    optimizer=estimator.optimizer,
                    loss=estimator.loss,
                    metrics=estimator.metrics_names
                )
                
                model_clone.fit(
                    X_train, y_train,
                    epochs=10,  # Reduced for speed
                    verbose=0,
                    validation_data=(X_val, y_val)
                )
                
                # Evaluate
                train_score = model_clone.evaluate(X_train, y_train, verbose=0)[1]
                val_score = model_clone.evaluate(X_val, y_val, verbose=0)[1]
                
                fold_train_scores.append(train_score)
                fold_val_scores.append(val_score)
            
            train_scores.append(np.mean(fold_train_scores))
            val_scores.append(np.mean(fold_val_scores))
        
        # Plot the learning curve
        plt.figure(figsize=(12, 6))
        plt.plot(train_sizes, train_scores, 'o-', label='Training score')
        plt.plot(train_sizes, val_scores, 'o-', label='Validation score')
        plt.xlabel('Training Examples')
        plt.ylabel('Score')
        plt.title('Learning Curve')
        plt.legend(loc='best')
        return plt

def plot_feature_importance(model, feature_names, model_type='sklearn'):
    """Plot feature importance for different types of models."""
    if model_type == 'sklearn':
        if hasattr(model, 'feature_importances_'):  # Tree-based models
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):  # Linear models
            importance = np.abs(model.coef_[0])
        else:
            return None
            
    elif model_type == 'keras':
        # For neural networks, we'll use permutation importance
        # This is a simplified version - in practice, you might want to use a proper implementation
        return None  # Skip for now as it requires test data
    else:
        return None
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    indices = np.argsort(importance)[::-1]
    
    # Plot
    plt.bar(range(len(importance)), importance[indices], align='center')
    plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.tight_layout()
    return plt

def save_metrics(y_true, y_pred, y_proba, output_dir):
    """Save evaluation metrics to a JSON file."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'classification_report': classification_report(y_true, y_pred, output_dict=True)
    }
    
    # Save metrics to JSON
    metrics_path = os.path.join(output_dir, 'evaluation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return metrics_path

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
        config_path: Path to custom configuration file
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
        if use_mlflow:
            mlflow.log_params({"missing_values_handled": bool(missing_info)})
            if missing_info:
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
        results = automl.fit(
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
        
        # Generate visualizations
        _generate_visualizations(automl, X_processed, y, run_dir)
        
        # Log to MLflow
        if use_mlflow:
            # Log parameters
            mlflow.log_params({
                'best_model': best_model_name,
                'model_type': model_type,
                'test_size': test_size,
                'random_state': random_state
            })
            
            # Log metrics
            mlflow.log_metrics(test_metrics)
            
            # Log model
            if model_type == 'keras':
                mlflow.keras.log_model(best_model, "model")
            else:
                mlflow.sklearn.log_model(best_model, "model")
            
            # Log artifacts
            mlflow.log_artifacts(str(run_dir / 'reports'), "reports")
            
            # Log feature importances if available
            if hasattr(automl, 'feature_importances_') and automl.feature_importances_ is not None:
                importances = pd.DataFrame({
                    'feature': preprocessor.feature_names,
                    'importance': automl.feature_importances_
                }).sort_values('importance', ascending=False)
                
                importance_path = str(run_dir / 'feature_importances.csv')
                importances.to_csv(importance_path, index=False)
                mlflow.log_artifact(importance_path)
        
        # Prepare results
        results = {
            'best_model': best_model_name,
            'model_path': model_path,
            'model_type': model_type,
            'metrics': test_metrics,
            'feature_importances': getattr(automl, 'feature_importances_', None),
            'feature_names': preprocessor.feature_names,
            'run_dir': str(run_dir)
        }
        
        # Save results to JSON
        results_path = str(run_dir / 'training_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*50}")
        print("Training completed successfully!")
        print(f"Best model: {best_model_name}")
        print(f"Test metrics: {test_metrics}")
        print(f"Results saved to: {run_dir}")
        print(f"{'='*50}")
        
        return results
    
    except Exception as e:
        print(f"\n{'='*50}")
        print(f"Error during training: {str(e)}")
        print(f"{'='*50}")
        raise
    
    finally:
        if use_mlflow:
            mlflow.end_run()

def _generate_visualizations(
    self,
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

    # Preprocess the data
    print("Preprocessing data...")
    X_processed, y_processed, preprocessor = preprocess_data(X, y, task_type)
    
    # Split data into training and testing sets
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_processed, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y if task_type == 'classification' and num_classes > 1 else None
    )
    
    # Set up MLflow if enabled
    if use_mlflow:
        mlflow.set_experiment(experiment_name)
        mlflow.start_run(run_name=f"{model_type}_{timestamp}")
    
    try:
        # Build and train the model
        print(f"Training {model_type} model...")
        
        if model_type in MODEL_CONFIG and MODEL_CONFIG[model_type]['class'] == 'neural_network':
            # Train neural network
            model = build_neural_network(
                input_shape=(X_train.shape[1],),
                num_classes=num_classes if task_type == 'classification' else 1,
                params=MODEL_CONFIG[model_type]['params']
            )
            
            # Train the model
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ModelCheckpoint(
                    os.path.join(run_dir, 'model', 'best_model.h5'),
                    save_best_only=True,
                    save_weights_only=False
                ),
                TensorBoard(log_dir=os.path.join('tensorboard_logs', f"{experiment_name}_{timestamp}"))
            ]
            
            history = model.fit(
                X_train, y_train,
                validation_split=0.2,
                epochs=MODEL_CONFIG[model_type]['params'].get('epochs', [100])[0],
                batch_size=MODEL_CONFIG[model_type]['params'].get('batch_size', [32])[0],
                callbacks=callbacks,
                verbose=1
            )
            
            # Make predictions
            if task_type == 'classification':
                y_pred = np.argmax(model.predict(X_test), axis=1)
                y_proba = model.predict_proba(X_test)
            else:
                y_pred = model.predict(X_test).flatten()
                y_proba = None
        else:
            # Train scikit-learn model
            model = build_model(model_type, X_train.shape, num_classes if task_type == 'classification' else None)
            
            # Hyperparameter tuning with GridSearchCV
            if model_type in MODEL_CONFIG and 'params' in MODEL_CONFIG[model_type]:
                print("Performing hyperparameter tuning...")
                grid_search = GridSearchCV(
                    model,
                    MODEL_CONFIG[model_type]['params'],
                    cv=5,
                    scoring='accuracy' if task_type == 'classification' else 'neg_mean_squared_error',
                    n_jobs=-1,
                    verbose=1
                )
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_
                print(f"Best parameters: {grid_search.best_params_}")
            else:
                model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        if task_type == 'classification':
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Model trained with accuracy: {accuracy:.4f}")
            
            # Classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            
            # Log metrics
            metrics = {
                'accuracy': accuracy,
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1_score': report['weighted avg']['f1-score']
            }
        else:
            # Regression metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"\nRegression Metrics:")
            print(f"MSE: {mse:.4f}")
            print(f"MAE: {mae:.4f}")
            print(f"R²: {r2:.4f}")
            
            metrics = {
                'mse': mse,
                'mae': mae,
                'r2': r2
            }
        
        # Save metrics
        metrics_path = os.path.join(run_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Log to MLflow if enabled
        if use_mlflow:
            mlflow.log_params({
                'model_type': model_type,
                'task_type': task_type,
                'test_size': test_size,
                'random_state': random_state
            })
            
            if model_type in MODEL_CONFIG and 'params' in MODEL_CONFIG[model_type] and hasattr(model, 'best_params_'):
                mlflow.log_params(model.best_params_)
            
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(metrics_path)
        
        # Generate and save plots
        plots_dir = os.path.join(run_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # 1. Confusion Matrix (for classification)
        if task_type == 'classification':
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(10, 8))
            labels = sorted(np.unique(y_test))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=labels, yticklabels=labels)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            cm_path = os.path.join(plots_dir, 'confusion_matrix.png')
            plt.savefig(cm_path, bbox_inches='tight')
            plt.close()
        
        # 2. Learning Curve
        lc_plot = plot_learning_curve(
            model, X_train, y_train, 
            model_type='keras' if model_type == 'dnn' else 'sklearn'
        )
        lc_path = os.path.join(plots_dir, 'learning_curve.png')
        lc_plot.savefig(lc_path, bbox_inches='tight')
        plt.close()
        
        # 3. Feature Importance (for tree-based models)
        if model_type != 'dnn':
            fi_plot = plot_feature_importance(model, feature_names)
            if fi_plot:
                fi_path = os.path.join(plots_dir, 'feature_importance.png')
                fi_plot.savefig(fi_path, bbox_inches='tight')
                plt.close()
        
        # 4. For regression, plot actual vs predicted
        if task_type == 'regression':
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, y_pred, alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.title('Actual vs Predicted Values')
            reg_plot_path = os.path.join(plots_dir, 'actual_vs_predicted.png')
            plt.savefig(reg_plot_path, bbox_inches='tight')
            plt.close()
        
        # Log all plots to MLflow
        if use_mlflow:
            for plot_file in os.listdir(plots_dir):
                mlflow.log_artifact(os.path.join(plots_dir, plot_file), "plots")
        
        # Save the model
        model_dir = os.path.join(run_dir, 'model')
        os.makedirs(model_dir, exist_ok=True)
        
        if model_type == 'dnn':
            # Save Keras model
            model_path = os.path.join(model_dir, 'model.h5')
            model.save(model_path)
            
            # Log to MLflow
            if use_mlflow:
                mlflow.keras.log_model(model, "model", registered_model_name=f"{experiment_name}_{model_type}")
        else:
            # Save scikit-learn model
            model_path = os.path.join(model_dir, 'model.joblib')
            joblib.dump(model, model_path)
            
            # Log to MLflow
            if use_mlflow:
                signature = infer_signature(X_train, model.predict(X_train))
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path=f"{experiment_name}-model",
                    signature=signature,
                    registered_model_name=f"{experiment_name}_{model_type}"
                )
        
        print(f"\nModel and artifacts saved to: {run_dir}")
        if use_mlflow:
            print(f"MLflow run ID: {mlflow.active_run().info.run_id}")
        
        return model, metrics
    
    
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
