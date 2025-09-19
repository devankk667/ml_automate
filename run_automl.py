#!/usr/bin/env python3
"""
AutoML Pipeline Runner

This script provides a command-line interface to run the AutoML pipeline.
It handles data loading, preprocessing, model training, and evaluation.
"""

import os
import argparse
import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Union

# Import the AutoML pipeline
from src.auto_ml import AutoML
from src.data_handling import AutoDataPreprocessor


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'automl_config.yaml')
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run AutoML Pipeline')
    
    # Required arguments
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to the input data file (CSV)')
    parser.add_argument('--target_column', type=str, required=True,
                        help='Name of the target column')
    
    # Optional arguments
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save outputs (default: outputs/)')
    parser.add_argument('--task', type=str, default='classification',
                        choices=['classification', 'regression'],
                        help='Type of machine learning task (default: classification)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to custom config file (default: config/automl_config.yaml)')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Fraction of data to use for testing (default: 0.2)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--use_mlflow', action='store_true',
                        help='Enable MLflow experiment tracking')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='MLflow experiment name (default: autolog)')
    
    return parser.parse_args()


def setup_mlflow(experiment_name: Optional[str] = None):
    """Set up MLflow for experiment tracking."""
    import mlflow
    
    if experiment_name is None:
        experiment_name = f'automl_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    
    mlflow.set_experiment(experiment_name)
    mlflow.start_run()
    
    return mlflow.active_run().info.run_id


def main():
    """Main function to run the AutoML pipeline."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up MLflow if enabled
    if args.use_mlflow:
        run_id = setup_mlflow(args.experiment_name)
        mlflow.log_params(vars(args))
        mlflow.log_artifact(args.config if args.config else 
                           os.path.join('config', 'automl_config.yaml'))
    
    try:
        # Load data
        print(f"Loading data from {args.input_path}...")
        data = pd.read_csv(args.input_path)
        
        # Separate features and target
        X = data.drop(columns=[args.target_column])
        y = data[args.target_column]
        
        # Initialize and fit the AutoML pipeline
        print("Initializing AutoML pipeline...")
        automl = AutoML(
            task=args.task,
            config=config,
            random_state=args.random_state
        )
        
        # Run the pipeline
        print("Running AutoML pipeline...")
        results = automl.fit(
            X, y,
            test_size=args.test_size,
            use_mlflow=args.use_mlflow
        )
        
        # Save the model
        model_path = output_dir / 'model'
        automl.save(model_path)
        print(f"Model saved to {model_path}")
        
        # Log results
        metrics = results.get('metrics', {})
        print("\nModel Performance:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        if args.use_mlflow:
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(model_path))
            mlflow.end_run()
        
        print("\nAutoML pipeline completed successfully!")
        
    except Exception as e:
        if args.use_mlflow and 'mlflow' in locals():
            mlflow.log_param('error', str(e))
            mlflow.end_run(status='FAILED')
        raise e


if __name__ == "__main__":
    main()
