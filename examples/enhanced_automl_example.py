"""
Enhanced AutoML Pipeline Example
-------------------------------
This example demonstrates the enhanced AutoML pipeline with advanced features
including intelligent task detection, ensemble methods, and SHAP interpretability.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from auto_ml import AutoML, auto_train
from data_handling import AutoDataPreprocessor
from sklearn.datasets import make_classification, make_regression, load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split

# Set up plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_output_directory():
    """Create output directory for results."""
    output_dir = Path('output/enhanced_automl_demo')
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def demo_classification_with_missing_data():
    """Demonstrate classification with missing data and advanced preprocessing."""
    print("\n" + "="*60)
    print("CLASSIFICATION DEMO WITH MISSING DATA")
    print("="*60)
    
    # Create a synthetic dataset with missing values
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=3,
        n_classes=3,
        class_sep=0.8,
        random_state=42
    )
    
    # Convert to DataFrame
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Introduce missing values
    np.random.seed(42)
    missing_mask = np.random.random(X_df.shape) < 0.15  # 15% missing values
    X_df[missing_mask] = np.nan
    
    # Add some categorical features
    X_df['category_A'] = np.random.choice(['cat1', 'cat2', 'cat3', 'cat4'], size=len(X_df))
    X_df['category_B'] = np.random.choice(['type1', 'type2'], size=len(X_df))
    
    print(f"Dataset shape: {X_df.shape}")
    print(f"Missing values: {X_df.isnull().sum().sum()}")
    print(f"Target classes: {np.unique(y)}")
    
    # Initialize enhanced AutoML
    automl = AutoML(
        task='auto',  # Auto-detect task
        use_ensembles=True,
        use_optuna=True,
        optuna_trials=50,
        verbose=1
    )
    
    # Train the model
    results = automl.fit(X_df, y, test_size=0.2, cv=5)
    
    # Print results
    print(f"\nBest model: {results['best_model']}")
    print(f"Best score: {results['best_score']:.4f}")
    print(f"Models trained: {results['models_trained']}")
    
    # Evaluate on test set
    test_metrics = automl.evaluate()
    print(f"\nTest metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Get SHAP values if available
    shap_values = automl.get_shap_values()
    if shap_values is not None:
        print(f"\nSHAP values calculated for interpretability")
    
    return automl, results

def demo_regression_with_ensemble():
    """Demonstrate regression with ensemble methods."""
    print("\n" + "="*60)
    print("REGRESSION DEMO WITH ENSEMBLE METHODS")
    print("="*60)
    
    # Load diabetes dataset
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    feature_names = diabetes.feature_names
    
    X_df = pd.DataFrame(X, columns=feature_names)
    
    print(f"Dataset shape: {X_df.shape}")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
    
    # Initialize AutoML with ensemble focus
    automl = AutoML(
        task='regression',
        use_ensembles=True,
        use_optuna=True,
        optuna_trials=30,
        verbose=1
    )
    
    # Train the model
    results = automl.fit(X_df, y, test_size=0.2, cv=5)
    
    # Print results
    print(f"\nBest model: {results['best_model']}")
    print(f"Best score: {results['best_score']:.4f}")
    
    # Evaluate on test set
    test_metrics = automl.evaluate()
    print(f"\nTest metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Feature importance
    if automl.feature_importances_ is not None:
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': automl.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 5 most important features:")
        print(importance_df.head())
    
    return automl, results

def demo_advanced_preprocessing():
    """Demonstrate advanced preprocessing capabilities."""
    print("\n" + "="*60)
    print("ADVANCED PREPROCESSING DEMO")
    print("="*60)
    
    # Create a complex dataset
    X, y = make_classification(
        n_samples=500,
        n_features=15,
        n_informative=10,
        n_redundant=3,
        n_classes=2,
        random_state=42
    )
    
    # Convert to DataFrame
    feature_names = [f'numeric_{i}' for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Add various data quality issues
    np.random.seed(42)
    
    # Missing values
    missing_mask = np.random.random(X_df.shape) < 0.2
    X_df[missing_mask] = np.nan
    
    # Categorical features
    X_df['category_high_card'] = np.random.choice([f'cat_{i}' for i in range(50)], size=len(X_df))
    X_df['category_low_card'] = np.random.choice(['A', 'B', 'C'], size=len(X_df))
    
    # Outliers
    outlier_indices = np.random.choice(len(X_df), size=int(0.05 * len(X_df)), replace=False)
    X_df.iloc[outlier_indices, 0] = X_df.iloc[outlier_indices, 0] * 10
    
    print(f"Dataset shape: {X_df.shape}")
    print(f"Missing values: {X_df.isnull().sum().sum()}")
    print(f"Categorical features: {X_df.select_dtypes(include=['object']).columns.tolist()}")
    
    # Initialize preprocessor with advanced settings
    config_path = Path(__file__).parent.parent / 'config' / 'preprocessing_config.yaml'
    preprocessor = AutoDataPreprocessor(config_path=str(config_path))
    
    # Preprocess the data
    X_processed, missing_info = preprocessor.fit_transform(X_df, y)
    
    print(f"\nAfter preprocessing:")
    print(f"Shape: {X_processed.shape}")
    print(f"Data types: {X_processed.dtypes.value_counts().to_dict()}")
    
    if missing_info:
        print(f"\nMissing value handling:")
        for col, info in missing_info.items():
            if isinstance(info, dict):
                print(f"  {col}: {info['strategy']} (missing: {info['missing_count']})")
    
    return X_processed, y, preprocessor

def demo_model_comparison():
    """Compare different models and show detailed results."""
    print("\n" + "="*60)
    print("MODEL COMPARISON DEMO")
    print("="*60)
    
    # Load breast cancer dataset
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target
    feature_names = cancer.feature_names
    
    X_df = pd.DataFrame(X, columns=feature_names)
    
    print(f"Dataset: {cancer.DESCR.split('.')[0]}")
    print(f"Shape: {X_df.shape}")
    print(f"Classes: {cancer.target_names}")
    
    # Initialize AutoML
    automl = AutoML(
        task='classification',
        use_ensembles=True,
        use_optuna=True,
        optuna_trials=20,
        verbose=1
    )
    
    # Train models
    results = automl.fit(X_df, y, test_size=0.2, cv=5)
    
    # Create comparison DataFrame
    model_comparison = pd.DataFrame({
        'Model': list(results['model_scores'].keys()),
        'CV Score': list(results['model_scores'].values()),
        'Training Time (s)': list(results['training_times'].values())
    }).sort_values('CV Score', ascending=False)
    
    print(f"\nModel Comparison:")
    print(model_comparison.to_string(index=False, float_format='%.4f'))
    
    # Test set evaluation
    test_metrics = automl.evaluate()
    print(f"\nBest model ({results['best_model']}) test performance:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return automl, model_comparison

def create_visualizations(automl, output_dir):
    """Create visualizations for the AutoML results."""
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    # Feature importance plot
    if automl.feature_importances_ is not None:
        plt.figure(figsize=(12, 8))
        
        # Get top 15 features
        n_features = min(15, len(automl.feature_importances_))
        indices = np.argsort(automl.feature_importances_)[-n_features:]
        
        plt.barh(range(n_features), automl.feature_importances_[indices])
        plt.yticks(range(n_features), [f'Feature {i}' for i in indices])
        plt.xlabel('Feature Importance')
        plt.title('Top Feature Importances')
        plt.tight_layout()
        
        plt.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Feature importance plot saved")
    
    # SHAP summary plot (if available)
    try:
        import shap
        shap_values = automl.get_shap_values()
        if shap_values is not None and hasattr(automl, 'X_test'):
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, automl.X_test, show=False)
            plt.savefig(output_dir / 'shap_summary.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ SHAP summary plot saved")
    except Exception as e:
        print(f"Could not create SHAP plot: {e}")
    
    # Model comparison plot
    if hasattr(automl, 'results_'):
        model_names = list(automl.results_.keys())
        scores = [automl.results_[name]['score'] for name in model_names]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(model_names, scores)
        plt.xlabel('Models')
        plt.ylabel('Cross-Validation Score')
        plt.title('Model Performance Comparison')
        plt.xticks(rotation=45, ha='right')
        
        # Highlight best model
        best_idx = scores.index(max(scores))
        bars[best_idx].set_color('red')
        bars[best_idx].set_alpha(0.8)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Model comparison plot saved")

def main():
    """Run all demos."""
    print("Enhanced AutoML Pipeline Demo")
    print("="*60)
    
    # Create output directory
    output_dir = create_output_directory()
    print(f"Output directory: {output_dir}")
    
    try:
        # Demo 1: Classification with missing data
        automl_clf, results_clf = demo_classification_with_missing_data()
        
        # Demo 2: Regression with ensembles
        automl_reg, results_reg = demo_regression_with_ensemble()
        
        # Demo 3: Advanced preprocessing
        X_processed, y_processed, preprocessor = demo_advanced_preprocessing()
        
        # Demo 4: Model comparison
        automl_comp, comparison_df = demo_model_comparison()
        
        # Create visualizations
        create_visualizations(automl_comp, output_dir)
        
        # Save results
        comparison_df.to_csv(output_dir / 'model_comparison.csv', index=False)
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Results saved to: {output_dir}")
        print("\nKey Features Demonstrated:")
        print("✓ Intelligent task detection")
        print("✓ Advanced missing value handling")
        print("✓ Ensemble methods (voting, stacking)")
        print("✓ Optuna hyperparameter optimization")
        print("✓ SHAP model interpretability")
        print("✓ Memory-efficient preprocessing")
        print("✓ Comprehensive model comparison")
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()