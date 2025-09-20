"""
Utility functions for the AutoML pipeline.
"""

import os
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve, auc, 
    ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file.
                    If None, loads the default config.
    
    Returns:
        dict: Configuration dictionary.
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / 'config' / 'automl_config.yaml'
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config: Configuration dictionary.
        config_path: Path to save the configuration file.
    """
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def create_directory(path: Union[str, Path]) -> Path:
    """
    Create a directory if it doesn't exist.
    
    Args:
        path: Directory path.
    
    Returns:
        Path: Path object of the created directory.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_metrics(metrics: Dict[str, float], filepath: Union[str, Path]) -> None:
    """
    Save metrics to a JSON file.
    
    Args:
        metrics: Dictionary of metrics.
        filepath: Path to save the metrics file.
    """
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)


def load_metrics(filepath: Union[str, Path]) -> Dict[str, float]:
    """
    Load metrics from a JSON file.
    
    Args:
        filepath: Path to the metrics file.
    
    Returns:
        dict: Dictionary of metrics.
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = 'Confusion Matrix',
    cmap: str = 'Blues',
    normalize: str = 'true',
    save_path: Optional[Union[str, Path]] = None,
    **kwargs
) -> None:
    """
    Plot a confusion matrix.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        labels: List of class labels.
        title: Plot title.
        cmap: Color map.
        normalize: Normalization mode ('true', 'pred', 'all', or None).
        save_path: Path to save the plot. If None, displays the plot.
        **kwargs: Additional arguments for ConfusionMatrixDisplay.
    """
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    
    if normalize == 'true':
        fmt = '.2f'
        vmin = 0
        vmax = 1
    else:
        fmt = 'd'
        vmin = None
        vmax = None
    
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )
    
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(
        cmap=cmap,
        ax=ax,
        values_format=fmt,
        colorbar=False,
        **kwargs
    )
    
    plt.title(title, pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    pos_label: Union[int, str] = 1,
    title: str = 'ROC Curve',
    save_path: Optional[Union[str, Path]] = None,
    **kwargs
) -> None:
    """
    Plot a ROC curve.
    
    Args:
        y_true: True binary labels.
        y_score: Target scores, can be probability estimates or decision function.
        pos_label: Label of the positive class.
        title: Plot title.
        save_path: Path to save the plot. If None, displays the plot.
        **kwargs: Additional arguments for RocCurveDisplay.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)
    
    disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, **kwargs)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, name=f'ROC curve (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_feature_importance(
    feature_importances: np.ndarray,
    feature_names: List[str],
    top_n: int = 20,
    title: str = 'Feature Importance',
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[Union[str, Path]] = None,
    **kwargs
) -> None:
    """
    Plot feature importances.
    
    Args:
        feature_importances: Array of feature importances.
        feature_names: List of feature names.
        top_n: Number of top features to display.
        title: Plot title.
        figsize: Figure size (width, height).
        save_path: Path to save the plot. If None, displays the plot.
        **kwargs: Additional arguments for barplot.
    """
    # Sort feature importances in descending order
    indices = np.argsort(feature_importances)[::-1]
    
    # Take top N features
    if top_n is not None:
        indices = indices[:top_n]
    
    # Get top feature names and importances
    top_features = [feature_names[i] for i in indices]
    top_importances = feature_importances[indices]
    
    # Create a horizontal bar plot
    plt.figure(figsize=figsize)
    sns.barplot(x=top_importances, y=top_features, **kwargs)
    
    plt.title(title)
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_learning_curve(
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    train_sizes: np.ndarray,
    title: str = 'Learning Curve',
    ylim: Optional[Tuple[float, float]] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Union[str, Path]] = None,
    **kwargs
) -> None:
    """
    Plot a learning curve.
    
    Args:
        train_scores: Training scores for each training set size.
        val_scores: Validation scores for each training set size.
        train_sizes: Training set sizes.
        title: Plot title.
        ylim: Tuple of (min, max) for the y-axis.
        figsize: Figure size (width, height).
        save_path: Path to save the plot. If None, displays the plot.
        **kwargs: Additional arguments for plot.
    """
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=figsize)
    
    plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
    plt.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.1,
        color='r'
    )
    
    plt.plot(train_sizes, val_mean, 'o-', color='g', label='Cross-validation score')
    plt.fill_between(
        train_sizes,
        val_mean - val_std,
        val_mean + val_std,
        alpha=0.1,
        color='g'
    )
    
    plt.title(title)
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.grid(True)
    
    if ylim is not None:
        plt.ylim(ylim)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def get_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
    average: str = 'macro'
) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_score: Predicted probabilities or decision function.
        average: Averaging strategy for multiclass classification.
    
    Returns:
        dict: Dictionary of classification metrics.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0),
    }
    
    if y_score is not None:
        if len(np.unique(y_true)) == 2:
            # For binary classification, y_score is likely (n, 2), take prob of positive class
            if y_score.ndim == 2 and y_score.shape[1] == 2:
                metrics['roc_auc'] = roc_auc_score(y_true, y_score[:, 1])
            else: # Already 1D
                metrics['roc_auc'] = roc_auc_score(y_true, y_score)
        else: # Multiclass
             metrics['roc_auc'] = roc_auc_score(y_true, y_score, multi_class='ovr')

    return metrics


def get_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate regression metrics.
    
    Args:
        y_true: True values.
        y_pred: Predicted values.
    
    Returns:
        dict: Dictionary of regression metrics.
    """
    return {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }


def save_yaml(data: Dict[str, Any], filepath: Union[str, Path]):
    """
    Save data to a YAML file.
    
    Args:
        data: Data to save.
        filepath: Path to save the YAML file.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


def save_model(
    model: Any,
    filepath: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save a trained model with metadata.
    
    Args:
        model: Trained model object.
        filepath: Path to save the model.
        metadata: Additional metadata to save with the model.
    """
    import joblib
    
    model_dir = Path(filepath).parent
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the model
    joblib.dump(model, filepath)
    
    # Save metadata if provided
    if metadata is not None:
        metadata_path = model_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)


def load_model(
    filepath: Union[str, Path]
) -> Tuple[Any, Dict[str, Any]]:
    """
    Load a trained model with metadata.
    
    Args:
        filepath: Path to the saved model.
    
    Returns:
        tuple: (model, metadata)
    """
    import joblib
    
    model = joblib.load(filepath)
    
    # Load metadata if it exists
    model_dir = Path(filepath).parent
    metadata_path = model_dir / 'metadata.json'
    
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    return model, metadata
