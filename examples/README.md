# AutoML Pipeline Examples

This directory contains example scripts demonstrating how to use the AutoML pipeline.

## Prerequisites

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Examples

### Basic Usage

Run the AutoML pipeline on a sample dataset:

```bash
python run_automl_pipeline.py --dataset iris
```

### Available Datasets

The example script supports the following built-in datasets:

- `iris`: Fisher's Iris dataset (classification)
- `wine`: Wine recognition dataset (classification)
- `breast_cancer`: Wisconsin breast cancer dataset (classification)
- `boston`: Boston housing dataset (regression)
- `diabetes`: Diabetes dataset (regression)

### Custom Configuration

You can specify custom configuration files for the pipeline:

```bash
python run_automl_pipeline.py \
    --dataset wine \
    --config ../config/automl_config.yaml \
    --preprocessing_config ../config/preprocessing_config.yaml \
    --output_dir my_outputs
```

### Command Line Options

```
usage: run_automl_pipeline.py [-h] [--dataset DATASET] [--output_dir OUTPUT_DIR]
                             [--config CONFIG] [--preprocessing_config PREPROCESSING_CONFIG]
                             [--test_size TEST_SIZE] [--random_state RANDOM_STATE]
                             [--n_jobs N_JOBS] [--verbose VERBOSE]

Run AutoML Pipeline on a sample dataset

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Name of the dataset to use (default: iris)
  --output_dir OUTPUT_DIR
                        Directory to save outputs (default: outputs/)
  --config CONFIG       Path to config file (default: config/automl_config.yaml)
  --preprocessing_config PREPROCESSING_CONFIG
                        Path to preprocessing config file (default:
                        config/preprocessing_config.yaml)
  --test_size TEST_SIZE
                        Fraction of data to use for testing (default: 0.2)
  --random_state RANDOM_STATE
                        Random seed for reproducibility (default: 42)
  --n_jobs N_JOBS       Number of jobs to run in parallel (default: -1, use all
                        cores)
  --verbose VERBOSE     Verbosity level (0: silent, 1: info, 2: debug)
```

## Outputs

The script will create the following outputs in the specified output directory:

- `pipeline/`: Saved pipeline including the trained model and preprocessor
- `metrics.json`: Evaluation metrics on the test set
- `feature_importances.png`: Plot of feature importances (if available)
- `confusion_matrix.png`: Confusion matrix (for classification)
- `roc_curve.png`: ROC curve (for binary classification)

## Custom Datasets

To use your own dataset, modify the script to load your data as a pandas DataFrame and pass it to the pipeline:

```python
import pandas as pd
from src.pipeline import create_pipeline

# Load your data
df = pd.read_csv('your_data.csv')

# Split into features and target
X = df.drop(columns=['target_column'])
y = df['target_column']

# Create and train the pipeline
pipeline = create_pipeline(task='classification')  # or 'regression'
pipeline.fit(X, y)

# Make predictions
y_pred = pipeline.predict(X)

# Evaluate the model
metrics = pipeline.evaluate(X, y)
print(metrics)
```

## Next Steps

- Experiment with different preprocessing configurations
- Try different hyperparameter search spaces
- Extend the pipeline with custom transformers or models
- Deploy the trained model as a web service
