# AutoML Pipeline

This module provides an automated machine learning pipeline that simplifies the process of training and evaluating machine learning models. It includes automated data preprocessing, feature engineering, model selection, hyperparameter tuning, and model evaluation.

## Features

- **Automated Data Preprocessing**:
  - Missing value imputation
  - Categorical encoding
  - Feature scaling
  - Outlier detection and handling
  - Feature selection

- **Model Training & Tuning**:
  - Multiple algorithms for classification and regression
  - Hyperparameter optimization using grid/random search
  - Cross-validation
  - Early stopping

- **Model Evaluation**:
  - Comprehensive metrics for classification and regression
  - Feature importance analysis
  - SHAP values for model interpretation
  - Partial dependence plots

- **Experiment Tracking**:
  - MLflow integration for experiment tracking
  - TensorBoard logging
  - Model versioning

- **Deployment Ready**:
  - Model serialization
  - API serving with FastAPI
  - Monitoring and logging

## Installation

1. Create a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements_automl.txt
   ```

## Quick Start

### 1. Prepare Your Data

Create a CSV file with your dataset. The last column should be the target variable, or you can specify the target column name.

### 2. Run the AutoML Pipeline

```python
from auto_ml import AutoML
import pandas as pd

# Load your data
data = pd.read_csv('your_data.csv')
X = data.drop('target', axis=1)  # Replace 'target' with your target column name
y = data['target']

# Initialize the AutoML pipeline
automl = AutoML(task='classification')  # or 'regression'

# Fit the model
results = automl.fit(X, y)

# Make predictions
predictions = automl.predict(X_test)

# Evaluate the model
metrics = automl.evaluate(X_test, y_test)
print(metrics)
```

### 3. Using the Command Line

You can also run the pipeline from the command line:

```bash
python -m src.train \
    --input_path data/your_data.csv \
    --output_dir outputs/ \
    --target_column target \
    --task classification  # or 'regression'
```

## Configuration

The pipeline can be configured using a YAML file. See `config/automl_config.yaml` for all available options.

## Model Interpretation

To interpret the model's predictions:

```python
# Get feature importances
importances = automl.get_feature_importances()

# Generate SHAP values (for tree-based models and neural networks)
shap_values = automl.explain(X_sample)

# Plot feature importance
automl.plot_feature_importance()

# Generate partial dependence plots
automl.plot_partial_dependence(features=['feature1', 'feature2'])
```

## Deployment

To deploy the trained model as a REST API:

```bash
python -m src.serve \
    --model_path outputs/model/ \
    --host 0.0.0.0 \
    --port 8000
```

## Monitoring

The pipeline includes monitoring capabilities through MLflow. To start the MLflow UI:

```bash
mlflow ui --backend-store-uri file:./mlruns
```

## Contributing

Contributions are welcome! Please read our [contributing guidelines](CONTRIBUTING.md) before submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
