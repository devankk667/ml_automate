# AutoML Pipeline Project

An end-to-end automated machine learning pipeline that handles data preprocessing, model training, evaluation, and deployment. This project is designed to be flexible, scalable, and easy to use for both classification and regression tasks.

## 🚀 Features

- **Automated Data Preprocessing**: Handles missing values, outliers, and feature engineering
- **Model Selection & Tuning**: Automatically selects and tunes the best model for your data
- **Comprehensive Evaluation**: Provides detailed metrics and visualizations
- **Easy Integration**: Simple CLI interface for running the pipeline
- **Reproducible Results**: Configurable random seeds and version control

## 📁 Project Structure

```
ml_automate/
├── configs/                # Configuration files
│   ├── preprocessing.yaml  # Preprocessing configurations
│   └── model_config.yaml   # Model training configurations
├── data/                   # Data files
│   ├── raw/               # Raw data files
│   └── processed/         # Processed data files
├── models/                # Saved models and artifacts
├── notebooks/             # Jupyter notebooks for exploration
├── src/                   # Source code
│   ├── data/             # Data handling and preprocessing
│   ├── models/           # Model definitions and training
│   ├── visualization/    # Plotting and visualization utilities
│   └── pipeline.py       # Main pipeline implementation
└── tests/                # Test files
    ├── unit/            # Unit tests
    └── integration/     # Integration tests
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ml_automate
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\Activate  # On Windows
   # or
   source .venv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## 🚀 Quick Start

### Running the Pipeline

1. **Prepare your data**
   - Place your dataset in `data/raw/`
   - Ensure your target column is specified or can be auto-detected

2. **Run the pipeline**
   ```bash
   python -m src.pipeline --input_path data/raw/your_data.csv --target_column target
   ```

### Example with Sample Data

```bash
# Generate sample Iris dataset
python -c "
from sklearn.datasets import load_iris
import pandas as pd
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df.to_csv('data/raw/iris.csv', index=False)
"

# Run the pipeline
python -m src.pipeline --input_path data/raw/iris.csv --target_column target
```

## 📊 Outputs

The pipeline generates the following outputs in the `output/` directory:

- `cleaning_report.json`: Summary of data cleaning steps
- `results.json`: Model evaluation metrics
- `predictions.csv`: Model predictions on test set
- `plots/`: Visualizations including:
  - `target_distribution.png`
  - `correlation_heatmap.png`
  - `pair_plot.png`

## 🤖 Available Models

The pipeline automatically selects from the following models:

- **Classification**:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - LightGBM
  - Neural Networks

- **Regression**:
  - Linear Regression
  - Random Forest Regressor
  - XGBoost Regressor
  - Neural Networks

## ⚙️ Configuration

Customize the pipeline behavior by creating/modifying files in the `configs/` directory:

- `preprocessing.yaml`: Data preprocessing parameters
- `model_config.yaml`: Model training parameters

## 🧪 Testing

Run tests to ensure everything is working correctly:

```bash
# Run unit tests
python -m pytest tests/unit/

# Run integration tests
python -m pytest tests/integration/
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with ❤️ using Python and popular ML libraries
- Inspired by best practices in MLOps and AutoML
This single command will automatically execute the data ingestion, data preparation, and model training steps in sequence.

### 2. How to Add a New Dataset

To add a new dataset to the project, you only need to add a new entry to the `datasets.yaml` file.

For example, to add a new dataset called `my_dataset`, you would add the following to `datasets.yaml`:
```yaml
my_dataset:
  url: "http://path/to/your/dataset.csv"
  target_column: "name_of_the_target_column"
  dataset_type: "iris" # or "wine", or a new type you define in data_handling.py
  # ... any other necessary metadata
```
Once added, you can run the entire pipeline with `python run_pipeline.py --dataset my_dataset`.

## Experiment Tracking with MLflow

This project uses **MLflow** to track experiments. The training script is instrumented to automatically log:
-   Hyperparameters (e.g., `max_iter`)
-   Performance metrics (e.g., `accuracy`)
-   **A confusion matrix plot** as a visual artifact.
-   The trained model as a versioned artifact.

To view the MLflow UI, you can run `mlflow ui` from the project root. This will start a local server to browse all the tracked experiments.

## Advanced Usage: Orchestration with Kubeflow

The `pipelines/main_pipeline.py` script uses the Kubeflow Pipelines (KFP) SDK to define a generic, end-to-end workflow that can be run in a production environment.

To compile the pipeline definition, run:
`python pipelines/main_pipeline.py`

This generates a `pipeline.yaml` file. This file can be uploaded to the Kubeflow UI. When creating a run from the UI, you would provide the parameters for the specific dataset, for example:
- **data_url**: `https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data`
- **dataset_type**: `iris`
- **target_column**: `species`
- **experiment_name**: `Iris Classification`
