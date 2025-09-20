# End-to-End AutoML Pipeline

This project provides a robust and scalable end-to-end Automated Machine Learning (AutoML) pipeline. It handles everything from data ingestion and preprocessing to automated model selection, hyperparameter tuning, and evaluation. The pipeline is designed for both local development and production deployment on Kubeflow.

![Project Architecture](https://i.imgur.com/example.png)  <!-- Placeholder for a future architecture diagram -->

## 🚀 Features

-   **Automated Preprocessing**: A powerful and configurable pipeline that handles missing values, categorical encoding, feature scaling, and outlier detection.
-   **Automated Model Selection**: Automatically trains and evaluates multiple models (e.g., Logistic Regression, RandomForest, XGBoost, DNNs) to find the best one for your data.
-   **Hyperparameter Tuning**: Uses Randomized Search to efficiently find the best hyperparameters for the top-performing models.
-   **Experiment Tracking**: Integrated with **MLflow** to log parameters, metrics, and model artifacts for full reproducibility and easy comparison of runs.
-   **Dual Execution Modes**:
    -   **Local Mode**: A simple `run_pipeline.py` script for quick local development and testing.
    -   **Production Mode**: A **Kubeflow Pipelines (KFP)** definition for orchestrated, scalable, and containerized runs in a production environment.
-   **Configurable**: All aspects of the pipeline, from preprocessing steps to the models and their hyperparameter search spaces, are controlled via simple YAML configuration files.

## 📁 Project Structure

```
├── config/                   # Configuration files
│   ├── automl_config.yaml
│   ├── models.yaml             # Model and hyperparameter definitions
│   └── preprocessing_config.yaml
├── data/                     # Data (managed by DVC, not in git)
├── infra/                    # Dockerfiles for Kubeflow components
├── pipelines/                # Kubeflow pipeline definitions
├── src/                      # Source code
│   ├── __init__.py
│   ├── auto_ml.py            # Core AutoML logic
│   ├── data_handling.py      # Automated data preprocessor
│   ├── data_ingestion.py     # Data download utility
│   ├── train.py              # Main training script
│   └── utils.py              # Utility functions
├── tests/                    # Unit and integration tests
├── run_pipeline.py           # Entry point for local runs
├── pipeline.yaml             # Compiled Kubeflow pipeline
└── README.md
```

## 🛠️ Getting Started

### 1. Installation

First, clone the repository and navigate into the project directory:

```bash
git clone <repository-url>
cd <project-directory>
```

It is recommended to use a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\Activate.ps1  # On Windows (PowerShell)
```

Install the required dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

To install the project in editable mode, which is recommended for development and required for running tests, run:
```bash
pip install -e .
```

### 2. Configuration

The pipeline's behavior is controlled by several YAML files in the `config/` directory:

-   `datasets.yaml`: Define your datasets here. For each dataset, specify its URL, target column, and type.
-   `models.yaml`: Configure the models to be used for classification and regression tasks, along with their hyperparameter search spaces.
-   `preprocessing_config.yaml`: Customize the automated data preprocessing steps.

### 3. Local Execution

There are two ways to run the pipeline locally: with a remote dataset defined by a URL, or with a local dataset on your machine.

**a. Running with a Remote Dataset**

To run the pipeline on a remote dataset, define it in `datasets.yaml` with a `url` key, like the `iris` example. Then, run the pipeline:

```bash
python run_pipeline.py --dataset <dataset_name>
```

For example, to run the pipeline on the `iris` dataset:

```bash
python run_pipeline.py --dataset iris
```

This script will automatically:
1.  Ingest the data from the URL specified in `datasets.yaml`.
2.  Run the training script (`src/train.py`), which handles preprocessing, model selection, and evaluation.
3.  Log the experiment to MLflow.

**b. Running with a Local Dataset**

To run the pipeline with a local dataset, you first need to download the data to your machine. For this example, we'll use the [PIRvision Fog Presence Detection dataset](https://archive.ics.uci.edu/dataset/1101/pirvision_fog_presence_detection).

1.  **Download the data:** Create a directory (e.g., `temp_data`) and download the dataset into it.
2.  **Update `datasets.yaml`:** Add a new entry for your local dataset. Instead of a `url`, provide a `local_path` to the CSV file.

    ```yaml
    pirvision:
      local_path: "temp_data/pirvision_office_dataset1.csv"
      target_column: "Label"
      dataset_type: "custom"
      experiment_name: "PIRvision Fog Detection"
    ```

3.  **Run the pipeline:**
    ```bash
    python run_pipeline.py --dataset pirvision
    ```
The pipeline will skip the download step and use your local file directly.

### 4. Prediction Example

After a pipeline run is complete, a `predict_example.py` script is created to demonstrate how to load the trained model and make predictions on new data. To run it, simply execute the script:

```bash
python predict_example.py
```

This will load the latest model, preprocess a few samples from the dataset, and print the predictions.

### 5. Experiment Tracking with MLflow

This project uses MLflow to track experiments. To view the results, launch the MLflow UI from the project root:

```bash
mlflow ui
```

This will start a local server (usually at `http://127.0.0.1:5000`) where you can browse all tracked experiments, compare runs, and view artifacts like confusion matrices and feature importance plots.

### 5. Production Deployment with Kubeflow

For production runs, the pipeline can be deployed to a Kubeflow cluster.

**a. Compile the Pipeline**

The Kubeflow pipeline is defined in `pipelines/main_pipeline.py`. To compile it into a static YAML file, run:

```bash
python pipelines/main_pipeline.py
```

This will generate `pipeline.yaml`.

**b. Run on Kubeflow**

1.  Upload the generated `pipeline.yaml` to your Kubeflow Pipelines UI.
2.  Create a new run from the uploaded pipeline.
3.  When creating the run, you will need to provide the required parameters, such as:
    -   `data_url`: The URL of your dataset (e.g., `https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data`).
    -   `target_column`: The name of the target variable.
    -   `experiment_name`: The name for the MLflow experiment.

## 🧪 Testing

To ensure the reliability of the pipeline, we have a suite of unit and integration tests. To run the tests, first install the testing dependencies:

```bash
pip install -r tests/requirements-test.txt
```

Then, run the test suite using `pytest`:

```bash
python -m pytest tests/
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps to contribute:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix (`git checkout -b feature/your-amazing-feature`).
3.  Make your changes and commit them with a clear message (`git commit -m 'Add some amazing feature'`).
4.  Push your changes to your forked repository (`git push origin feature/your-amazing-feature`).
5.  Open a Pull Request to the main repository.
