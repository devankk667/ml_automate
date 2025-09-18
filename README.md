# End-to-End MLOps Demo Pipeline

This project demonstrates a sample end-to-end MLOps pipeline, showcasing key principles like componentization, automation, experiment tracking, and orchestration. The goal is to build a system that can automatically ingest data, prepare it, train a model, and track the entire process.

This implementation was built to address a high-level architectural design for a large-scale, automated ML system.

## Project Structure

- `/src/`: Contains the core Python source code for each step of the pipeline.
- `/infra/`: Holds `Dockerfile`s for containerizing each pipeline component.
- `/pipelines/`: Contains the Kubeflow Pipelines (KFP) definition script.
- `datasets.yaml`: A central configuration file to register new datasets.
- `run_pipeline.py`: The master script to run the entire pipeline for a configured dataset.
- `.gitignore`: Specifies files to be ignored by version control.

## Simplified Workflow

This project is designed to be data-centric. The entire pipeline for any given dataset can be executed with a single command, driven by a central configuration file.

### 1. How to Run a Pipeline

To run the pipeline for a pre-configured dataset, use the `run_pipeline.py` script with the `--dataset` flag.

**Run the Iris dataset pipeline:**
```bash
python run_pipeline.py --dataset iris
```

**Run the Wine dataset pipeline:**
```bash
python run_pipeline.py --dataset wine
```
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
