import argparse
import yaml
import subprocess
import sys
import os

def run_command(command: list):
    """Runs a command and checks for errors."""
    print(f"\nRunning command: {' '.join(command)}")
    result = subprocess.run(command, check=False, text=True)
    if result.returncode != 0:
        print(f"Error: Command failed with exit code {result.returncode}")
        sys.exit(1)

def main(dataset_name: str):
    """
    Runs the full MLOps pipeline for a given dataset.
    """
    # Load dataset configurations
    with open('datasets.yaml', 'r') as f:
        configs = yaml.safe_load(f)

    if dataset_name not in configs:
        print(f"Error: Dataset '{dataset_name}' not found in datasets.yaml")
        sys.exit(1)

    config = configs[dataset_name]

    # Define file paths
    raw_data_path = os.path.join("data", f"raw_{dataset_name}.csv")

    # --- Step 1: Data Ingestion ---
    ingestion_command = [
        sys.executable, "src/data_ingestion.py",
        "--url", config['url'],
        "--output_path", raw_data_path
    ]
    run_command(ingestion_command)

    # --- Step 2: Model Training (with integrated preprocessing) ---
    training_command = [
        sys.executable, "src/train.py",
        "--input_path", raw_data_path,
        "--target_column", config['target_column'],
        "--experiment_name", config.get('experiment_name', f"{dataset_name.title()} Classification")
    ]
    run_command(training_command)

    print(f"\nPipeline for dataset '{dataset_name}' completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master runner for the MLOps pipeline.")
    parser.add_argument("--dataset", type=str, required=True, help="The name of the dataset to process (e.g., 'iris', 'wine'). Must match an entry in datasets.yaml.")

    args = parser.parse_args()
    main(args.dataset)