import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import seaborn as sns
import matplotlib.pyplot as plt
import os

def train_model(input_path: str, target_column: str, experiment_name: str):
    """
    Loads processed data, trains a logistic regression model, and logs it with MLflow.
    """
    print(f"Loading processed data from {input_path}...")
    df = pd.read_csv(input_path)

    # Split data into features (X) and target (y)
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Set experiment name
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        print("Training a Logistic Regression model...")

        # Define model and its hyperparameters
        max_iter = 200
        model = LogisticRegression(max_iter=max_iter)
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Model trained with accuracy: {acc:.4f}")

        # Log parameters and metrics to MLflow
        print("Logging parameters and metrics to MLflow...")
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_metric("accuracy", acc)

        # Create and log confusion matrix plot
        print("Generating and logging confusion matrix...")
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 7))
        # Use sorted unique labels for consistent plot ordering
        labels = sorted(y.unique())
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')

        plot_path = "confusion_matrix.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path, "plots")
        print(f"Confusion matrix saved and logged to MLflow under 'plots/{plot_path}'.")
        plt.close()

        # Infer the model signature
        signature = infer_signature(X_train, model.predict(X_train))

        # Log the model
        print("Logging model to MLflow...")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=f"{experiment_name}-model",
            signature=signature
        )

        print("MLflow logging complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with MLflow tracking.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the processed data file.")
    parser.add_argument("--target_column", type=str, required=True, help="Name of the target column.")
    parser.add_argument("--experiment_name", type=str, required=True, help="Name of the MLflow experiment.")

    args = parser.parse_args()

    train_model(args.input_path, args.target_column, args.experiment_name)
