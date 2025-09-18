import argparse
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

def prepare_data(input_path: str, output_path: str, dataset_type: str):
    """
    Loads and prepares raw data based on the dataset type.
    """
    print(f"Loading {dataset_type} data from {input_path}...")

    if dataset_type == 'iris':
        column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
        df = pd.read_csv(input_path, header=None, names=column_names)
        print("Data loaded successfully. Processing...")

        # Encode the categorical species column
        label_encoder = LabelEncoder()
        df['species'] = label_encoder.fit_transform(df['species'])
        print("Species column encoded.")

    elif dataset_type == 'wine':
        column_names = [
            'class', 'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash',
            'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
            'proanthocyanins', 'color_intensity', 'hue',
            'od280_od315_of_diluted_wines', 'proline'
        ]
        df = pd.read_csv(input_path, header=None, names=column_names)
        print("Data loaded successfully. Processing...")

        # The target 'class' is 1-indexed. Make it 0-indexed.
        df['class'] = df['class'] - 1
        print("Class column adjusted to be 0-indexed.")

    else:
        raise ValueError("Unsupported dataset type. Please choose 'iris' or 'wine'.")

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save the processed data
    df.to_csv(output_path, index=False)

    print(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for training.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the raw data file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the processed data.")
    parser.add_argument("--dataset_type", type=str, required=True, choices=['iris', 'wine'], help="The type of the dataset to prepare.")

    args = parser.parse_args()

    prepare_data(args.input_path, args.output_path, args.dataset_type)
