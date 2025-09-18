import argparse
import requests
import os

def download_data(url: str, output_path: str):
    """
    Downloads data from a URL and saves it to a local path.
    """
    try:
        print(f"Downloading data from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Successfully downloaded data to {output_path}")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading data: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download data from a URL.")
    parser.add_argument("--url", type=str, required=True, help="URL of the data to download.")
    parser.add_argument("--output_path", type=str, required=True, help="Local path to save the downloaded data.")

    args = parser.parse_args()

    download_data(args.url, args.output_path)
