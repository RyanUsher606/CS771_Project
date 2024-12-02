import os

def download_kaggle_dataset(api_key_path, dataset_name, download_path):
    """
    Downloads a dataset from Kaggle using the Kaggle API.

    :param api_key_path: Path to the Kaggle API key file (kaggle.json).
    :param dataset_name: Dataset identifier (e.g., 'rounakbanik/the-movies-dataset').
    :param download_path: Local path to save the downloaded dataset.
    """
    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
    if not os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")):
        os.replace(api_key_path, os.path.expanduser("~/.kaggle/kaggle.json"))

    # Set download path
    os.makedirs(download_path, exist_ok=True)

    # Download the dataset
    os.system(f"kaggle datasets download -d {dataset_name} -p {download_path} --unzip")
    print(f"Dataset downloaded to {download_path}")


api_key_path =r"C:\Users\ryanu\.kaggle\kaggle.json"  # Path to the downloaded kaggle.json file
dataset_name = "rounakbanik/the-movies-dataset"
download_path = r"../dataset/unfiltered_dataset"

download_kaggle_dataset(api_key_path, dataset_name, download_path)

print("Data saved to:", download_path)