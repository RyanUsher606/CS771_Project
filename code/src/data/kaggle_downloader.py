import os

def download_kaggle_dataset(api_key_path, dataset_name, download_path):
    """
    Downloads a Kaggle dataset using the Kaggle API.
    """
    # Setup Kaggle API key
    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
    kaggle_config = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_config):
        os.replace(api_key_path, kaggle_config)

    os.makedirs(download_path, exist_ok=True)
    os.system(f"kaggle datasets download -d {dataset_name} -p {download_path} --unzip")
    print(f"Dataset downloaded to {download_path}")
