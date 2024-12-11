import os
import sys
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

# Add the project root to the Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.data.movie_dataset import MovieDataset, split_metadata  # Import your custom dataset and functions
from src.config import load_config  # Assuming you have a config loader function

def test_dataloader():
    """
    Test the dataloader functionality to ensure it correctly loads batches and individual samples.
    """
    # Load configuration
    config_path = os.path.join(PROJECT_ROOT, "configs", "model_config.yaml")
    config = load_config(config_path)

    # Paths from the config
    metadata_path = config["paths"]["metadata_path"]
    processed_folder = config["paths"]["processed_posters_folder"]

    # Dataset splitting
    train_dataset, val_dataset, test_dataset, genre_list = split_metadata(
        metadata_path=metadata_path,
        processed_folder=processed_folder,
        train_split=config["data_split"]["train_split"],
        val_split=config["data_split"]["val_split"],
        test_split=config["data_split"]["test_split"],
    )

    # Create DataLoader for testing
    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Test the DataLoader
    print("Testing DataLoader with training data...")
    for images, genres in train_loader:
        print("Batch Loaded Successfully!")
        print(f"Images Shape: {images.shape}")  # Should match (batch_size, 3, 224, 224)
        print(f"Genres Shape: {genres.shape}")  # Should match (batch_size, num_genres)
        print(f"First Batch Labels: {genres[0]}")
        break  # Test a single batch and exit

    # Test fetching a single sample
    print("\nTesting single sample loading...")
    sample_image, sample_genres = train_dataset[0]
    print("Single Sample Test:")
    print(f"Sample Image Shape: {sample_image.shape}")  # Should be (3, 224, 224)
    print(f"Sample Genres: {sample_genres}")  # Multi-hot encoded vector of genres


if __name__ == "__main__":
    test_dataloader()
