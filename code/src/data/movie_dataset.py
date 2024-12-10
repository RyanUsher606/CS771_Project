import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def movie_dataset(metadata_path):
    """
    Extract unique genres from the metadata file.
    Args:
        metadata_path (str): Path to the metadata CSV file.
    Returns:
        genre_list (list): List of unique genres.
    """
    # Load metadata
    metadata = pd.read_csv(metadata_path)

    # Extract unique genres from the 'Genre' column
    all_genres = set()
    metadata['Genre'].apply(lambda x: all_genres.update(x.split('|')))  # Split genres by pipe (|)
    genre_list = sorted(list(all_genres))  # Sort genres alphabetically

    return genre_list


def split_metadata(metadata_path, processed_folder, train_split=0.7, val_split=0.05, test_split=0.25, random_state=42):
    """
    Split metadata into training, validation, and testing sets with a 70/25/5 ratio.
    Args:
        metadata_path (str): Path to the metadata CSV file.
        processed_folder (str): Path to the folder with processed posters.
        train_split (float): Proportion of the dataset to use for training.
        val_split (float): Proportion of the dataset to use for validation.
        test_split (float): Proportion of the dataset to use for testing.
        random_state (int): Seed for reproducibility.
    Returns:
        train_dataset, val_dataset, test_dataset (Dataset): Split datasets.
        genre_list (list): List of unique genres.
    """
    assert train_split + val_split + test_split == 1.0, "Splits must add up to 1.0"

    # Load metadata
    metadata = pd.read_csv(metadata_path)

    # Extract unique genres
    genre_list = movie_dataset(metadata_path)

    # Split into train, val, test datasets
    train_metadata, temp_metadata = train_test_split(
        metadata, test_size=(1 - train_split), random_state=random_state
    )
    val_metadata, test_metadata = train_test_split(
        temp_metadata,
        test_size=(test_split / (test_split + val_split)),
        random_state=random_state,
    )

    # Create Dataset objects
    train_dataset = MovieDataset(train_metadata, processed_folder, genre_list)
    val_dataset = MovieDataset(val_metadata, processed_folder, genre_list)
    test_dataset = MovieDataset(test_metadata, processed_folder, genre_list)

    return train_dataset, val_dataset, test_dataset, genre_list


class MovieDataset(Dataset):
    def __init__(self, metadata, processed_folder, genre_list):
        """
        Initialize the MovieDataset.
        Args:
            metadata (pd.DataFrame): DataFrame containing movie metadata.
            processed_folder (str): Path to the folder with preprocessed poster .npy files.
            genre_list (list): List of all genres for multi-label classification.
        """
        self.metadata = metadata
        self.processed_folder = processed_folder
        self.genre_list = genre_list

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Extract data from the DataFrame
        row = self.metadata.iloc[idx]
        movie_id = row["imdbId"]
        poster_path = os.path.join(self.processed_folder, f"{movie_id}.npy")

        # Multi-label genres: split the 'Genre' column by '|' and create a multi-hot encoded vector
        genres = [1 if genre in row["Genre"].split('|') else 0 for genre in self.genre_list]

        # Load preprocessed poster image
        try:
            image = torch.tensor(np.load(poster_path), dtype=torch.float32)
            genres = torch.tensor(genres, dtype=torch.float32)
        except FileNotFoundError:
            raise ValueError(f"Poster file not found for ID: {movie_id}")

        return image, genres
