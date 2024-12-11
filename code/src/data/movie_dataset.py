import sys
import os

# Add the project root to the Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)

import torch
import numpy as np
import pandas as pd
from src.config import load_config  # Import your configuration loader
import ast
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

def get_genre_list(metadata_path):
    """
    Extract unique genres from the metadata file.
    """
    metadata = pd.read_csv(metadata_path)
    metadata['genres'] = metadata['genres'].apply(ast.literal_eval)

    all_genres = set()
    for g_list in metadata['genres']:
        all_genres.update(g_list)

    genre_list = sorted(list(all_genres))
    return genre_list

def split_metadata_and_save(metadata_path, processed_folder, 
                            train_path="train.csv", test_path="test.csv",
                            train_split=0.8, test_split=0.2, random_state=42):
    """
    Split metadata into training and testing sets, and save them as separate CSV files.
    """
    # Load metadata
    metadata = pd.read_csv(metadata_path)
    metadata['genres'] = metadata['genres'].apply(ast.literal_eval)

    # Filter metadata to only include rows that have corresponding .npy files
    def has_npy_file(row):
        movie_id = row['id']
        poster_npy_path = os.path.join(processed_folder, f"{movie_id}.npy")
        return os.path.exists(poster_npy_path)

    metadata = metadata[metadata.apply(has_npy_file, axis=1)]

    # Split into train and test datasets
    train_metadata, test_metadata = train_test_split(
        metadata, test_size=test_split, random_state=random_state
    )

    # Save each split to a separate CSV
    train_metadata.to_csv(train_path, index=False)
    test_metadata.to_csv(test_path, index=False)

    # Also return the genre list
    genre_list = get_genre_list(metadata_path)
    return genre_list

class MovieDataset(Dataset):
    def __init__(self, csv_path, processed_folder, genre_list, return_id=False):
        """
        Initialize the MovieDataset from a CSV file containing 'id', 'genres', and 'poster_path' columns.
        """
        self.metadata = pd.read_csv(csv_path)
        self.metadata['genres'] = self.metadata['genres'].apply(ast.literal_eval)

        self.processed_folder = processed_folder
        self.genre_list = genre_list
        self.return_id = return_id
        self.genre_to_idx = {g: i for i, g in enumerate(self.genre_list)}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        movie_id = row["id"]
        poster_npy_path = os.path.join(self.processed_folder, f"{movie_id}.npy")

        # One-hot encode the genres
        genres_list = row['genres']
        genre_vector = np.zeros(len(self.genre_list), dtype=np.float32)
        for g in genres_list:
            if g in self.genre_to_idx:
                genre_vector[self.genre_to_idx[g]] = 1.0

        if not os.path.exists(poster_npy_path):
            raise FileNotFoundError(f"Poster file not found for ID: {movie_id}")

        image = np.load(poster_npy_path).astype(np.float32)
        image_tensor = torch.from_numpy(image)
        genres_tensor = torch.from_numpy(genre_vector)

        if self.return_id:
            return image_tensor, genres_tensor, movie_id
        else:
            return image_tensor, genres_tensor

def main():
    config_path = os.path.join(PROJECT_ROOT, "configs", "model_config.yaml")
    config = load_config(config_path)

    # Split metadata into train and test only
    genre_list = split_metadata_and_save(
        metadata_path=config["paths"]["metadata_path"],
        processed_folder=config["paths"]["processed_posters_folder"],
        train_path=os.path.abspath(os.path.join(config["paths"]["train_testing"], "train.csv")),
        test_path=os.path.abspath(os.path.join(config["paths"]["train_testing"], "test.csv")),
        train_split=config['data_split']['train_split'],
        test_split=1 - config['data_split']['train_split'],
    )
    
    print("Splitting is now complete")

if __name__ == "__main__":
    main()
