import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

# Define paths
metadata_path = r"../dataset/filtered_dataset/filtered_movies_metadata.csv"
processed_posters_folder = r"../dataset/filtered_dataset/processed_posters/"

# Load metadata
metadata = pd.read_csv(metadata_path)

# Encode labels
# Multi-hot encode genres
genre_list = ['Action', 'Drama', 'Comedy', 'Thriller']  # Only have these for testing, add more genres for full training.
metadata['genre_labels'] = metadata['genres'].apply(
    lambda x: [1 if genre in x else 0 for genre in genre_list]
)

# Map decades to integers
decade_mapping = {'1990s': 0, '2000s': 1, '2010s': 2}
metadata['decade_label'] = metadata['decade'].map(decade_mapping)

# Define Dataset Class
class MovieDataset(Dataset):
    def __init__(self, metadata, processed_folder):
        self.metadata = metadata
        self.processed_folder = processed_folder

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        movie_id = row['id']
        genre_labels = torch.tensor(row['genre_labels'], dtype=torch.float32)
        decade_label = torch.tensor(row['decade_label'], dtype=torch.long)

        # Load preprocessed image
        poster_path = os.path.join(self.processed_folder, f"{movie_id}.npy")
        image = torch.tensor(np.load(poster_path))

        return image, genre_labels, decade_label

# Split the dataset
train_metadata = metadata.sample(frac=0.7, random_state=42)
val_metadata = metadata.drop(train_metadata.index)

# Create Dataset objects
train_dataset = MovieDataset(train_metadata, processed_posters_folder)
val_dataset = MovieDataset(val_metadata, processed_posters_folder)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
