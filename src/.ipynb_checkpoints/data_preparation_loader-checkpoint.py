import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import ast

metadata_path = r"../dataset/filtered_dataset/filtered_movies_metadata.csv"
processed_posters_folder = r"../dataset/filtered_dataset/processed_posters/"

# Load metadata
metadata = pd.read_csv(metadata_path)

# Safely parse genres
metadata['genres'] = metadata['genres'].apply(ast.literal_eval)
all_genres = set(genre for genres in metadata['genres'] for genre in genres)
genre_list = sorted(list(all_genres))

# Multi-hot encode genres
metadata['genre_labels'] = metadata['genres'].apply(
    lambda x: [1 if genre in x else 0 for genre in genre_list]
)

# Dynamically extract and map decades to integers
unique_decades = sorted(metadata['decade'].dropna().unique())  # Get all unique decades
decade_mapping = {decade: idx for idx, decade in enumerate(unique_decades)}

# Map decades to labels
metadata = metadata[metadata['decade'].isin(decade_mapping.keys())]  # Keep only valid decades
metadata['decade_label'] = metadata['decade'].map(decade_mapping)

# Ensure all decade_labels are integers
metadata['decade_label'] = metadata['decade_label'].astype(int)

# Filter for rows with valid poster images
metadata['poster_available'] = metadata['id'].apply(
    lambda movie_id: os.path.exists(os.path.join(processed_posters_folder, f"{movie_id}.npy"))
)
metadata = metadata[metadata['poster_available']]

# Split the dataset into training, validation, and test sets
train_metadata, temp_metadata = train_test_split(metadata, test_size=0.3)
val_metadata, test_metadata = train_test_split(temp_metadata, test_size=0.5)

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

        # Validate decade_label
        try:
            decade_label = torch.tensor(row['decade_label'], dtype=torch.long)
        except Exception as e:
            raise ValueError(f"Invalid decade_label value: {row['decade_label']} for movie ID {movie_id}")

        # Load preprocessed image
        poster_path = os.path.join(self.processed_folder, f"{movie_id}.npy")
        try:
            image = torch.tensor(np.load(poster_path))
        except FileNotFoundError:
            raise ValueError(f"Poster image not found for movie ID {movie_id} at {poster_path}")

        return image, genre_labels, decade_label

# Create Dataset objects
train_dataset = MovieDataset(train_metadata, processed_posters_folder)
val_dataset = MovieDataset(val_metadata, processed_posters_folder)
test_dataset = MovieDataset(test_metadata, processed_posters_folder)

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Print statistics
print("Extracted Genres:", genre_list)
print("Extracted Decades:", decade_mapping)
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Testing samples: {len(test_dataset)}")
