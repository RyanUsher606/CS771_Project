import sys
import os

# Add the project root to the Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import torch
import numpy as np
import pandas as pd
from src.models.vgg_model import VGGMultiLabel  # Import your model
from src.config import load_config  # Import your configuration loader

# Add the project root to the Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

def get_input_size_from_npy(processed_folder):
    """
    Dynamically determine the input size from the processed posters stored as .npy files.
    Assumes all .npy files have the same dimensions.
    """
    for filename in os.listdir(processed_folder):
        if filename.endswith(".npy"):
            poster_path = os.path.join(processed_folder, filename)
            poster = np.load(poster_path)
            return poster.shape  # Shape should be (channels, height, width)
    raise FileNotFoundError("No .npy poster files found in the processed folder.")

def main():
    # Load configuration
    config_path = os.path.join(PROJECT_ROOT, "configs", "model_config.yaml")
    config = load_config(config_path)

    # Extract paths and parameters from the configuration
    metadata_path = config["paths"]["metadata_path"]
    processed_folder = config["paths"]["processed_posters_folder"]

    # Load metadata and calculate the number of unique genres
    metadata = pd.read_csv(metadata_path)
    all_genres = set()
    metadata['Genre'].apply(lambda x: all_genres.update(x.split('|')))
    num_genres = len(all_genres)

    # Dynamically determine input size from the .npy posters
    input_size = get_input_size_from_npy(processed_folder)  # (channels, height, width)

    # Initialize the model
    model = VGGMultiLabel(num_classes=num_genres)  # Use dynamic genre count

    # Test parameters
    batch_size = config["training"]["batch_size"]

    # Generate random test input
    inputs = torch.randn(batch_size, *input_size)  # Random images matching the determined size

    # Set model to evaluation mode
    model.eval()

    # Perform a forward pass and print outputs
    with torch.no_grad():
        genre_logits = model(inputs)

    # Check output shapes
    print(f"Input shape: {inputs.shape}")
    print(f"Genre logits shape: {genre_logits.shape}")  # Expected: [batch_size, num_genres]

    # Additional test: Check intermediate outputs
    try:
        with torch.no_grad():
            # Extract backbone feature maps from VGG
            backbone_features = model.features(inputs)
            print(f"Backbone feature map shape: {backbone_features.shape}")

            # Flatten the features and test the classifier
            flattened_features = torch.flatten(backbone_features, start_dim=1)
            print(f"Flattened feature map shape: {flattened_features.shape}")

            classifier_output = model.classifier(flattened_features)
            print(f"Classifier output shape: {classifier_output.shape}")
    except Exception as e:
        print(f"Error during intermediate shape checks: {e}")

if __name__ == "__main__":
    main()
