import sys
import os

# Add the project root to the Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)


from src.config import load_config
from src.data.movie_poster_preprocessing import preprocess_posters
import pandas as pd

# Load configuration
config = load_config("../configs/config.yaml")


# Step 4: Preprocess posters
preprocess_posters(
    input_folder=config["paths"]["posters_folder"],
    output_folder=config["paths"]["processed_posters_folder"],
    image_size=(224, 224)  # Example resize dimensions
)