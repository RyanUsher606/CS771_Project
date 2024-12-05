import os
import requests
import pandas as pd

# TMDB API settings
BASE_URL = "https://image.tmdb.org/t/p/w500" 
API_KEY = "6246a2551e31257416ba93d40906c67a"  #TMDB API key

# Paths
metadata_path = r"../dataset/filtered_dataset/filtered_movies_metadata.csv"
output_folder = r"../dataset/filtered_dataset/posters"
os.makedirs(output_folder, exist_ok=True)

# Load metadata
metadata = pd.read_csv(metadata_path)

# Download posters
def download_posters(metadata, output_folder):
    for index, row in metadata.iterrows():
        poster_path = row['poster_path']
        movie_id = row['id']
        
        if not isinstance(poster_path, str) or not poster_path.strip():
            print(f"Skipping movie {movie_id}: No poster path available.")
            continue
        
        # Construct full poster URL
        poster_url = f"{BASE_URL}{poster_path}"
        output_path = os.path.join(output_folder, f"{movie_id}.jpg")
        
        try:
            # Fetch and save poster
            response = requests.get(poster_url, stream=True)
            if response.status_code == 200:
                with open(output_path, 'wb') as file:
                    for chunk in response.iter_content(1024):
                        file.write(chunk)
                print(f"Downloaded: {movie_id} -> {output_path}")
            else:
                print(f"Failed to download poster for {movie_id}: HTTP {response.status_code}")
        except Exception as e:
            print(f"Error downloading poster for {movie_id}: {e}")

# Run the download
download_posters(metadata, output_folder)
