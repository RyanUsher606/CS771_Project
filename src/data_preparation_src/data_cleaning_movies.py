import pandas as pd
import json

# Paths to input and output files
movies_metadata_path = r"../dataset/unfiltered_dataset/movies_metadata.csv"
output_cleaned_metadata_path = r"../dataset/filtered_dataset/filtered_movies_metadata.csv"

# Load the movies metadata
movies_metadata = pd.read_csv(movies_metadata_path, low_memory=False)

# Filter necessary columns
required_columns = ['id', 'title', 'genres', 'release_date', 'poster_path']
filtered_metadata = movies_metadata[required_columns]

# Drop rows with missing values
filtered_metadata = filtered_metadata.dropna(subset=['genres', 'release_date', 'poster_path'])

# Parse genres into a multi-label format
def parse_genres(genres_str):
    try:
        genres_list = json.loads(genres_str.replace("'", '"'))
        return [genre['name'] for genre in genres_list]
    except:
        return []

filtered_metadata['genres'] = filtered_metadata['genres'].apply(parse_genres)

# Drop rows where genres is an empty list
filtered_metadata = filtered_metadata[filtered_metadata['genres'].map(len) > 0]

# Extract release decade
def extract_decade(release_date):
    try:
        year = int(release_date.split('-')[0])
        return f"{(year // 10) * 10}s"
    except:
        return None

filtered_metadata['decade'] = filtered_metadata['release_date'].apply(extract_decade)
filtered_metadata = filtered_metadata.dropna(subset=['decade'])

# Save cleaned metadata
filtered_metadata.to_csv(output_cleaned_metadata_path, index=False)

print("Data saved to:", output_cleaned_metadata_path)
