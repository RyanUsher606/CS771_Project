import os
import urllib.request
from tqdm import tqdm

def download_imdb_posters(metadata, output_folder):
    """
    Downloads movie posters from IMDb and tracks IDs to remove from the dataset if downloading fails.
    
    Parameters:
    - metadata: DataFrame containing movie metadata with 'Poster' (URL) and 'imdbId'.
    - output_folder: Path to save downloaded posters.
    
    Returns:
    - not_found: List of IMDb IDs for which poster download failed.
    """
    not_found = []
    os.makedirs(output_folder, exist_ok=True)

    for index, row in tqdm(metadata.iterrows(), total=metadata.shape[0]):
        imdb_id = row['imdbId']
        poster_url = row['Poster']
        file_path = os.path.join(output_folder, f"{imdb_id}.jpg")

        if not isinstance(poster_url, str) or not poster_url.strip():
            print(f"Skipping IMDb ID {imdb_id}: No poster URL provided.")
            not_found.append(imdb_id)
            continue

        try:
            # Attempt to download the poster
            response = urllib.request.urlopen(poster_url)
            data = response.read()
            with open(file_path, 'wb') as file:
                file.write(bytearray(data))
            print(f"Downloaded: {imdb_id} -> {file_path}")
        except Exception as e:
            # If the download fails, log the ID in not_found
            print(f"Failed to download poster for IMDb ID {imdb_id}: {e}")
            not_found.append(imdb_id)

    return not_found