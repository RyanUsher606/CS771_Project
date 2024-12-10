import pandas as pd
import glob
import os

def filter_metadata_with_posters(input_csv, posters_folder, output_csv):
    """
    Filters movie metadata to include only rows with corresponding poster images.

    Parameters:
    - input_csv: Path to the input metadata CSV file.
    - posters_folder: Path to the folder containing poster images.
    - output_csv: Path to save the filtered metadata with image paths.
    """
    # Load the metadata with fallback encoding
    try:
        df = pd.read_csv(input_csv, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(input_csv, encoding='latin1')

    # Ensure imdbId is treated as a string to match file names
    df['imdbId'] = df['imdbId'].astype(str)

    # Prepare lists to hold filtered data
    image_paths = []
    imdb_ids = []
    genres = []
    titles = []

    # Iterate through poster files
    for file in glob.glob(os.path.join(posters_folder, "*.jpg")):
        # Extract IMDb ID from the file name
        img_id = os.path.basename(file).split('.')[0]

        # Check if the IMDb ID exists in the DataFrame
        if img_id in df['imdbId'].values:
            # Retrieve metadata for the matched IMDb ID
            row = df[df['imdbId'] == img_id].iloc[0]
            image_paths.append(file)
            imdb_ids.append(img_id)
            genres.append(row['Genre'])
            titles.append(row['Title'])

    # Create a filtered DataFrame
    filtered_df = pd.DataFrame({
        'imdbId': imdb_ids,
        'Genre': genres,
        'Title': titles,
        'Image_Paths': image_paths
    })

    # Save the filtered metadata to a new CSV file
    filtered_df.to_csv(output_csv, index=False)
    print(f"Filtered metadata with poster paths saved to {output_csv}")
