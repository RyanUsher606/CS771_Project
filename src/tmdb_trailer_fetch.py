import os
import requests
import random
import time
import pandas as pd
import yt_dlp
from googleapiclient.discovery import build

# TMDB and YouTube API keys
TMDB_API_KEY = '6246a2551e31257416ba93d40906c67a'
YOUTUBE_API_KEY = 'AIzaSyAou2aUKB1zhP5s4qjMBMKZtkG7B9YVqGU'

# Initialize YouTube API client
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

def fetch_random_tmdb_movies(num_movies=10): #Select number of movies wanted.
    """Fetch a random set of popular movies from TMDB."""
    random_movies = []
    max_pages = 500  # TMDB limits popular movies to 500 pages
    movies_per_page = 20  # Each page has 20 movies
    
    while len(random_movies) < num_movies:
        page = random.randint(1, max_pages)
        url = f"https://api.themoviedb.org/3/movie/popular"
        params = {
            'api_key': TMDB_API_KEY,
            'page': page,
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            movies = response.json().get('results', [])
            random.shuffle(movies)  # Shuffle to add randomness
            needed_movies = num_movies - len(random_movies)
            random_movies.extend(movies[:needed_movies])
        else:
            print("Failed to fetch movies from TMDB.")
            break
        time.sleep(0.5)  # To avoid hitting API rate limits

    return random_movies[:num_movies]

def fetch_tmdb_trailer_and_genres(tmdb_id):
    """Fetch the trailer URL and genre descriptions for a movie from TMDB using its TMDB movie ID."""
    # Fetch trailer
    trailer_url = None
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}/videos"
    params = {'api_key': TMDB_API_KEY}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        videos = response.json().get('results', [])
        for video in videos:
            if video['type'] == 'Trailer' and video['site'] == 'YouTube':
                trailer_url = f"https://www.youtube.com/watch?v={video['key']}"
                break

    # Fetch genres
    genres_url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
    genres_response = requests.get(genres_url, params=params)
    genres = []
    if genres_response.status_code == 200:
        genres_data = genres_response.json().get('genres', [])
        genres = [genre['name'] for genre in genres_data]  # List of genre names
    
    return trailer_url, ", ".join(genres)  # Return trailer URL and genres as a comma-separated string

def fetch_random_movie_trailers(num_movies=100):
    """Fetch trailers and genre descriptions for a random set of popular movies."""
    trailers = []
    random_movies = fetch_random_tmdb_movies(num_movies)
    for movie in random_movies:
        trailer_url, genres = fetch_tmdb_trailer_and_genres(movie['id'])
        trailers.append({
            'title': movie['title'],
            'release_year': movie['release_date'][:4] if 'release_date' in movie else 'N/A',
            'trailer_url': trailer_url,
            'genres': genres
        })
        print(f"Fetched trailer for {movie['title']}: {trailer_url} - Genres: {genres}")
        time.sleep(0.5)  # To avoid hitting API rate limits
    return trailers

def download_trailer_yt_dlp(trailer_url, title, save_dir='../data/trailers'):
    """Download a YouTube trailer using yt-dlp."""
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{title.replace(' ', '_')}.mp4"
    file_path = os.path.join(save_dir, filename)
    
    ydl_opts = {
        'format': 'best',
        'outtmpl': file_path,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([trailer_url])
            print(f"Downloaded: {title}")
    except Exception as e:
        print(f"Error downloading {trailer_url}: {e}")

def save_metadata_to_csv(trailers, filename='trailer_metadata.csv'):
    """Save trailer metadata to a CSV file in the output_data folder."""
    output_dir = '../data'
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)
    
    # Save the CSV file
    df = pd.DataFrame(trailers)
    df.to_csv(file_path, index=False)

#Fetch, download, and save metadata for a random set of movies
num_movies = 10
random_trailers = fetch_random_movie_trailers(num_movies=num_movies)
save_metadata_to_csv(random_trailers)

# Download each trailer using yt-dlp
for trailer in random_trailers:
    if trailer['trailer_url']:
        download_trailer_yt_dlp(trailer['trailer_url'], trailer['title'])
