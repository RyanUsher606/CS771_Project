import cv2
import os
import numpy as np
from tqdm import tqdm

def extract_frames(video_path, output_dir, frame_interval=30, target_size=(224, 224)):
    """
    Extract frames from a video file at a given interval, resize, and normalize them.
    
    Args:
    - video_path (str): Path to the video file.
    - output_dir (str): Directory to save the extracted frames.
    - frame_interval (int): Interval to extract frames (e.g., every 30 frames).
    - target_size (tuple): Target size (width, height) for resizing.
    
    Returns:
    - List of paths to saved frames.
    """
    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Extract every 'frame_interval' frame
        if frame_count % frame_interval == 0:
            # Resize frame
            frame = cv2.resize(frame, target_size)
            # Normalize frame
            frame = frame / 255.0  # Normalized to range [0, 1]
            # Save frame as an image
            frame_filename = os.path.join(output_dir, f"{video_name}_frame_{frame_count}.jpg")
            cv2.imwrite(frame_filename, (frame * 255).astype(np.uint8))  # Convert back to uint8 for saving
            saved_frames.append(frame_filename)
        
        frame_count += 1

    cap.release()
    return saved_frames

def preprocess_videos(video_dir, output_dir, frame_interval=30, target_size=(224, 224)):
    """
    Preprocess all videos in a directory by extracting frames.
    
    Args:
    - video_dir (str): Directory containing videos to preprocess.
    - output_dir (str): Directory to save preprocessed frames.
    - frame_interval (int): Interval to extract frames.
    - target_size (tuple): Target size for resizing.
    """
    video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
    for video_file in tqdm(video_files, desc="Processing videos"):
        video_path = os.path.join(video_dir, video_file)
        video_output_dir = os.path.join(output_dir, os.path.splitext(video_file)[0])
        extract_frames(video_path, video_output_dir, frame_interval, target_size)

video_dir = '../data/trailers'  # Directory where trailers are stored
output_dir = '../data/processed_frames'  # Directory to save processed frames
preprocess_videos(video_dir, output_dir, frame_interval=30, target_size=(224, 224))
