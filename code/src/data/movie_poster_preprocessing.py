import os
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

def preprocess_posters(input_folder, output_folder, image_size=(224, 224)):
    """
    Preprocesses poster images: resizing, normalization, and saving as NumPy arrays.

    Args:
        input_folder (str): Path to the input folder containing images.
        output_folder (str): Path to the output folder to save preprocessed images.
        image_size (tuple): Desired image size for resizing (default is (224, 224)).
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Define preprocessing pipeline
    transform = transforms.Compose([
        transforms.Resize(image_size),  # Resize image to match VGG input
        transforms.ToTensor(),         # Convert image to Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize using ImageNet means
                             std=[0.229, 0.224, 0.225])    # Normalize using ImageNet stds
    ])

    # List all files in the input folder
    image_files = [img for img in os.listdir(input_folder) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Process images with tqdm progress bar
    for img_name in tqdm(image_files, desc="Processing images"):
        input_path = os.path.join(input_folder, img_name)
        output_path = os.path.join(output_folder, img_name.replace('.jpg', '.npy'))

        try:
            # Load and preprocess image
            image = Image.open(input_path).convert('RGB')  # Convert to RGB
            transformed_image = transform(image)          # Apply transformations

            # Save preprocessed image as a NumPy array
            np.save(output_path, transformed_image.numpy())
        except Exception as e:
            print(f"Error processing image {img_name}: {e}")