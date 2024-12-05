import os
import numpy as np
from PIL import Image
from torchvision import transforms

# Paths
posters_folder = r"../dataset/unfiltered_dataset/posters/" 
output_folder = r"../dataset/filtered_dataset/processed_posters/"
os.makedirs(output_folder, exist_ok=True)

# Define preprocessing pipeline for ResNet
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Process poster images
def process_posters(posters_folder, output_folder):
    for img_name in os.listdir(posters_folder):
        input_path = os.path.join(posters_folder, img_name)
        output_path = os.path.join(output_folder, img_name.replace('.jpg', '.npy'))
        
        try:
            # Load and preprocess image
            image = Image.open(input_path).convert('RGB')  # Ensure 3-channel RGB
            transformed_image = image_transform(image)

            # Save processed image as NumPy array
            np.save(output_path, transformed_image.numpy())
            print(f"Processed: {img_name} -> {output_path}")
        except Exception as e:
            print(f"Error processing image {img_name}: {e}")

    print(f"Processed images saved to: {output_folder}")

# Run the preprocessing
process_posters(posters_folder, output_folder)
