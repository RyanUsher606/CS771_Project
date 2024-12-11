import sys
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # For TensorBoard
import numpy as np

# Add the project root to the Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.models.model import MovieClassifier  # Import your model definition
from src.data.movie_dataset import MovieDataset, get_genre_list  # Import your dataset code

def compute_class_weights(dataset):
    """
    Compute class weights for a multi-label dataset.
    dataset: a PyTorch Dataset that returns (image, target), where target is a multi-hot vector of shape [num_genres].
    """
    all_counts = []
    for i in range(len(dataset)):
        _, target = dataset[i]
        all_counts.append(target)

    all_counts = np.stack(all_counts, axis=0)  # Shape: [num_samples, num_genres]
    pos_count = all_counts.sum(axis=0)  # How many times each class is positive
    total_samples = len(dataset)

    # pos_weight formula: (N - pos_count) / pos_count
    # To avoid division by zero, ensure pos_count > 0
    pos_weight = (total_samples - pos_count) / np.clip(pos_count, a_min=1, a_max=None)
    return torch.tensor(pos_weight, dtype=torch.float32)

def main():
    # Paths to your CSV splits and processed posters
    train_csv = r"C:\Users\ryanu\Desktop\New folder\dataset\training_testing\train.csv"
    test_csv  = r"C:\Users\ryanu\Desktop\New folder\dataset\training_testing\test.csv"
    processed_folder = r"C:\Users\ryanu\Desktop\New folder\dataset\filtered_dataset\processed_posters"
    metadata_path = r"C:\Users\ryanu\Desktop\New folder\dataset\filtered_dataset\filtered_movies_metadata.csv"  
    checkpoint_dir = r"C:\Users\ryanu\Desktop\New folder\model_checkpoint"

    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    # TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, "tensorboard_logs"))

    # Get genre list
    genre_list = get_genre_list(metadata_path)

    # Create datasets
    train_dataset = MovieDataset(csv_path=train_csv, processed_folder=processed_folder, genre_list=genre_list)
    test_dataset  = MovieDataset(csv_path=test_csv, processed_folder=processed_folder, genre_list=genre_list)

    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    # Compute class weights for imbalanced classes
    pos_weight = compute_class_weights(train_dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_genres = len(genre_list)
    model = MovieClassifier(backbone="resnet18", num_genres=num_genres).to(device)

    # Apply class weighting through pos_weight
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training parameters
    num_epochs = 20

    for epoch in range(1, num_epochs + 1):
        # Training Loop
        model.train()
        running_train_loss = 0.0
        correct_preds = 0
        total_labels = 0

        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Training]"):
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

            # Accuracy calculation
            preds = torch.sigmoid(logits) > 0.5  # Convert logits to binary predictions
            correct_preds += (preds == targets).sum().item()
            total_labels += targets.numel()

        avg_train_loss = running_train_loss / len(train_loader)
        train_accuracy = correct_preds / total_labels

        print(f"Epoch {epoch}/{num_epochs}: Train Loss = {avg_train_loss:.4f}, Train Accuracy = {train_accuracy:.4f}")

        # Log to TensorBoard
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_accuracy, epoch)

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved at: {checkpoint_path}")

    writer.close()

if __name__ == '__main__':
    main()
