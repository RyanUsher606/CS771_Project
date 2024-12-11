import sys
import os
# Add the project root to the Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from tqdm import tqdm  # For progress bar
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter  # TensorBoard for metrics
from src.models.vgg_model import VGGMultiLabel
from src.data.movie_dataset import MovieDataset, split_metadata
from src.config import load_config
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
import numpy as np


def train_model(config_path):
    # Load configuration
    config = load_config(config_path)

    # Paths
    metadata_path = config["paths"]["metadata_path"]
    processed_folder = config["paths"]["processed_posters_folder"]
    logs_dir = config["paths"]["logs"]
    model_dir = config["paths"]["models"]

    # Ensure output directories exist
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # TensorBoard writer
    writer = SummaryWriter(log_dir=logs_dir)

    # Training parameters
    batch_size = config["training"]["batch_size"]
    learning_rate = config["training"]["learning_rate"]
    num_epochs = config["training"]["num_epochs"]

    # Data split
    train_split = config["data_split"]["train_split"]
    val_split = config["data_split"]["val_split"]
    test_split = config["data_split"]["test_split"]

    # Prepare datasets and dataloaders
    train_dataset, val_dataset, _, genre_list = split_metadata(
        metadata_path, processed_folder, train_split, val_split, test_split
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Flatten the multi-hot encoded genre labels to calculate class frequencies
    all_labels = []
    for _, row in train_dataset.metadata.iterrows():
        all_labels.extend([genre for genre in row['Genre'].split('|')])

    # Calculate class frequencies
    unique_labels, label_counts = np.unique(all_labels, return_counts=True)

    # Calculate class weights based on frequencies (inverse frequency)
    class_weights = compute_class_weight('balanced', classes=unique_labels, y=all_labels)

    # Convert class weights to tensor
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # Move class_weights to CPU for NumPy conversion
    class_weights_cpu = class_weights.cpu()

    # Debug: Print genre_list and class_weights to inspect their contents
    print(f"genre_list: {genre_list}")
    print(f"class_weights_cpu: {class_weights_cpu}")

    # Weighted sampler
    samples_weight = []
    for _, row in train_dataset.metadata.iterrows():
        weight_sum = 0
        for genre in row['Genre'].split('|'):
            if genre in genre_list:  # Ensure genre exists in genre_list
                genre_index = genre_list.index(genre)
                weight_sum += class_weights_cpu[genre_index].item()  # Retrieve the weight using the genre index
            else:
                print(f"Warning: Genre '{genre}' not found in genre_list! Skipping this genre.")
        samples_weight.append(weight_sum)

    samples_weight = torch.from_numpy(np.array(samples_weight)).to(device)

    sampler = WeightedRandomSampler(samples_weight, num_samples=len(samples_weight), replacement=True)

    # DataLoader with sampler
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model setup
    num_classes = len(genre_list)
    model = VGGMultiLabel(num_classes=num_classes)

    # Freeze backbone if specified
    if config["model"]["freeze_backbone"]:
        for param in model.features.parameters():
            param.requires_grad = False

    # Move model to GPU if available
    model.to(device)

    # Loss function with class weights
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)  # Use BCEWithLogitsLoss for better stability
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Training loop
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # Add progress bar
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate accuracy
            predicted = (outputs > 0.5).float()
            correct_train += (predicted == labels).sum().item()
            total_train += labels.numel()

        # Calculate training accuracy
        train_accuracy = 100 * correct_train / total_train

        # Validate the model
        val_loss, val_f1 = validate_model(model, val_loader, criterion, device)

        # Log metrics to TensorBoard
        writer.add_scalar("Loss/Train", running_loss / len(train_loader), epoch)
        writer.add_scalar("Accuracy/Train", train_accuracy, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("F1/Validation", val_f1, epoch)
        writer.add_scalar("Learning_Rate", scheduler.get_last_lr()[0], epoch)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {running_loss/len(train_loader):.4f}, "
              f"Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, "
              f"Val F1: {val_f1:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save the model if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(model_dir, f"best_model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

        # Step the scheduler
        scheduler.step()

    writer.close()


def validate_model(model, val_loader, criterion, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Collect predictions and true labels for F1 score calculation
            preds = (outputs > 0.5).float()
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Flatten the list of predictions and true labels
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Calculate F1 score
    val_f1 = f1_score(all_labels, all_preds, average='macro')  # or 'weighted' based on your preference

    val_loss = total_loss / len(val_loader)
    return val_loss, val_f1


if __name__ == "__main__":
    # Path to the configuration file
    config_path = os.path.join(os.path.dirname(__file__), "../configs/model_config.yaml")
    train_model(config_path)
