import sys
import os
# Add the project root to the Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from tqdm import tqdm  # For progress bar
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # TensorBoard for metrics
from src.models.vgg_model import VGGMultiLabel
from src.data.movie_dataset import MovieDataset, split_metadata
from src.config import load_config


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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model setup
    num_classes = len(genre_list)
    model = VGGMultiLabel(num_classes=num_classes)

    # Freeze backbone if specified
    if config["model"]["freeze_backbone"]:
        for param in model.features.parameters():
            param.requires_grad = False

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.BCELoss()  # For multi-label classification
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
        val_loss, val_accuracy = validate_model(model, val_loader, criterion, device)

        # Log metrics to TensorBoard
        writer.add_scalar("Loss/Train", running_loss / len(train_loader), epoch)
        writer.add_scalar("Accuracy/Train", train_accuracy, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("Accuracy/Validation", val_accuracy, epoch)
        writer.add_scalar("Learning_Rate", scheduler.get_last_lr()[0], epoch)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {running_loss/len(train_loader):.4f}, "
              f"Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, "
              f"Val Accuracy: {val_accuracy:.2f}%, LR: {scheduler.get_last_lr()[0]:.6f}")

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
    """
    Validate the model on the validation set.
    """
    model.eval()
    total_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Calculate accuracy
            predicted = (outputs > 0.5).float()
            correct_val += (predicted == labels).sum().item()
            total_val += labels.numel()

    val_loss = total_loss / len(val_loader)
    val_accuracy = 100 * correct_val / total_val
    return val_loss, val_accuracy


if __name__ == "__main__":
    # Path to the configuration file
    config_path = os.path.join(os.path.dirname(__file__), "../configs/model_config.yaml")
    train_model(config_path)
