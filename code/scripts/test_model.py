import sys
import os

# Add the project root to the Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from tqdm import tqdm  # For progress bar
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # TensorBoard for metrics
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from src.models.vgg_model import VGGMultiLabel
from src.data.movie_dataset import MovieDataset, split_metadata
from src.config import load_config

def test_model(config_path, model_path):
    """
    Test the trained model on the test set.
    Args:
        config_path (str): Path to the configuration file.
        model_path (str): Path to the trained model's weights.
    """
    # Load configuration
    config = load_config(config_path)

    # Paths
    metadata_path = config["paths"]["metadata_path"]
    processed_folder = config["paths"]["processed_posters_folder"]
    logs_dir = config["paths"]["logs"]

    # Ensure output directories exist
    os.makedirs(logs_dir, exist_ok=True)

    # TensorBoard writer
    writer = SummaryWriter(log_dir=logs_dir)

    # Test parameters
    batch_size = config["training"]["batch_size"]

    # Prepare datasets and dataloaders (only test data)
    train_dataset, val_dataset, test_dataset, genre_list = split_metadata(
        metadata_path, processed_folder, train_split=0.7, val_split=0.05, test_split=0.25  # Usual splitting
    )
    
    # Use only the test dataset
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model setup
    num_classes = len(genre_list)
    model = VGGMultiLabel(num_classes=num_classes)

    # Load the trained model
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define loss function
    criterion = nn.BCELoss()  # For multi-label classification

    # Variables to track all predictions and ground truths
    all_predictions = []
    all_labels = []

    # Test loop
    total_loss = 0.0
    correct_test = 0
    total_test = 0

    # Add progress bar
    for images, labels in tqdm(test_loader, desc="Testing", ncols=100):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item()

        # Calculate accuracy
        predicted = (outputs > 0.5).float()
        correct_test += (predicted == labels).sum().item()
        total_test += labels.numel()

        # Store predictions and labels for evaluation metrics
        all_predictions.append(predicted.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    # Convert lists to numpy arrays
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Calculate test accuracy and average loss
    test_accuracy = 100 * correct_test / total_test
    avg_test_loss = total_loss / len(test_loader)

    # Calculate Precision, Recall, and F1-Score
    precision = precision_score(all_labels, all_predictions, average='micro')
    recall = recall_score(all_labels, all_predictions, average='micro')
    f1 = f1_score(all_labels, all_predictions, average='micro')

    # Compute a confusion matrix for each label
    overall_conf_matrix = np.zeros((num_classes, num_classes))

    for i in range(num_classes):
        # Compute confusion matrix for each label
        label_conf_matrix = confusion_matrix(all_labels[:, i], all_predictions[:, i], labels=[0, 1])
        
        # Assign values to overall confusion matrix (row i and column i is TP)
        overall_conf_matrix[i, i] = label_conf_matrix[1, 1]  # True Positives
        for j in range(num_classes):
            if i != j:
                overall_conf_matrix[i, j] += label_conf_matrix[0, 1]  # False Positives for the row

    # Log metrics to TensorBoard
    writer.add_scalar("Loss/Test", avg_test_loss, 0)
    writer.add_scalar("Accuracy/Test", test_accuracy, 0)
    writer.add_scalar("Precision/Test", precision, 0)
    writer.add_scalar("Recall/Test", recall, 0)
    writer.add_scalar("F1_Score/Test", f1, 0)

    # Print results
    print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

    # Print overall confusion matrix for all labels
    print(f"Confusion Matrix (All Labels):\n{overall_conf_matrix}")

    writer.close()


if __name__ == "__main__":
    # Path to the configuration file
    config_path = os.path.join(os.path.dirname(__file__), "../configs/model_config.yaml")
    # Path to the trained model file
    model_path = os.path.join(os.path.dirname(__file__), "C:/Users/ryanu/Desktop/CS771_New/model_checkpoints/best_model_epoch_20.pth")

    test_model(config_path, model_path)
