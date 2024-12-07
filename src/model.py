import time
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torch.optim import AdamW
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import yaml
from data_preparation_loader import train_loader, val_loader, test_loader, decade_mapping, genre_list


class ResNetClassifier(nn.Module):
    def __init__(self, num_genres, num_decades):
        super(ResNetClassifier, self).__init__()
        
        # Pre-trained ResNet backbone
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Identity()  # Remove default classification head

        # Custom genre head with convolutional layers
        self.genre_head = nn.Sequential(
            nn.Conv1d(2048, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(1024, num_genres)
        )

        # Custom decade head with convolutional layers
        self.decade_head = nn.Sequential(
            nn.Conv1d(2048, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(1024, num_decades)
        )

    def forward(self, x):
        # Extract features from ResNet backbone
        features = self.resnet(x).unsqueeze(2)  # Add a dimension for Conv1d (B, 2048, 1)

        # Task-specific outputs
        genre_logits = self.genre_head(features)
        decade_logits = self.decade_head(features)
        
        return genre_logits, decade_logits


def train_model_with_metrics(model, train_loader, val_loader, num_epochs, learning_rate, device):
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # Decay LR by 0.1 every 10 epochs
    genre_loss_fn = BCEWithLogitsLoss()
    decade_loss_fn = CrossEntropyLoss()

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_start = time.time()
        train_loss = 0

        for images, genres, decades in train_loader:
            images, genres, decades = images.to(device), genres.to(device), decades.to(device)

            optimizer.zero_grad()
            genre_logits, decade_logits = model(images)

            genre_loss = genre_loss_fn(genre_logits, genres)
            decade_loss = decade_loss_fn(decade_logits, decades)
            loss = genre_loss + decade_loss

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Average training loss
        train_loss /= len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for images, genres, decades in val_loader:
                images, genres, decades = images.to(device), genres.to(device), decades.to(device)
                genre_logits, decade_logits = model(images)

                genre_loss = genre_loss_fn(genre_logits, genres)
                decade_loss = decade_loss_fn(decade_logits, decades)
                val_loss += (genre_loss + decade_loss).item()

        # Average validation loss
        val_loss /= len(val_loader)

        # Step the scheduler
        scheduler.step()

        # Print metrics for the epoch
        current_lr = scheduler.get_last_lr()[0]
        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Time: {epoch_time:.2f}s | "
            f"Learning Rate: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

    return model


def evaluate_model(model, test_loader, device):
    model.eval()
    model = model.to(device)
    
    all_genres, all_decades = [], []
    pred_genres, pred_decades = [], []

    # Evaluation loop
    with torch.no_grad():
        for images, genres, decades in test_loader:
            images = images.to(device)
            genre_logits, decade_logits = model(images)

            # Collect true labels and predictions
            all_genres.append(genres.cpu())
            all_decades.append(decades.cpu())
            pred_genres.append((genre_logits.sigmoid() > 0.5).cpu())  # Multi-label threshold
            pred_decades.append(decade_logits.argmax(dim=1).cpu())  # Multi-class prediction

    # Concatenate all batches
    all_genres = torch.cat(all_genres).numpy()
    all_decades = torch.cat(all_decades).numpy()
    pred_genres = torch.cat(pred_genres).numpy()
    pred_decades = torch.cat(pred_decades).numpy()

    # Metrics for Decades (Multi-class classification)
    decade_accuracy = accuracy_score(all_decades, pred_decades)
    decade_precision = precision_score(all_decades, pred_decades, average='weighted', zero_division=0)
    decade_recall = recall_score(all_decades, pred_decades, average='weighted', zero_division=0)
    decade_f1 = f1_score(all_decades, pred_decades, average='weighted', zero_division=0)

    # Metrics for Genres (Multi-label classification)
    genre_precision = precision_score(all_genres, pred_genres, average='samples', zero_division=0)
    genre_recall = recall_score(all_genres, pred_genres, average='samples', zero_division=0)
    genre_f1 = f1_score(all_genres, pred_genres, average='samples', zero_division=0)

    print("\nEvaluation Metrics:")
    print("Decades (Multi-class):")
    print(f"  Accuracy: {decade_accuracy:.4f}")
    print(f"  Precision: {decade_precision:.4f}")
    print(f"  Recall: {decade_recall:.4f}")
    print(f"  F1-Score: {decade_f1:.4f}")

    print("Genres (Multi-label):")
    print(f"  Precision: {genre_precision:.4f}")
    print(f"  Recall: {genre_recall:.4f}")
    print(f"  F1-Score: {genre_f1:.4f}")

    return {
        "decades": {
            "accuracy": decade_accuracy,
            "precision": decade_precision,
            "recall": decade_recall,
            "f1_score": decade_f1,
        },
        "genres": {
            "precision": genre_precision,
            "recall": genre_recall,
            "f1_score": genre_f1,
        },
    }


if __name__ == "__main__":
    # Load configuration
    with open("config.yaml", 'r') as file:
        config = yaml.safe_load(file)

    # Initialize the model
    num_genres = len(genre_list)
    num_decades = len(decade_mapping)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetClassifier(num_genres, num_decades)

    # Train the model
    model = train_model_with_metrics(
        model,
        train_loader,
        val_loader,
        num_epochs=config['training']['epochs'],
        learning_rate=config['training']['learning_rate'],
        device=device
    )

    # Evaluate the model
    metrics = evaluate_model(model, test_loader, device)
