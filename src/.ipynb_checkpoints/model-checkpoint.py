import time
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torch.optim import AdamW
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
import yaml
from data_preparation_loader import train_loader, val_loader, test_loader, decade_mapping, genre_list  # Import loaders and mappings


class ResNetClassifier(nn.Module):
    def __init__(self, num_genres, num_decades):
        super(ResNetClassifier, self).__init__()
        
        # Pre-trained ResNet backbone using the updated 'weights' parameter
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Identity()  # Remove default classification head

        # Custom heads
        self.genre_head = nn.Linear(2048, num_genres)  # Multi-label classification
        self.decade_head = nn.Linear(2048, num_decades)  # Multi-class classification

    def forward(self, x):
        # Extract features from ResNet backbone
        features = self.resnet(x)
        
        # Task-specific outputs
        genre_logits = self.genre_head(features)
        decade_logits = self.decade_head(features)
        
        return genre_logits, decade_logits


def train_model_with_metrics(model, train_loader, val_loader, num_epochs, learning_rate, device):
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    genre_loss_fn = BCEWithLogitsLoss()
    decade_loss_fn = CrossEntropyLoss()

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_start = time.time()
        train_loss = 0
        train_acc1, train_acc5 = 0, 0
        data_time, batch_time = 0, 0
        start_time = time.time()

        for i, (images, genres, decades) in enumerate(train_loader):
            data_time += time.time() - start_time

            images, genres, decades = images.to(device), genres.to(device), decades.to(device)

            optimizer.zero_grad()
            genre_logits, decade_logits = model(images)

            genre_loss = genre_loss_fn(genre_logits, genres)
            decade_loss = decade_loss_fn(decade_logits, decades)
            loss = genre_loss + decade_loss

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            batch_time += time.time() - start_time
            start_time = time.time()

        # Average metrics over all batches
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

        # Average metrics over all batches
        val_loss /= len(val_loader)

        # Print metrics for the epoch
        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Time: {epoch_time:.2f}s | "
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
            pred_genres.append((genre_logits.sigmoid() > 0.5).cpu())
            pred_decades.append(decade_logits.argmax(dim=1).cpu())

    return all_genres, all_decades, pred_genres, pred_decades


if __name__ == "__main__":
    # Load configuration
    with open("config.yaml", 'r') as file:
        config = yaml.safe_load(file)

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
    all_genres, all_decades, pred_genres, pred_decades = evaluate_model(model, test_loader, device)