from transformers import ViTForImageClassification
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
import torch.nn as nn
import torch

# Load a pre-trained Vision Transformer
class MultiTaskViT(nn.Module):
    def __init__(self, num_genres, num_decades):
        super().__init__()
        self.vit = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k"
        )
        self.genre_head = nn.Linear(self.vit.config.hidden_size, num_genres)
        self.decade_head = nn.Linear(self.vit.config.hidden_size, num_decades)

    def forward(self, x):
        features = self.vit.vit(x).last_hidden_state.mean(dim=1)  # Extract features
        genre_logits = self.genre_head(features)
        decade_logits = self.decade_head(features)
        return genre_logits, decade_logits

# Instantiate model
num_genres = len(genre_list)
num_decades = len(decade_mapping)
model = MultiTaskViT(num_genres, num_decades)

#Training begin
genre_loss_fn = BCEWithLogitsLoss()
decade_loss_fn = CrossEntropyLoss()