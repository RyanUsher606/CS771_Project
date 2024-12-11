import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss

from torchvision.models import resnet

class ResNetGenreConvHead(nn.Module):
    def __init__(self, num_genres, pretrained=True):
        super(ResNetGenreConvHead, self).__init__()

        # Convolutional head
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)  # First convolutional layer
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)  # Second convolutional layer
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)   # Third convolutional layer

        # Global Average Pooling (GAP)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Outputs a 1x1 feature map

        # Final fully connected layer for genre classification
        self.fc = nn.Linear(64, num_genres)  # Output layer for multi-label classification

        self.dropout = nn.Dropout(0.5)  # Dropout layer for regularization

    def forward(self, x):
        # Pass through the convolutional layers
        x = F.relu(self.conv1(x))  # ReLU activation after first convolutional layer
        x = F.relu(self.conv2(x))  # ReLU activation after second convolutional layer
        x = F.relu(self.conv3(x))  # ReLU activation after third convolutional layer

        # Apply Global Average Pooling (GAP) to reduce spatial dimensions
        x = self.global_avg_pool(x)  # Shape: [batch_size, 64, 1, 1]
        x = x.view(x.size(0), -1)    # Flatten the output to (batch_size, 64)

        # Apply dropout for regularization
        x = self.dropout(x)

        # Apply the final fully connected layer
        x = self.fc(x)

        return x

class MovieClassifier(nn.Module):
    def __init__(self, backbone, backbone_freeze_bn, num_genres):
        super().__init__()
        assert backbone in ("resnet18", "resnet34")
        self.backbone_name = backbone
        self.backbone_freeze_bn = backbone_freeze_bn
        self.backbone = resnet.__dict__[backbone](weights="IMAGENET1K_V1")
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        self.genre_head = ResNetGenreConvHead(num_genres)


    @staticmethod
    def freeze_bn(module):
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)

        if self.backbone_freeze_bn:
            self.apply(self.freeze_bn)
        return self

    def forward(self, images):

        features = self.backbone(images)

        genre_logits = self.genre_head(features)

        return genre_logits