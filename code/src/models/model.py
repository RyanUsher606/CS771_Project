import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class VGGGenreConvHead(nn.Module):
    def __init__(self, num_genres, pretrained=True):
        super(VGGGenreConvHead, self).__init__()
        
        # Load the pre-trained VGG model (VGG16 in this case)
        self.vgg = models.vgg16(pretrained=pretrained)
        
        # Retain only the feature extraction layers (convolutional layers)
        self.vgg_features = self.vgg.features  # This outputs 4D feature maps
        
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
        # Forward pass through VGG backbone (feature extractor)
        x = self.vgg_features(x)  # Shape: [batch_size, 512, height, width]
        
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
        
        # Sigmoid activation for multi-label classification (output between 0 and 1)
        x = torch.sigmoid(x)
        
        return x