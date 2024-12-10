import torch
import torch.nn as nn
import torch.nn.functional as F

class VGGMultiLabel(nn.Module):
    def __init__(self, num_classes=20):
        super(VGGMultiLabel, self).__init__()
        
        # Convolutional layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # Conv1
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # Conv2
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Conv3
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.4)
        )
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),  # Adjust dimensions based on input size
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
            nn.Sigmoid()  # For multi-label classification
        )

    def forward(self, x):
        x = self.features(x)  # Pass through convolutional layers
        x = self.classifier(x)  # Pass through fully connected layers
        return x
