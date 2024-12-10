import torch
import torch.nn as nn
import torchvision.models as models

class VGGMultiLabel(nn.Module):
    def __init__(self, num_classes):
        super(VGGMultiLabel, self).__init__()
        # Load a pre-trained VGG16 model
        self.vgg = models.vgg16(pretrained=True)
        
        # Replace the classifier with a custom head
        self.vgg.classifier = nn.Sequential(
            nn.Linear(25088, 4096),  # First layer of the VGG16 classifier
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),  # Second layer of the VGG16 classifier
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes) # Final layer for num_classes outputs
        )
        
    def forward(self, x):
        return self.vgg(x)