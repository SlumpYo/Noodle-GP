import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class ResNet18Backbone(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        # change the first layer to accept our custom channels (cqt, log_cqt and (if enabled) mel)
        self.resnet.conv1 = nn.Conv2d(
            in_channels, 
            64,
            kernel_size=7, 
            stride=2, 
            padding=3, 
            bias=False
        )
        # Keep everything but the avgpool + fc
        self.feature_extractor = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4,
            # outputs tensor of shape (B, C_out, H_out, W_out)
        )

    def forward(self, x):
        return self.feature_extractor(x)