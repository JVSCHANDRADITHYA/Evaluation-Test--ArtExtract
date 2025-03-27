import torch
import torch.nn as nn
import torchvision.models as models

class CNNFeatureExtractor(nn.Module):
    def __init__(self, backbone="resnet50", pretrained=True, output_dim=512):
        super(CNNFeatureExtractor, self).__init__()
        
        if backbone == "resnet50":
            self.model = models.resnet50(pretrained=pretrained)
            self.model.fc = nn.Identity()  # Remove classification head
            feature_dim = self.model.fc.in_features if hasattr(self.model.fc, 'in_features') else output_dim
        elif backbone == "vgg16":
            self.model = models.vgg16(pretrained=pretrained)
            self.model.classifier = nn.Identity()  # Remove FC layers
            feature_dim = 4096  # VGG16 outputs a 4096-dim feature
        else:
            raise ValueError("Unsupported backbone. Choose from 'resnet50' or 'vgg16'")
        
        self.feature_dim = feature_dim

    def forward(self, x):
        features = self.model(x)  # Extract CNN features
        return features
