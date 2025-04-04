import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

class ConvolutionalFeatureExtractorBackbone(nn.Module):
    def __init__(self):
        super(ConvolutionalFeatureExtractorBackbone, self).__init__()
        resnet_backbone = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet_backbone.children())[:-2])
    
    def forward(self, x):
        features = self.feature_extractor(x)
        batch, channels, height, width = features.shape
        features = features.view(batch, height * width, channels)  # Reshape to (batch, seq_length, feature_dim)
        
        return features
        
    

if __name__ == "__main__":
    img_path = r"F:\GSoc_2025\wiki_art_dataset\wikiart\Abstract_Expressionism\aki-kuroda_untitled-1988-1.jpg"

    # Initialize models
    feature_extractor = ConvolutionalFeatureExtractorBackbone()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match ResNet input
        transforms.ToTensor()
    ])
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0)  # Add batch dimension
    
    features = feature_extractor(img)



    
    