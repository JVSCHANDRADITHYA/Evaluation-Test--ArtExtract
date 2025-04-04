import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from ConvFeatureExtraction import ConvolutionalFeatureExtractorBackbone

class LSTMModel(nn.Module):
    def __init__(self, input_size=512, hidden_size=256, num_layers=2, num_classes=10):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)  # Final classification layer

    def forward(self, x):
        out, _ = self.lstm(x)  # LSTM processing
        out = self.fc(out[:, -1, :])  # Take the last time-step output
        return out
    
if __name__ == '__main__':
    featute_extractor = ConvolutionalFeatureExtractorBackbone()
    lstm_model = LSTMModel()
    
    img_path = r"F:\GSoc_2025\wiki_art_dataset\wikiart\Abstract_Expressionism\aki-kuroda_untitled-1988-1.jpg"
    transform = transforms.Compose([   
        transforms.Resize((224, 224)),  # Resize to match ResNet input
        transforms.ToTensor()
    ])
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0)  # Add batch dimension
    
    features = featute_extractor(img)
    output = lstm_model(features)
    print(output.shape)  # Should be (1, num_classes)
    print(output)
    
    print("LSTM Output:", output)
    print("Predicted Class:", torch.argmax(output, dim=1).item())