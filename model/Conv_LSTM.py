import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import os

class CNN_LSTM_Model(nn.Module):
    def __init__(self, hidden_size=256, num_layers=2, num_classes=(128, 11, 27)):  
        super(CNN_LSTM_Model, self).__init__()

        # Load Pretrained ResNet (Feature Extractor)
        resnet = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])  # Remove FC layer
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM for sequence processing
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        # Fully Connected Layers for Artist, Genre, Style
        self.fc_artist = nn.Linear(hidden_size, num_classes[0])   # Number of artists
        self.fc_genre = nn.Linear(hidden_size, num_classes[1])   # Number of genres
        self.fc_style = nn.Linear(hidden_size, num_classes[2])   # Number of styles

    def forward(self, x):
        batch_size = x.shape[0]

        # Extract CNN Features (Output shape: batch_size x 512 x 7 x 7)
        features = self.feature_extractor(x)
        features = features.view(batch_size, 512, -1)  # Flatten spatial dimensions

        # Process features with LSTM
        lstm_out, _ = self.lstm(features.permute(0, 2, 1))  # Permute for LSTM (batch, seq_len, input_size)

        # Get the last time step output
        lstm_out = lstm_out[:, -1, :]

        # Predict Artist, Genre, Style
        artist_pred = self.fc_artist(lstm_out)
        genre_pred = self.fc_genre(lstm_out)
        style_pred = self.fc_style(lstm_out)

        return artist_pred, genre_pred, style_pred

if __name__ == "__main__":
    img_path = r"F:\GSoc_2025\wiki_art_dataset\wikiart\Abstract_Expressionism\aki-kuroda_untitled-1988-1.jpg"

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model and move to device
    feature_extractor = CNN_LSTM_Model().to(device)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match ResNet input
        transforms.ToTensor()
    ])

    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)  # Add batch dim and move to device

    features = feature_extractor(img) 

    # Output stuff
    print("Artist Prediction Shape:", features[0].shape)
    print("Genre Prediction Shape:", features[1].shape)
    print("Style Prediction Shape:", features[2].shape)
    print("Predicted Artist:", torch.argmax(features[0], dim=1).item())
    print("Predicted Genre:", torch.argmax(features[1], dim=1).item())
    print("Predicted Style:", torch.argmax(features[2], dim=1).item())
