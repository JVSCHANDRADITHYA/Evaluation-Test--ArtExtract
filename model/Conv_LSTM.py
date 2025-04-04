import torch
import torch.nn as nn
from torchvision import models

class ConvLSTMModel(nn.Module):
    def __init__(self, num_artists, num_genres, num_styles):
        super(ConvLSTMModel, self).__init__()
        
        # Pretrained CNN backbone (ResNet50)
        self.cnn_backbone = models.resnet50(pretrained=True)
        self.cnn_backbone.fc = nn.Identity()  # Remove the final fully connected layer
        
        # RNN (GRU) layer
        self.rnn = nn.GRU(input_size=2048, hidden_size=512, num_layers=2, batch_first=True)
        
        # Fully connected layers for each output (Artist, Genre, Style)
        self.fc_artist = nn.Linear(512, num_artists)
        self.fc_genre = nn.Linear(512, num_genres)
        self.fc_style = nn.Linear(512, num_styles)

    def forward(self, x):
        # CNN feature extraction
        x = self.cnn_backbone(x)
        
        # Reshape to feed into RNN
        x = x.unsqueeze(1)  # (batch_size, seq_len=1, feature_size)
        
        # RNN layer
        rnn_out, _ = self.rnn(x)
        
        # Use the output of the last time step
        rnn_out = rnn_out[:, -1, :]
        
        # Classification layers
        artist_pred = self.fc_artist(rnn_out)
        genre_pred = self.fc_genre(rnn_out)
        style_pred = self.fc_style(rnn_out)
        
        return artist_pred, genre_pred, style_pred
