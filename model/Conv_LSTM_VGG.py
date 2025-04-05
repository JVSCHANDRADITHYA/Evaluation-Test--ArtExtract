import torch 
import torch.nn as nn
import torchvision.models as models

class VGG_LSTM_Model(nn.Module):
    def __init__(self, hidden_size=256, num_layers=2, num_classes=(128, 11, 27)):
        super(VGG_LSTM_Model, self).__init__()
        vgg = models.vgg16(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg.features.children()))  # output: (B, 512, 7, 7)

        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        self.fc_artist = nn.Linear(hidden_size, num_classes[0])
        self.fc_genre = nn.Linear(hidden_size, num_classes[1])
        self.fc_style = nn.Linear(hidden_size, num_classes[2])

    def forward(self, x):
        batch_size = x.shape[0]
        features = self.feature_extractor(x)       # (B, 512, 7, 7)
        features = features.view(batch_size, 512, -1)  # (B, 512, 49)
        features = features.permute(0, 2, 1)       # (B, 49, 512)

        lstm_out, _ = self.lstm(features)
        last_hidden = lstm_out[:, -1, :]           # (B, hidden_size)

        artist_pred = self.fc_artist(last_hidden)
        genre_pred = self.fc_genre(last_hidden)
        style_pred = self.fc_style(last_hidden)

        return artist_pred, genre_pred, style_pred