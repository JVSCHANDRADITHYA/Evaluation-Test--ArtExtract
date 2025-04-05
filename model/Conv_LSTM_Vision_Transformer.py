import torch
import torch.nn as nn
import torchvision.models as models

class ConvLSTM_VisionTransformer(nn.Module):
    def __init__(self, hidden_size=256, num_layers=2, num_classes=(128, 11, 27)):
        super(ConvLSTM_VisionTransformer, self).__init__()

        # Vision Transformer (remove classification head to get CLS embeddings)
        self.vit = models.vit_b_16(weights='DEFAULT')
        self.vit.heads = nn.Identity()  # Now vit(x) outputs (B, 768)

        # LSTM over CLS embedding
        self.lstm = nn.LSTM(input_size=768, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        # Classification heads
        self.fc_artist = nn.Linear(hidden_size, num_classes[0])
        self.fc_genre = nn.Linear(hidden_size, num_classes[1])
        self.fc_style = nn.Linear(hidden_size, num_classes[2])

    def forward(self, x):
        batch_size = x.shape[0]

        # Get CLS token embedding (B, 768)
        cls_embedding = self.vit(x)  # No classifier, just CLS output

        # Reshape to (B, 1, 768) for LSTM
        lstm_input = cls_embedding.unsqueeze(1)

        # Pass through LSTM
        lstm_out, _ = self.lstm(lstm_input)
        last_hidden = lstm_out[:, -1, :]  # (B, hidden_size)

        # Predict classes
        artist_pred = self.fc_artist(last_hidden)
        genre_pred = self.fc_genre(last_hidden)
        style_pred = self.fc_style(last_hidden)

        return artist_pred, genre_pred, style_pred
