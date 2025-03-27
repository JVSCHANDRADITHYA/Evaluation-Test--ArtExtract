import torch
import torch.nn as nn
from cnn_feature_extractor import CNNFeatureExtractor
from rnn_sequence_model import RNNSequenceModel
from classification_heads import ClassificationHeads

class CRNNModel(nn.Module):
    def __init__(self, cnn_backbone="resnet50", rnn_type="gru", hidden_dim=256,
                 num_classes_artist=100, num_classes_genre=50, num_classes_style=30):
        super(CRNNModel, self).__init__()
        
        self.cnn = CNNFeatureExtractor(backbone=cnn_backbone)
        self.rnn = RNNSequenceModel(input_dim=self.cnn.feature_dim, hidden_dim=hidden_dim, rnn_type=rnn_type)
        self.classifier = ClassificationHeads(input_dim=hidden_dim * 2,  # Bidirectional RNN
                                              num_classes_artist=num_classes_artist,
                                              num_classes_genre=num_classes_genre,
                                              num_classes_style=num_classes_style)
    
    def forward(self, x):
        features = self.cnn(x)  # CNN feature extraction
        rnn_out = self.rnn(features.unsqueeze(1))  # Add sequence dimension
        artist_out, genre_out, style_out = self.classifier(rnn_out)
        return artist_out, genre_out, style_out
