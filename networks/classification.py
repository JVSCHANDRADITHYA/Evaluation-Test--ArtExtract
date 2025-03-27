import torch
import torch.nn as nn

class ClassificationHeads(nn.Module):
    def __init__(self, input_dim, num_classes_artist, num_classes_genre, num_classes_style):
        super(ClassificationHeads, self).__init__()
        
        self.artist_head = nn.Linear(input_dim, num_classes_artist)
        self.genre_head = nn.Linear(input_dim, num_classes_genre)
        self.style_head = nn.Linear(input_dim, num_classes_style)
    
    def forward(self, x):
        artist_out = self.artist_head(x)
        genre_out = self.genre_head(x)
        style_out = self.style_head(x)
        return artist_out, genre_out, style_out
