import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import os
import random
import matplotlib.pyplot as plt

# Import the ConvLSTMModel
from model.Conv_LSTM import ConvLSTMModel  # Replace with actual filename

# ======= 1️⃣ Generate a Small Sample Dataset =======
# Create a small dataset CSV with random labels
def generate_sample_csv(csv_file, num_samples=3000):
    data = {
        'image_path': [f"image_{i}.jpg" for i in range(num_samples)],
        'artist': np.random.randint(0, 128, num_samples),
        'genre': np.random.randint(0, 11, num_samples),
        'style': np.random.randint(0, 27, num_samples),
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)

# Create fake dataset CSV
csv_file = r'F:\GSoc_2025\Evaluation-Test--ArtExtract\csv_files\val_data.csv'
generate_sample_csv(csv_file, num_samples=3000)

# ======= 2️⃣ Create a Dummy Dataset Class =======
class DummyWikiArtDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        # Generate random image tensor (C, H, W) -> (3, 224, 224)
        image = torch.rand(3, 224, 224)

        # Generate labels (artist, genre, style)
        labels = torch.tensor([int(row['artist']), int(row['genre']), int(row['style'])], dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, labels

# ======= 3️⃣ Define Transformations & Load Data =======
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dataset = DummyWikiArtDataset(csv_file=csv_file, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# ======= 4️⃣ Initialize Model & Move to Device =======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvLSTMModel().to(device)

# ======= 5️⃣ Run a Forward Pass (Sanity Check) =======
for images, labels in dataloader:
    images, labels = images.to(device), labels.to(device)
    
    # Forward pass
    artist_pred, genre_pred, style_pred = model(images)

    # Print output shapes
    print("Artist Prediction Shape:", artist_pred.shape)  # Expected: (batch_size, num_artists)
    print("Genre Prediction Shape:", genre_pred.shape)   # Expected: (batch_size, num_genres)
    print("Style Prediction Shape:", style_pred.shape)   # Expected: (batch_size, num_styles)
    
    # Convert to class indices
    artist_class = torch.argmax(artist_pred, dim=1)
    genre_class = torch.argmax(genre_pred, dim=1)
    style_class = torch.argmax(style_pred, dim=1)
    
    print(f"Predicted Artist: {artist_class.tolist()}")
    print(f"Predicted Genre: {genre_class.tolist()}")
    print(f"Predicted Style: {style_class.tolist()}")
    
    break
