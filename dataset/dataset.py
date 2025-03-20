import os
import numpy as np
import cv2
import pandas as pd
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset

class WikiArtDataset(Dataset):
    def __init__(self, artist_csv, genre_csv, style_csv, img_dir, transform=None):
        # Load CSVs We have 3 lables for each image and hence create a dataframe accordingly
        self.artist_labels = pd.read_csv(artist_csv)
        self.genre_labels = pd.read_csv(genre_csv)
        self.style_labels = pd.read_csv(style_csv)
        
        self.data = self.artist_labels.merge(self.genre_labels, on='image_path').merge(self.style_labels, on='image_path')

        self.img_dir = img_dir
        self.transform = transform # None taken if None given

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_id = self.data.iloc[index]['image_path']
        img_path = os.path.join(self.img_dir, img_id)

        # Read image
        image = read_image(img_path).float()  # Convert to float tensor for better handling

        artist_label = self.data.iloc[index]['artist']
        genre_label = self.data.iloc[index]['genre']
        style_label = self.data.iloc[index]['style']

        labels = torch.tensor([int(self.data.iloc[index]['artist']), 
                               int(self.data.iloc[index]['genre']), 
                               int(self.data.iloc[index]['style'])], dtype=torch.long)

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        return image, labels


# EXAMPLE USAGE
if __name__ == '__main__' : 
    artist_csv = r"F:\GSoc_2025\wiki_art_dataset\wikiart\artist_train.csv"
    styles_csv = r"F:\GSoc_2025\wiki_art_dataset\wikiart\style_train.csv"  
    genres_csv = r"F:\GSoc_2025\wiki_art_dataset\wikiart\genre_train.csv"
    img_dir = r"F:\GSoc_2025\wiki_art_dataset\wikiart"

    # Create dataset instance
    dataset = WikiArtDataset(artist_csv=artist_csv, genre_csv=genres_csv, style_csv=styles_csv, img_dir=img_dir)

