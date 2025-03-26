'''
Th
'''

import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torchvision.io import read_image
from torchvision.transforms import transforms
from torch.utils.data import Dataset

class WikiArtDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file containing image paths and labels (artist, genre, style).
            img_dir (str): Path to the directory containing images.
            transform (callable, optional): Optional transformations to apply on the images.
        """
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform  # Default is None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Returns:
            image (torch.Tensor): Image tensor in (C, H, W) format.
            labels (torch.Tensor): Tensor of (artist, genre, style).
        """
        row = self.data.iloc[index]
        img_path = os.path.join(self.img_dir, row['image_path'])

        # Check if image file exists
        if not os.path.exists(img_path):
            print(f" File not found: {img_path}")

        # Read and normalize image
        image = read_image(img_path).float() / 255.0  

        # Convert labels to tensor
        labels = torch.tensor([int(row['artist']), int(row['genre']), int(row['style'])], dtype=torch.long)

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        return image, labels


# EXAMPLE USAGE
if __name__ == '__main__':
    csv_file = r"make_dataset\files\class_wiki_art.csv"
    img_dir = r"F:\GSoc_2025\wiki_art_dataset\wikiart"

    transform = transforms.Compose([
        transforms.Resize((225, 225)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Create dataset instance
    dataset = WikiArtDataset(csv_file=csv_file, img_dir=img_dir, transform=None)

    print(f"Dataset size: {len(dataset)}")
    
    # Sample image and labels
    image, labels = dataset[100]  # Change index as needed
    print("Labels:", labels)

    # Display image
    plt.imshow(image.permute(1, 2, 0))
    plt.show()
