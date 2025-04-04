'''
This code implements on top of the torch.utils.data.Dataset class to create a
custom dataset for the WikiArt dataset. The dataset class is called WikiArtDataset and consists of three labels artist, genre and style. 
The dataset class takes in the path to the artist, genre and style csv files and the data directory where multiple subfolders each contianing images are present. 
The __len__ method returns the length of the dataset and the __getitem__ method returns the image and the labels as tensors. 
The image is read using the torchvision.io.read_image function and is converted to a float tensor. The labels are converted to a tensor as well. 
The __getitem__ method also applies the transformations if provided. 

The image is then returned along with the labels in the form of a torch tensor of float type.

The example usage of the WikiArtDataset class is also provided in the code. 
The artist, genre and style csv files and the image directory are provided as input to the WikiArtDataset class.
The image and labels are then printed and the image is displayed using the matplotlib.pyplot.imshow function.

This file is created by Chandradithya Janaswami. This code returns the image and labels as tensors.
'''

import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torchvision.io import read_image
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

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

        if not os.path.exists(img_path):
            print(f" File not found: {img_path}")

        image = read_image(img_path).float() / 255.0  

        # Convert labels to tensor
        labels = torch.tensor([int(row['artist']), int(row['genre']), int(row['style'])], dtype=torch.long)

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
        

        return image, labels


# Helper function to create DataLoader instances
def create_dataloader(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=20)


# EXAMPLE USAGE
if __name__ == '__main__':
    csv_file = r"F:\GSoc_2025\Evaluation-Test--ArtExtract\csv_files\test_data.csv"
    img_dir = r"F:\GSoc_2025\wiki_art_dataset\wikiart"

    transform = transforms.Compose([
        transforms.Resize((225, 225)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Create dataset instance
    dataset = WikiArtDataset(csv_file=csv_file, img_dir=img_dir, transform=transform)
    image, labels = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Labels: {labels}")
    # Display image
    plt.imshow(image.permute(1, 2, 0).numpy())
    plt.show()