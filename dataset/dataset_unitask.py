'''
The WikiArtDataset class is a multi-task dataset class that loads the artist, genre, and style labels for each image from separate CSV files but is largely unlabelled for specific tasks.
This requires seperate tasks to be created for each label and thus is done by using the SingleTaskWikiArtDataset class.
Unline WikiArtDataset, SingleTaskWikiArtDataset is a single-task dataset class that loads the artist, genre, or style labels for each image from a single CSV file.
'''

import os
import pandas as pd
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

class SingleTaskWikiArtDataset(Dataset):
    def __init__(self, csv_file, img_dir, label_column, transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file containing image paths and labels.
            img_dir (str): Path to the image directory.
            label_column (str): The name of the column to use as labels (artist, genre, or style).
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_column = label_column
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_id = self.data.iloc[index]['image_path']
        img_path = os.path.join(self.img_dir, img_id)

        if not os.path.exists(img_path):
            print(f"File not found: {img_path}")

        # Read image and convert to float tensor
        image = read_image(img_path).float() / 255.0  

        # Extract the label
        label = torch.tensor(int(self.data.iloc[index][self.label_column]), dtype=torch.long)

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        return image, label


# EXAMPLE USAGE
if __name__ == '__main__': 
    # Define dataset paths for artist, genre, and style separately
    artist_csv = r"csv_files\artist\artist_train.csv"
    genre_csv = r"csv_files\genre\genre_train.csv"
    style_csv = r"csv_files\style\style_train.csv"
    img_dir = r"F:\GSoc_2025\wiki_art_dataset\wikiart"

    transform = transforms.Compose([
        transforms.Resize((225, 225)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Create dataset instances separately for each task
    artist_dataset = SingleTaskWikiArtDataset(csv_file=artist_csv, img_dir=img_dir, label_column='artist', transform=transform)
    genre_dataset = SingleTaskWikiArtDataset(csv_file=genre_csv, img_dir=img_dir, label_column='genre', transform=transform)
    style_dataset = SingleTaskWikiArtDataset(csv_file=style_csv, img_dir=img_dir, label_column='style', transform=transform)

    print("Artist Dataset Size:", len(artist_dataset))
    print("Genre Dataset Size:", len(genre_dataset))
    print("Style Dataset Size:", len(style_dataset))

    # Example visualization
    image, label = artist_dataset[0]  # Change to genre_dataset or style_dataset as needed
    print("Label:", label.item())
    plt.imshow(image.permute(1, 2, 0))
    plt.show()
