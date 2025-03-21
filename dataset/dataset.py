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
'''

import os
import numpy as np
import cv2
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
    
class WikiArtDataset(Dataset):
    def __init__(self, artist_csv, genre_csv, style_csv, img_dir, transform=None):
        '''
        Args:
            artist_csv (str): Path to the CSV file containing artist labels for each image.  
            genre_csv (str): Path to the CSV file containing genre labels for each image.
            style_csv (str): Path to the CSV file containing style labels for each image.   
            
            img_dir (str): Path to the image directory. 
            transform (callable, optional): Optional transform to be applied on an image.
        '''
        self.artist_labels = pd.read_csv(artist_csv)
        self.genre_labels = pd.read_csv(genre_csv)
        self.style_labels = pd.read_csv(style_csv)
        
        self.data = self.artist_labels.merge(self.genre_labels, on='image_path').merge(self.style_labels, on='image_path')

        self.img_dir = img_dir
        self.transform = transform # None taken if None given

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        # returns image and labels as tensor 
        # image is in (C, H, W) format i.e Channel, Height, Width and needs to be converted bcak to (H, W, C) format for plotting
        
        img_id = self.data.iloc[index]['image_path']
        img_path = os.path.join(self.img_dir, img_id)
        
        if not os.path.exists(img_path):
            print(f"File not found: {img_path}")
             
        # Read image
        # Convert to float tensor for better handling you can just omit the float / 255.0 if you want to keep it as int tensor
        image = read_image(img_path).float() /255.0  
        
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

    tran = transforms.Compose([
        transforms.Resize((225, 225)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    
    # Create dataset instance
    dataset = WikiArtDataset(artist_csv=artist_csv, genre_csv=genres_csv, style_csv=styles_csv, img_dir=img_dir, transform=None)

    print(len(dataset))
    
    image, labels = dataset[8000]
    print(image, labels)
    plt.imshow(image.permute(1, 2, 0))
    plt.show()
