'''
This script visualizer.py is used to visualize the images and their corresponding labels from the WikiArt dataset. 
The WikiArtVisualizer class is created which takes in the dataset instance as input. It imports the WikiartDataset class from the dataset.py file.
The show_random_images method is used to display random images from the dataset along with their labels and size.
This can be used to visualize the dataset and check if the images and labels are loaded correctly.
The example usage of the WikiArtVisualizer class is also provided in the code.
'''

from dataset import WikiArtDataset
from PIL import Image
import random
import os
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import ToPILImage
from torchvision.transforms import transforms


to_pil = ToPILImage()

class WikiArtVisualizer:
    def __init__(self, dataset):
        self.dataset = dataset
    
    def show_random_images(self, num_rows=4, num_cols=4):
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))
        
        for ax in axes.flatten():
            index = random.randint(0, len(self.dataset) - 1)
            image, labels = self.dataset[index]
            
            if isinstance(image, torch.Tensor):
                img_pil = to_pil(image)
            else:
                img_pil = image
                                
            img_size = img_pil.size  # (width, height)
            img_path = self.dataset.data.iloc[index]['image_path']
            
            # Extract label details
            artist, genre, style = labels.tolist()
            ax.imshow(img_pil)
            ax.set_title(f"Artist: {artist}\nGenre: {genre}\nStyle: {style}\nSize: {img_size}")
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        
# EXAMPLE USAGE
if __name__ == '__main__':

    img_dir = r"F:\GSoc_2025\wiki_art_dataset\wikiart"


    tran = transforms.Compose([
        transforms.Resize((225, 225)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Create dataset instance
    dataset = WikiArtDataset(img_dir=img_dir, transform=None)
    
    visualizer = WikiArtVisualizer(dataset)
    visualizer.show_random_images(2, 2)