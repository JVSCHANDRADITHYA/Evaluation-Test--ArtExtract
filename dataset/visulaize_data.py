from dataset import WikiArtDataset
from PIL import Image
import random
import os
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage


to_pil = ToPILImage()

class WikiArtVisualizer:
    def __init__(self, dataset):
        self.dataset = dataset
    
    def show_random_images(self, num_rows=4, num_cols=4):
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))
        
        for ax in axes.flatten():
            index = random.randint(0, len(self.dataset) - 1)
            image, labels = self.dataset[index]
            img_pil = to_pil(image)
            img_size = img_pil.size  # (width, height)
            img_path = self.dataset.data.iloc[index]['image_path']
            
            # Extract label details
            artist, genre, style = labels.tolist()
            ax.imshow(img_pil)
            ax.set_title(f"Artist: {artist}\nGenre: {genre}\nStyle: {style}\nSize: {img_size}")
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        
if __name__ == '__main__':
    artist_csv = r"F:\GSoc_2025\wiki_art_dataset\wikiart\artist_train.csv"
    styles_csv = r"F:\GSoc_2025\wiki_art_dataset\wikiart\style_train.csv"  
    genres_csv = r"F:\GSoc_2025\wiki_art_dataset\wikiart\genre_train.csv"
    img_dir = r"F:\GSoc_2025\wiki_art_dataset\wikiart"



    
    # Create dataset instance
    dataset = WikiArtDataset(artist_csv=artist_csv, genre_csv=genres_csv, style_csv=styles_csv, img_dir=img_dir)
    
    visualizer = WikiArtVisualizer(dataset)
    visualizer.show_random_images()