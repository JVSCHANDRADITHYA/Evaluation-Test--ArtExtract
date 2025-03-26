'''
In the original dataset, the genre of the images is labelled but only with 10 classes. The updated completed labeled dataset has 11 classes in the artist causing the genre labels to be shifted by 1.
This script is used to determine the genre of the images in the dataset. The genre is determined by analyzing the images in it. This script can be used to check the images of every class.

However, I was not able to determine a name for the 139th class and hence named is as uncla
'''


import os
import cv2
import pandas as pd
import time
import matplotlib.pyplot as plt

# Path to CSV file and image directory
csv_file = r"make_dataset\class_wiki_art.csv"  # Update with actual path
img_dir = r"F:\GSoc_2025\wiki_art_dataset\wikiart"  # Update with actual path

# Read the CSV file
data = pd.read_csv(csv_file)    
data = data[data['genre'] == 136]  # Filter for genre 139

for i in range(len(data)):  # Display 5 images
    img_path = os.path.join(img_dir, data.iloc[i]['image_path'])
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title(data.iloc[i]['genre'])
    plt.show()
