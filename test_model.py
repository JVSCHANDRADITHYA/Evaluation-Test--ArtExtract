import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm

from model.Conv_LSTM_flatten import CNN_LSTM_Model
from dataset.dataset import WikiArtDataset

test_data_path = r"csv_files\test_data.csv"
img_dir = r"F:\GSoc_2025\wiki_art_dataset\wikiart"
model_path = r"cnn_lstm_patch_model.pth"

tran = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match ResNet input
])

test_data = WikiArtDataset(test_data_path, img_dir, transform=tran)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN_LSTM_Model().to(device)
model.load_state_dict(torch.load(model_path), strict=False)
model.eval()

correct_artist = 0
correct_genre = 0
correct_style = 0
total = 0

with torch.no_grad():
    for images, labels in tqdm(test_loader):
        images = images.to(device)
        artist_labels = labels[:, 0].to(device)
        genre_labels = labels[:, 1].to(device)
        style_labels = labels[:, 2].to(device)

        artist_pred, genre_pred, style_pred = model(images)

        _, artist_pred_labels = torch.max(artist_pred, 1)
        _, genre_pred_labels = torch.max(genre_pred, 1)
        _, style_pred_labels = torch.max(style_pred, 1)

        correct_artist += (artist_pred_labels == artist_labels).sum().item()
        correct_genre += (genre_pred_labels == genre_labels).sum().item()
        correct_style += (style_pred_labels == style_labels).sum().item()
        total += labels.size(0)

# === Final Accuracy ===
print(f" Artist Accuracy: {100 * correct_artist / total:.2f}%")
print(f" Genre Accuracy:  {100 * correct_genre / total:.2f}%")
print(f" Style Accuracy:  {100 * correct_style / total:.2f}%")