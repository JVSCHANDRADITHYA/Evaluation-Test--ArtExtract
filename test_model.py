import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import csv

# ======= Load model and dataset classes =======
from model.Conv_LSTM_patch import CNN_LSTM_Model
from dataset.dataset import WikiArtDataset

# ======= Paths =======
test_data_path = r"csv_files\test_data.csv"
img_dir = r"F:\GSoc_2025\wiki_art_dataset\wikiart"
model_path = r"checkpoints\cnn_lstm_patch_model.pth"

# ======= Transforms =======
tran = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match ResNet input
])

# ======= Load label names =======
def load_class_names(csv_path):
    with open(csv_path, "r", encoding='utf-8') as f:
        reader = csv.reader(f)
        class_dict = {int(row[0]): row[1] for row in reader}
    return [class_dict[i] for i in sorted(class_dict)]

artist_classes = load_class_names(r"csv_files\class_lables\artist_class.csv")
genre_classes = load_class_names(r"csv_files\class_lables\genre_class.csv")
style_classes = load_class_names(r"csv_files\class_lables\style_class.csv")

# ======= Load test data =======
test_data = WikiArtDataset(test_data_path, img_dir, transform=tran)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======= Load model =======
model = CNN_LSTM_Model().to(device)
model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
model.eval()

# ======= Accuracies =======
correct_artist = 0
correct_genre = 0
correct_style = 0
total = 0

# ======= For visualizing 3 correctly classified samples =======
correct_samples = []

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

        # Accuracy counters
        correct_artist += (artist_pred_labels == artist_labels).sum().item()
        correct_genre += (genre_pred_labels == genre_labels).sum().item()
        correct_style += (style_pred_labels == style_labels).sum().item()
        total += labels.size(0)

        # Save correct all-3 predictions (max 3)
        if (artist_labels == artist_pred_labels and
            genre_labels == genre_pred_labels and
            style_labels == style_pred_labels):
            if len(correct_samples) < 3:
                img_np = images[0].detach().cpu().permute(1, 2, 0).numpy()
                correct_samples.append((
                    img_np,
                    artist_labels.item(), artist_pred_labels.item(),
                    genre_labels.item(), genre_pred_labels.item(),
                    style_labels.item(), style_pred_labels.item()
                ))

# ======= Final Accuracy =======
print(f"\nFinal Results on Test Set")
print(f" Artist Accuracy: {100 * correct_artist / total:.2f}%")
print(f" Genre Accuracy:  {100 * correct_genre / total:.2f}%")
print(f" Style Accuracy:  {100 * correct_style / total:.2f}%")

# ======= Plot 3 Correct Samples =======
if len(correct_samples) > 0:
    fig, axs = plt.subplots(1, len(correct_samples), figsize=(18, 6))
    for idx, (img_np, a_true, a_pred, g_true, g_pred, s_true, s_pred) in enumerate(correct_samples):
        axs[idx].imshow(img_np)
        axs[idx].axis("off")
        axs[idx].set_title(
            f"Artist: {artist_classes[a_pred]}\nGenre: {genre_classes[g_pred]}\nStyle: {style_classes[s_pred]}",
            fontsize=10
        )
        axs[idx].set_xlabel(
            f"(GT: {artist_classes[a_true]}, {genre_classes[g_true]}, {style_classes[s_true]})",
            fontsize=9
        )

    plt.tight_layout()
    plt.suptitle("✅ Correctly Classified (All 3)", fontsize=16, y=1.05)
    plt.show()
else:
    print("No completely correct predictions to display.")
