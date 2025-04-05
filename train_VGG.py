import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset.dataset import WikiArtDataset
from model.Conv_LSTM_VGG import VGG_LSTM_Model
from tqdm import tqdm

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
model = VGG_LSTM_Model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
])

# Directories
img_dir = r"F:\GSoc_2025\wiki_art_dataset\wikiart"

# Datasets
train_dataset = WikiArtDataset(csv_file=r"csv_files\train_data.csv", img_dir=img_dir, transform=transform)
test_dataset = WikiArtDataset(csv_file=r"csv_files\test_data.csv", img_dir=img_dir, transform=transform)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

# Accuracy function
def evaluate(model, loader):
    model.eval()
    correct_artist, correct_genre, correct_style = 0, 0, 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            artist_labels, genre_labels, style_labels = labels[:, 0], labels[:, 1], labels[:, 2]
            images = images.to(device)
            artist_labels = artist_labels.to(device)
            genre_labels = genre_labels.to(device)
            style_labels = style_labels.to(device)

            artist_pred, genre_pred, style_pred = model(images)

            _, artist_preds = torch.max(artist_pred, 1)
            _, genre_preds = torch.max(genre_pred, 1)
            _, style_preds = torch.max(style_pred, 1)

            correct_artist += (artist_preds == artist_labels).sum().item()
            correct_genre += (genre_preds == genre_labels).sum().item()
            correct_style += (style_preds == style_labels).sum().item()
            total += artist_labels.size(0)

    return (
        100 * correct_artist / total,
        100 * correct_genre / total,
        100 * correct_style / total,
    )

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for images, labels in progress_bar:
        artist_labels, genre_labels, style_labels = labels[:, 0], labels[:, 1], labels[:, 2]
        images = images.to(device)
        artist_labels = artist_labels.to(device)
        genre_labels = genre_labels.to(device)
        style_labels = style_labels.to(device)

        optimizer.zero_grad()
        artist_pred, genre_pred, style_pred = model(images)

        loss1 = criterion(artist_pred, artist_labels)
        loss2 = criterion(genre_pred, genre_labels)
        loss3 = criterion(style_pred, style_labels)

        loss = loss1 + loss2 + loss3
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

    avg_loss = running_loss / len(train_loader)

    # Evaluate on test set
    artist_acc, genre_acc, style_acc = evaluate(model, test_loader)

    print(f"\n📊 Epoch [{epoch+1}/{num_epochs}]")
    print(f"Loss: {avg_loss:.4f}")
    print(f"🎨 Test Artist Accuracy: {artist_acc:.2f}%")
    print(f"🎼 Test Genre Accuracy: {genre_acc:.2f}%")
    print(f"🎭 Test Style Accuracy: {style_acc:.2f}%\n")

# Save model
    torch.save(model.state_dict(), f"cnn_lstm_patch_model{epoch}.pth")
    print(f"Model saved to cnn_lstm_patch_model.pth at epoch {epoch}")
