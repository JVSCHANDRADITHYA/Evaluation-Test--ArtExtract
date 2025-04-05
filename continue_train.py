# Same setup
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset.dataset import WikiArtDataset
from model.Conv_LSTM_patch import CNN_LSTM_Model
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model + weights
model = CNN_LSTM_Model().to(device)
model.load_state_dict(torch.load("cnn_lstm_patch_model.pth"))

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
])

# Dataset paths
img_dir = r"F:\GSoc_2025\wiki_art_dataset\wikiart"
train_dataset = WikiArtDataset(csv_file=r"csv_files\train_data.csv", img_dir=img_dir, transform=transform)
test_dataset = WikiArtDataset(csv_file=r"csv_files\test_data.csv", img_dir=img_dir, transform=transform)

# Dataloaders
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
            correct_artist += (artist_pred.argmax(1) == artist_labels).sum().item()
            correct_genre += (genre_pred.argmax(1) == genre_labels).sum().item()
            correct_style += (style_pred.argmax(1) == style_labels).sum().item()
            total += artist_labels.size(0)
    return (
        100 * correct_artist / total,
        100 * correct_genre / total,
        100 * correct_style / total,
    )

# Resume training from epoch 11
start_epoch = 11
end_epoch = 20  # or go more

for epoch in range(start_epoch, end_epoch + 1):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{end_epoch}")

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
    artist_acc, genre_acc, style_acc = evaluate(model, test_loader)

    print(f"\n📊 Epoch [{epoch}/{end_epoch}]")
    print(f"Loss: {avg_loss:.4f}")
    print(f"🎨 Test Artist Accuracy: {artist_acc:.2f}%")
    print(f"🎼 Test Genre Accuracy: {genre_acc:.2f}%")
    print(f"🎭 Test Style Accuracy: {style_acc:.2f}%\n")

    # Save model for every epoch
    torch.save(model.state_dict(), f"cnn_lstm_patch_model_epoch{epoch}.pth")
    print(f"✅ Model saved for epoch {epoch}")
