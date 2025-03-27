import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm


# ================== Dataset Class ==================
class WikiArtDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        img_path = os.path.join(self.img_dir, row['image_path'])

        if not os.path.exists(img_path):
            print(f"File not found: {img_path}")

        image = read_image(img_path).float() / 255.0
        labels = torch.tensor([int(row['artist']), int(row['genre']), int(row['style'])], dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, labels


# ================== CNN Feature Extractor ==================
class CNNFeatureExtractor(nn.Module):
    def __init__(self, backbone="resnet50"):
        super(CNNFeatureExtractor, self).__init__()
        from torchvision.models import resnet50, resnet18

        if backbone == "resnet50":
            model = resnet50(pretrained=True)
            self.feature_dim = 2048
        elif backbone == "resnet18":
            model = resnet18(pretrained=True)
            self.feature_dim = 512
        else:
            raise ValueError("Unsupported CNN Backbone")

        self.cnn = nn.Sequential(*list(model.children())[:-1])  # Remove final FC layer

    def forward(self, x):
        x = self.cnn(x)
        return x.view(x.size(0), -1)  # Flatten


# ================== RNN Sequence Model ==================
class RNNSequenceModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, rnn_type="gru"):
        super(RNNSequenceModel, self).__init__()
        self.rnn_type = rnn_type
        self.hidden_dim = hidden_dim

        if rnn_type == "gru":
            self.rnn = nn.GRU(input_dim, hidden_dim, bidirectional=True, batch_first=True)
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(input_dim, hidden_dim, bidirectional=True, batch_first=True)
        else:
            raise ValueError("Unsupported RNN type")

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        return rnn_out[:, -1, :]  # Take last time-step output


# ================== Classification Heads ==================
class ClassificationHeads(nn.Module):
    def __init__(self, input_dim, num_classes_artist, num_classes_genre, num_classes_style):
        super(ClassificationHeads, self).__init__()
        self.fc_artist = nn.Linear(input_dim, num_classes_artist)
        self.fc_genre = nn.Linear(input_dim, num_classes_genre)
        self.fc_style = nn.Linear(input_dim, num_classes_style)

    def forward(self, x):
        return self.fc_artist(x), self.fc_genre(x), self.fc_style(x)


# ================== CRNN Model ==================
class CRNNModel(nn.Module):
    def __init__(self, cnn_backbone="resnet50", rnn_type="gru", hidden_dim=256,
                 num_classes_artist=100, num_classes_genre=50, num_classes_style=30):
        super(CRNNModel, self).__init__()

        self.cnn = CNNFeatureExtractor(backbone=cnn_backbone)
        self.rnn = RNNSequenceModel(input_dim=self.cnn.feature_dim, hidden_dim=hidden_dim, rnn_type=rnn_type)
        self.classifier = ClassificationHeads(input_dim=hidden_dim * 2,  # Bidirectional RNN
                                              num_classes_artist=num_classes_artist,
                                              num_classes_genre=num_classes_genre,
                                              num_classes_style=num_classes_style)

    def forward(self, x):
        features = self.cnn(x)  # CNN feature extraction
        rnn_out = self.rnn(features.unsqueeze(1))  # Add sequence dimension
        artist_out, genre_out, style_out = self.classifier(rnn_out)
        return artist_out, genre_out, style_out


# ================== Training and Validation ==================
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0

    for images, labels in tqdm(dataloader, desc="Training"):
        images, artist_labels, genre_labels, style_labels = images.to(device), labels[:, 0].to(device), labels[:, 1].to(device), labels[:, 2].to(device)

        # Forward pass
        artist_preds, genre_preds, style_preds = model(images)

        # Compute loss
        loss_artist = criterion["artist"](artist_preds, artist_labels)
        loss_genre = criterion["genre"](genre_preds, genre_labels)
        loss_style = criterion["style"](style_preds, style_labels)

        total_loss = loss_artist + loss_genre + loss_style

        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.item()

    return epoch_loss / len(dataloader)


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    correct_artist, correct_genre, correct_style = 0, 0, 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images, artist_labels, genre_labels, style_labels = images.to(device), labels[:, 0].to(device), labels[:, 1].to(device), labels[:, 2].to(device)

            # Forward pass
            artist_preds, genre_preds, style_preds = model(images)

            # Compute loss
            loss_artist = criterion["artist"](artist_preds, artist_labels)
            loss_genre = criterion["genre"](genre_preds, genre_labels)
            loss_style = criterion["style"](style_preds, style_labels)

            total_loss = loss_artist + loss_genre + loss_style
            epoch_loss += total_loss.item()

            # Compute accuracy
            _, predicted_artist = artist_preds.max(1)
            _, predicted_genre = genre_preds.max(1)
            _, predicted_style = style_preds.max(1)

            correct_artist += (predicted_artist == artist_labels).sum().item()
            correct_genre += (predicted_genre == genre_labels).sum().item()
            correct_style += (predicted_style == style_labels).sum().item()

            total_samples += artist_labels.size(0)

    acc_artist = correct_artist / total_samples
    acc_genre = correct_genre / total_samples
    acc_style = correct_style / total_samples

    return epoch_loss / len(dataloader), acc_artist, acc_genre, acc_style


# ================== Main Training Script ==================
if __name__ == "__main__":
    # Paths
    csv_file = r"make_dataset\files\class_wiki_art.csv"
    img_dir = r"F:\GSoc_2025\wiki_art_dataset\wikiart"

    # Dataset
    transform = transforms.Compose([
        transforms.Resize((225, 225)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = WikiArtDataset(csv_file, img_dir, transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    # Model, Loss & Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNNModel().to(device)
    criterion = {"artist": nn.CrossEntropyLoss(), "genre": nn.CrossEntropyLoss(), "style": nn.CrossEntropyLoss()}
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # Train Loop
    num_epochs = 20
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc_artist, val_acc_genre, val_acc_style = validate_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        torch.save(model.state_dict(), f"crnn_epoch_{epoch+1}.pth")
