import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset.dataset import WikiArtDataset
from model.Conv_LSTM import CNN_LSTM_Model
from torchvision import transforms


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN_LSTM_Model().to(device)

criterion_artist = nn.CrossEntropyLoss()
criterion_genre = nn.CrossEntropyLoss()
criterion_style = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

tran = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                         std=[0.229, 0.224, 0.225])
])

csv_file = r"F:\GSoc_2025\Evaluation-Test--ArtExtract\csv_files\train_data.csv"
dataset = WikiArtDataset(csv_file=csv_file, img_dir=r"F:\GSoc_2025\wiki_art_dataset\wikiart", transform=tran)  
# Dataloader (example)
train_loader = DataLoader(dataset=dataset, batch_size=256, shuffle=True)

# Training
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, labels in tqdm(train_loader):
        (artist_labels, genre_labels, style_labels) = labels[:, 0], labels[:, 1], labels[:, 2]
        
        images = images.to(device)
        artist_labels = artist_labels.to(device)
        genre_labels = genre_labels.to(device)
        style_labels = style_labels.to(device)

        optimizer.zero_grad()

        # Forward
        out_artist, out_genre, out_style = model(images)

        # Compute losses
        loss_artist = criterion_artist(out_artist, artist_labels)
        loss_genre = criterion_genre(out_genre, genre_labels)
        loss_style = criterion_style(out_style, style_labels)

        # Total loss (you can weight them if needed)
        loss = loss_artist + loss_genre + loss_style

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss:.4f}")
    #print test accuracies at every epoch
    
