import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Define a model that extracts feature maps
class CNN_FeatureExtractor(nn.Module):
    def __init__(self, cnn_arch="resnet18"):
        super(CNN_FeatureExtractor, self).__init__()

        # Load Pretrained ResNet18
        self.cnn = models.resnet18(pretrained=True)

        # Remove the final classification layer
        self.cnn.fc = nn.Identity()  

        # Hook to store intermediate feature maps
        self.feature_maps = {}

        # Define layers where we want to extract feature maps
        layers_to_hook = {
            "conv1": self.cnn.conv1,            # First convolutional layer (low-level edges)
            "layer1": self.cnn.layer1[0].conv1, # First residual block (mid-level textures)
            "layer2": self.cnn.layer2[0].conv1, # Second residual block
            "layer3": self.cnn.layer3[0].conv1, # Third residual block
            "layer4": self.cnn.layer4[0].conv1, # Fourth residual block (high-level features)
        }

        # Register hooks for each selected layer
        for name, layer in layers_to_hook.items():
            layer.register_forward_hook(self._hook_fn(name))

    def _hook_fn(self, layer_name):
        """Hook function to save feature maps"""
        def hook(module, input, output):
            self.feature_maps[layer_name] = output.detach()
        return hook

    def forward(self, x):
        self.feature_maps = {}  # Reset feature maps before forward pass
        features = self.cnn(x)  # Forward pass through the network
        return features, self.feature_maps

# Load Image
img_path = r"F:\GSoc_2025\wiki_art_dataset\wikiart\Abstract_Expressionism\aki-kuroda_untitled-1988-1.jpg"
img = Image.open(img_path).convert("RGB")

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match ResNet input
    transforms.ToTensor()
])
img = transform(img).unsqueeze(0)  # Add batch dimension

# Create model and get feature maps
model = CNN_FeatureExtractor()
model.eval()

print(model)  # Print model architecture

with torch.no_grad():
    output, feature_maps = model(img)

# Print Feature Shapes
for layer_name, fmap in feature_maps.items():
    print(f"{layer_name} Feature Map Shape: {fmap.shape}")  

# Function to visualize feature maps
def visualize_feature_maps(feature_maps, layer_name, num_maps=8):
    fmap = feature_maps[layer_name].squeeze(0)  # Remove batch dimension (C, H, W)
    num_features = min(fmap.shape[0], num_maps)  # Select first `num_maps` feature maps

    fig, axes = plt.subplots(2, num_features // 2, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        if i >= num_features: break
        ax.imshow(fmap[i].cpu().numpy(), cmap="viridis")
        ax.set_title(f"{layer_name} - {i}")
        ax.axis("off")
    plt.show()

# Visualize different feature map levels
visualize_feature_maps(feature_maps, "conv1")   # Low-level edges
visualize_feature_maps(feature_maps, "layer2")  # Mid-level textures
visualize_feature_maps(feature_maps, "layer4")  # High-level objects
