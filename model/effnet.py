import torch.nn as nn
import torch
import torchvision.models as models
import torch.nn.functional as F

class EfficientNet(nn.Module):
    def __init__(self, num_classes=128):
        super(EfficientNet, self).__init__()
        # Load a pre-trained EfficientNet model
        self.base_model = models.efficientnet_b0(pretrained=True)
        
        # Replace the classifier with a new one for our specific task
        self.base_model.classifier = nn.Linear(self.base_model.classifier., num_classes)
        # Optionally, freeze the base model parameters
        # for param in self.base_model.parameters():
        #     param.requires_grad = False
        # Unfreeze the classifier layer
        for param in self.base_model.classifier.parameters():
            param.requires_grad = True
        # Add a dropout layer for regularization   
        self.dropout = nn.Dropout(p=0.5)
        # Add a batch normalization layer
        self.batch_norm = nn.BatchNorm1d(num_classes)
        # Add a ReLU activation function
        self.relu = nn.ReLU()
        # Add a final fully connected layer
        self.fc = nn.Linear(num_classes, num_classes)
        # Initialize weights
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.fc.bias, 0)
        # Initialize batch normalization
        nn.init.constant_(self.batch_norm.weight, 1)
        nn.init.constant_(self.batch_norm.bias, 0)
        # Initialize dropout
        nn.init.constant_(self.dropout.weight, 1)
        nn.init.constant_(self.dropout.bias, 0)
        # Initialize ReLU
        nn.init.constant_(self.relu.weight, 1)
        nn.init.constant_(self.relu.bias, 0)
        # Initialize classifier
        nn.init.kaiming_normal_(self.base_model.classifier.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.base_model.classifier.bias, 0)
        # Initialize base model
        nn.init.kaiming_normal_(self.base_model.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.base_model.bias, 0)
      
        
    def forward(self, x):
        x = self.base_model(x)
        return x
    
# Example usage
if __name__ == "__main__":
    model = EfficientNet(num_classes=128)
    print(model)
    
    # Dummy input
    x = torch.randn(1, 3, 224, 224)  # Batch size of 1, 3 channels, 224x224 image
    output = model(x)
    print(output.shape)  # Should be [1, num_classes]