'''
This is the __init__.py file for the model module.
It contains ConvLSTMModel class which is a PyTorch model that combines a CNN backbone (ResNet50) with an RNN (GRU) layer for multi-label classification tasks.  
It is designed to classify images into multiple categories: artist, genre, and style.
The model uses a pretrained ResNet50 for feature extraction, followed by a GRU layer to capture temporal dependencies in the features. 
Finally, it has fully connected layers to output predictions for each category.
'''