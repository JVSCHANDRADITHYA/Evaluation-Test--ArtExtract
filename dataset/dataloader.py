'''
This file creates and returns dataloader for the datasets WikiArtDataset, inWikiArtDataset, and SingleTaskWikiArtDataset.
    batch_size (int): Number of samples per batch.
    shuffle (bool): Whether to shuffle the data at every epoch.
'''

from torch.utils.data import DataLoader
from dataset.dataset_unitask import SingleTaskWikiArtDataset
from dataset.dataset import WikiArtDataset
from dataset.dataset_incomplete import inWikiArtDataset

def create_dataloader(dataset, batch_size=32, shuffle=True):
    """
    Create a DataLoader for the dataset.

    Args:
        dataset (Dataset): The dataset to load data from.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        DataLoader: DataLoader object for the dataset.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def create_WikiArtDataloader(csv_file, img_dir, label_column, transform):
    """
    Create a DataLoader for the WikiArtDataset.

    Returns:
        DataLoader: DataLoader object for the WikiArtDataset.
    """
    # Define paths to your dataset and labels
    
    dataset = WikiArtDataset(csv_file=csv_file, img_dir=img_dir, transform=transform)
    return create_dataloader(dataset)

def create_inWikiArtDataloader(csv_file, img_dir, label_column, transform):
    """
    Create a DataLoader for the inWikiArtDataset.

    Returns:
        DataLoader: DataLoader object for the inWikiArtDataset.
    """
    # Define paths to your dataset and labels
    
    dataset = inWikiArtDataset(csv_file=csv_file, img_dir=img_dir, label_column=label_column, transform=transform)
    return create_dataloader(dataset)

def create_SingleWikiArtDataloader(csv_file, img_dir, label_column, transform):
    """
    Create a DataLoader for the SingleTaskWikiArtDataset.

    Args:
        csv_file (str): Path to the CSV file containing image paths and labels.
        img_dir (str): Path to the image directory.
        label_column (str): The name of the column to use as labels (artist, genre, or style).
        transform (callable, optional): Optional transform to be applied on an image.

    Returns:
        DataLoader: DataLoader object for the SingleTaskWikiArtDataset.
    """
    dataset = SingleTaskWikiArtDataset(csv_file=csv_file, img_dir=img_dir, label_column=label_column, transform=transform)
    return create_dataloader(dataset)