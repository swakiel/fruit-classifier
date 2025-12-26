import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

from data.dataset import FruitDataset 

class FruitTorchDataset(Dataset):
    def __init__(self, fruit_dataset, transform=None):
        """
        Wraps existing FruitDataset to work with PyTorch
        """
        self.fruit_dataset = fruit_dataset
        self.transform = transform

    def __len__(self):
        return len(self.fruit_dataset)

    def __getitem__(self, idx):
        image, label = self.fruit_dataset[idx]  # PIL Image

        if self.transform:
            image = self.transform(image)  # apply transforms

        return image, label
