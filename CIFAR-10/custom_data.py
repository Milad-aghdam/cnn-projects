import matplotlib.pyplot as plt
import torch
from torchvision import datasets
from torch.utils.data import Dataset


class CustomCifarDataset(Dataset):
    def __init__(self, train=True, transform=None, root="./data"):
        self.cifar_data = datasets.CIFAR10(root=root, train=train, download=True)
        self.transform = transform

    def __len__(self):
        return len(self.cifar_data)
    
    def __getitem__(self, idx):
        image, lable = self.cifar_data[idx]

        if self.transform:
            image = self.transform(image)
        return image, lable