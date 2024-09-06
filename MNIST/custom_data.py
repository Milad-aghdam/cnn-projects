import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

class CustomMNISTDataset(Dataset):
    def __init__(self, train=True, transform=None):
        self.mnist_data = datasets.MNIST(root='./MNIST/data', train=train, download=True)
        self.transform = transform

    def __len__(self):
        return len(self.mnist_data)

    def __getitem__(self, idx):
        image, label = self.mnist_data[idx]

        if self.transform:
            image = self.transform(image)
        
        print(f"Image shape: {image.shape}, Label shape: {label}")
        return image, label
