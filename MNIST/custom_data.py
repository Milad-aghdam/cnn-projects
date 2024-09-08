import matplotlib.pyplot as plt
import torch
from torchvision import datasets
from torch.utils.data import Dataset

class CustomMNISTDataset(Dataset):
    def __init__(self, train=True, transform=None, root='./MNIST/data'):
        self.mnist_data = datasets.MNIST(root=root , train=train, download=True)
        self.transform = transform

    def __len__(self):
        return len(self.mnist_data)

    def __getitem__(self, idx):
        image, lable  = self.mnist_data[idx]

        if self.transform:
            image = self.transform(image)
        
        # print(f"Image shape: {image.shape}, lable  shape: {label}")
        return image, lable