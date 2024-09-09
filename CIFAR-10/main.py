import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

import matplotlib.pyplot as plt
from custom_data import CustomCifarDataset
from model import CnnModel

num_classes = 10  #
batch_size = 64
num_epochs = 10
learning_rate = 0.001
random_seed = 42  # For reproducibility
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Augmentation
    transforms.RandomCrop(32, padding=4),  # Augmentation
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalization
])

train_dataset = CustomCifarDataset(transform=transform)
train_size = int(0.8 * len(train_dataset)) 
val_size = len(train_dataset) - train_size

random_seed = 42
generator = torch.Generator().manual_seed(random_seed)

# Perform the split
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=generator)

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Same normalization as training
])
test_dataset = CustomCifarDataset(train=False, transform=test_transform)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Initialize the model, loss function, and optimizer
model = CnnModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
