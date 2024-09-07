import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from custom_data import CustomMNISTDataset  # Import your dataset
from model import CnnModel  # Import your model

# Model Training Parameters
batch_size = 64
num_epochs = 10
learning_rate = 0.001
random_seed = 42  # For reproducibility
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # GPU or CPU

transform = transforms.ToTensor()


train_dataset = CustomMNISTDataset(train=True, transform=transform)
train_size = int(0.7 * len(train_dataset))  # 70% for training
val_size = len(train_dataset) - train_size   # Remaining 30% for validation

random_seed = 42
generator = torch.Generator().manual_seed(random_seed)

# Perform the split
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=generator)

#Load the testing dataset (pre-split by MNIST)
test_dataset = CustomMNISTDataset(train=False, transform=transform)

# Create DataLoaders for training, validation, and testing sets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)