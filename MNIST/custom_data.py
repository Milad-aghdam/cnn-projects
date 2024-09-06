import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class CustomMNISTDataset(Dataset):
    def __init__(self, train=True, transform=None):
        """
        Initialize the dataset by loading MNIST using torchvision.datasets.
        Args:
        - train (bool): Whether to load the training or testing dataset.
        - transform (callable): Optional transform to be applied on a sample.
        """
        # Load the MNIST dataset from torchvision
        self.mnist_data = datasets.MNIST(root='./MNIST/data', train=train, download=True, transform=transform)
        self.transform = transform

    def __len__(self):
        """
        Return the size of the dataset.
        """
        return len(self.mnist_data)

    def __getitem__(self, idx):
        """
        Retrieve a specific image and its corresponding label by index.
        Args:
        - idx (int): Index of the sample to retrieve.
        Returns:
        - (feature, target): Tuple of image (as a tensor) and its label (as a tensor).
        """
        # Use torchvision's MNIST dataset to get the item at index `idx`
        image, label = self.mnist_data[idx]
        
        # Apply transformations if any (already handled by torchvision dataset)
        if self.transform:
            image = self.transform(image)

        # Return the image and label
        return image, label

# Define transformations for the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),                # Convert image to PyTorch tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize to mean=0.5, std=0.5 for 1 channel (grayscale)
])

# Instantiate the custom dataset
train_dataset = CustomMNISTDataset(train=True, transform=transform)
test_dataset = CustomMNISTDataset(train=False, transform=transform)

# Create DataLoaders for batching
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Test: Visualize a sample from the custom dataset
def visualize_sample(dataset, idx):
    """
    Visualize a sample from the dataset at the given index.
    Args:
    - dataset (CustomMNISTDataset): The custom dataset instance.
    - idx (int): Index of the sample to visualize.
    """
    feature, label = dataset[idx]  # Retrieve the sample
    
    # Plot the image
    plt.imshow(feature.squeeze(0), cmap='gray')  # Remove the channel dimension for plotting
    plt.title(f'Label: {label.item()}')
    plt.show()

# Visualize the first sample from the training dataset
visualize_sample(train_dataset, idx=0)
