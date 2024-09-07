import torch
import torch.nn as nn 

class CnnModel(nn.Model):
    def __init__(self):
        super(CnnModel, self).__init__()
        # First Convolutional Layer: 
        # - Input channels: 1 (for grayscale images like MNIST)
        # - Output channels: 32 (number of filters)
        # - Kernel size: 3x3
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)

        # Second Convolutional Layer: 
        # - Input channels: 32 (from the output of conv1)
        # - Output channels: 64
        # - Kernel size: 3x3
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        # Pooling layer to downsample after convolutions
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layer
        # Assuming the input image size is 28x28 (MNIST size), after two 2x2 pooling operations,
        # the spatial size will be reduced to 7x7.
        # 64 filters of size 7x7 will be the input to the fully connected layer.
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=128)

        # Output layer: 10 output classes for MNIST digits (0-9)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

        def forward(self, x):
            # 1st Conv Layer + ReLU + Pooling
            x = self.pool(torch.relu(self.conv1(x)))  # Output: 32 channels, 14x14
            
            # 2nd Conv Layer + ReLU + Pooling
            x = self.pool(torch.relu(self.conv2(x)))  # Output: 64 channels, 7x7
            
            # Flatten for fully connected layer
            x = x.view(-1, 64 * 7 * 7)
            
            # 1st Fully Connected Layer + ReLU
            x = torch.relu(self.fc1(x))
            
            # Output Layer (for classification)
            x = self.fc2(x)
            
            return x
