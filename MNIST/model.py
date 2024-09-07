import torch
import torch.nn as nn

class CnnModel(nn.Module):  # Ensure it inherits from nn.Module
    def __init__(self, input_channels=1, num_classes=10):
        super(CnnModel, self).__init__()

        # First Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)

        # Second Convolutional Layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        # Pooling layer to downsample
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layer, assuming input image size is 28x28
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=128)

        # Output layer: num_classes controls the number of output classes
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    # The forward method must be defined for every nn.Module
    def forward(self, x):
        # Apply the first convolution, followed by ReLU activation and pooling
        x = self.pool(torch.relu(self.conv1(x)))

        # Apply the second convolution, followed by ReLU activation and pooling
        x = self.pool(torch.relu(self.conv2(x)))

        # Flatten the tensor for the fully connected layer
        x = x.view(-1, 64 * 7 * 7)  # Flatten the tensor to (batch_size, 64*7*7)

        # Apply the fully connected layers
        x = torch.relu(self.fc1(x))

        # Final output layer (no activation here as it's usually applied outside for classification)
        x = self.fc2(x)

        return x  # Return the output
