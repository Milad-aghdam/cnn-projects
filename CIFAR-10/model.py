import torch
import torch.nn as nn 

class CnnModel(nn.Module):
    def __init__(self):
        super(CnnModel, self).__init__()
        
        # First convolutional block: Conv1 + BatchNorm + ReLU + Pooling
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # BatchNorm after conv1
        
        # Second convolutional block: Conv2 + BatchNorm + ReLU + Pooling
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # BatchNorm after conv2
        
        # Third convolutional block: Conv3 + BatchNorm + ReLU + Pooling
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)  # BatchNorm after conv3
        
        # Max pooling layer to reduce spatial dimensions
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=128 * 4 * 4, out_features=512)
        self.dropout1 = nn.Dropout(0.5)  # Dropout for regularization
        
        self.fc2 = nn.Linear(in_features=512, out_features=128)
        self.dropout2 = nn.Dropout(0.5)  # Dropout for regularization
        
        self.fc3 = nn.Linear(in_features=128, out_features=10)  # Output for 10 classes

    def forward(self, x):
        # First convolutional block
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        
        # Second convolutional block
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        
        # Third convolutional block
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))

        # Flatten the tensor for the fully connected layers
        x = x.view(-1, 128 * 4 * 4)
        
        # Fully connected layers with dropout
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        
        x = self.fc3(x)  # Final output, no activation here (CrossEntropyLoss will handle it)
        return x
