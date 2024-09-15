import torch
import torch.nn as nn

class CnnModel(nn.Module):
    def __init__(self):
        super(CnnModel, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Adjusted based on your input size, assuming 128x128 input image
        self.fc1 = nn.Linear(256 * 8 * 8, 512)    # Corrected to 16384
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(in_features=512, out_features=128)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(in_features=128, out_features=64)
        self.dropout3 = nn.Dropout(0.5)

        # Output layer for binary classification with sigmoid activation
        self.fc4 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))

        # Print the shape before flattening
        # print(f"Shape before flattening: {x.shape}")
        
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)

        x = torch.relu(self.fc3(x))
        x = self.dropout3(x)
        
        x = torch.sigmoid(self.fc4(x))  # Use sigmoid for binary classification
        x = x.squeeze(1)
        return x
