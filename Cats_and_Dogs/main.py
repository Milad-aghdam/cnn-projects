import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

import matplotlib.pyplot as plt
from custom_data import CustomCatsDogsDataset
from model import CnnModel

num_classes = 2  #
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

train_dataset = CustomCatsDogsDataset(root_dir='data/training_set', transform=transform)
print(len(train_dataset))
test_dataset = CustomCatsDogsDataset(root_dir='data/test_set', transform=transform)
print(len(test_dataset))

train_size = int(.70 * len(train_dataset))
val_size = len(train_dataset) - train_size

train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])


# Create DataLoader for train, validation, and test sets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = CnnModel().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    train_correct = 0
    train_total = 0

    for x_batch, y_batch in train_loader:
        x_batch, y_batch= x_batch.to(device), y_batch.to(device).float()
        outputs = model(x_batch)
        loss_fn = criterion(outputs, y_batch)
        optimizer.zero_grad()
        loss_fn.backward()
        optimizer.step()
        running_loss += loss_fn.item()

        predicted = (outputs > 0.5).float()

        # Update accuracy metrics
        train_total += y_batch.size(0)  # Total number of samples
        train_correct += (predicted == y_batch).sum().item()  # Correct predictions

    train_accuracy = 100 * train_correct / train_total
    train_losses.append(running_loss / len(train_loader))
    train_accuracies.append(train_accuracy)

    # Set model to evaluation mode
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient calculation during validation
        for val_images, val_labels in val_loader:
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            val_outputs = model(val_images)
            val_loss += criterion(val_outputs, val_labels).item()

            # Calculate validation accuracy
            _, predicted = torch.max(val_outputs, 1)
            total += val_labels.size(0)
            correct += (predicted == val_labels).sum().item()


    # Print the results for this epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {running_loss / len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%, "
          f"Validation Loss: {val_loss / len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%")

# After training, plot the learning curves
epochs = range(1, num_epochs + 1)

# Create subplots for both Loss and Accuracy
plt.figure(figsize=(14, 6))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Training Loss', marker='o', color='b')
plt.plot(epochs, val_losses, label='Validation Loss', marker='o', color='r')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, label='Training Accuracy', marker='o', color='b')
plt.plot(epochs, val_accuracies, label='Validation Accuracy', marker='o', color='r')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.show()


model.eval()  # Set the model to evaluation mode
test_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for test_images, test_labels in test_loader:
        test_images, test_labels = test_images.to(device), test_labels.to(device)
        test_outputs = model(test_images)
        test_loss += criterion(test_outputs, test_labels).item()

        # Calculate test accuracy
        _, predicted = torch.max(test_outputs, 1)
        total += test_labels.size(0)
        correct += (predicted == test_labels).sum().item()

test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")