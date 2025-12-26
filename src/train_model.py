import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T

from utils.constants import DATASET_PATHS, CLASSES
from data.dataset import FruitDataset
from data.split import train_val_split
from data.pytorch_dataset import FruitTorchDataset
from data.transforms import train_transform, val_transform

# -------------------
# Define CNN
# -------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=len(CLASSES)):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(64*25*25, 128)  # input size 100x100
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))  # first pool: 100→50
        x = self.pool(x)                       # second pool: 50→25
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# -------------------
# Define training function
# -------------------
def train(dataset_name, epochs=10, batch_size=32, lr=1e-3, val_ratio=0.2, results_dir="results"):
    # Load dataset
    dataset_path = DATASET_PATHS[dataset_name]
    fruit_dataset = FruitDataset(dataset_path)

    # Train/val split
    train_ds, val_ds = train_val_split(fruit_dataset, val_ratio=val_ratio)

    # Wrap for PyTorch
    train_ds = FruitTorchDataset(train_ds, transform=train_transform)
    val_ds   = FruitTorchDataset(val_ds, transform=val_transform)

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Model, optimizer, loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()*images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        train_acc = correct/total

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()*images.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct/val_total

        print(f"Epoch {epoch+1} | "
              f"Train Loss: {train_loss/total:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss/val_total:.4f} | Val Acc: {val_acc:.4f}")

    # Save model
    os.makedirs(results_dir, exist_ok=True)
    model_file = os.path.join(results_dir, f"{dataset_name}_cnn.pth")
    torch.save(model.state_dict(), model_file)
    print(f"Model saved to {model_file}")
    return model
