import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from utils.constants import DATASET_PATHS, CLASSES
from data.dataset import FruitDataset
from data.split import train_val_split
from data.pytorch_dataset import FruitTorchDataset
from train_model import SimpleCNN
from data.transforms import val_transform


def evaluate(dataset_name, batch_size=32, val_ratio=0.2, results_dir="results"):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # -------------------
    # Dataset & split
    # -------------------
    dataset_path = DATASET_PATHS[dataset_name]
    full_dataset = FruitDataset(dataset_path)
    _, val_ds = train_val_split(full_dataset, val_ratio=val_ratio)

    val_ds = FruitTorchDataset(val_ds, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # -------------------
    # Load model
    # -------------------
    model = SimpleCNN().to(device)
    model_path = os.path.join(results_dir, f"{dataset_name}_cnn.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # -------------------
    # Evaluation
    # -------------------
    all_preds = []
    all_labels = []

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"\nValidation Accuracy ({dataset_name}): {accuracy:.4f}")

    # -------------------
    # Confusion Matrix
    # -------------------
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)

    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title(f"Confusion Matrix - {dataset_name}")
    plt.xticks(rotation=45)
    plt.tight_layout()

    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"{dataset_name}_confusion_matrix.png")
    plt.savefig(out_path)
    plt.close()

    print(f"Confusion matrix saved to {out_path}")


if __name__ == "__main__":
    evaluate(dataset_name="fruitnet")
    evaluate(dataset_name="fruits360")
