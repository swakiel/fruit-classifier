import sys
from pathlib import Path
import setup_path
from data.dataset import FruitDataset
from utils.constants import DATASET_PATHS, CLASSES

dataset = FruitDataset(DATASET_PATHS["fruitnet"])

print("Dataset size:", len(dataset))

img, label = dataset[0]
print("Label index:", label)
print("Label name:", CLASSES[label])
print("Image size:", img.size)
img.show()