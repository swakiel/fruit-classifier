import setup_path
from data.dataset import FruitDataset
from data.split import train_val_split
from utils.constants import DATASET_PATHS, CLASSES

dataset = FruitDataset(DATASET_PATHS["fruits360"])
train_ds, val_ds = train_val_split(dataset, val_ratio=0.2)

print("Total:", len(dataset))
print("Train:", len(train_ds))
print("Val:", len(val_ds))

img, label = train_ds[0]
print("Train sample label:", CLASSES[label])
img.show()
