from pathlib import Path
from PIL import Image
import random

from utils.constants import CLASSES, RANDOM_SEED

class FruitDataset:
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.samples = []
        self.transform = transform
        self._build_index()

    def _build_index(self):
        for label_idx, fruit in enumerate(CLASSES):
            fruit_dir = self.root_dir / fruit

            if not fruit_dir.exists():
                raise ValueError(f"Missing folder: {fruit_dir}")

            for img_path in fruit_dir.glob("*.jpg"):
                self.samples.append((img_path, label_idx))

        random.seed(RANDOM_SEED)
        random.shuffle(self.samples)


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.tranform(image)

        return image, label
