import sys
from pathlib import Path
import os
import setup_path
from utils.constants import CLASSES, DATASET_PATHS

# ---- sanity check ----
def count_images(dataset_name, dataset_path):
    print(f"\nDataset: {dataset_name}")
    total = 0
    file_types = set()

    for fruit in CLASSES:
        fruit_dir = Path(dataset_path) / fruit
        num_images = len(list(fruit_dir.iterdir()))
        total += num_images
        print(f"  {fruit}: {num_images}")
        file_types |= {p.suffix.lower() for p in fruit_dir.iterdir()}

    print(f"  TOTAL: {total}")
    print(f"  File Types: {file_types}")  # should be only .jpg, .png etc.

def main():
    for name, path in DATASET_PATHS.items():
        count_images(name, path)

if __name__ == "__main__":
    main()
