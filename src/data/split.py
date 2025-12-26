import random
from collections import defaultdict
from utils.constants import RANDOM_SEED


def train_val_split(dataset, val_ratio=0.2):
    random.seed(RANDOM_SEED)

    label_to_indices = defaultdict(list)

    for idx, (_, label) in enumerate(dataset.samples):
        label_to_indices[label].append(idx)

    train_indices = []
    val_indices = []

    for label, indices in label_to_indices.items():
        random.shuffle(indices)

        split_point = int(len(indices) * (1 - val_ratio))

        train_indices.extend(indices[:split_point])
        val_indices.extend(indices[split_point:])

    train_dataset = DatasetSubset(dataset, train_indices)
    val_dataset = DatasetSubset(dataset, val_indices)

    return train_dataset, val_dataset


class DatasetSubset:
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]
