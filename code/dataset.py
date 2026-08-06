"""
Dataset for MIL training on cached CAMELYON16 features.
Each item is one whole slide: a bag of patch feature vectors plus its label.
Bags vary in size (different patch counts), so batch size is effectively one slide.
"""
import os
import csv
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class SlideFeatureDataset(Dataset):
    def __init__(self, split_csv, features_dir):
        """
        split_csv:    a fold_X_train.csv / _val.csv / test.csv with slide_name,label
        features_dir: directory of per-slide .h5 feature files
        """
        self.features_dir = features_dir
        self.items = []
        with open(split_csv) as f:
            for r in csv.DictReader(f):
                self.items.append((r["slide_name"], int(r["label"])))

        # Keep only slides whose feature file actually exists (safety)
        kept = []
        for name, label in self.items:
            if os.path.exists(os.path.join(features_dir, f"{name}.h5")):
                kept.append((name, label))
            else:
                print(f"  WARNING: missing features for {name}, skipping")
        self.items = kept

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        name, label = self.items[idx]
        path = os.path.join(self.features_dir, f"{name}.h5")
        with h5py.File(path, "r") as f:
            feats = f["features"][:]        # (N, 4096)
            coords = f["coords"][:]         # (N, 2)
        feats = torch.from_numpy(feats).float()
        return {
            "features": feats,
            "label": torch.tensor(label, dtype=torch.long),
            "coords": torch.from_numpy(np.array(coords)),
            "slide_name": name,
        }


def make_loader(split_csv, features_dir, shuffle=False):
    """
    A loader that yields one slide at a time. We use batch_size=1 and a
    trivial collate that unwraps the single item, since bags vary in size.
    """
    ds = SlideFeatureDataset(split_csv, features_dir)
    from torch.utils.data import DataLoader
    return DataLoader(ds, batch_size=1, shuffle=shuffle,
                      collate_fn=lambda batch: batch[0])