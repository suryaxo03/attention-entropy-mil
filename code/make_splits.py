"""
Create stratified k-fold cross-validation splits for CAMELYON16 training slides.
The 270 train slides are split into k folds preserving the pos/neg ratio.
The 129 test slides are held out entirely (written to their own list).
Splits are saved as CSV so every experiment uses identical folds (reproducible).
"""
import os
import csv
import argparse
import numpy as np
from sklearn.model_selection import StratifiedKFold


def load_manifest(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def make_splits(manifest_path, out_dir, k=10, seed=42):
    rows = load_manifest(manifest_path)

    train_rows = [r for r in rows if r["split"] == "train"]
    test_rows = [r for r in rows if r["split"] == "test"]

    names = np.array([r["slide_name"] for r in train_rows])
    labels = np.array([int(r["label"]) for r in train_rows])

    os.makedirs(out_dir, exist_ok=True)

    # Held-out test list (never used for training/validation)
    with open(os.path.join(out_dir, "test.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["slide_name", "label"])
        for r in test_rows:
            w.writerow([r["slide_name"], r["label"]])

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(names, labels)):
        # training slides for this fold
        with open(os.path.join(out_dir, f"fold_{fold}_train.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["slide_name", "label"])
            for i in tr_idx:
                w.writerow([names[i], labels[i]])
        # validation slides for this fold
        with open(os.path.join(out_dir, f"fold_{fold}_val.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["slide_name", "label"])
            for i in va_idx:
                w.writerow([names[i], labels[i]])

        n_tr_pos = int(labels[tr_idx].sum())
        n_va_pos = int(labels[va_idx].sum())
        print(f"fold {fold}: train {len(tr_idx)} ({n_tr_pos} pos), "
              f"val {len(va_idx)} ({n_va_pos} pos)")

    print(f"\n{k} folds written to {out_dir}")
    print(f"test set: {len(test_rows)} slides (held out)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="../data/manifest.csv")
    p.add_argument("--out_dir", default="../data/splits")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    make_splits(args.manifest, args.out_dir, k=args.k, seed=args.seed)