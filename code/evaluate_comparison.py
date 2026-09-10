"""
Final aggregator comparison: cross-validated AUC and Dice for every model.
For each config (model, lambda), loads its 10 fold-checkpoints, computes
validation AUC per fold, and Dice on the 49 test tumour slides per fold.
Writes a per-fold CSV and prints a mean +/- sd summary table.
"""
import os, csv, glob, argparse
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from dataset import make_loader
from train import build_model
from evaluate_heatmap import evaluate as heatmap_eval

CONFIGS = [
    ("mean", "0.0"), ("max", "0.0"), ("abmil", "0.0"),
    ("transmil", "0.0"), ("clam", "0.0"), ("clam", "0.05"),
]
FOLDS = list(range(10))
FEATURES = "../features"
SPLITS = "../data/splits"
CKPT = "../outputs/checkpoints"
EVAL_DIR = "../data/eval_slides"


def val_auc(model_name, lam, fold, device):
    ckpt = f"{CKPT}/{model_name}_fold{fold}_lam{lam}_best.pt"
    if not os.path.exists(ckpt):
        return None
    val_csv = f"{SPLITS}/fold_{fold}_val.csv"
    loader = make_loader(val_csv, FEATURES, shuffle=False)
    model = build_model(model_name, in_dim=4096).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for b in loader:
            h = b["features"].to(device)
            out = model(h)
            probs.append(torch.softmax(out["logits"], 1)[0, 1].item())
            labels.append(b["label"].item())
    if len(set(labels)) < 2:
        return None
    return roc_auc_score(labels, probs)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    xmls = sorted(glob.glob(os.path.join(EVAL_DIR, "test_*.xml")))

    rows = []  # (model, lam, fold, auc)
    print("Computing validation AUC per fold...")
    for model_name, lam in CONFIGS:
        for fold in FOLDS:
            a = val_auc(model_name, lam, fold, device)
            if a is not None:
                rows.append({"model": model_name, "lambda": lam,
                             "fold": fold, "auc": a})

    # summary
    print("\n=== AUC SUMMARY (10-fold CV) ===")
    print(f"{'model':>16} {'lambda':>7} {'mean AUC':>10} {'sd':>7}")
    summary = {}
    for model_name, lam in CONFIGS:
        aucs = [r["auc"] for r in rows
                if r["model"] == model_name and r["lambda"] == lam]
        if aucs:
            m, s = np.mean(aucs), np.std(aucs, ddof=1)
            label = f"{model_name}+entropy" if lam != "0.0" else model_name
            print(f"{label:>16} {lam:>7} {m:>10.4f} {s:>7.4f}")
            summary[(model_name, lam)] = (m, s)

    # save
    with open("../outputs/comparison_auc.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "lambda", "fold", "auc"])
        w.writeheader(); w.writerows(rows)
    print("\nsaved -> ../outputs/comparison_auc.csv")
    print("\n(Dice comparison runs separately: it needs the 49 test slides downloaded.)")


if __name__ == "__main__":
    main()