"""
Compute precision, recall, F1, and confusion matrix for the CV checkpoints.
For each lambda, runs each fold's best checkpoint over that fold's validation
slides, pools predictions across folds, and reports metrics at a 0.5 threshold.
"""
import os
import glob
import numpy as np
import torch
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score)
from dataset import make_loader
from models import CLAM_SB

LAMBDAS = ["0.0", "0.05", "0.1"]
FOLDS = list(range(10))
FEATURES = "../features"
SPLITS = "../data/splits"
CKPT = "../outputs/checkpoints"


def eval_checkpoint(ckpt_path, val_csv, device):
    loader = make_loader(val_csv, FEATURES, shuffle=False)
    model = CLAM_SB(in_dim=4096).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for batch in loader:
            h = batch["features"].to(device)
            out = model(h)
            p = torch.softmax(out["logits"], dim=1)[0, 1].item()
            probs.append(p)
            labels.append(batch["label"].item())
    return np.array(probs), np.array(labels)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for lam in LAMBDAS:
        all_probs, all_labels = [], []
        for fold in FOLDS:
            ckpt = os.path.join(CKPT, f"fold{fold}_lam{lam}_best.pt")
            val_csv = os.path.join(SPLITS, f"fold_{fold}_val.csv")
            if not os.path.exists(ckpt):
                print(f"  missing {ckpt}, skipping fold {fold}")
                continue
            p, y = eval_checkpoint(ckpt, val_csv, device)
            all_probs.append(p); all_labels.append(y)

        probs = np.concatenate(all_probs)
        labels = np.concatenate(all_labels)
        preds = (probs >= 0.5).astype(int)

        auc = roc_auc_score(labels, probs)
        prec = precision_score(labels, preds, zero_division=0)
        rec = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        cm = confusion_matrix(labels, preds)

        print(f"\n=== lambda = {lam} (pooled over {len(labels)} val slides) ===")
        print(f"  AUC       : {auc:.4f}")
        print(f"  Precision : {prec:.4f}")
        print(f"  Recall    : {rec:.4f}")
        print(f"  F1        : {f1:.4f}")
        print(f"  Confusion matrix [rows=true, cols=pred]:")
        print(f"        pred_neg  pred_pos")
        print(f"  true_neg  {cm[0,0]:5d}   {cm[0,1]:5d}")
        print(f"  true_pos  {cm[1,0]:5d}   {cm[1,1]:5d}")


if __name__ == "__main__":
    main()