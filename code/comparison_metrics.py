"""
Extended classification metrics per aggregator for the comparison table (#15).
For each config, pools out-of-fold validation predictions across 10 folds and
computes AUC, precision, recall, F1 at threshold 0.5.
"""
import os
import numpy as np
import torch
from sklearn.metrics import (roc_auc_score, precision_score,
                             recall_score, f1_score)
from dataset import make_loader
from train import build_model

CONFIGS = [("mean","0.0"),("max","0.0"),("abmil","0.0"),
           ("transmil","0.0"),("clam","0.0"),("clam","0.05")]
FOLDS = list(range(10))
FEATURES = "../features"; SPLITS = "../data/splits"; CKPT = "../outputs/checkpoints"


def preds_for(model_name, lam, device):
    probs, labels = [], []
    for fold in FOLDS:
        ckpt = f"{CKPT}/{model_name}_fold{fold}_lam{lam}_best.pt"
        if not os.path.exists(ckpt):
            continue
        loader = make_loader(f"{SPLITS}/fold_{fold}_val.csv", FEATURES, shuffle=False)
        model = build_model(model_name, in_dim=4096).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        with torch.no_grad():
            for b in loader:
                h = b["features"].to(device)
                probs.append(torch.softmax(model(h)["logits"],1)[0,1].item())
                labels.append(b["label"].item())
    return np.array(probs), np.array(labels)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{'model':>16} {'AUC':>7} {'Prec':>7} {'Recall':>7} {'F1':>7}")
    for model_name, lam in CONFIGS:
        p, y = preds_for(model_name, lam, device)
        if len(p) == 0:
            continue
        pred = (p >= 0.5).astype(int)
        auc = roc_auc_score(y, p)
        prec = precision_score(y, pred, zero_division=0)
        rec = recall_score(y, pred, zero_division=0)
        f1 = f1_score(y, pred, zero_division=0)
        label = f"{model_name}+ent" if lam!="0.0" else model_name
        print(f"{label:>16} {auc:>7.3f} {prec:>7.3f} {rec:>7.3f} {f1:>7.3f}")


if __name__ == "__main__":
    main()