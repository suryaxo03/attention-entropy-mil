"""
Train one CLAM (optionally entropy-regularised) model on one CV fold.
Loss = CLAM classification loss + bag-clustering loss - lambda * attention entropy.
Evaluates validation AUC each epoch, keeps the best checkpoint.
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from dataset import make_loader
from models import CLAM_SB


def run_epoch(model, loader, optimizer, device, lam, train=True):
    model.train() if train else model.eval()
    ce = nn.CrossEntropyLoss()

    total_loss = 0.0
    all_probs, all_labels = [], []

    for batch in loader:
        h = batch["features"].to(device)
        label = batch["label"].to(device)

        with torch.set_grad_enabled(train):
            out = model(h, label=label.item(), instance_eval=True)
            logits = out["logits"]
            inst_loss = out.get("instance_loss", torch.tensor(0.0, device=device))
            entropy = out["attention_entropy"]

            cls_loss = ce(logits, label.unsqueeze(0))
            # CLAM combines classification + instance loss; we subtract lambda*entropy
            loss = cls_loss + 0.3 * inst_loss - lam * entropy

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        prob = torch.softmax(logits, dim=1)[0, 1].item()  # P(tumour)
        all_probs.append(prob)
        all_labels.append(label.item())

    avg_loss = total_loss / len(loader)
    # AUC needs both classes present
    if len(set(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_probs)
    else:
        auc = float("nan")
    return avg_loss, auc


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feats = args.features_dir
    splits = args.splits_dir

    train_csv = os.path.join(splits, f"fold_{args.fold}_train.csv")
    val_csv = os.path.join(splits, f"fold_{args.fold}_val.csv")

    train_loader = make_loader(train_csv, feats, shuffle=True)
    val_loader = make_loader(val_csv, feats, shuffle=False)

    model = CLAM_SB(in_dim=4096).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.wd)

    print(f"Fold {args.fold} | lambda={args.lam} | "
          f"{len(train_loader)} train, {len(val_loader)} val slides")

    best_auc = 0.0
    best_path = os.path.join(args.out_dir, f"fold{args.fold}_lam{args.lam}_best.pt")
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(args.epochs):
        tr_loss, tr_auc = run_epoch(model, train_loader, optimizer, device,
                                    args.lam, train=True)
        with torch.no_grad():
            va_loss, va_auc = run_epoch(model, val_loader, optimizer, device,
                                        args.lam, train=False)

        flag = ""
        if va_auc > best_auc:
            best_auc = va_auc
            torch.save(model.state_dict(), best_path)
            flag = "  <-- best"
        print(f"  epoch {epoch+1:2d}/{args.epochs} | "
              f"train loss {tr_loss:.3f} auc {tr_auc:.3f} | "
              f"val loss {va_loss:.3f} auc {va_auc:.3f}{flag}")

    print(f"Best val AUC: {best_auc:.3f}  ({best_path})")
    return best_auc


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--lam", type=float, default=0.0,
                   help="entropy regularisation weight (0 = standard CLAM)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-5)
    p.add_argument("--features_dir", default="../features")
    p.add_argument("--splits_dir", default="../data/splits")
    p.add_argument("--out_dir", default="../outputs/checkpoints")
    args = p.parse_args()
    main(args)