"""
Parse per-epoch metrics from the comparison training logs and produce:
  (1) a convergence figure for one representative fold per model
  (2) a comparison of validation-AUC training curves across the models
Data source: ../outputs/logs/compare_<jobid>_<task>.log
Task index -> config: task//10 = config, task%10 = fold.
CONFIGS order: mean, max, abmil, transmil, clam, clam (last = clam+entropy lam0.05)
"""
import re
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONFIG_NAMES = ["mean", "max", "abmil", "transmil", "clam", "clam+entropy"]
EPOCH_RE = re.compile(
    r"epoch\s+(\d+)\s*/\s*\d+\s*\|\s*train loss\s+(-?[\d.]+)\s+auc\s+([\d.]+)\s*\|\s*val loss\s+(-?[\d.]+)\s+auc\s+([\d.]+)")


def parse_log(path):
    """Return dict of arrays: epoch, tr_loss, tr_auc, va_loss, va_auc."""
    ep, tl, ta, vl, va = [], [], [], [], []
    for line in open(path):
        m = EPOCH_RE.search(line)
        if m:
            ep.append(int(m.group(1)))
            tl.append(float(m.group(2))); ta.append(float(m.group(3)))
            vl.append(float(m.group(4))); va.append(float(m.group(5)))
    return dict(epoch=np.array(ep), tr_loss=np.array(tl), tr_auc=np.array(ta),
                va_loss=np.array(vl), va_auc=np.array(va))


def find_log(config_idx, fold):
    task = config_idx * 10 + fold
    hits = glob.glob(f"../outputs/logs/compare_*_{task}.log")
    return hits[0] if hits else None


# ---- Figure 1: convergence of one representative fold (CLAM+entropy, fold 0) ----
def convergence_figure(config_idx=5, fold=0, name="clam+entropy"):
    log = find_log(config_idx, fold)
    d = parse_log(log)
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(d["epoch"], d["tr_loss"], "b-", label="train loss")
    ax1.plot(d["epoch"], d["va_loss"], "b--", label="val loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    ax2 = ax1.twinx()
    ax2.plot(d["epoch"], d["tr_auc"], "r-", label="train AUC")
    ax2.plot(d["epoch"], d["va_auc"], "r--", label="val AUC")
    ax2.set_ylabel("AUC", color="r"); ax2.tick_params(axis="y", labelcolor="r")
    ax2.set_ylim(0.5, 1.02)
    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, loc="center right", fontsize=8)
    plt.title(f"Training convergence: {name} (fold {fold})")
    plt.tight_layout()
    plt.savefig("../outputs/fig_convergence.png", dpi=130)
    plt.close()
    print("saved ../outputs/fig_convergence.png")


# ---- Figure 2: val-AUC curves across models, averaged over folds with band ----
def comparison_figure():
    plt.figure(figsize=(7.5, 4.8))
    styles = {
        "mean": ("#7b3294", "-"), "max": ("#c2a5cf", "-"),
        "abmil": ("#a6dba0", "-"), "transmil": ("#5aae61", "-"),
        "clam": ("#1b7837", "-"), "clam+entropy": ("#00441b", "--"),
    }
    for ci, name in enumerate(CONFIG_NAMES):
        curves = []
        for fold in range(10):
            log = find_log(ci, fold)
            if not log:
                continue
            d = parse_log(log)
            if len(d["va_auc"]) >= 18:
                curves.append(d["va_auc"][:20])
        if not curves:
            continue
        mean = np.array(curves).mean(0)
        epochs = np.arange(1, len(mean) + 1)
        color, ls = styles.get(name, ("gray", "-"))
        plt.plot(epochs, mean, color=color, linestyle=ls, label=name, linewidth=2)
    plt.xlabel("epoch"); plt.ylabel("validation AUC (mean over 10 folds)")
    plt.ylim(0.6, 0.98)
    plt.title("Validation AUC during training, by aggregator")
    plt.legend(fontsize=9, loc="lower right", framealpha=0.9)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig("../outputs/fig_training_comparison.png", dpi=130)
    plt.close()
    print("saved ../outputs/fig_training_comparison.png")


if __name__ == "__main__":
    convergence_figure()
    comparison_figure()