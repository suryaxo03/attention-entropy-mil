"""
Knee plot: cross-validated Dice and AUC vs lambda, with error bars.
Dice from cv_dice_knee.csv; AUC hardcoded from the cross-validated log analysis.
"""
import csv, collections, statistics
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Dice from CSV (per-fold mean, then mean+/-sd over folds) ---
rows = list(csv.DictReader(open("../outputs/cv_dice/cv_dice_knee.csv")))
byfold = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    byfold[r["lambda"]][r["fold"]].append(float(r["dice"]))

lams = sorted(byfold, key=float)
lam_f = [float(x) for x in lams]
dice_mean, dice_sd = [], []
for la in lams:
    fm = [statistics.mean(v) for v in byfold[la].values()]
    dice_mean.append(statistics.mean(fm))
    dice_sd.append(statistics.pstdev(fm))

# --- AUC (cross-validated means +/- sd) from earlier log analysis ---
auc_map = {0.0:(0.940,0.065), 0.01:(0.929,0.057), 0.02:(0.931,0.056),
           0.05:(0.916,0.065), 0.1:(0.911,0.069), 0.15:(0.907,0.065),
           0.2:(0.887,0.070)}
auc_mean = [auc_map[l][0] for l in lam_f]
auc_sd   = [auc_map[l][1] for l in lam_f]

fig, ax1 = plt.subplots(figsize=(8, 4.8))
c_dice, c_auc = "#0f6e56", "#b0300b"

# shade the "no significant AUC cost" region (lambda <= 0.05)
ax1.axvspan(-0.005, 0.05, color="green", alpha=0.05)

# Dice: line + light shaded band instead of heavy error bars
dice_mean = np.array(dice_mean); dice_sd = np.array(dice_sd)
ax1.plot(lam_f, dice_mean, marker="o", color=c_dice, linewidth=2, label="Heatmap Dice", zorder=3)
ax1.fill_between(lam_f, dice_mean - dice_sd, dice_mean + dice_sd, color=c_dice, alpha=0.12)
ax1.set_xlabel("regularisation strength  λ")
ax1.set_ylabel("heatmap Dice", color=c_dice)
ax1.tick_params(axis="y", labelcolor=c_dice)
ax1.set_ylim(0.14, 0.21)

ax2 = ax1.twinx()
auc_mean = np.array(auc_mean); auc_sd = np.array(auc_sd)
ax2.plot(lam_f, auc_mean, marker="s", color=c_auc, linewidth=2, linestyle="--",
         label="Classification AUC", zorder=3)
ax2.fill_between(lam_f, auc_mean - auc_sd, auc_mean + auc_sd, color=c_auc, alpha=0.10)
ax2.set_ylabel("classification AUC", color=c_auc)
ax2.tick_params(axis="y", labelcolor=c_auc)
ax2.set_ylim(0.80, 0.98)

# mark chosen operating point, annotation moved to top-left, out of the way
ax1.axvline(0.05, color="gray", linestyle=":", linewidth=1.3, zorder=2)
ax1.annotate("λ = 0.05: most of the Dice gain,\nno significant AUC cost",
             xy=(0.05, 0.180), xytext=(0.065, 0.150),
             fontsize=8.5, color="dimgray",
             arrowprops=dict(arrowstyle="->", color="gray", lw=1))

lines1, lab1 = ax1.get_legend_handles_labels()
lines2, lab2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, lab1 + lab2, loc="upper center", fontsize=9, framealpha=0.9)
plt.title("Heatmap quality and classification accuracy versus λ (10-fold CV)")
plt.tight_layout()
plt.savefig("../outputs/fig_knee.png", dpi=130)
plt.close()
print("saved ../outputs/fig_knee.png")