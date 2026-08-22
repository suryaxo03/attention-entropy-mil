"""
Small-focus analysis (RQ3): does entropy regularisation help more on
micro-metastases than macro-metastases?
Joins cv_dice_results.csv (per-slide Dice per fold per lambda) to the
metastasis-size class from the manifest, then compares baseline vs entropy
Dice within each size stratum.
"""
import pandas as pd
import numpy as np
from scipy import stats

dice = pd.read_csv("../outputs/cv_dice/cv_dice_results.csv")
manifest = pd.read_csv("../data/manifest.csv")

# map slide_name -> class (negative/micro/macro)
size = manifest.set_index("slide_name")["class"].to_dict()
dice["size"] = dice["slide"].map(size)

print("Slides per size class in the evaluation set:")
print(dice[dice["lambda"] == 0.0].groupby("size")["slide"].nunique())
print()

# For each size stratum, compare baseline (0.0) vs entropy (0.05) Dice,
# paired by (slide, fold)
for stratum in ["micro", "macro"]:
    sub = dice[dice["size"] == stratum]
    # pivot: index (slide,fold), columns lambda, values dice
    piv = sub.pivot_table(index=["slide", "fold"], columns="lambda", values="dice")
    if 0.0 not in piv.columns or 0.05 not in piv.columns:
        print(f"{stratum}: missing lambda columns"); continue
    b = piv[0.0].dropna()
    e = piv[0.05].dropna()
    common = b.index.intersection(e.index)
    b = b.loc[common].values; e = e.loc[common].values
    delta = e - b
    t, p = stats.ttest_rel(e, b)
    try:
        w = stats.wilcoxon(e, b)[1]
    except Exception:
        w = float("nan")
    d = delta.mean() / delta.std(ddof=1) if delta.std(ddof=1) > 0 else 0
    print(f"=== {stratum.upper()} metastases (n={len(common)} slide-fold pairs) ===")
    print(f"  baseline Dice : {b.mean():.4f}")
    print(f"  entropy  Dice : {e.mean():.4f}")
    print(f"  delta         : {delta.mean():+.4f}")
    print(f"  t-p = {p:.4f}   Wilcoxon-p = {w:.4f}   Cohen's d = {d:+.3f}")
    print()