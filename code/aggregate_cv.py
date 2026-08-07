"""
Aggregate cross-validation validation AUCs across folds for each lambda.
Reads the 'Best val AUC' line from each cv training log.
"""
import re
import glob
import numpy as np
from collections import defaultdict

results = defaultdict(list)  # lambda -> [auc per fold]

for log in glob.glob("../outputs/logs/cv_*.log"):
    text = open(log).read()
    m_auc = re.search(r"Best val AUC:\s*([0-9.]+)", text)
    m_cfg = re.search(r"lambda=([0-9.]+)", text)
    if m_auc and m_cfg:
        lam = float(m_cfg.group(1))
        auc = float(m_auc.group(1))
        results[lam].append(auc)

print(f"{'lambda':>8} {'n_folds':>8} {'mean AUC':>10} {'std':>8}")
for lam in sorted(results):
    aucs = np.array(results[lam])
    print(f"{lam:>8} {len(aucs):>8} {aucs.mean():>10.4f} {aucs.std(ddof=1):>8.4f}")

# paired comparison: does lambda>0 differ from baseline per fold?
from scipy import stats
if 0.0 in results:
    base = np.array(results[0.0])
    print("\nPaired vs baseline (per fold):")
    for lam in sorted(results):
        if lam == 0.0: continue
        arr = np.array(results[lam])
        if len(arr) == len(base):
            t, p = stats.ttest_rel(arr, base)
            print(f"  lambda={lam}: delta {arr.mean()-base.mean():+.4f}  p={p:.4f}")