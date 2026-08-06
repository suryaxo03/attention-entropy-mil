"""
Evaluate all lambda checkpoints on the 49 annotated test slides.
Downloads each slide once, evaluates every checkpoint on it, then deletes.
Writes a per-slide-per-lambda results CSV and a summary.
"""
import os
import csv
import glob
import argparse
import subprocess
import numpy as np
from evaluate_heatmap import evaluate

S3_IMG = "s3://camelyon-dataset/CAMELYON16/images"

# lambda -> checkpoint path
LAMBDAS = {
    "0.15":  "../outputs/checkpoints/fold0_lam0.15_best.pt",
    "0.2":   "../outputs/checkpoints/fold0_lam0.2_best.pt",
    "0.3":   "../outputs/checkpoints/fold0_lam0.3_best.pt",
}


def main(args):
    xmls = sorted(glob.glob(os.path.join(args.eval_dir, "test_*.xml")))
    print(f"{len(xmls)} annotated test slides x {len(LAMBDAS)} lambdas")

    rows = []
    for xml in xmls:
        name = os.path.splitext(os.path.basename(xml))[0]
        feat = os.path.join(args.features_dir, f"{name}.h5")
        if not os.path.exists(feat):
            continue
        raw = os.path.join(args.eval_dir, f"{name}.tif")
        try:
            if not os.path.exists(raw):
                subprocess.run(["aws", "s3", "cp", "--no-sign-request",
                                f"{S3_IMG}/{name}.tif", raw], check=True,
                               stdout=subprocess.DEVNULL)
            for lam, ckpt in LAMBDAS.items():
                if not os.path.exists(ckpt):
                    continue
                dice, iou, ent = evaluate(ckpt, feat, raw, xml,
                                          args.out_dir, tag=f"lam{lam}")
                rows.append({"slide": name, "lambda": float(lam),
                             "dice": dice, "iou": iou, "entropy": ent})
        except Exception as e:
            print(f"  {name}: ERROR {e}")
        finally:
            if os.path.exists(raw):
                os.remove(raw)

    out_csv = os.path.join(args.out_dir, "sweep_results_high.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["slide", "lambda", "dice", "iou", "entropy"])
        w.writeheader(); w.writerows(rows)

    # summary per lambda
    print("\n=== SUMMARY (mean over slides) ===")
    print(f"{'lambda':>8} {'dice':>14} {'iou':>14} {'entropy':>14}")
    for lam in sorted(set(r["lambda"] for r in rows)):
        sub = [r for r in rows if r["lambda"] == lam]
        d = np.array([r["dice"] for r in sub])
        i = np.array([r["iou"] for r in sub])
        e = np.array([r["entropy"] for r in sub])
        print(f"{lam:>8} {d.mean():.3f}±{d.std():.3f} "
              f"{i.mean():.3f}±{i.std():.3f} {e.mean():.3f}±{e.std():.3f}")
    print(f"\nsaved -> {out_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--features_dir", default="../features")
    p.add_argument("--eval_dir", default="../data/eval_slides")
    p.add_argument("--out_dir", default="../outputs/sweep_eval")
    args = p.parse_args()
    main(args)