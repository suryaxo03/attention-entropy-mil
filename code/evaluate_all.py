"""
Batch heatmap evaluation across all annotated (tumour) test slides.
For each slide: download raw .tif, evaluate baseline + entropy checkpoints,
record Dice/IoU/entropy, delete raw slide. Writes a results CSV.
"""
import os
import csv
import glob
import argparse
import subprocess
import numpy as np

from evaluate_heatmap import evaluate

S3_IMG = "s3://camelyon-dataset/CAMELYON16/images"


def main(args):
    xmls = sorted(glob.glob(os.path.join(args.eval_dir, "test_*.xml")))
    print(f"{len(xmls)} annotated test slides to evaluate")

    checkpoints = {
        "baseline": args.baseline_ckpt,
        "entropy": args.entropy_ckpt,
    }

    rows = []
    for xml in xmls:
        name = os.path.splitext(os.path.basename(xml))[0]
        feat = os.path.join(args.features_dir, f"{name}.h5")
        if not os.path.exists(feat):
            print(f"  {name}: no features, skipping")
            continue

        raw = os.path.join(args.eval_dir, f"{name}.tif")
        try:
            if not os.path.exists(raw):
                subprocess.run(["aws", "s3", "cp", "--no-sign-request",
                                f"{S3_IMG}/{name}.tif", raw], check=True,
                               stdout=subprocess.DEVNULL)

            row = {"slide": name}
            for tag, ckpt in checkpoints.items():
                dice, iou, ent = evaluate(ckpt, feat, raw, xml,
                                          args.out_dir, tag=tag)
                row[f"{tag}_dice"] = dice
                row[f"{tag}_iou"] = iou
                row[f"{tag}_entropy"] = ent
            rows.append(row)
        except Exception as e:
            print(f"  {name}: ERROR {e}")
        finally:
            if os.path.exists(raw):
                os.remove(raw)

    # write CSV
    out_csv = os.path.join(args.out_dir, "heatmap_results.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "slide", "baseline_dice", "baseline_iou", "baseline_entropy",
            "entropy_dice", "entropy_iou", "entropy_entropy"])
        w.writeheader(); w.writerows(rows)

    # summary
    def col(key): return np.array([r[key] for r in rows])
    print("\n=== SUMMARY over", len(rows), "slides ===")
    for metric in ["dice", "iou", "entropy"]:
        b = col(f"baseline_{metric}"); e = col(f"entropy_{metric}")
        print(f"{metric:8s}: baseline {b.mean():.3f}±{b.std():.3f}   "
              f"entropy {e.mean():.3f}±{e.std():.3f}   "
              f"delta {e.mean()-b.mean():+.3f}")
    print(f"\nsaved -> {out_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--baseline_ckpt", default="../outputs/checkpoints/fold0_lam0.0_best.pt")
    p.add_argument("--entropy_ckpt", default="../outputs/checkpoints/fold0_lam0.01_best.pt")
    p.add_argument("--features_dir", default="../features")
    p.add_argument("--eval_dir", default="../data/eval_slides")
    p.add_argument("--out_dir", default="../outputs/heatmap_eval")
    args = p.parse_args()
    main(args)