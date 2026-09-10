"""
Cross-validated heatmap evaluation.
For each of the 49 test tumour slides: download once, evaluate every
(fold, lambda) checkpoint, record Dice/IoU/entropy, then delete the slide.
Writes one row per (slide, fold, lambda) to a CSV for later aggregation.
"""
import os
import csv
import glob
import argparse
import subprocess
from evaluate_heatmap import evaluate

S3_IMG = "s3://camelyon-dataset/CAMELYON16/images"
LAMBDAS = ["0.0", "0.01", "0.02", "0.05", "0.1", "0.15", "0.2"]
FOLDS = list(range(10))


def main(args):
    xmls = sorted(glob.glob(os.path.join(args.eval_dir, "test_*.xml")))
    print(f"{len(xmls)} slides x {len(FOLDS)} folds x {len(LAMBDAS)} lambdas "
          f"= {len(xmls)*len(FOLDS)*len(LAMBDAS)} evals", flush=True)

    out_csv = os.path.join(args.out_dir, "cv_dice_knee.csv")
    os.makedirs(args.out_dir, exist_ok=True)

    # resume support: skip (slide,fold,lam) already in CSV
    done = set()
    if os.path.exists(out_csv):
        with open(out_csv) as f:
            for r in csv.DictReader(f):
                done.add((r["slide"], r["fold"], r["lambda"]))
        print(f"resuming: {len(done)} evals already done", flush=True)

    write_header = not os.path.exists(out_csv)
    fcsv = open(out_csv, "a", newline="")
    writer = csv.writer(fcsv)
    if write_header:
        writer.writerow(["slide", "fold", "lambda", "dice", "iou", "entropy"])
        fcsv.flush()

    for xi, xml in enumerate(xmls):
        name = os.path.splitext(os.path.basename(xml))[0]
        feat = os.path.join(args.features_dir, f"{name}.h5")
        if not os.path.exists(feat):
            continue

        # skip slide entirely if all its evals are already done
        need = [(fo, la) for fo in FOLDS for la in LAMBDAS
                if (name, str(fo), la) not in done]
        if not need:
            print(f"[{xi+1}/{len(xmls)}] {name}: all done, skip", flush=True)
            continue

        raw = os.path.join(args.eval_dir, f"{name}.tif")
        try:
            if not os.path.exists(raw):
                print(f"[{xi+1}/{len(xmls)}] {name}: downloading", flush=True)
                subprocess.run(["aws", "s3", "cp", "--no-sign-request",
                                f"{S3_IMG}/{name}.tif", raw], check=True,
                               stdout=subprocess.DEVNULL)

            for fo, la in need:
 
                if not os.path.exists(ckpt):
                    print(f"    missing ckpt {ckpt}, skip", flush=True)
                    continue
                dice, iou, ent = evaluate(ckpt, feat, raw, xml,
                                          args.out_dir, tag=f"f{fo}_l{la}")
                writer.writerow([name, fo, la, dice, iou, ent])
                fcsv.flush()
        except Exception as e:
            print(f"    {name}: ERROR {e}", flush=True)
        finally:
            if os.path.exists(raw):
                os.remove(raw)

    fcsv.close()
    print("done", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--features_dir", default="../features")
    p.add_argument("--eval_dir", default="../data/eval_slides")
    p.add_argument("--out_dir", default="../outputs/cv_dice")
    args = p.parse_args()
    main(args)