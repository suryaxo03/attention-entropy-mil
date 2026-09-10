"""
Aggregator comparison, Dice side. For each of the 49 test tumour slides:
download once, evaluate all six model configs (each across its 10 folds),
record Dice, delete. Writes per-slide-per-model-per-fold results + summary.
Resumable and incremental.
"""
import os, csv, glob, subprocess
import numpy as np
from evaluate_heatmap import evaluate

S3_IMG = "s3://camelyon-dataset/CAMELYON16/images"
CONFIGS = [("mean","0.0"),("max","0.0"),("abmil","0.0"),
           ("transmil","0.0"),("clam","0.0"),("clam","0.05")]
FOLDS = list(range(10))
CKPT = "../outputs/checkpoints"
EVAL_DIR = "../data/eval_slides"
FEATURES = "../features"
OUT = "../outputs/comparison_dice"


def main():
    os.makedirs(OUT, exist_ok=True)
    out_csv = f"{OUT}/comparison_dice_results.csv"
    done = set()
    if os.path.exists(out_csv):
        with open(out_csv) as f:
            for r in csv.DictReader(f):
                done.add((r["slide"], r["model"], r["lambda"], r["fold"]))
    write_header = not os.path.exists(out_csv)
    fout = open(out_csv, "a", newline="")
    writer = csv.writer(fout)
    if write_header:
        writer.writerow(["slide","model","lambda","fold","dice","iou","entropy"]); fout.flush()

    xmls = sorted(glob.glob(f"{EVAL_DIR}/test_*.xml"))
    for xi, xml in enumerate(xmls):
        name = os.path.splitext(os.path.basename(xml))[0]
        feat = f"{FEATURES}/{name}.h5"
        if not os.path.exists(feat):
            continue
        need = [(m,l,fo) for (m,l) in CONFIGS for fo in FOLDS
                if (name,m,l,str(fo)) not in done]
        if not need:
            print(f"[{xi+1}/{len(xmls)}] {name}: done, skip", flush=True); continue
        raw = f"{EVAL_DIR}/{name}.tif"
        try:
            if not os.path.exists(raw):
                print(f"[{xi+1}/{len(xmls)}] {name}: downloading", flush=True)
                subprocess.run(["aws","s3","cp","--no-sign-request",
                                f"{S3_IMG}/{name}.tif", raw], check=True,
                               stdout=subprocess.DEVNULL)
            for (m,l,fo) in need:
                ckpt = f"{CKPT}/{m}_fold{fo}_lam{l}_best.pt"
                if not os.path.exists(ckpt): continue
                # evaluate() rebuilds CLAM_SB by default; we need the right model class.
                dice, iou, ent = evaluate(ckpt, feat, raw, xml, OUT,
                                          tag=f"{m}_l{l}_f{fo}", model_name=m)
                writer.writerow([name,m,l,fo,dice,iou,ent]); fout.flush()
        except Exception as e:
            print(f"  {name}: ERROR {e}", flush=True)
        finally:
            if os.path.exists(raw): os.remove(raw)
    fout.close()
    print("done", flush=True)


if __name__ == "__main__":
    main()