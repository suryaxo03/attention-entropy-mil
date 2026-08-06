"""
Build a single manifest CSV driving batch preprocessing.
Columns: slide_name, split, label, s3_path
- label: 0 = negative (normal), 1 = positive (tumour metastasis)
- split: 'train' for normal_/tumor_ slides, 'test' for test_ slides
Labels come from evaluation/reference.csv (covers all slides).
"""
import csv
import argparse

S3_BASE = "s3://camelyon-dataset/CAMELYON16/images"


def build(reference_csv, out_csv):
    rows = []
    with open(reference_csv) as f:
        reader = csv.DictReader(f)
        for r in reader:
            image = r["image"].strip()              # e.g. normal_001.tif
            name = image.replace(".tif", "")
            cls = r["class"].strip().lower()         # negative / positive (itc/micro/macro?)
            # Any non-negative class counts as positive (tumour present)
            label = 0 if cls == "negative" else 1
            split = "test" if name.startswith("test_") else "train"
            rows.append({
                "slide_name": name,
                "split": split,
                "label": label,
                "class": cls,
                "center": r.get("center", "").strip(),
                "s3_path": f"{S3_BASE}/{image}",
            })

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "slide_name", "split", "label", "class", "center", "s3_path"])
        writer.writeheader()
        writer.writerows(rows)

    # summary
    n = len(rows)
    n_pos = sum(r["label"] for r in rows)
    n_train = sum(r["split"] == "train" for r in rows)
    n_test = n - n_train
    print(f"manifest: {n} slides -> {out_csv}")
    print(f"  train: {n_train}  test: {n_test}")
    print(f"  positive: {n_pos}  negative: {n - n_pos}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--reference", default="../data/reference.csv")
    p.add_argument("--out", default="../data/manifest.csv")
    args = p.parse_args()
    build(args.reference, args.out)