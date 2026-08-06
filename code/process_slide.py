"""
Process one CAMELYON16 slide end to end, unattended:
  download from S3 -> segment -> patch -> extract features -> save -> delete raw .tif
Designed to keep peak disk low: the raw slide is removed once features are cached.
Skips work already done (idempotent), so re-running is safe.
"""
import os
import sys
import csv
import argparse
import subprocess
import traceback

from extract_patches import extract_patches
from extract_features import extract_features


def load_manifest(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def process_one(row, dirs, batch_size=128):
    name = row["slide_name"]
    s3 = row["s3_path"]

    feat_path = os.path.join(dirs["features"], f"{name}.h5")
    if os.path.exists(feat_path):
        print(f"[{name}] features already exist, skipping.")
        return "skipped"

    raw_path = os.path.join(dirs["data"], f"{name}.tif")
    patch_path = os.path.join(dirs["patches"], f"{name}.h5")

    try:
        # 1. Download raw slide (if not present)
        if not os.path.exists(raw_path):
            print(f"[{name}] downloading ...")
            subprocess.run(
                ["aws", "s3", "cp", "--no-sign-request", s3, raw_path],
                check=True,
            )

        # 2. Patch extraction (segments internally)
        if not os.path.exists(patch_path):
            print(f"[{name}] extracting patches ...")
            extract_patches(raw_path, dirs["patches"])

        # 3. Feature extraction
        print(f"[{name}] extracting features ...")
        extract_features(patch_path, dirs["features"], batch_size=batch_size)

    except Exception as e:
        print(f"[{name}] ERROR: {e}")
        traceback.print_exc()
        return "failed"

    finally:
        # 4. Always delete the raw slide to save disk, keep small feature file
        if os.path.exists(raw_path):
            os.remove(raw_path)
            print(f"[{name}] removed raw slide.")
        # Optionally remove the intermediate patch file too (features are what we need)
        if os.path.exists(patch_path) and os.path.exists(feat_path):
            os.remove(patch_path)
            print(f"[{name}] removed intermediate patches.")

    return "done"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="../data/manifest.csv")
    p.add_argument("--index", type=int, required=True,
                   help="row index into the manifest (0-based)")
    p.add_argument("--base", default="/mnt/scratch/users/plx526/camelyon")
    args = p.parse_args()

    dirs = {
        "data": os.path.join(args.base, "data"),
        "patches": os.path.join(args.base, "patches"),
        "features": os.path.join(args.base, "features"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    rows = load_manifest(args.manifest)
    if args.index < 0 or args.index >= len(rows):
        print(f"index {args.index} out of range (0..{len(rows)-1})")
        sys.exit(1)

    row = rows[args.index]
    print(f"=== processing #{args.index}: {row['slide_name']} "
          f"(label={row['label']}, {row['class']}) ===")
    status = process_one(row, dirs)
    print(f"=== {row['slide_name']}: {status} ===")