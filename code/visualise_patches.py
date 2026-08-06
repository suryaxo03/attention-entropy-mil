"""
Sanity-check multi-scale patch extraction.
Loads an .h5 patch file and shows a few high-res / context pairs side by side.
"""
import os
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt


def visualise(h5_path, out_dir, n=6, seed=0):
    with h5py.File(h5_path, "r") as f:
        hi = f["patches_hi"][:]
        lo = f["patches_lo"][:]
        coords = f["coords"][:]
        name = f.attrs["slide_name"]
        total = len(hi)

    rng = np.random.default_rng(seed)
    idx = rng.choice(total, size=min(n, total), replace=False)

    fig, axes = plt.subplots(2, len(idx), figsize=(2.2 * len(idx), 4.6))
    for col, i in enumerate(idx):
        axes[0, col].imshow(hi[i])
        axes[0, col].set_title(f"#{i}\n20x (hi-res)", fontsize=9)
        axes[1, col].imshow(lo[i])
        axes[1, col].set_title("5x (context)", fontsize=9)
        for row in (0, 1):
            axes[row, col].axis("off")

    plt.suptitle(f"{name}: {total} patches total, showing {len(idx)}", fontsize=11)
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}_patch_check.png")
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--h5", required=True)
    p.add_argument("--out_dir", default="../outputs")
    p.add_argument("--n", type=int, default=6)
    args = p.parse_args()
    visualise(args.h5, args.out_dir, n=args.n)