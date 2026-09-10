"""
Evaluate attention-heatmap quality for a trained model on a tumour slide.
Scatters per-patch attention onto a level-6 grid, compares to the tumour mask
via Dice and IoU, and reports attention entropy. Saves a visual overlay.
"""
import os
import argparse
import numpy as np
import h5py
import torch
import openslide
import matplotlib.pyplot as plt

from models import CLAM_SB
from annotation_to_mask import rasterise_mask
from train import build_model

def build_attention_grid(coords, attn, slide_path, level=6, patch_level=1, patch_size=256):
    """
    Scatter per-patch attention weights onto a level-`level` grid.
    coords are level-0 top-left pixels of each patch.
    """
    slide = openslide.OpenSlide(slide_path)
    W, H = slide.level_dimensions[level]
    ds_level = slide.level_dimensions[0][0] / W   # level-0 -> level scale
    # size of one patch (at patch_level) expressed in level-0 pixels
    patch_l0 = patch_size * int(slide.level_downsamples[patch_level])
    slide.close()

    grid = np.zeros((H, W), dtype=np.float32)
    counts = np.zeros((H, W), dtype=np.float32)

    step = max(1, int(round(patch_l0 / ds_level)))  # patch footprint at this level
    for (x0, y0), a in zip(coords, attn):
        gx = int(x0 / ds_level)
        gy = int(y0 / ds_level)
        gx2 = min(W, gx + step); gy2 = min(H, gy + step)
        if gx < W and gy < H:
            grid[gy:gy2, gx:gx2] += a
            counts[gy:gy2, gx:gx2] += 1
    counts[counts == 0] = 1
    grid = grid / counts
    return grid


def dice_iou(pred_bin, gt_bin):
    inter = np.logical_and(pred_bin, gt_bin).sum()
    p, g = pred_bin.sum(), gt_bin.sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    dice = (2 * inter) / (p + g) if (p + g) > 0 else 0.0
    iou = inter / union if union > 0 else 0.0
    return dice, iou


def evaluate(checkpoint, feat_path, slide_path, xml_path, out_dir,
             level=6, thresh_pct=90, tag="", model_name="clam"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with h5py.File(feat_path, "r") as f:
        feats = torch.from_numpy(f["features"][:]).float().to(device)
        coords = f["coords"][:]

    model = build_model(model_name, in_dim=feats.shape[1]).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    with torch.no_grad():
        out = model(feats)
        attn = out["attention"].cpu().numpy()
        entropy = out["attention_entropy"].item()

    # normalise attention to [0,1] for thresholding
    a = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)

    grid = build_attention_grid(coords, a, slide_path, level=level)
    mask, _ = rasterise_mask(slide_path, xml_path, level=level)

    # match grid to mask shape if off by a row/col
    h = min(grid.shape[0], mask.shape[0]); w = min(grid.shape[1], mask.shape[1])
    grid = grid[:h, :w]; mask = mask[:h, :w]

    # threshold attention at a high percentile to get the "attended" region
    tval = np.percentile(grid[grid > 0], thresh_pct) if (grid > 0).any() else 1.0
    pred = (grid >= tval).astype(np.uint8)

    dice, iou = dice_iou(pred, mask)
    norm_ent = entropy / np.log(len(attn))

    name = os.path.splitext(os.path.basename(slide_path))[0]
    print(f"{name} [{tag}]: Dice={dice:.3f}  IoU={iou:.3f}  "
          f"norm_entropy={norm_ent:.3f}")

    # overlay figure
    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    axes[0].imshow(grid, cmap="hot"); axes[0].set_title(f"attention ({tag})")
    axes[1].imshow(mask, cmap="gray"); axes[1].set_title("tumour mask")
    axes[2].imshow(mask, cmap="gray")
    axes[2].imshow(grid, cmap="hot", alpha=0.5)
    axes[2].set_title(f"overlay  Dice={dice:.3f}")
    for ax in axes: ax.axis("off")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{name}_heatmap_{tag}.png")
    plt.savefig(out, dpi=100, bbox_inches="tight"); plt.close()
    print(f"  saved -> {out}")
    return dice, iou, norm_ent


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--slide", required=True)
    p.add_argument("--xml", required=True)
    p.add_argument("--out_dir", default="../outputs")
    p.add_argument("--tag", default="")
    args = p.parse_args()
    evaluate(args.checkpoint, args.features, args.slide, args.xml,
             args.out_dir, tag=args.tag)