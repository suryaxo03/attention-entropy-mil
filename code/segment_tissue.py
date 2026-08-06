"""
Tissue segmentation for CAMELYON16 WSIs.
Reads a slide at low resolution, separates tissue from glass background
using Otsu thresholding on saturation, cleans the mask with morphology,
and saves a side-by-side overlay for visual inspection.
"""
import os
import argparse
import numpy as np
import openslide
import cv2
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt


def segment_tissue(slide_path, seg_level=6, min_region_frac=0.02):
    """
    Returns (thumbnail_rgb, tissue_mask) both as numpy arrays at seg_level.
    seg_level: pyramid level to work at (higher = lower resolution). (Level 6 is the downsample-64 level)
    min_region_frac: drop connected components smaller than this fraction
                     of the largest one (removes dust/pen specks).
    """
    slide = openslide.OpenSlide(slide_path)

    # Guard: some slides may have fewer levels than requested
    seg_level = min(seg_level, slide.level_count - 1)
    dims = slide.level_dimensions[seg_level]

    # Read the whole slide at this low level (fits in memory)
    img = slide.read_region((0, 0), seg_level, dims).convert("RGB")
    img = np.array(img)

    # Convert to HSV; tissue is distinguished from glass mainly by saturation
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]

    # Otsu threshold on the saturation channel
    thresh = threshold_otsu(sat)
    mask = (sat > thresh).astype(np.uint8)

    # Morphological cleanup: close small holes, then remove small specks
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Keep only connected components above a fraction of the largest one
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]  # skip background label 0
        largest = areas.max()
        keep = {i + 1 for i, a in enumerate(areas) if a >= min_region_frac * largest}
        mask = np.isin(labels, list(keep)).astype(np.uint8)

    slide.close()
    return img, mask


def save_overlay(img, mask, out_path, slide_name):
    """Save a 3-panel figure: thumbnail, mask, overlay."""
    overlay = img.copy()
    overlay[mask == 1] = (0.6 * overlay[mask == 1] +
                          0.4 * np.array([0, 255, 0])).astype(np.uint8)

    tissue_pct = 100.0 * mask.mean()

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    axes[0].imshow(img);     axes[0].set_title(f"{slide_name}\noriginal")
    axes[1].imshow(mask, cmap="gray"); axes[1].set_title(f"tissue mask\n{tissue_pct:.1f}% tissue")
    axes[2].imshow(overlay); axes[2].set_title("overlay")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  saved overlay -> {out_path}  ({tissue_pct:.1f}% tissue)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--slide", required=True, help="path to .tif slide")
    p.add_argument("--out_dir", default="../outputs", help="where to save overlay")
    p.add_argument("--seg_level", type=int, default=6)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.slide))[0]
    print(f"Segmenting {name} ...")

    img, mask = segment_tissue(args.slide, seg_level=args.seg_level)
    out_path = os.path.join(args.out_dir, f"{name}_segmentation.png")
    save_overlay(img, mask, out_path, name)