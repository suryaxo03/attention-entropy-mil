"""
Multi-scale patch extraction for CAMELYON16.
For every tissue location on the grid, extracts a 256x256 patch at 20x (level 1)
and a context patch at 5x (level 3) centred on the same point, resized to 256x256.
Saves patch coordinates + both feature-ready patches to an HDF5 file.
"""
import os
import argparse
import numpy as np
import openslide
import h5py
from segment_tissue import segment_tissue


def extract_patches(slide_path, out_dir,
                    patch_level=1, context_level=3,
                    patch_size=256, seg_level=6,
                    tissue_thresh=0.5):
    """
    patch_level:    pyramid level for the high-res patch (1 = 20x here).
    context_level:  pyramid level for the low-res context patch (3 = 5x).
    patch_size:     output patch edge in pixels (both scales resized to this).
    tissue_thresh:  minimum fraction of a patch that must be tissue to keep it.
    """
    slide = openslide.OpenSlide(slide_path)
    name = os.path.splitext(os.path.basename(slide_path))[0]

    # Guard against slides with fewer levels than expected
    patch_level = min(patch_level, slide.level_count - 1)
    context_level = min(context_level, slide.level_count - 1)

    # Tissue mask (at low seg_level) tells us where to extract
    _, mask = segment_tissue(slide_path, seg_level=seg_level)
    mask_dims = slide.level_dimensions[seg_level]  # (w, h) at seg_level

    # Downsample factors relative to level 0
    ds_patch = int(slide.level_downsamples[patch_level])      # e.g. 2
    ds_seg = int(slide.level_downsamples[seg_level])          # e.g. 64

    # Patch size at level 0 = patch_size * downsample of patch_level
    patch_size_l0 = patch_size * ds_patch

    # Step across level 0 in strides of one patch (non-overlapping)
    w0, h0 = slide.level_dimensions[0]

    coords = []          # top-left (x, y) at level 0 for each kept patch
    for y0 in range(0, h0 - patch_size_l0 + 1, patch_size_l0):
        for x0 in range(0, w0 - patch_size_l0 + 1, patch_size_l0):
            # Map the patch centre to the segmentation mask to test tissue
            cx = (x0 + patch_size_l0 // 2) // ds_seg
            cy = (y0 + patch_size_l0 // 2) // ds_seg
            if cy >= mask.shape[0] or cx >= mask.shape[1]:
                continue
            # Check a small window in the mask around the centre
            y_lo = max(0, cy - 1); y_hi = min(mask.shape[0], cy + 2)
            x_lo = max(0, cx - 1); x_hi = min(mask.shape[1], cx + 2)
            if mask[y_lo:y_hi, x_lo:x_hi].mean() >= tissue_thresh:
                coords.append((x0, y0))

    print(f"  {name}: {len(coords)} tissue patches on grid")

    if len(coords) == 0:
        print(f"  WARNING: no patches for {name}, skipping")
        slide.close()
        return

    # Extract both scales for each coordinate
    n = len(coords)
    patches_hi = np.zeros((n, patch_size, patch_size, 3), dtype=np.uint8)
    patches_lo = np.zeros((n, patch_size, patch_size, 3), dtype=np.uint8)

    ctx_size_l0 = patch_size * int(slide.level_downsamples[context_level])

    for i, (x0, y0) in enumerate(coords):
        # High-res patch at patch_level
        hi = slide.read_region((x0, y0), patch_level,
                               (patch_size, patch_size)).convert("RGB")
        patches_hi[i] = np.array(hi)

        # Context patch: wider region centred on same point, at context_level
        cx0 = x0 + patch_size_l0 // 2 - ctx_size_l0 // 2
        cy0 = y0 + patch_size_l0 // 2 - ctx_size_l0 // 2
        cx0 = max(0, min(cx0, w0 - ctx_size_l0))
        cy0 = max(0, min(cy0, h0 - ctx_size_l0))
        lo = slide.read_region((cx0, cy0), context_level,
                               (patch_size, patch_size)).convert("RGB")
        patches_lo[i] = np.array(lo)

    slide.close()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}.h5")
    with h5py.File(out_path, "w") as f:
        f.create_dataset("coords", data=np.array(coords))
        f.create_dataset("patches_hi", data=patches_hi, compression="gzip")
        f.create_dataset("patches_lo", data=patches_lo, compression="gzip")
        f.attrs["slide_name"] = name
        f.attrs["patch_level"] = patch_level
        f.attrs["context_level"] = context_level
        f.attrs["patch_size"] = patch_size
    print(f"  saved -> {out_path}  ({n} patches, both scales)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--slide", required=True)
    p.add_argument("--out_dir", default="../patches")
    p.add_argument("--patch_level", type=int, default=1)
    p.add_argument("--context_level", type=int, default=3)
    p.add_argument("--patch_size", type=int, default=256)
    args = p.parse_args()

    print(f"Extracting patches from {os.path.basename(args.slide)} ...")
    extract_patches(args.slide, args.out_dir,
                    patch_level=args.patch_level,
                    context_level=args.context_level,
                    patch_size=args.patch_size)