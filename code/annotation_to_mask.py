"""
Rasterise a CAMELYON16 ASAP XML annotation into a binary tumour mask
at a chosen pyramid level, matching the resolution of our attention maps.
Keeps only polygons in the 'Tumor' group.
"""
import os
import argparse
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import openslide


def parse_tumor_polygons(xml_path):
    """Return a list of (N,2) float arrays of level-0 (x,y) polygon vertices."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    polygons = []
    for ann in root.iter("Annotation"):
        group = ann.get("PartOfGroup", "")
        # Keep tumour polygons; skip exclusions / other groups
        if group.lower() != "tumor":
            continue
        coords = []
        for c in ann.iter("Coordinate"):
            coords.append((float(c.get("X")), float(c.get("Y"))))
        if len(coords) >= 3:
            polygons.append(np.array(coords, dtype=np.float64))
    return polygons


def rasterise_mask(slide_path, xml_path, level=6):
    """
    Build a binary mask (H,W) at the given pyramid level, where 1 = tumour.
    Polygon coords are level-0 pixels; we scale them down by the level downsample.
    """
    slide = openslide.OpenSlide(slide_path)
    level = min(level, slide.level_count - 1)
    W, H = slide.level_dimensions[level]
    downsample = slide.level_downsamples[level]
    slide.close()

    mask = np.zeros((H, W), dtype=np.uint8)
    polygons = parse_tumor_polygons(xml_path)
    for poly in polygons:
        scaled = np.round(poly / downsample).astype(np.int32)  # (N,2) x,y
        cv2.fillPoly(mask, [scaled], color=1)
    return mask, len(polygons)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--slide", required=True)
    p.add_argument("--xml", required=True)
    p.add_argument("--level", type=int, default=6)
    p.add_argument("--out_dir", default="../outputs")
    args = p.parse_args()

    name = os.path.splitext(os.path.basename(args.slide))[0]
    mask, n_poly = rasterise_mask(args.slide, args.xml, level=args.level)
    print(f"{name}: {n_poly} tumour polygons, mask {mask.shape}, "
          f"{100*mask.mean():.2f}% tumour at level {args.level}")

    # save a preview
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5, 9))
    plt.imshow(mask, cmap="hot"); plt.axis("off")
    plt.title(f"{name}\ntumour mask ({100*mask.mean():.2f}%)")
    out = os.path.join(args.out_dir, f"{name}_tumour_mask.png")
    plt.savefig(out, dpi=100, bbox_inches="tight"); plt.close()
    print(f"saved -> {out}")