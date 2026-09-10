"""
Appendix heatmap gallery: baseline vs entropy attention for several test slides.
One row per slide: tissue, annotation, baseline (lambda=0), entropy (lambda=0.05).
"""
import os, numpy as np, h5py, torch, openslide
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from train import build_model
from annotation_to_mask import rasterise_mask
from evaluate_heatmap import build_attention_grid

# EDIT this list with the slides you choose:
SLIDES = ["test_001", "test_016", "test_004"]
LEVEL = 6
EVAL = "../data/eval_slides"; FEAT = "../features"; CKPT = "../outputs/checkpoints"

def attention(model_name, lam, feats, device):
    ck = f"{CKPT}/clam_fold0_lam{lam}_best.pt"
    m = build_model(model_name, in_dim=feats.shape[1]).to(device)
    m.load_state_dict(torch.load(ck, map_location=device)); m.eval()
    with torch.no_grad():
        a = m(feats)["attention"].cpu().numpy()
    return (a - a.min())/(a.max()-a.min()+1e-8)

device = "cuda" if torch.cuda.is_available() else "cpu"
n = len(SLIDES)
fig, axes = plt.subplots(n, 4, figsize=(14, 3.4*n))
if n == 1: axes = axes[None, :]

for r, name in enumerate(SLIDES):
    tif = f"{EVAL}/{name}.tif"; xml = f"{EVAL}/{name}.xml"; h5 = f"{FEAT}/{name}.h5"
    if not os.path.exists(tif):
        print(f"downloading {name}")
        os.system(f"aws s3 cp --no-sign-request s3://camelyon-dataset/CAMELYON16/images/{name}.tif {EVAL}/ >/dev/null 2>&1")
    with h5py.File(h5) as f:
        feats = torch.from_numpy(f["features"][:]).float().to(device)
        coords = f["coords"][:]
    slide = openslide.OpenSlide(tif)
    lvl = min(LEVEL, slide.level_count-1)
    thumb = np.array(slide.read_region((0,0), lvl, slide.level_dimensions[lvl]).convert("RGB"))
    slide.close()
    mask, _ = rasterise_mask(tif, xml, level=LEVEL)
    h, w = mask.shape

    ab = attention("clam", "0.0", feats, device)
    ae = attention("clam", "0.05", feats, device)
    gb = build_attention_grid(coords, ab, tif, level=LEVEL)[:h,:w]
    ge = build_attention_grid(coords, ae, tif, level=LEVEL)[:h,:w]

    axes[r,0].imshow(thumb[:h,:w]); axes[r,0].set_ylabel(name, fontsize=11)
    axes[r,1].imshow(mask, cmap="gray")
    for ax, g, t in [(axes[r,2], gb, "baseline"), (axes[r,3], ge, "entropy 0.05")]:
        ax.imshow(thumb[:h,:w])
        gm = np.ma.masked_where(np.power(g,0.5) < 0.15, np.power(g,0.5))
        ax.imshow(gm, cmap="jet", alpha=0.6)
        if r == 0: ax.set_title(t)
    for c in range(4): axes[r,c].set_xticks([]); axes[r,c].set_yticks([])
    if r == 0:
        axes[0,0].set_title("tissue"); axes[0,1].set_title("annotation")

plt.tight_layout()
plt.savefig("../outputs/fig_heatmap_gallery.png", dpi=120, bbox_inches="tight")
print("saved ../outputs/fig_heatmap_gallery.png")