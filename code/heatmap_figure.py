"""
Generate a standalone attention heatmap overlay for a single slide,
for use in the pipeline diagram. No annotation / Dice needed.
"""
import numpy as np, h5py, torch, openslide
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from train import build_model
from evaluate_heatmap import build_attention_grid

SLIDE = "../data/eval_slides/tumor_001.tif"   # adjust path if tumor_001 lives elsewhere
FEAT  = "../features/tumor_001.h5"
CKPT  = "../outputs/checkpoints/clam_fold0_lam0.05_best.pt"
LEVEL = 5

device = "cuda" if torch.cuda.is_available() else "cpu"
with h5py.File(FEAT) as f:
    feats = torch.from_numpy(f["features"][:]).float().to(device)
    coords = f["coords"][:]

model = build_model("clam", in_dim=feats.shape[1]).to(device)
model.load_state_dict(torch.load(CKPT, map_location=device))
model.eval()
with torch.no_grad():
    a = model(feats)["attention"].cpu().numpy()
a = (a - a.min()) / (a.max() - a.min() + 1e-8)

grid = build_attention_grid(coords, a, SLIDE, level=LEVEL)
slide = openslide.OpenSlide(SLIDE)
lvl = min(LEVEL, slide.level_count-1)
thumb = np.array(slide.read_region((0,0), lvl, slide.level_dimensions[lvl]).convert("RGB"))
slide.close()
h,w = grid.shape

fig, ax = plt.subplots(figsize=(5,5))
ax.imshow(thumb[:h,:w])
# apply a mild gamma to spread the concentrated attention, then mask only true zeros
grid_vis = np.power(grid, 0.5)  # sqrt boosts mid/low values so they're visible
gm = np.ma.masked_where(grid_vis < 0.02, grid_vis)
ax.imshow(gm, cmap="jet", alpha=0.7)
ax.axis("off")
plt.tight_layout(pad=0)
plt.savefig("../outputs/fig_diagram_heatmap.png", dpi=150, bbox_inches="tight")
print("saved ../outputs/fig_diagram_heatmap.png")