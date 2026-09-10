import h5py, torch
from models import MeanMaxPooling, TransMIL, ABMIL, CLAM_SB

with h5py.File("../features/tumor_001.h5") as f:
    h = torch.from_numpy(f["features"][:]).float()
device = "cuda" if torch.cuda.is_available() else "cpu"
h = h.to(device)

models = {
    "mean": MeanMaxPooling(mode="mean"),
    "max":  MeanMaxPooling(mode="max"),
    "abmil": ABMIL(),
    "transmil": TransMIL(),
    "clam": CLAM_SB(),
}
for name, m in models.items():
    m = m.to(device).eval()
    with torch.no_grad():
        out = m(h) if name!="clam" else m(h)
        if isinstance(out, tuple): out = {"logits": out[0], "attention": out[1]}
    lg = out["logits"]; A = out["attention"]
    print(f"{name:9s}: logits {tuple(lg.shape)}, attn {tuple(A.shape)}, attn_sum {A.sum().item():.3f}")