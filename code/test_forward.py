"""
Sanity-check a single forward pass of ABMIL on one slide's cached features.
No training — just confirms the mechanism produces the right shapes.
"""
import argparse
import h5py
import torch
from models import ABMIL


def main(feat_path):
    with h5py.File(feat_path, "r") as f:
        feats = f["features"][:]
        name = f.attrs["slide_name"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    h = torch.from_numpy(feats).float().to(device)

    from models import CLAM_SB
    model = CLAM_SB(in_dim=h.shape[1]).to(device)
    model.eval()

    # infer a pseudo-label from the filename just for this test
    label = 1 if name.startswith("tumor") else 0

    with torch.no_grad():
        out = model(h, label=label, instance_eval=True)

    logits = out["logits"]
    attn = out["attention"]
    probs = torch.softmax(logits, dim=1)

    print(f"slide            : {name}  (test label={label})")
    print(f"input bag        : {tuple(h.shape)}")
    print(f"attention weights: {tuple(attn.shape)}  sum={attn.sum().item():.4f}")
    print(f"  min / max attn : {attn.min().item():.2e} / {attn.max().item():.2e}")
    print(f"logits           : {tuple(logits.shape)}")
    print(f"slide probability: {probs.squeeze().tolist()}  [P(normal), P(tumour)]")
    print(f"instance loss    : {out['instance_loss'].item():.4f}")
    ent = out["attention_entropy"].item()
    import math
    max_ent = math.log(h.shape[0])   # entropy of a uniform distribution over N patches
    print(f"attention entropy: {ent:.3f}   (max possible = ln(N) = {max_ent:.3f})")
    print(f"  normalised      : {ent / max_ent:.3f}   (1.0 = perfectly uniform)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    args = p.parse_args()
    main(args.features)