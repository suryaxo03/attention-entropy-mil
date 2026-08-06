"""
Feature extraction for multi-scale CAMELYON16 patches.
Runs ResNet-50 (ImageNet-pretrained, frozen) over the 20x and 5x patches,
concatenates the two feature vectors per location, and caches to HDF5.
Each patch -> 2048-dim vector (1024 from each scale).
"""
import os
import argparse
import numpy as np
import h5py
import torch
import torch.nn as nn
from torchvision import models, transforms


def build_encoder(device):
    """ResNet-50 truncated after global average pooling -> 2048-dim... 
    we take the penultimate (2048) then project? No: torchvision ResNet-50
    avgpool output is 2048. We keep that."""
    net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    # Drop the final classification fc layer; keep everything up to avgpool.
    net.fc = nn.Identity()
    net.eval().to(device)
    for p in net.parameters():
        p.requires_grad = False
    return net


# ImageNet normalisation (what the pretrained weights expect)
_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def encode_batch(net, imgs, device):
    """imgs: (B, H, W, 3) uint8 numpy -> (B, 2048) float32 numpy."""
    batch = torch.stack([_tf(im) for im in imgs]).to(device)
    with torch.no_grad():
        feats = net(batch)          # (B, 2048)
    return feats.cpu().numpy().astype(np.float32)


def extract_features(h5_in, out_dir, batch_size=128, device="cuda"):
    name = os.path.splitext(os.path.basename(h5_in))[0]
    with h5py.File(h5_in, "r") as f:
        hi = f["patches_hi"][:]
        lo = f["patches_lo"][:]
        coords = f["coords"][:]

    n = len(hi)
    print(f"  {name}: encoding {n} patches (both scales) ...")

    net = build_encoder(device)

    feats_hi = np.zeros((n, 2048), dtype=np.float32)
    feats_lo = np.zeros((n, 2048), dtype=np.float32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        feats_hi[start:end] = encode_batch(net, hi[start:end], device)
        feats_lo[start:end] = encode_batch(net, lo[start:end], device)
        print(f"    {end}/{n}", end="\r")
    print()

    # Concatenate the two scales -> 4096-dim combined feature per patch
    feats = np.concatenate([feats_hi, feats_lo], axis=1)  # (n, 4096)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}.h5")
    with h5py.File(out_path, "w") as f:
        f.create_dataset("features", data=feats, compression="gzip")
        f.create_dataset("coords", data=coords)
        f.attrs["slide_name"] = name
        f.attrs["feat_dim"] = feats.shape[1]
        f.attrs["encoder"] = "resnet50_imagenet"
    print(f"  saved -> {out_path}  ({n} x {feats.shape[1]})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--h5", required=True, help="patch .h5 from extract_patches.py")
    p.add_argument("--out_dir", default="../features")
    p.add_argument("--batch_size", type=int, default=128)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    extract_features(args.h5, args.out_dir, batch_size=args.batch_size, device=device)