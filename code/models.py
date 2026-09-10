"""
MIL aggregators for WSI classification.
Starts with ABMIL (Ilse et al., 2018): gated-attention pooling over patch
features, producing a slide-level prediction plus per-patch attention weights.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MeanMaxPooling(nn.Module):
    """
    Non-learned control aggregators. mode='mean' or 'max'.
    A small linear classifier sits on top of the pooled feature so it can train,
    but the aggregation itself has no learned attention.
    """
    def __init__(self, in_dim=4096, hidden=512, n_classes=2, mode="mean"):
        super().__init__()
        self.mode = mode
        self.compress = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.25))
        self.classifier = nn.Linear(hidden, n_classes)

    def forward(self, h, label=None, instance_eval=False):
        h = self.compress(h)                       # (N, hidden)
        if self.mode == "mean":
            z = h.mean(dim=0, keepdim=True)        # (1, hidden)
        else:
            z = h.max(dim=0, keepdim=True).values  # (1, hidden)
        logits = self.classifier(z)
        # uniform "attention" so heatmap eval still works (all patches equal)
        A = torch.full((h.shape[0],), 1.0 / h.shape[0], device=h.device)
        ent = -torch.sum(A * torch.log(A + 1e-8))
        return {"logits": logits, "attention": A, "attention_entropy": ent}

class GatedAttention(nn.Module):
    """
    Gated attention (Ilse et al., 2018).
    Maps a bag of patch features (N, D) to attention logits (N, 1).
    """
    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.V = nn.Linear(in_dim, hidden)   # tanh branch
        self.U = nn.Linear(in_dim, hidden)   # sigmoid gate
        self.w = nn.Linear(hidden, 1)        # to a single score per patch

    def forward(self, h):
        # h: (N, in_dim)
        a = torch.tanh(self.V(h))            # (N, hidden)
        g = torch.sigmoid(self.U(h))         # (N, hidden)
        scores = self.w(a * g)               # (N, 1)  gated
        return scores


class ABMIL(nn.Module):
    """
    Attention-based MIL.
    - optional feature compression (4096 -> hidden) to keep the head small
    - gated attention over patches
    - softmax-normalised attention weights
    - weighted sum -> bag representation -> linear classifier
    """
    def __init__(self, in_dim=4096, hidden=512, attn_hidden=256, n_classes=2):
        super().__init__()
        self.compress = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.25),
        )
        self.attention = GatedAttention(hidden, attn_hidden)
        self.classifier = nn.Linear(hidden, n_classes)

    def forward(self, h, label=None, instance_eval=False):
        # h: (N, in_dim) features for ONE slide (one bag)
        h = self.compress(h)                 # (N, hidden)
        scores = self.attention(h)           # (N, 1)
        A = torch.softmax(scores, dim=0)     # (N, 1) attention weights, sum to 1
        z = torch.sum(A * h, dim=0, keepdim=True)  # (1, hidden) bag representation
        logits = self.classifier(z)          # (1, n_classes)
        return {"logits": logits, "attention": A.squeeze(1),
                "attention_entropy": -torch.sum(A.squeeze(1) * torch.log(A.squeeze(1) + 1e-8))}

class CLAM_SB(ABMIL):
    """
    Single-branch CLAM (Lu et al., 2021), built on ABMIL.
    Adds instance-level clustering: the top-k highest-attention patches are
    treated as pseudo-positives and the bottom-k lowest-attention as
    pseudo-negatives, and a small instance classifier learns to separate them.
    This constrains the attention to be diagnostically meaningful, not just
    label-correct.
    """
    def __init__(self, in_dim=4096, hidden=512, attn_hidden=256,
                 n_classes=2, k_sample=8):
        super().__init__(in_dim, hidden, attn_hidden, n_classes)
        self.k_sample = k_sample
        # instance classifier: does this patch look like tumour or not?
        self.instance_classifier = nn.Linear(hidden, 2)
        self.instance_loss_fn = nn.CrossEntropyLoss()

    def forward(self, h, label=None, instance_eval=False):
        # h: (N, in_dim) for one slide
        h = self.compress(h)                       # (N, hidden)
        scores = self.attention(h)                 # (N, 1)
        A = torch.softmax(scores, dim=0)           # (N, 1)
        z = torch.sum(A * h, dim=0, keepdim=True)  # (1, hidden)
        logits = self.classifier(z)                # (1, n_classes)

        result = {"logits": logits, "attention": A.squeeze(1)}
        result["attention_entropy"] = self.attention_entropy(A.squeeze(1))

        if instance_eval and label is not None:
            inst_loss = self._instance_loss(h, A.squeeze(1), label)
            result["instance_loss"] = inst_loss

        return result

    def _instance_loss(self, h, attn, label):
        """
        Take the top-k and bottom-k patches by attention, label them
        pseudo-positive / pseudo-negative, and classify them.
        Only applied when the slide is positive (label == 1); for negative
        slides all patches are genuinely negative.
        """
        k = min(self.k_sample, h.shape[0] // 2)
        if k < 1:
            return torch.tensor(0.0, device=h.device)

        # indices of highest- and lowest-attention patches
        top_idx = torch.topk(attn, k).indices
        bot_idx = torch.topk(-attn, k).indices

        top_feats = h[top_idx]                 # (k, hidden)
        bot_feats = h[bot_idx]                 # (k, hidden)

        if label == 1:
            # positive slide: high-attn patches -> pseudo-tumour (1),
            #                 low-attn patches  -> pseudo-normal (0)
            feats = torch.cat([top_feats, bot_feats], dim=0)
            targets = torch.cat([
                torch.ones(k, dtype=torch.long, device=h.device),
                torch.zeros(k, dtype=torch.long, device=h.device),
            ])
        else:
            # negative slide: every patch is genuinely normal (0)
            feats = torch.cat([top_feats, bot_feats], dim=0)
            targets = torch.zeros(2 * k, dtype=torch.long, device=h.device)

        inst_logits = self.instance_classifier(feats)   # (2k, 2)
        return self.instance_loss_fn(inst_logits, targets)

    def attention_entropy(self, attn, eps=1e-8):
        """
        Shannon entropy of the attention distribution.
        attn: (N,) softmax attention weights summing to 1.
        High entropy = spread-out attention; low = concentrated.
        This is the quantity the proposed regularisation rewards.
        """
        return -torch.sum(attn * torch.log(attn + eps))

class TransMIL(nn.Module):
    """
    TransMIL (Shao et al., 2021), compact implementation.
    Projects features, prepends a class token, applies two transformer layers
    with standard multi-head self-attention, and classifies from the class token.
    Attention for the heatmap is taken from the class-token attention over patches.
    (Full TransMIL uses Nystrom approximation + PPEG; this uses standard attention,
    which is tractable here because bags are a few thousand patches and we run on A40.)
    """
    def __init__(self, in_dim=4096, hidden=512, n_classes=2, n_heads=8):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=hidden*2,
            dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(hidden)
        self.classifier = nn.Linear(hidden, n_classes)
        self.n_heads = n_heads

    def forward(self, h, label=None, instance_eval=False):
        # h: (N, in_dim)
        x = self.proj(h).unsqueeze(0)                  # (1, N, hidden)
        cls = self.cls_token                            # (1, 1, hidden)
        x = torch.cat([cls, x], dim=1)                  # (1, N+1, hidden)
        x = self.transformer(x)                         # (1, N+1, hidden)
        x = self.norm(x)
        logits = self.classifier(x[:, 0])               # from class token -> (1, n_classes)

        # Approximate per-patch attention: cosine similarity of each patch token
        # to the class token, softmaxed. Gives a heatmap comparable to the others.
        cls_out = x[0, 0]                                # (hidden,)
        patch_out = x[0, 1:]                             # (N, hidden)
        sim = torch.matmul(patch_out, cls_out) / (cls_out.norm() + 1e-8)
        A = torch.softmax(sim, dim=0)                   # (N,)
        ent = -torch.sum(A * torch.log(A + 1e-8))
        return {"logits": logits, "attention": A, "attention_entropy": ent}