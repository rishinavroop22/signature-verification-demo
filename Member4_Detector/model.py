"""Siamese CNN + Transformer.

Both signatures go through the SAME pretrained CNN (shared weights). The Transformer then reads the
feature tokens of the two signatures together. Two outputs:
  1. class: Genuine (0), Forged (1), AI-generated (2)
  2. same person / different person
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

CLASSES = ["Genuine", "Forged", "AI-generated"]


class SiameseCNNTransformer(nn.Module):
    def __init__(self, dim=192, layers=3, heads=6, pretrained=True, drop=0.2):
        super().__init__()
        r = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.cnn = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3)   # 256 x 8 x 16
        self.proj = nn.Conv2d(256, dim, 1)
        self.pos = nn.Parameter(torch.randn(1, 8 * 16, dim) * 0.02)
        self.seg = nn.Parameter(torch.randn(2, 1, 1, dim) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        enc = nn.TransformerEncoderLayer(dim, heads, 3 * dim, drop, "gelu", batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.emb = nn.Linear(256, 128)
        self.head = nn.Sequential(nn.Linear(dim + 3 * 128, 192), nn.GELU(), nn.Dropout(0.3), nn.Linear(192, 3))
        self.same = nn.Sequential(nn.Linear(dim + 2 * 128 + 1, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, 1))

    def branch(self, x):
        """x: ink map (B,1,128,256) in [0,1]."""
        f = self.cnn(((x - 0.1) / 0.28).expand(-1, 3, -1, -1))
        return f, F.normalize(self.emb(f.mean((2, 3)) + f.amax((2, 3))), dim=1)

    def forward(self, ref, query):
        n = ref.size(0)
        f, e = self.branch(torch.cat([ref, query]))
        t = self.proj(f).flatten(2).transpose(1, 2) + self.pos
        t = torch.cat([self.cls.expand(n, -1, -1), t[:n] + self.seg[0], t[n:] + self.seg[1]], 1)
        c = self.norm(self.tr(t)[:, 0])
        a, b = e[:n], e[n:]
        logits = self.head(torch.cat([c, (a - b).abs(), a * b, b], 1))
        dist = (a - b).norm(dim=1)
        same = self.same(torch.cat([c, (a - b).abs(), a * b, dist[:, None]], 1)).squeeze(1)
        return logits, dist, same


def contrastive(dist, same, margin=1.0):
    """Genuine pairs are pulled together; forged and AI pairs are pushed apart."""
    return (same * dist ** 2 + (1 - same) * F.relu(margin - dist) ** 2).mean()
