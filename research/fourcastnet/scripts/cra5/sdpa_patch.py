"""Runtime patch: replace VAEformer's memory-hungry 'math' attention with
torch's scaled_dot_product_attention.

The original 'math' branch materializes the full (B, heads, N, N) attention
matrix -- 6.4 GiB for a single CRA5 frame, which does not fit on a 10 GB card.
SDPA computes the same result without ever building that matrix.

Equivalence: the 'math' branch applies no relative position bias and no mask,
so plain SDPA with the same `scale` is mathematically identical.
"""
import torch
import torch.nn.functional as F
from einops import rearrange


def _sdpa_forward(self, x, H, W):
    B, N, C = x.shape
    qkv = self.qkv(x)
    qkv = rearrange(qkv, "B N (K H D) -> K B H N D", K=3, H=self.num_heads)
    q, k, v = qkv.unbind(0)
    x = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
    x = x.transpose(1, 2).reshape(B, N, C)
    return self.proj(x)


def apply():
    # Imported lazily: touching cra5.models.vaeformer before cra5.api has been
    # imported trips a circular import (vaeformer <-> compressai.zoo.image).
    from cra5.models.vaeformer import vit_nlc

    vit_nlc.Attention.forward = _sdpa_forward
    return "SDPA patch applied"
