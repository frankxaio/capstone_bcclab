import torch
import numpy as np
import os
from torch.nn.functional import elu
import math
from torch import nn, einsum
from torch.nn import MultiheadAttention
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# classes
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x,mask = None,y=None):
        return self.fn(self.norm(x),mask)
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x,mask = None):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 48, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x,mask = None):
        # ============================== TODO [Ex 3.4a] ==============================
        # Multi-head scaled dot-product attention. x: (b, n, dim)
        #   1. to_qkv -> split into q, k, v -> reshape each to (b, heads, n, dim_head)
        #      (einops: rearrange(t, "b n (h d) -> b h n d", h=self.heads))
        #   2. scores = q @ k^T * self.scale
        #   3. if mask is given: set scores where mask == 0 to -1e9
        #   4. softmax (self.attend) + dropout, then weight v
        #   5. merge heads back to (b, n, heads*dim_head) and apply self.to_out
        # ============================================================================
        raise NotImplementedError("TODO [Ex 3.4a]")
    
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x,mask = None):
        # ============================== TODO [Ex 3.4b] ==============================
        # Pre-norm Transformer encoder. For every (attn, ff) pair in self.layers
        # apply both sub-layers with RESIDUAL connections (x = f(x) + x),
        # passing `mask` to the attention block. Return x.
        # ============================================================================
        raise NotImplementedError("TODO [Ex 3.4b]")

