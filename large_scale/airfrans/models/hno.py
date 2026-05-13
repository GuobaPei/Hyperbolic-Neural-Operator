"""
Hyperbolic Neural Operator (HNO) for AirfRANS

This implementation uses an M-token interaction core (via slice/deslice
aggregation) and replaces Euclidean dot-product token mixing with stabilized
Lorentz-hyperbolic distance attention.
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import trunc_normal_


ACTIVATION = {
    'gelu': nn.GELU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'relu': nn.ReLU,
    'leaky_relu': lambda: nn.LeakyReLU(0.1),
    'softplus': nn.Softplus,
    'ELU': nn.ELU,
    'silu': nn.SiLU,
}


class MLP(nn.Module):
    def __init__(self, n_input: int, n_hidden: int, n_output: int, n_layers: int = 1, act: str = 'gelu', res: bool = True):
        super().__init__()
        act_cls = ACTIVATION.get(act, None)
        if act_cls is None:
            raise NotImplementedError(f'Unknown activation: {act}')
        if callable(act_cls) and not isinstance(act_cls, type):
            # leaky_relu lambda
            act_layer = act_cls
        else:
            act_layer = act_cls

        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act_layer())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act_layer()) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_pre(x)
        for layer in self.linears:
            x = layer(x) + x if self.res else layer(x)
        return self.linear_post(x)


class LorentzManifold:
    """
    Lorentz ops (adapted from history/ela.py) for numerical stability.
    """
    ACOSH_EPS = 1e-4

    @staticmethod
    def to_lorentz(x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f = x.float()
        norm_sq = (x_f * x_f).sum(dim=-1, keepdim=True)
        t = torch.sqrt(1.0 + norm_sq)
        out = torch.cat([t, x_f], dim=-1)
        return out.to(orig_dtype)

    @staticmethod
    def pairwise_lorentz_distance_multihead(x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B,H,N,d+1), y: (B,H,M,d+1) -> (B,H,N,M)
        """
        if y is None:
            y = x
        orig_dtype = x.dtype
        x_f = x.float()
        y_f = y.float()
        inner = -torch.einsum('bhni,bhmi->bhnm', x_f[..., :1], y_f[..., :1]) + \
                torch.einsum('bhni,bhmi->bhnm', x_f[..., 1:], y_f[..., 1:])
        dist = torch.acosh((-inner).clamp(min=1.0 + LorentzManifold.ACOSH_EPS))
        return dist.to(orig_dtype)


class HyperbolicSliceAttention(nn.Module):
    """
    Slice/deslice aggregation with Lorentz-distance attention among slice tokens.

    Input: x (B,N,C)
    Output: (B,N,C)
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 32,
        dropout: float = 0.0,
        slice_num: int = 64,
        hyp_dim: int = 16,
   ):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        # slicing temperature
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        nn.init.orthogonal_(self.in_project_slice.weight)

        # hyperbolic Q/K projection
        self.hyp_dim = hyp_dim
        self.to_q_h = nn.Linear(dim_head, hyp_dim, bias=False)
        self.to_k_h = nn.Linear(dim_head, hyp_dim, bias=False)

        # value projection stays in dim_head
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
       )

        # learnable temperature for hyperbolic token attention (per-head)
        self.temp_token = nn.Parameter(torch.ones([1, heads, 1, 1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        fx_mid = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()

        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)  # (B,H,N,G)
        slice_norm = slice_weights.sum(2)  # (B,H,G)
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        # token attention in hyperbolic space
        q = self.to_q_h(slice_token)  # (B,H,G,hyp)
        k = self.to_k_h(slice_token)  # (B,H,G,hyp)
        q_l = LorentzManifold.to_lorentz(q)
        k_l = LorentzManifold.to_lorentz(k)
        dist = LorentzManifold.pairwise_lorentz_distance_multihead(q_l, k_l)  # (B,H,G,G)

        temp = self.temp_token.clamp(0.1, 3.0)
        attn = F.softmax((-dist / temp).float(), dim=-1).to(slice_token.dtype)
        attn = self.dropout(attn)

        v = self.to_v(slice_token)  # (B,H,G,D)
        out_slice = attn @ v  # (B,H,G,D)

        # deslice back
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)


class HNOBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        act: str = 'gelu',
        mlp_ratio: float = 2.0,
        last_layer: bool = False,
        out_dim: int = 4,
        slice_num: int = 64,
        hyp_dim: int = 16,
   ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = HyperbolicSliceAttention(
            hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            slice_num=slice_num,
            hyp_dim=hyp_dim,
       )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, int(hidden_dim * mlp_ratio), hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx: torch.Tensor) -> torch.Tensor:
        fx = self.attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.proj(self.ln_3(fx))
        return fx


class HNO(nn.Module):
    """
    HNO for AirfRANS with an M-token interaction core.
    """

    def __init__(
        self,
        space_dim: int = 7,
        n_layers: int = 8,
        n_hidden: int = 256,
        dropout: float = 0.0,
        n_head: int = 8,
        act: str = 'gelu',
        mlp_ratio: float = 2.0,
        fun_dim: int = 0,
        out_dim: int = 4,
        slice_num: int = 64,
        hyp_dim: int = 16,
        ref: int = 8,
        unified_pos: int = 1,
   ):
        super().__init__()
        self.__name__ = "HNO"
        self.ref = int(ref)
        self.unified_pos = bool(unified_pos)
        self.space_dim = int(space_dim)
        self.n_hidden = int(n_hidden)

        # preprocess
        if self.unified_pos:
            # append ref^2 distance features (2D grid) to x
            in_dim = fun_dim + space_dim + self.ref * self.ref
            self.preprocess = MLP(in_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)
        else:
            in_dim = fun_dim + space_dim
            self.preprocess = MLP(in_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)

        self.blocks = nn.ModuleList([
            HNOBlock(
                num_heads=n_head,
                hidden_dim=n_hidden,
                dropout=dropout,
                act=act,
                mlp_ratio=mlp_ratio,
                out_dim=out_dim,
                slice_num=slice_num,
                hyp_dim=hyp_dim,
                last_layer=(i == n_layers - 1),
           )
            for i in range(n_layers)
        ])

        self.placeholder = nn.Parameter((1.0 / n_hidden) * torch.rand(n_hidden, dtype=torch.float))
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _get_grid_feat_2d(self, pos: torch.Tensor) -> torch.Tensor:
        """
        pos: (B,N,2) or (B,N,>=2) uses first 2 dims.
        returns: (B,N,ref^2) distances to a fixed 2D grid.
        """
        B, N, _ = pos.shape
        device = pos.device
        pos2 = pos[..., :2]
        gridx = torch.tensor(np.linspace(-2.0, 4.0, self.ref), dtype=torch.float32, device=device)
        gridy = torch.tensor(np.linspace(-1.5, 1.5, self.ref), dtype=torch.float32, device=device)
        gridx = gridx.reshape(1, self.ref, 1, 1).repeat(B, 1, self.ref, 1)
        gridy = gridy.reshape(1, 1, self.ref, 1).repeat(B, self.ref, 1, 1)
        grid = torch.cat((gridx, gridy), dim=-1).reshape(B, self.ref * self.ref, 2)
        dist_feat = torch.sqrt(torch.sum((pos2[:, :, None, :] - grid[:, None, :, :]) ** 2, dim=-1))
        return dist_feat.contiguous()

    def forward(self, data) -> torch.Tensor:
        # upstream uses batch_size=1 and wraps as (1,N,C)
        x = data.x[None, :, :]  # (1,N,C_in)
        if self.unified_pos:
            pos = data.pos[None, :, :]
            grid_feat = self._get_grid_feat_2d(pos)
            x = torch.cat((x, grid_feat), dim=-1)
        fx = self.preprocess(x) + self.placeholder[None, None, :]
        for block in self.blocks:
            fx = block(fx)
        return fx[0]
