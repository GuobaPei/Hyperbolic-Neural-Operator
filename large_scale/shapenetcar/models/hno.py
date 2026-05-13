import math
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
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
        act_layer = act_cls if isinstance(act_cls, type) else act_cls  # allow lambda for leaky_relu
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
    Minimal Lorentz model ops for hyperbolic attention.
    """
    ACOSH_EPS = 1e-4

    @staticmethod
    def to_lorentz(x: torch.Tensor) -> torch.Tensor:
        # x: (..., d) -> (..., d+1) with time-like coordinate
        orig_dtype = x.dtype
        x_f = x.float()
        norm_sq = (x_f * x_f).sum(dim=-1, keepdim=True)
        t = torch.sqrt(1.0 + norm_sq)
        out = torch.cat([t, x_f], dim=-1)
        return out.to(orig_dtype)

    @staticmethod
    def pairwise_lorentz_distance_multihead(x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B,H,N,d+1), y: (B,H,M,d+1) -> dist: (B,H,N,M)
        """
        if y is None:
            y = x
        # Minkowski inner product: <x,y>_L = -t_x t_y + sum_i x_i y_i
        tx = x[..., :1]  # (B,H,N,1)
        xx = x[..., 1:]  # (B,H,N,d)
        ty = y[..., :1]  # (B,H,M,1)
        yy = y[..., 1:]  # (B,H,M,d)
        mink = -(tx @ ty.transpose(-1, -2)) + (xx @ yy.transpose(-1, -2))
        z = (-mink).clamp(min=1.0 + LorentzManifold.ACOSH_EPS)
        return torch.acosh(z)


class HyperbolicSliceAttention(nn.Module):
    """
    Slice -> hyperbolic attention among slice tokens -> deslice.
    """
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0, slice_num: int = 64, hyp_dim: int = 16):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        torch.nn.init.orthogonal_(self.in_project_slice.weight)

        # project slice tokens to low-dim hyperbolic space
        self.to_q = nn.Linear(dim_head, hyp_dim, bias=False)
        self.to_k = nn.Linear(dim_head, hyp_dim, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        B, N, C = x.shape

        fx_mid = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()  # B H N D
        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()    # B H N D

        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)  # B H N G
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        # hyperbolic attention on Lorentz manifold
        q = self.to_q(slice_token)  # B H G hyp
        k = self.to_k(slice_token)
        v = self.to_v(slice_token)  # B H G D
        qL = LorentzManifold.to_lorentz(q)  # B H G hyp+1
        kL = LorentzManifold.to_lorentz(k)
        dist = LorentzManifold.pairwise_lorentz_distance_multihead(qL, kL)  # B H G G
        # temperature: (1,H,1,1) -> broadcast to (B,H,G,G)
        attn = self.softmax(-dist / self.temperature)
        attn = self.dropout(attn)
        out_slice = torch.matmul(attn, v)  # B H G D

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
        mlp_ratio: int = 4,
        last_layer: bool = False,
        out_dim: int = 1,
        slice_num: int = 64,
        hyp_dim: int = 16,
   ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = HyperbolicSliceAttention(
            hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads, dropout=dropout, slice_num=slice_num, hyp_dim=hyp_dim
       )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
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
    Car-Design compatible:
      - forward(data): data has.x and.pos (torch_geometric Data)
      - forward((cfd_data, geom_data)): uses cfd_data
    """
    def __init__(
        self,
        n_hidden: int = 256,
        n_layers: int = 8,
        space_dim: int = 7,
        fun_dim: int = 0,
        n_head: int = 8,
        mlp_ratio: int = 2,
        out_dim: int = 4,
        slice_num: int = 64,
        hyp_dim: int = 16,
        unified_pos: int = 0,
        ref: int = 8,
        dropout: float = 0.0,
        act: str = 'gelu',
        cond_dim: int = 0,
   ):
        super().__init__()
        self.ref = int(ref)
        self.unified_pos = bool(unified_pos)
        self.n_hidden = int(n_hidden)

        if self.unified_pos:
            self.preprocess = None  # lazily built based on pos dim
        else:
            self.preprocess = MLP(fun_dim + space_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)

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
        self.cond_embed = nn.Linear(cond_dim, n_hidden) if cond_dim and cond_dim > 0 else None
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

    def _get_grid_feat(self, pos: torch.Tensor) -> torch.Tensor:
        """
        pos: (B, N, d), d in {2,3}
        returns: (B, N, ref^d)
        """
        B, N, d = pos.shape
        device = pos.device
        if d == 2:
            gridx = torch.linspace(-2.0, 4.0, self.ref, device=device, dtype=torch.float32)
            gridy = torch.linspace(-1.5, 1.5, self.ref, device=device, dtype=torch.float32)
            gridx = gridx.reshape(1, self.ref, 1, 1).repeat(B, 1, self.ref, 1)
            gridy = gridy.reshape(1, 1, self.ref, 1).repeat(B, self.ref, 1, 1)
            grid = torch.cat((gridx, gridy), dim=-1).reshape(B, self.ref ** 2, 2)
        elif d == 3:
            gridx = torch.linspace(-1.5, 1.5, self.ref, device=device, dtype=torch.float32)
            gridy = torch.linspace(0.0, 2.0, self.ref, device=device, dtype=torch.float32)
            gridz = torch.linspace(-4.0, 4.0, self.ref, device=device, dtype=torch.float32)
            gridx = gridx.reshape(1, self.ref, 1, 1, 1).repeat(B, 1, self.ref, self.ref, 1)
            gridy = gridy.reshape(1, 1, self.ref, 1, 1).repeat(B, self.ref, 1, self.ref, 1)
            gridz = gridz.reshape(1, 1, 1, self.ref, 1).repeat(B, self.ref, self.ref, 1, 1)
            grid = torch.cat((gridx, gridy, gridz), dim=-1).reshape(B, self.ref ** 3, 3)
        else:
            raise ValueError(f"unified_pos expects pos dim 2 or 3, got {d}")

        dist_feat = torch.sqrt(torch.sum((pos[:, :, None, :] - grid[:, None, :, :]) ** 2, dim=-1))
        return dist_feat.contiguous()

    def forward(self, data):
        if isinstance(data, (tuple, list)):
            data = data[0]

        x = data.x[None, :, :]  # (1, N, C_in)
        pos = data.pos[None, :, :] if hasattr(data, "pos") else None

        if self.unified_pos:
            if pos is None:
                raise ValueError("unified_pos=True requires data.pos")
            grid_feat = self._get_grid_feat(pos)
            x = torch.cat((x, grid_feat), dim=-1)
            if self.preprocess is None:
                self.preprocess = MLP(x.shape[-1], self.n_hidden * 2, self.n_hidden, n_layers=0, res=False, act='gelu').to(x.device)

        fx = self.preprocess(x) + self.placeholder[None, None, :]

        cond = getattr(data, "condition", None)
        if cond is not None and self.cond_embed is not None:
            if cond.dim() == 1:
                cond = cond[None, :]
            fx = fx + self.cond_embed(cond).unsqueeze(1)

        for block in self.blocks:
            fx = block(fx)
        return fx[0]
