"""Adaptive HNO variant for regular grids and irregular meshes."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pdebench.hno.attention import CrossAttentionBlock, PerceiverBlock
from pdebench.hno.lorentz import LorentzManifold


class RelativePositionBias2D(nn.Module):
    """2D relative position bias(dynamic size)"""
    
    def __init__(self, num_heads: int, max_grid_size: int = 20):
        super().__init__()
        self.num_heads = num_heads
        self.max_grid_size = max_grid_size
        
        num_relative_distance = (2 * max_grid_size - 1) ** 2
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(num_relative_distance, num_heads)
       )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
    
    def _get_relative_position_index(self, grid_size: int, device: torch.device) -> torch.Tensor:
        coords_h = torch.arange(grid_size, device=device)
        coords_w = torch.arange(grid_size, device=device)
        try:
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        except TypeError:
            coords = torch.stack(torch.meshgrid(coords_h, coords_w))
        coords_flatten = coords.view(2, -1)
        
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.max_grid_size - 1
        relative_coords[:, :, 1] += self.max_grid_size - 1
        relative_coords[:, :, 0] *= 2 * self.max_grid_size - 1
        return relative_coords.sum(-1)
    
    def forward(self, grid_size: int, device: torch.device) -> torch.Tensor:
        relative_position_index = self._get_relative_position_index(grid_size, device)
        bias = self.relative_position_bias_table[relative_position_index.view(-1)].view(
            grid_size ** 2, grid_size ** 2, -1
       )
        return bias.permute(2, 0, 1).contiguous()


class HyperbolicPatchAttention(nn.Module):
    """Hyperbolic patch attention for regular grids."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        hyp_dim: int = 16,
        dropout: float = 0.0,
   ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.hyp_dim = hyp_dim
        self.head_dim = hidden_dim // num_heads
        
        self.to_q = nn.Linear(hidden_dim, num_heads * hyp_dim)
        self.to_k = nn.Linear(hidden_dim, num_heads * hyp_dim)
        self.to_v = nn.Linear(hidden_dim, hidden_dim)
        self.to_out = nn.Linear(hidden_dim, hidden_dim)
        
        self.rel_pos_bias = RelativePositionBias2D(num_heads, max_grid_size=20)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, P, C = x.shape
        H = self.num_heads
        
        q = self.to_q(x).view(B, P, H, self.hyp_dim).permute(0, 2, 1, 3)
        k = self.to_k(x).view(B, P, H, self.hyp_dim).permute(0, 2, 1, 3)
        
        # Norm clipping for numerical stability.
        q = LorentzManifold.clip_norm(q)
        k = LorentzManifold.clip_norm(k)
        
        q_lorentz = LorentzManifold.to_lorentz(q)
        k_lorentz = LorentzManifold.to_lorentz(k)
        
        # vectorizedHyperbolic distance
        dist = LorentzManifold.pairwise_lorentz_distance_multihead(q_lorentz, k_lorentz)
        
        grid_size = int(math.sqrt(P))
        rel_pos = self.rel_pos_bias(grid_size, x.device)
        
        temp = self.temperature.clamp(0.1, 3.0)
        attn_logits = -dist / temp + rel_pos.unsqueeze(0)
        attn = F.softmax(attn_logits.float(), dim=-1).to(x.dtype)  #  float32 softmax
        attn = self.dropout(attn)
        
        v = self.to_v(x).view(B, P, H, self.head_dim).permute(0, 2, 1, 3)
        out = attn @ v
        out = out.permute(0, 2, 1, 3).reshape(B, P, C)
        out = self.to_out(out)
        
        return out


class LocalFeatureExtractor(nn.Module):
    """Local feature extraction"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.view(B, C, N).permute(0, 2, 1)
        x = self.norm(x)
        return x


class HyperbolicPatchBlock(nn.Module):
    """Patch-token block = hyperbolic attention + local mixing + FFN."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        hyp_dim: int = 16,
        mlp_ratio: float = 2.5,
        dropout: float = 0.0,
   ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = HyperbolicPatchAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            hyp_dim=hyp_dim,
            dropout=dropout,
       )
        self.local_feat = LocalFeatureExtractor(hidden_dim)
        
        self.norm2 = nn.LayerNorm(hidden_dim)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
       )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, P, C = x.shape
        grid_size = int(math.sqrt(P))
        
        x = x + self.attn(self.norm1(x))
        x = x + self.local_feat(x, grid_size, grid_size)
        x = x + self.ffn(self.norm2(x))
        return x


class AdaptiveHyperbolicNO(nn.Module):
    """
    Adaptive Hyperbolic Neural Operator
    
    Regular grids -> patch tokenization
    Irregular meshes -> latent-set tokenization
    """
    
    def __init__(
        self,
        space_dim: int = 2,
        in_channels: int = 1,
        out_channels: int = 1,
        hidden_dim: int = 96,
        patch_size: int = 5,
        num_latents: int = 64,
        num_heads: int = 4,
        num_layers: int = 4,
        hyp_dim: int = 16,
        mlp_ratio: float = 2.5,
        dropout: float = 0.0,
   ):
        super().__init__()
        
        self.space_dim = space_dim
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        
        # Lifting
        self.lifting = nn.Sequential(
            nn.Linear(in_channels + space_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
       )
        self.lifting_no_pos = nn.Sequential(
            nn.Linear(space_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
       )
        
        # =============== regular-grid branch (patch tokens) ===============
        self.patch_embed = nn.Conv2d(
            hidden_dim, hidden_dim,
            kernel_size=patch_size, stride=patch_size
       )
        
        self.patch_blocks = nn.ModuleList([
            HyperbolicPatchBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                hyp_dim=hyp_dim,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
           )
            for _ in range(num_layers)
        ])
        
        self.unpatch = nn.ConvTranspose2d(
            hidden_dim, hidden_dim,
            kernel_size=patch_size, stride=patch_size
       )
        
        # =============== irregular-mesh branch (latent-set tokens) ===============
        self.latents = nn.Parameter(torch.randn(1, num_latents, hidden_dim) * 0.02)
        
        self.encoder = CrossAttentionBlock(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            hyp_dim=hyp_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
       )
        
        self.processor = nn.ModuleList([
            PerceiverBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                hyp_dim=hyp_dim,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
           )
            for _ in range(num_layers)
        ])
        
        self.decoder = CrossAttentionBlock(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            hyp_dim=hyp_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
       )
        
        # Projection
        self.projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_channels),
       )
    
    def _is_regular_grid(self, x: torch.Tensor) -> bool:
        """Detect whether the input is a regular grid"""
        B, N, D = x.shape
        H = W = int(math.sqrt(N))
        
        # Condition 1: N must be a perfect square
        if H * W != N:
            return False
        
        # Condition 2: H*W must be divisible by patch_size
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            return False
        
        # Condition 3: check whether coordinates are uniformly distributed (simple heuristic).
        # Use the first sample and inspect the standard deviation of coordinate differences.
        coords = x[0]  # (N, D)
        coords_2d = coords.view(H, W, D)
        
        # Check y-coordinate differences along rows.
        dy = coords_2d[1:, :, 1] - coords_2d[:-1, :, 1]
        # Check x-coordinate differences along columns.
        dx = coords_2d[:, 1:, 0] - coords_2d[:, :-1, 0]
        
        # treat as regular if the std is small enough
        eps = 1e-4
        return dy.std() < eps and dx.std() < eps
    
    def _forward_regular(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Regular-grid path: patch tokenization."""
        B, N, C = h.shape
        H = W = int(math.sqrt(N))
        
        # Reshape to 2D
        h = h.permute(0, 2, 1).view(B, self.hidden_dim, H, W)
        
        # Patchify
        patches = self.patch_embed(h)
        nH, nW = patches.shape[2], patches.shape[3]
        patches = patches.view(B, self.hidden_dim, -1).permute(0, 2, 1)
        
        # Patch-token blocks
        for block in self.patch_blocks:
            patches = block(patches)
        
        # Unpatchify
        patches = patches.permute(0, 2, 1).view(B, self.hidden_dim, nH, nW)
        h = self.unpatch(patches)
        
        # Reshape back
        h = h.view(B, self.hidden_dim, N).permute(0, 2, 1)
        
        return h
    
    def _forward_irregular(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Irregular-mesh path: latent-set tokenization."""
        B = h.shape[0]
        
        # Latent Patches
        latents = self.latents.expand(B, -1, -1)
        
        # Encode: points -> latent
        latents = self.encoder(latents, h)
        
        # Process: hyperbolic self-attention among latents.
        for block in self.processor:
            latents = block(latents)
        
        # Decode: latent -> points
        h = self.decoder(h, latents)
        
        return h
    
    def forward(
        self,
        x: torch.Tensor,
        fx: Optional[torch.Tensor] = None,
        T: Optional[torch.Tensor] = None,
   ) -> torch.Tensor:
        """
        Args:
            x: coordinates (B, N, space_dim)
            fx: input features (B, N, in_channels) or None
        """
        B, N, _ = x.shape
        
        # 1. Lifting
        if fx is not None:
            inp = torch.cat([x, fx], dim=-1)
            h = self.lifting(inp)
        else:
            h = self.lifting_no_pos(x)
        
        # 2. Choose the path based on grid type
        if self._is_regular_grid(x):
            h = self._forward_regular(x, h)
        else:
            h = self._forward_irregular(x, h)
        
        # 3. Projection
        out = self.projection(h)
        
        return out
