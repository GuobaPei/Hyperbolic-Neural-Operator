"""Latent-set Hyperbolic Neural Operator."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pdebench.hno.attention import SharedWeightBlock
from pdebench.hno.lorentz import LorentzManifold


class HyperbolicPerceiverNO(nn.Module):
    """
    HNO with latent-set tokenization (shared cross-attention weights).

    High-level structure:
    1) Lift inputs to `hidden_dim`
    2) Compute cross-attention weights once (MxN) and reuse across layers
    3) Alternate latent self-attention (hyperbolic distance) and decode back to points
    4) Project to outputs
    """
    
    def __init__(
        self,
        space_dim: int = 2,
        in_channels: int = 1,
        out_channels: int = 1,
        hidden_dim: int = 96,
        num_latents: int = 64,        # M: number of latent patches
        num_heads: int = 4,
        num_layers: int = 4,          # number of symmetric interaction layers
        hyp_dim: int = 16,
        mlp_ratio: float = 2.5,
        dropout: float = 0.0,
   ):
        super().__init__()
        
        self.space_dim = space_dim
        self.hidden_dim = hidden_dim
        self.num_latents = num_latents
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hyp_dim = hyp_dim
        
        # Lifting(supports fx=None)
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
        
        # Learnable latent patches
        self.latents = nn.Parameter(torch.randn(1, num_latents, hidden_dim) * 0.02)
        
        # === Shared attention-weight module ===
        # Compute once and reuse across all layers.
        self.to_q_shared = nn.Linear(hidden_dim, num_heads * hyp_dim)  # for latents
        self.to_k_shared = nn.Linear(hidden_dim, num_heads * hyp_dim)  # for points
        
        # Orthogonal initialization
        nn.init.orthogonal_(self.to_q_shared.weight)
        nn.init.orthogonal_(self.to_k_shared.weight)
        
        # Temperature initialized to 0.5
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) * 0.5)
        
        # N shared-weight interaction blocks
        self.blocks = nn.ModuleList([
            SharedWeightBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                hyp_dim=hyp_dim,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
           )
            for _ in range(num_layers)
        ])
        
        # Projection
        self.projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_channels),
       )
    
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
        
        Returns:
            out: (B, N, out_channels)
        """
        B, N, _ = x.shape
        M = self.num_latents
        H = self.num_heads
        
        # 1. Lifting
        if fx is not None:
            inp = torch.cat([x, fx], dim=-1)
            h = self.lifting(inp)  # (B, N, hidden_dim)
        else:
            h = self.lifting_no_pos(x)  # (B, N, hidden_dim)
        
        # 2. Initialize latent patches
        latents = self.latents.expand(B, -1, -1)  # (B, M, hidden_dim)
        
        # 3. Compute shared attention weights once.
        q = self.to_q_shared(latents).view(B, M, H, self.hyp_dim).permute(0, 2, 1, 3)  # (B, H, M, hyp_dim)
        k = self.to_k_shared(h).view(B, N, H, self.hyp_dim).permute(0, 2, 1, 3)        # (B, H, N, hyp_dim)
        
        # Norm clipping for numerical stability.
        q = LorentzManifold.clip_norm(q)
        k = LorentzManifold.clip_norm(k)
        
        q_lorentz = LorentzManifold.to_lorentz(q)
        k_lorentz = LorentzManifold.to_lorentz(k)
        
        # vectorizedHyperbolic distance(without Python loops)
        dist = LorentzManifold.pairwise_lorentz_distance_multihead(q_lorentz, k_lorentz)  # (B, H, M, N)
        
        temp = self.temperature.clamp(0.1, 3.0)
        attn_weights = F.softmax((-dist / temp).float(), dim=-1).to(h.dtype)  # (B, H, M, N), shared by all layers
        
        # 4. Symmetric interactions: each layer uses the same attn_weights.
        for block in self.blocks:
            latents, h = block(latents, h, attn_weights)
        
        # 5. Projection
        out = self.projection(h)  # (B, N, out_channels)
        
        return out


# ============================================================================
# (Optional) Adaptive variant: patchify for grids, latent-set for irregular
# ============================================================================

HNO = HyperbolicPerceiverNO


def build_hno(
    *,
    space_dim: int = 2,
    fun_dim: int = 0,
    out_dim: int = 1,
    hidden_dim: int = 384,
    num_layers: int = 6,
    num_heads: int = 8,
    num_latents: int = 96,
    hyp_dim: int = 16,
    mlp_ratio: float = 2.5,
    dropout: float = 0.0,
) -> HNO:
    """
    Convenience constructor used by the PDEBench Elasticity training script.
    """
    return HNO(
        space_dim=space_dim,
        in_channels=int(fun_dim),
        out_channels=int(out_dim),
        hidden_dim=hidden_dim,
        num_latents=num_latents,
        num_heads=num_heads,
        num_layers=num_layers,
        hyp_dim=hyp_dim,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
   )
