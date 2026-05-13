"""
Hyperbolic Neural Operator (HNO) - Spatiotemporal Patch Tokenization

Variant with stronger local convolutional processing and time conditioning,
used for spatiotemporal PDEBench settings (e.g., Plasticity).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from pdebench.hno.embedding import timestep_embedding
from pdebench.utils.clamp_stats import record_acosh_clamp


class LorentzManifold:
    """Lorentz manifold utilities"""
    
    @staticmethod
    def to_lorentz(x: torch.Tensor) -> torch.Tensor:
        """Euclidean -> Lorentz: x -> (sqrt(1 + ||x||^2), x)"""
        norm_sq = (x * x).sum(dim=-1, keepdim=True)
        time = torch.sqrt(1 + norm_sq)
        return torch.cat([time, x], dim=-1)


class RelativePositionBias2D(nn.Module):
    """2D relative position bias (supports rectangular grids)."""
    
    def __init__(self, num_heads: int, max_h: int = 50, max_w: int = 20):
        super().__init__()
        self.num_heads = num_heads
        self.max_h = max_h
        self.max_w = max_w
        
        num_relative_distance = (2 * max_h - 1) * (2 * max_w - 1)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(num_relative_distance, num_heads)
       )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
    
    def _get_relative_position_index(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Compute relative-position indices dynamically"""
        coords_h = torch.arange(H, device=device)
        coords_w = torch.arange(W, device=device)
        try:
            coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # (2, H, W)
        except TypeError:
            coords = torch.stack(torch.meshgrid(coords_h, coords_w))  # (2, H, W)
        coords_flatten = coords.view(2, -1)  # (2, H*W)
        
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, P, P)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (P, P, 2)
        
        relative_coords[:, :, 0] += self.max_h - 1
        relative_coords[:, :, 1] += self.max_w - 1
        relative_coords[:, :, 0] *= 2 * self.max_w - 1
        
        relative_position_index = relative_coords.sum(-1)  # (P, P)
        return relative_position_index
    
    def forward(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Return relative-position bias with shape (num_heads, P, P)."""
        relative_position_index = self._get_relative_position_index(H, W, device)
        
        relative_position_index = relative_position_index.clamp(
            0, self.relative_position_bias_table.size(0) - 1
       )
        
        relative_position_bias = self.relative_position_bias_table[
            relative_position_index.view(-1)
        ].view(H * W, H * W, -1)
        
        return relative_position_bias.permute(2, 0, 1).contiguous()


class EnhancedLocalFeature(nn.Module):
    """
    Enhanced local feature extraction (ConvNeXt-style).
    
    Structure:
    1. DWConv 7x7(large receptive field)
    2. LayerNorm
    3. Linear expansion
    4. GELU
    5. Linear projection
    
    stronger than a simple 3x3 depthwise convolution for local spatiotemporal details
    """
    
    def __init__(self, dim: int, expansion: float = 2.0):
        super().__init__()
        hidden_dim = int(dim * expansion)
        
        # large-kernel depthwise convolution
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        
        # Pointwise expansion/projection (FFN-style)
        self.pwconv1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden_dim, dim)
        
        # learnable scaling
        self.gamma = nn.Parameter(torch.ones(dim) * 0.1)
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        x: (B, N, C)
        Returns: (B, N, C)
        """
        B, N, C = x.shape
        
        # Reshape to 2D
        h = x.permute(0, 2, 1).view(B, C, H, W)
        h = self.dwconv(h)
        h = h.view(B, C, N).permute(0, 2, 1)  # (B, N, C)
        
        # Norm + FFN
        h = self.norm(h)
        h = self.pwconv1(h)
        h = self.act(h)
        h = self.pwconv2(h)
        
        return self.gamma * h


class LocalConvBlock(nn.Module):
    """
    Local convolution block
    
    apply convolutional preprocessing before attention
    """
    
    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        h = x.permute(0, 2, 1).view(B, C, H, W)
        h = self.conv(h)
        h = h.view(B, C, N).permute(0, 2, 1)
        h = self.norm(h)
        h = self.act(h)
        return h


class HyperbolicPatchAttentionV3(nn.Module):
    """
    Hyperbolic patch attention v3
    
    Design choices:
    1. LocalConv preprocessing before attention
    2. Separate Q/K projections
    3. 2D relative position bias
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        hyp_dim: int = 16,
        dropout: float = 0.0,
        max_h: int = 50,
        max_w: int = 20,
        use_local_conv: bool = True,  # whether to apply convolution before attention
   ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.hyp_dim = hyp_dim
        self.head_dim = hidden_dim // num_heads
        self.use_local_conv = use_local_conv
        
        # Optional local-convolution preprocessing
        if use_local_conv:
            self.local_conv = LocalConvBlock(hidden_dim, kernel_size=3)
        
        # Separate Q/K projections
        self.to_q = nn.Linear(hidden_dim, num_heads * hyp_dim)
        self.to_k = nn.Linear(hidden_dim, num_heads * hyp_dim)
        self.to_v = nn.Linear(hidden_dim, hidden_dim)
        self.to_out = nn.Linear(hidden_dim, hidden_dim)
        
        # Relative-position bias
        self.rel_pos_bias = RelativePositionBias2D(num_heads, max_h=max_h, max_w=max_w)
        
        # Learnable temperature
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, pH: int, pW: int) -> torch.Tensor:
        B, P, C = x.shape
        H = self.num_heads
        
        # Optional local-convolution preprocessing
        if self.use_local_conv:
            x_conv = self.local_conv(x, pH, pW)
        else:
            x_conv = x
        
        # Q/K projection
        q = self.to_q(x_conv).view(B, P, H, self.hyp_dim).permute(0, 2, 1, 3)
        k = self.to_k(x_conv).view(B, P, H, self.hyp_dim).permute(0, 2, 1, 3)
        
        # Lorentz lift
        q_lorentz = LorentzManifold.to_lorentz(q)
        k_lorentz = LorentzManifold.to_lorentz(k)
        
        # Hyperbolic distance
        inner = -torch.einsum('bhpi,bhqi->bhpq', q_lorentz[..., :1], k_lorentz[..., :1]) + \
                torch.einsum('bhpi,bhqi->bhpq', q_lorentz[..., 1:], k_lorentz[..., 1:])
        record_acosh_clamp((-inner), eps=1e-4, tag="hno_patch_time_attn")
        dist = torch.acosh((-inner).clamp(min=1.0 + 1e-4))
        
        # Relative-position bias
        rel_pos = self.rel_pos_bias(pH, pW, x.device)
        
        # attention weights
        temp = self.temperature.clamp(0.1, 3.0)
        attn_logits = -dist / temp + rel_pos.unsqueeze(0)
        attn = F.softmax(attn_logits.float(), dim=-1).to(x.dtype)
        attn = self.dropout(attn)
        
        # Aggregate values
        v = self.to_v(x).view(B, P, H, self.head_dim).permute(0, 2, 1, 3)
        out = attn @ v
        out = out.permute(0, 2, 1, 3).reshape(B, P, C)
        out = self.to_out(out)
        
        return out


class HyperbolicPatchBlockV3(nn.Module):
    """
    HNO block = hyperbolic attention + stronger local mixing + FFN
    
    Design choice: use EnhancedLocalFeature instead of a simple depthwise convolution.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        hyp_dim: int = 16,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        max_h: int = 50,
        max_w: int = 20,
        local_expansion: float = 2.0,  # Expansion ratio for local features
        use_attn_conv: bool = True,  # Whether to apply convolution before attention
   ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = HyperbolicPatchAttentionV3(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            hyp_dim=hyp_dim,
            dropout=dropout,
            max_h=max_h,
            max_w=max_w,
            use_local_conv=use_attn_conv,
       )
        
        # Enhanced local features
        self.local_feat = EnhancedLocalFeature(hidden_dim, expansion=local_expansion)
        
        self.norm2 = nn.LayerNorm(hidden_dim)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
       )
    
    def forward(self, x: torch.Tensor, pH: int, pW: int) -> torch.Tensor:
        # Hyperbolic attention
        x = x + self.attn(self.norm1(x), pH, pW)
        # Enhanced local features
        x = x + self.local_feat(x, pH, pW)
        # FFN
        x = x + self.ffn(self.norm2(x))
        return x


class HyperbolicPatchNOv3(nn.Module):
    """
    Hyperbolic Patch Neural Operator v3(enhanced local-convolution variant)
    
    Design choices:
    1. Use EnhancedLocalFeature(7x7 DWConv + FFN)
    2. LocalConv preprocessing before attention
    3. stronger local modeling capacity
    """
    
    def __init__(
        self,
        space_dim: int = 2,
        in_channels: int = 1,
        out_channels: int = 1,
        hidden_dim: int = 128,
        patch_size: int = 5,
        num_heads: int = 8,
        num_layers: int = 6,
        hyp_dim: int = 16,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        grid_h: int = None,
        grid_w: int = None,
        local_expansion: float = 2.0,  # Expansion ratio for local features
        use_attn_conv: bool = True,  # Whether to apply convolution before attention
        time_embed: str = "scalar",  # scalar | timestep | none
   ):
        super().__init__()
        
        self.space_dim = space_dim
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Compute the maximum patch-grid size
        max_h = (grid_h // patch_size + 1) if grid_h else 50
        max_w = (grid_w // patch_size + 1) if grid_w else 20

        # Time conditioning (Plasticity is 2D+Time).
        # Default keeps the previous behavior (MLP on scalar T).
        # Optional: use sinusoidal timestep embedding.
        # - scalar: MLP(1 -> C) on raw scalar T
        # - timestep: timestep_embedding(T, C) then MLP(C -> C)
        # - none: disable time conditioning
        self.time_embed = (time_embed or "scalar").lower()
        if self.time_embed == "none":
            self.time_fc = None
        elif self.time_embed == "timestep":
            self.time_fc = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
           )
        else:
            # scalar
            self.time_fc = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
           )
        
        # Lifting
        self.lifting = nn.Sequential(
            nn.Linear(in_channels + space_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
       )
        self.lifting_coord_only = nn.Sequential(
            nn.Linear(space_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
       )
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            hidden_dim, hidden_dim, 
            kernel_size=patch_size, stride=patch_size
       )
        
        # HNO blocks
        self.blocks = nn.ModuleList([
            HyperbolicPatchBlockV3(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                hyp_dim=hyp_dim,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                max_h=max_h,
                max_w=max_w,
                local_expansion=local_expansion,
                use_attn_conv=use_attn_conv,
           )
            for _ in range(num_layers)
        ])
        
        # Unpatch
        self.unpatch = nn.ConvTranspose2d(
            hidden_dim, hidden_dim,
            kernel_size=patch_size, stride=patch_size
       )
        
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
        fx: torch.Tensor,
        T: Optional[torch.Tensor] = None,
   ) -> torch.Tensor:
        # Normalize input layout
        if x.dim() == 4:
            B, H, W, _ = x.shape
            x = x.reshape(B, H * W, -1)
            fx = fx.reshape(B, H * W, -1)
            reshape_back = True
        else:
            B, N, _ = x.shape
            reshape_back = False
            
            if self.grid_h is not None and self.grid_w is not None:
                H, W = self.grid_h, self.grid_w
            else:
                H = W = int(math.sqrt(N))
                if H * W != N:
                    raise ValueError(
                        f"Non-square token count N={N}; pass grid_h/grid_w when constructing the model."
                   )
            
            assert H * W == N, f"N={N} must equal H*W={H}*{W}={H*W}"
        
        N = H * W
        
        # 1. Lifting
        if fx is None:
            h = self.lifting_coord_only(x)
        else:
            inp = torch.cat([x, fx], dim=-1)
            h = self.lifting(inp)

        # 1.1 Time conditioning
        if T is not None and self.time_fc is not None:
            if self.time_embed == "timestep":
                # T expected as (B,1) or (B,)
                t = T.squeeze(-1) if T.ndim == 2 else T  # (B,)
                t_emb = timestep_embedding(t, self.hidden_dim)  # (B,C)
                h = h + self.time_fc(t_emb).unsqueeze(1)  # (B,1,C) broadcast over N
            else:
                # scalar MLP on raw T
                T_in = T.unsqueeze(-1) if T.ndim == 1 else T
                h = h + self.time_fc(T_in).unsqueeze(1)  # (B,1,C) broadcast over N
        
        # 2. Reshape to 2D
        h = h.permute(0, 2, 1).view(B, self.hidden_dim, H, W)
        
        # 3. Padding(reflect)
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            h = F.pad(h, (0, pad_w, 0, pad_h), mode='reflect')
        H_pad, W_pad = H + pad_h, W + pad_w
        
        # 4. Patch embedding
        h = self.patch_embed(h)  # (B, C, pH, pW)
        pH, pW = h.shape[2], h.shape[3]
        h = h.view(B, self.hidden_dim, pH * pW).permute(0, 2, 1)  # (B, P, C)
        
        # 5. HNO blocks
        for block in self.blocks:
            h = block(h, pH, pW)
        
        # 6. Unpatch
        h = h.permute(0, 2, 1).view(B, self.hidden_dim, pH, pW)
        h = self.unpatch(h)  # (B, C, H_pad, W_pad)
        
        # 7. Crop back to the original size
        h = h[:, :, :H, :W]
        
        # 8. Reshape
        h = h.reshape(B, self.hidden_dim, N).permute(0, 2, 1)  # (B, N, C)
        
        # 9. Projection
        out = self.projection(h)
        
        if reshape_back:
            out = out.view(B, H, W, -1)
        
        return out


HNO = HyperbolicPatchNOv3


def build_hno(
    *,
    space_dim: int = 2,
    fun_dim: int = 1,
    out_dim: int = 4,
    hidden_dim: int = 128,
    num_layers: int = 5,
    num_heads: int = 8,
    patch_size: int = 3,
    hyp_dim: int = 16,
    mlp_ratio: float = 2.0,
    dropout: float = 0.0,
    H: Optional[int] = None,
    W: Optional[int] = None,
    local_expansion: float = 2.0,
    use_attn_conv: bool = True,
    time_embed: str = "scalar",
) -> HNO:
    """
    Convenience constructor used by the PDEBench Plasticity training script.
    """
    return HNO(
        space_dim=space_dim,
        in_channels=int(fun_dim),
        out_channels=int(out_dim),
        hidden_dim=hidden_dim,
        patch_size=patch_size,
        num_heads=num_heads,
        num_layers=num_layers,
        hyp_dim=hyp_dim,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        grid_h=H,
        grid_w=W,
        local_expansion=local_expansion,
        use_attn_conv=use_attn_conv,
        time_embed=time_embed,
   )
