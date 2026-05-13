"""
Hyperbolic Neural Operator (HNO) - Patch Tokenization

Implements a patchified neural operator where token interactions are scored by
Lorentz-hyperbolic (geodesic) distances instead of Euclidean dot products.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from pdebench.utils.clamp_stats import record_acosh_clamp


class LorentzManifold:
    """Lorentz manifold utilities"""
    
    @staticmethod
    def to_lorentz(x: torch.Tensor) -> torch.Tensor:
        """Euclidean -> Lorentz: x -> (sqrt(1 + ||x||^2), x)"""
        norm_sq = (x * x).sum(dim=-1, keepdim=True)
        time = torch.sqrt(1 + norm_sq)
        return torch.cat([time, x], dim=-1)
    
    @staticmethod
    def pairwise_lorentz_distance(x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        """
        Compute pairwise Lorentz distances
        x: (B, N, d+1), y: (B, M, d+1) or None
        Returns: (B, N, M) distance matrix
        """
        if y is None:
            y = x
        inner = -torch.einsum('bni,bmi->bnm', x[..., :1], y[..., :1]) + \
                torch.einsum('bni,bmi->bnm', x[..., 1:], y[..., 1:])
        record_acosh_clamp((-inner), eps=1e-6, tag="hno_pairwise")
        return torch.acosh((-inner).clamp(min=1.0 + 1e-6))


class RelativePositionBias2D(nn.Module):
    """2D relative position bias (supports rectangular grids)."""
    
    # max_h/max_w must cover the patch-grid height/width (pH/pW).
    def __init__(self, num_heads: int, max_h: int = 256, max_w: int = 256):
        super().__init__()
        self.num_heads = num_heads
        self.max_h = max_h
        self.max_w = max_w
        
        # Relative offset ranges:
        # H: [-(max_h - 1), max_h - 1], W: [-(max_w - 1), max_w - 1].
        num_relative_distance = (2 * max_h - 1) * (2 * max_w - 1)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(num_relative_distance, num_heads)
       )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
    
    def _get_relative_position_index(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Compute relative-position indices dynamically(rectangular-grid aware)"""
        coords_h = torch.arange(H, device=device)
        coords_w = torch.arange(W, device=device)
        try:
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # (2, H, W)
        except TypeError:
            coords = torch.stack(torch.meshgrid(coords_h, coords_w))  # (2, H, W)
        coords_flatten = coords.view(2, -1)  # (2, H*W)
        
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, H*W, H*W)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (H*W, H*W, 2)
        relative_coords[:, :, 0] += self.max_h - 1
        relative_coords[:, :, 1] += self.max_w - 1
        relative_coords[:, :, 0] *= 2 * self.max_w - 1
        return relative_coords.sum(-1)  # (H*W, H*W)
    
    def forward(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Returns: (num_heads, H*W, H*W)"""
        relative_position_index = self._get_relative_position_index(H, W, device)
        table_len = self.relative_position_bias_table.shape[0]
        relative_position_index = relative_position_index.clamp(min=0, max=table_len - 1)
        N = H * W
        bias = self.relative_position_bias_table[relative_position_index.view(-1)].view(
            N, N, -1
       )  # (H*W, H*W, num_heads)
        return bias.permute(2, 0, 1).contiguous()  # (num_heads, H*W, H*W)


class LocalFeatureExtractor(nn.Module):
    """Local feature extraction(DepthwiseConv)"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        x: (B, N, C)
        Returns: (B, N, C)
        """
        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.view(B, C, N).permute(0, 2, 1)
        x = self.norm(x)
        return x


class HyperbolicPatchAttentionV2(nn.Module):
    """
    Hyperbolic patch attention v2
    
    Design choices:
    1. Separate Q/K projections
    2. 2D relative position bias
    3. temperature initialized to 1.0
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        hyp_dim: int = 16,
        dropout: float = 0.0,
        max_h: int = 256,
        max_w: int = 256,
   ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.hyp_dim = hyp_dim
        self.head_dim = hidden_dim // num_heads
        
        # Project Q/K features into hyperbolic coordinates.
        self.to_q = nn.Linear(hidden_dim, num_heads * hyp_dim)
        self.to_k = nn.Linear(hidden_dim, num_heads * hyp_dim)
        
        # Value projection
        self.to_v = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.to_out = nn.Linear(hidden_dim, hidden_dim)
        
        # 2D relative-position bias for rectangular patch grids.
        self.rel_pos_bias = RelativePositionBias2D(num_heads, max_h=max_h, max_w=max_w)
        
        # Learnable temperature initialized to 1.0.
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, pH: int, pW: int) -> torch.Tensor:
        """
        x: (B, P, C) - P patch tokens
        Returns: (B, P, C)
        """
        B, P, C = x.shape
        H = self.num_heads
        
        # 1. Project Q/K features into hyperbolic coordinates.
        q = self.to_q(x).view(B, P, H, self.hyp_dim).permute(0, 2, 1, 3)  # (B, H, P, hyp_dim)
        k = self.to_k(x).view(B, P, H, self.hyp_dim).permute(0, 2, 1, 3)
        
        # Map to the Lorentz manifold
        q_lorentz = LorentzManifold.to_lorentz(q)  # (B, H, P, hyp_dim + 1)
        k_lorentz = LorentzManifold.to_lorentz(k)
        
        # 2. Compute the hyperbolic distance matrix (vectorized over heads).
        # Lorentz inner product: -t1*t2 + x1*x2, where t is the first component.
        # q_lorentz: (B, H, P, hyp_dim+1), k_lorentz: (B, H, P, hyp_dim+1)
        inner = -torch.einsum('bhpi,bhqi->bhpq', q_lorentz[..., :1], k_lorentz[..., :1]) + \
                torch.einsum('bhpi,bhqi->bhpq', q_lorentz[..., 1:], k_lorentz[..., 1:])
        if record_acosh_clamp is not None:
            record_acosh_clamp((-inner), eps=1e-6, tag="hpatchv2_attn")
        dist = torch.acosh((-inner).clamp(min=1.0 + 1e-6))  # (B, H, P, P)
        
        # 3. Add relative-position bias(rectangular-grid support)
        if pH * pW != P:
            raise RuntimeError(f"[HNO] P={P} must equal pH*pW={pH}*{pW}={pH*pW}")
        rel_pos = self.rel_pos_bias(pH, pW, x.device)  # (H, P, P)
        
        # 4. Distance -> attention weights
        temp = self.temperature.clamp(0.1, 3.0)  # allow a wider temperature range
        # attention = softmax(-dist/temp + rel_pos)
        attn_logits = -dist / temp + rel_pos.unsqueeze(0)
        attn = F.softmax(attn_logits.float(), dim=-1).to(x.dtype)  # (B, H, P, P)
        attn = self.dropout(attn)
        
        # 5. Aggregate values
        v = self.to_v(x).view(B, P, H, self.head_dim).permute(0, 2, 1, 3)  # (B, H, P, head_dim)
        out = attn @ v  # (B, H, P, head_dim)
        out = out.permute(0, 2, 1, 3).reshape(B, P, C)  # (B, P, C)
        
        # 6. Output projection
        out = self.to_out(out)
        
        return out


class HyperbolicPatchBlockV2(nn.Module):
    """HNO block = hyperbolic attention + local mixing + FFN."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        hyp_dim: int = 16,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        max_h: int = 256,
        max_w: int = 256,
   ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = HyperbolicPatchAttentionV2(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            hyp_dim=hyp_dim,
            dropout=dropout,
            max_h=max_h,
            max_w=max_w,
       )
        
        # Local feature extraction
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
    
    def forward(self, x: torch.Tensor, pH: int, pW: int) -> torch.Tensor:
        """x: (B, P, C), P=pH*pW"""
        # Hyperbolic attention
        x = x + self.attn(self.norm1(x), pH, pW)
        # Local features
        x = x + self.local_feat(x, pH, pW)
        # FFN
        x = x + self.ffn(self.norm2(x))
        return x


class HyperbolicPatchNO(nn.Module):
    """
    HNO with patch tokenization (grid/structured-mesh setting).
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
        use_temporal_conv: bool = False,
        rel_pos_max_h: Optional[int] = None,
        rel_pos_max_w: Optional[int] = None,
   ):
        super().__init__()
        
        self.space_dim = space_dim
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.grid_h = grid_h
        self.grid_w = grid_w

        # Optional: temporal 1D conv along the input-channel axis (e.g., NS where
        # in_channels == T_in). This is disabled by default and can be enabled
        # explicitly by callers.
        self.use_temporal_conv = bool(use_temporal_conv)
        if self.use_temporal_conv:
            # fx: (B, N, T_in) -> (B, N, hidden_dim)
            # treat the T_in sequence at each spatial point as a one-channel 1D signal
            t_width = max(16, hidden_dim // 4)
            self.temporal_conv = nn.Sequential(
                nn.Conv1d(1, t_width, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(t_width, t_width, kernel_size=3, padding=1),
                nn.GELU(),
           )
            self.temporal_proj = nn.Linear(t_width * in_channels, hidden_dim)
            self.coord_proj = nn.Linear(space_dim, hidden_dim)
            self.fusion = nn.Sequential(
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
           )
        
        # Lifting
        if not self.use_temporal_conv:
            self.lifting = nn.Sequential(
                nn.Linear(in_channels + space_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
           )
        
        # Patch embedding via Conv2d.
        self.patch_embed = nn.Conv2d(
            hidden_dim, hidden_dim, 
            kernel_size=patch_size, stride=patch_size
       )
        
        # HNO blocks (support rectangular patch grids)
        max_h = int(rel_pos_max_h) if rel_pos_max_h is not None else ((grid_h // patch_size + 1) if grid_h else 256)
        max_w = int(rel_pos_max_w) if rel_pos_max_w is not None else ((grid_w // patch_size + 1) if grid_w else 256)
        self.blocks = nn.ModuleList([
            HyperbolicPatchBlockV2(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                hyp_dim=hyp_dim,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                max_h=max_h,
                max_w=max_w,
           )
            for _ in range(num_layers)
        ])
        
        # Unpatch (use ConvTranspose2d)
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
        """
        Args:
            x: coordinates (B, N, space_dim)
            fx: input features (B, N, in_channels)
        
        Returns:
            out: (B, N, out_channels)
        """
        B, N, _ = x.shape
        
        # Grid size:prefer grid_h/grid_w supplied at construction time(supports rectangular cases such as Airfoil 221x51)
        if self.grid_h is not None and self.grid_w is not None:
            H, W = self.grid_h, self.grid_w
        else:
            H = W = int(math.sqrt(N))
        assert H * W == N, f"N={N} must equal H*W={H}*{W}={H*W}"
        
        # 1. Lifting
        if self.use_temporal_conv:
            # fx expected: (B, N, T_in=self.in_channels)
            if fx.shape[-1] != self.in_channels:
                raise RuntimeError(
                    f"[HNO] use_temporal_conv=True expects fx last dim={self.in_channels}, got {fx.shape[-1]}"
               )
            b, n, t = fx.shape
            seq = fx.reshape(b * n, 1, t)  # (B*N, 1, T)
            feat = self.temporal_conv(seq)  # (B*N, t_width, T)
            feat = feat.reshape(b * n, -1)  # (B*N, t_width*T)
            fx_emb = self.temporal_proj(feat).reshape(b, n, self.hidden_dim)
            x_emb = self.coord_proj(x)
            h = self.fusion(fx_emb + x_emb)  # (B, N, hidden_dim)
        else:
            # For some datasets(for example Pipe/Elasticity),the training script passes fx=None and fun_dim=0
            # treat fx=None as an empty feature tensor(in_channels==0),to keep the interface compatible
            if fx is None:
                if self.in_channels != 0:
                    raise RuntimeError(f"[HNO] fx is None but in_channels={self.in_channels} != 0")
                fx = x.new_zeros((B, N, 0))
            inp = torch.cat([x, fx], dim=-1)
            h = self.lifting(inp)  # (B, N, hidden_dim)
        
        # 2. Reshape to 2D grid
        h = h.permute(0, 2, 1).view(B, self.hidden_dim, H, W)  # (B, C, H, W)
        
        # 3. Padding(reflect),make the grid divisible by patch_size(especially important for rectangular grids)
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            h = F.pad(h, (0, pad_w, 0, pad_h), mode='reflect')
        H_pad, W_pad = H + pad_h, W + pad_w
        
        # 4. Patchify with Conv2d.
        patches = self.patch_embed(h)  # (B, C, pH, pW)
        pH, pW = patches.shape[2], patches.shape[3]
        patches = patches.view(B, self.hidden_dim, -1).permute(0, 2, 1)  # (B, P, C)
        
        # 5. HNO blocks
        for block in self.blocks:
            patches = block(patches, pH, pW)
        
        # 6. Unpatchify (use ConvTranspose2d)
        patches = patches.permute(0, 2, 1).view(B, self.hidden_dim, pH, pW)  # (B, C, pH, pW)
        h = self.unpatch(patches)  # (B, C, H_pad, W_pad)
        
        # 7. Crop back to the original size
        h = h[:, :, :H, :W]
        
        # 8. Reshape back
        h = h.reshape(B, self.hidden_dim, N).permute(0, 2, 1)  # (B, N, C)
        
        # 9. Projection
        out = self.projection(h)  # (B, N, out_channels)
        
        return out


HNO = HyperbolicPatchNO


def build_hno(
    *,
    space_dim: int = 2,
    fun_dim: int = 1,
    out_dim: int = 1,
    hidden_dim: int = 96,
    num_layers: int = 4,
    num_heads: int = 4,
    patch_size: int = 5,
    hyp_dim: int = 16,
    mlp_ratio: float = 2.5,
    dropout: float = 0.0,
    H: Optional[int] = None,
    W: Optional[int] = None,
    use_temporal_conv: bool = False,
) -> HNO:
    """
    Convenience constructor used by the PDEBench training scripts.
    """
    return HNO(
        space_dim=space_dim,
        in_channels=fun_dim,
        out_channels=out_dim,
        hidden_dim=hidden_dim,
        patch_size=patch_size,
        num_heads=num_heads,
        num_layers=num_layers,
        hyp_dim=hyp_dim,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        grid_h=H,
        grid_w=W,
        use_temporal_conv=use_temporal_conv,
   )
