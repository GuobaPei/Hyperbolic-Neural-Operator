"""
Hyperbolic Neural Operator (HNO) - Latent-Set Tokenization

Perceiver-style latent tokens serve as an interaction core (M tokens), enabling
O(N*M + M^2) complexity for point-cloud / irregular-mesh settings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from pdebench.utils.clamp_stats import record_acosh_clamp


class LorentzManifold:
    """
    Lorentz manifold utilities
    
    Numerical-stability choices:
    1. force critical computations to float32(avoid NaNs under fp16)
    2. clamp acosh input to 1 + 1e-4(1e-6 may still yield unstable gradients)
    3. provide optional norm clipping for Q/K vectors
    """
    
    # Numerical-stability constants
    ACOSH_EPS = 1e-4  # minimum offset for acosh input(safer than 1e-6)
    MAX_NORM = 10.0   # maximum norm for Q/K vectors(soft-clipping scale; larger values preserve hierarchy)
    
    @staticmethod
    def to_lorentz(x: torch.Tensor) -> torch.Tensor:
        """Euclidean -> Lorentz: x -> (sqrt(1 + ||x||^2), x)"""
        # Force float32 computation
        orig_dtype = x.dtype
        x = x.float()
        norm_sq = (x * x).sum(dim=-1, keepdim=True)
        time = torch.sqrt(1 + norm_sq)
        result = torch.cat([time, x], dim=-1)
        return result.to(orig_dtype)
    
    @staticmethod
    def clip_norm(x: torch.Tensor, max_norm: float = None, mode: str = 'none') -> torch.Tensor:
        """
        Apply optional norm control to Q/K vectors
        
        Args:
            x: input tensor
            max_norm: maximum norm(used by soft/hard modes)
            mode: 
                - 'none': no clipping; keep the original norm(recommended; lets the model learn freely)
                - 'soft': smooth compression with tanh
                - 'hard': hard clipping(not recommended; may lose information)
        """
        if mode == 'none':
            # No clipping: let the model learn the hierarchy freely.
            return x
        
        if max_norm is None:
            max_norm = LorentzManifold.MAX_NORM
        
        norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        
        if mode == 'soft':
            # Soft clipping with tanh
            target_norm = torch.tanh(norm / max_norm) * max_norm
            scale = target_norm / norm
        else:  # 'hard'
            # Hard clipping: truncate vectors whose norm exceeds max_norm.
            scale = torch.clamp(max_norm / norm, max=1.0)
        
        return x * scale
    
    @staticmethod
    def pairwise_lorentz_distance(x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        """
        Compute pairwise Lorentz distances
        x: (B, N, d+1), y: (B, M, d+1) or None
        Returns: (B, N, M) distance matrix
        """
        if y is None:
            y = x
        # Force float32 computation
        orig_dtype = x.dtype
        x, y = x.float(), y.float()
        
        inner = -torch.einsum('bni,bmi->bnm', x[..., :1], y[..., :1]) + \
                torch.einsum('bni,bmi->bnm', x[..., 1:], y[..., 1:])
        record_acosh_clamp((-inner), eps=LorentzManifold.ACOSH_EPS, tag="hno_latent_pairwise")
        
        # Use a larger clamp value for numerical stability
        result = torch.acosh((-inner).clamp(min=1.0 + LorentzManifold.ACOSH_EPS))
        return result.to(orig_dtype)
    
    @staticmethod
    def pairwise_lorentz_distance_multihead(x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        """
        x: (B, H, N, d+1), y: (B, H, M, d+1) or None
        Returns: (B, H, N, M) distance matrix
        
        Numerical stability: float32 computation and safer clamping
        """
        if y is None:
            y = x
        # Force float32 computation; fp16 can produce NaNs near acosh.
        orig_dtype = x.dtype
        x, y = x.float(), y.float()
        
        # Lorentz inner product: -t1*t2 + x1*x2
        inner = -torch.einsum('bhni,bhmi->bhnm', x[..., :1], y[..., :1]) + \
                torch.einsum('bhni,bhmi->bhnm', x[..., 1:], y[..., 1:])
        record_acosh_clamp((-inner), eps=LorentzManifold.ACOSH_EPS, tag="hno_latent_multihead")
        
        # acosh(z) has large gradients near z=1, so use a larger epsilon
        result = torch.acosh((-inner).clamp(min=1.0 + LorentzManifold.ACOSH_EPS))
        return result.to(orig_dtype)


class HyperbolicCrossAttention(nn.Module):
    """
    Hyperbolic cross-attention
    
    Q comes from one sequence; K/V from another.
    Hyperbolic distance replaces dot-product similarity.
    """
    
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
        
        # Project Q/K into hyperbolic coordinates
        self.to_q = nn.Linear(hidden_dim, num_heads * hyp_dim)
        self.to_k = nn.Linear(hidden_dim, num_heads * hyp_dim)
        
        # Value projection
        self.to_v = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.to_out = nn.Linear(hidden_dim, hidden_dim)
        
        # Learnable temperature
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: (B, Q, C) - query sequence
            context: (B, K, C) - context sequence
        
        Returns:
            out: (B, Q, C)
            
        Numerical stability:
        - apply norm control to Q/K(to avoid unstable gradients)
        - run softmax in float32
        """
        B, Q, C = query.shape
        _, K, _ = context.shape
        H = self.num_heads
        
        # 1. Project Q/K into hyperbolic coordinates
        q = self.to_q(query).view(B, Q, H, self.hyp_dim).permute(0, 2, 1, 3)  # (B, H, Q, hyp_dim)
        k = self.to_k(context).view(B, K, H, self.hyp_dim).permute(0, 2, 1, 3)  # (B, H, K, hyp_dim)
        
        # Norm clipping for stable acosh gradients.
        q = LorentzManifold.clip_norm(q)
        k = LorentzManifold.clip_norm(k)
        
        # Map to the Lorentz manifold
        q_lorentz = LorentzManifold.to_lorentz(q)  # (B, H, Q, hyp_dim + 1)
        k_lorentz = LorentzManifold.to_lorentz(k)  # (B, H, K, hyp_dim + 1)
        
        # 2. Compute the hyperbolic distance matrix.
        dist = LorentzManifold.pairwise_lorentz_distance_multihead(q_lorentz, k_lorentz)  # (B, H, Q, K)
        
        # 3. Distance -> attention weights (float32 softmax).
        temp = self.temperature.clamp(0.1, 3.0)
        attn_logits = -dist / temp
        attn = F.softmax(attn_logits.float(), dim=-1).to(query.dtype)  # (B, H, Q, K)
        attn = self.dropout(attn)
        
        # 4. Aggregate values
        v = self.to_v(context).view(B, K, H, self.head_dim).permute(0, 2, 1, 3)  # (B, H, K, head_dim)
        out = attn @ v  # (B, H, Q, head_dim)
        out = out.permute(0, 2, 1, 3).reshape(B, Q, C)  # (B, Q, C)
        
        # 5. Output projection
        out = self.to_out(out)
        
        return out


class HyperbolicSelfAttention(nn.Module):
    """Hyperbolic self-attention among latent patches."""
    
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
        
        # Project Q/K features into hyperbolic coordinates.
        self.to_q = nn.Linear(hidden_dim, num_heads * hyp_dim)
        self.to_k = nn.Linear(hidden_dim, num_heads * hyp_dim)
        
        # Value projection
        self.to_v = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.to_out = nn.Linear(hidden_dim, hidden_dim)
        
        # Learnable temperature
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, M, C)
        Returns: (B, M, C)
        
        Numerical stability:
        - apply norm control to Q/K(to avoid unstable gradients)
        - run softmax in float32
        """
        B, M, C = x.shape
        H = self.num_heads
        
        # 1. Q/K projection
        q = self.to_q(x).view(B, M, H, self.hyp_dim).permute(0, 2, 1, 3)
        k = self.to_k(x).view(B, M, H, self.hyp_dim).permute(0, 2, 1, 3)
        
        #  Norm clipping(paper stability trick)
        q = LorentzManifold.clip_norm(q)
        k = LorentzManifold.clip_norm(k)
        
        # Map to the Lorentz manifold
        q_lorentz = LorentzManifold.to_lorentz(q)
        k_lorentz = LorentzManifold.to_lorentz(k)
        
        # 2. Compute hyperbolic distances(vectorized)
        dist = LorentzManifold.pairwise_lorentz_distance_multihead(q_lorentz, k_lorentz)  # (B, H, M, M)
        
        # 3. Distance -> attention (float32 softmax).
        temp = self.temperature.clamp(0.1, 3.0)
        attn = F.softmax((-dist / temp).float(), dim=-1).to(x.dtype)
        attn = self.dropout(attn)
        
        # 4. Aggregate values
        v = self.to_v(x).view(B, M, H, self.head_dim).permute(0, 2, 1, 3)
        out = attn @ v
        out = out.permute(0, 2, 1, 3).reshape(B, M, C)
        
        # 5. Output projection
        out = self.to_out(out)
        
        return out


class PerceiverBlock(nn.Module):
    """
    Perceiver Block = Self Attention + FFN
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        hyp_dim: int = 16,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
   ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = HyperbolicSelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            hyp_dim=hyp_dim,
            dropout=dropout,
       )
        
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
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class CrossAttentionBlock(nn.Module):
    """
    Cross Attention Block = Cross Attention + FFN
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        hyp_dim: int = 16,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
   ):
        super().__init__()
        
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)
        self.cross_attn = HyperbolicCrossAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            hyp_dim=hyp_dim,
            dropout=dropout,
       )
        
        self.norm2 = nn.LayerNorm(hidden_dim)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
       )
    
    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        query = query + self.cross_attn(self.norm_q(query), self.norm_kv(context))
        query = query + self.ffn(self.norm2(query))
        return query


class SymmetricCrossAttention(nn.Module):
    """
    Symmetric cross-attention with shared encode/decode weights.
    
    Key idea:
    - Encode: attn_weights (MxN) aggregates points into latents
    - Decode: attn_weights.T (NxM) distributes latents back to points
    - the two directions use transposed versions of the same attention weights
    """
    
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
        
        # Project Q/K into hyperbolic coordinates - for computing attention weights
        self.to_q = nn.Linear(hidden_dim, num_heads * hyp_dim)  # for latents
        self.to_k = nn.Linear(hidden_dim, num_heads * hyp_dim)  # for points
        
        # Separate value projections for encode/decode
        self.to_v_points = nn.Linear(hidden_dim, hidden_dim)   # points -> latents
        self.to_v_latents = nn.Linear(hidden_dim, hidden_dim)  # latents -> points
        
        # Output projection
        self.to_out_encode = nn.Linear(hidden_dim, hidden_dim)
        self.to_out_decode = nn.Linear(hidden_dim, hidden_dim)
        
        # Learnable temperature
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, latents: torch.Tensor, points: torch.Tensor):
        """
        Symmetric encode/decode
        
        Args:
            latents: (B, M, C) - learnable latent tokens
            points: (B, N, C) - point features
        
        Returns:
            new_latents: (B, M, C) - updated latents
            attn_weights: (B, H, M, N) - used for later decoding
        """
        B, M, C = latents.shape
        _, N, _ = points.shape
        H = self.num_heads
        
        # 1. Project Q/K into hyperbolic coordinates
        q = self.to_q(latents).view(B, M, H, self.hyp_dim).permute(0, 2, 1, 3)  # (B, H, M, hyp_dim)
        k = self.to_k(points).view(B, N, H, self.hyp_dim).permute(0, 2, 1, 3)   # (B, H, N, hyp_dim)
        
        # Norm clipping for numerical stability.
        q = LorentzManifold.clip_norm(q)
        k = LorentzManifold.clip_norm(k)
        
        # Map to the Lorentz manifold
        q_lorentz = LorentzManifold.to_lorentz(q)  # (B, H, M, hyp_dim + 1)
        k_lorentz = LorentzManifold.to_lorentz(k)  # (B, H, N, hyp_dim + 1)
        
        # 2. Compute the hyperbolic distance matrix (M x N, vectorized).
        dist = LorentzManifold.pairwise_lorentz_distance_multihead(q_lorentz, k_lorentz)  # (B, H, M, N)
        
        # 3. Distance -> attention weights (float32 softmax).
        temp = self.temperature.clamp(0.1, 3.0)
        attn_logits = -dist / temp
        attn_weights = F.softmax(attn_logits.float(), dim=-1).to(latents.dtype)  # (B, H, M, N)
        attn_weights = self.dropout(attn_weights)
        
        return attn_weights
    
    def encode(self, latents: torch.Tensor, points: torch.Tensor, attn_weights: torch.Tensor) -> torch.Tensor:
        """Encode points into latents using attn_weights"""
        B, M, C = latents.shape
        _, N, _ = points.shape
        H = self.num_heads
        
        # V from points
        v = self.to_v_points(points).view(B, N, H, self.head_dim).permute(0, 2, 1, 3)  # (B, H, N, D)
        
        # Aggregate: attn_weights (M, N) @ V (N, D) -> (M, D)
        out = attn_weights @ v  # (B, H, M, D)
        out = out.permute(0, 2, 1, 3).reshape(B, M, C)
        
        return latents + self.to_out_encode(out)
    
    def decode(self, points: torch.Tensor, latents: torch.Tensor, attn_weights: torch.Tensor) -> torch.Tensor:
        """Decode: latents -> points using the transposed attention weights."""
        B, N, C = points.shape
        _, M, _ = latents.shape
        H = self.num_heads
        
        # V from latents
        v = self.to_v_latents(latents).view(B, M, H, self.head_dim).permute(0, 2, 1, 3)  # (B, H, M, D)
        
        # Use transposed attention weights: (N, M) @ V (M, D) -> (N, D)
        attn_weights_T = attn_weights.transpose(-2, -1)  # (B, H, N, M), symmetric transpose
        out = attn_weights_T @ v  # (B, H, N, D)
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        
        return points + self.to_out_decode(out)


class SharedWeightBlock(nn.Module):
    """
    Shared-weight symmetric interaction block
    
    Key idea: attention weights are provided externally and reused across layers.
    Reuse the same attention weights across layers
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        hyp_dim: int = 16,
        mlp_ratio: float = 2.5,
        dropout: float = 0.0,
   ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Layer-local value projections for encode/decode
        self.to_v_points = nn.Linear(hidden_dim, hidden_dim)   # points -> latents
        self.to_v_latents = nn.Linear(hidden_dim, hidden_dim)  # latents -> points
        
        # Output projection
        self.to_out_encode = nn.Linear(hidden_dim, hidden_dim)
        self.to_out_decode = nn.Linear(hidden_dim, hidden_dim)
        
        # === Latent Self-Attention ===
        self.norm_self = nn.LayerNorm(hidden_dim)
        self.to_q_self = nn.Linear(hidden_dim, num_heads * hyp_dim)
        self.to_k_self = nn.Linear(hidden_dim, num_heads * hyp_dim)
        self.to_v_self = nn.Linear(hidden_dim, hidden_dim)
        self.to_out_self = nn.Linear(hidden_dim, hidden_dim)
        self.temp_self = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.hyp_dim = hyp_dim
        
        # === FFNs ===
        self.norm_latent = nn.LayerNorm(hidden_dim)
        self.ffn_latent = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
            nn.Dropout(dropout),
       )
        
        self.norm_points = nn.LayerNorm(hidden_dim)
        self.ffn_points = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
            nn.Dropout(dropout),
       )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, latents: torch.Tensor, points: torch.Tensor, attn_weights: torch.Tensor):
        """
        Args:
            latents: (B, M, C)
            points: (B, N, C)
            attn_weights: (B, H, M, N) - provided externally and shared by all layers
        Returns:
            new_latents: (B, M, C)
            new_points: (B, N, C)
        """
        B, M, C = latents.shape
        _, N, _ = points.shape
        H = self.num_heads
        
        # ========== 1. Encode: points -> latents (using shared attn_weights) ==========
        v_enc = self.to_v_points(points).view(B, N, H, self.head_dim).permute(0, 2, 1, 3)
        enc_out = attn_weights @ v_enc  # (B, H, M, D)
        enc_out = enc_out.permute(0, 2, 1, 3).reshape(B, M, C)
        latents = latents + self.to_out_encode(enc_out)
        
        # ========== 2. Latent Self-Attention ==========
        latents_normed = self.norm_self(latents)
        q_s = self.to_q_self(latents_normed).view(B, M, H, self.hyp_dim).permute(0, 2, 1, 3)
        k_s = self.to_k_self(latents_normed).view(B, M, H, self.hyp_dim).permute(0, 2, 1, 3)
        v_s = self.to_v_self(latents_normed).view(B, M, H, self.head_dim).permute(0, 2, 1, 3)
        
        # Norm clipping for numerical stability.
        q_s = LorentzManifold.clip_norm(q_s)
        k_s = LorentzManifold.clip_norm(k_s)
        
        q_s_lor = LorentzManifold.to_lorentz(q_s)
        k_s_lor = LorentzManifold.to_lorentz(k_s)
        
        # vectorizedHyperbolic distance
        dist_s = LorentzManifold.pairwise_lorentz_distance_multihead(q_s_lor, k_s_lor)
        
        temp_s = self.temp_self.clamp(0.1, 3.0)
        attn_s = F.softmax((-dist_s / temp_s).float(), dim=-1).to(latents.dtype)  #  float32 softmax
        attn_s = self.dropout(attn_s)
        
        self_out = attn_s @ v_s
        self_out = self_out.permute(0, 2, 1, 3).reshape(B, M, C)
        latents = latents + self.to_out_self(self_out)
        
        # Latent FFN
        latents = latents + self.ffn_latent(self.norm_latent(latents))
        
        # ========== 3. Decode: latents -> points (using transposed attn_weights) ==========
        v_dec = self.to_v_latents(latents).view(B, M, H, self.head_dim).permute(0, 2, 1, 3)
        attn_weights_T = attn_weights.transpose(-2, -1)  # (B, H, N, M), symmetric transpose
        dec_out = attn_weights_T @ v_dec  # (B, H, N, D)
        dec_out = dec_out.permute(0, 2, 1, 3).reshape(B, N, C)
        points = points + self.to_out_decode(dec_out)
        
        # Points FFN
        points = points + self.ffn_points(self.norm_points(points))
        
        return latents, points


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
