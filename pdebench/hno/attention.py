"""Attention and interaction blocks used by the latent-set HNO."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pdebench.hno.lorentz import LorentzManifold


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
