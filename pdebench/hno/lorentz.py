"""Shared Lorentz-manifold utilities for HNO modules."""

import torch

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
