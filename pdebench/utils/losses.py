"""Loss components for Hyperbolic-SSM (PINO + weak shock constraints)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _pad_periodic(u: Tensor, padding: int = 1) -> Tensor:
    if padding == 0:
        return u
    return F.pad(u, (0, 0, padding, padding, padding, padding), mode="circular")


def finite_difference_laplacian(u: Tensor, spacing: float, periodic: bool = False) -> Tensor:
    """Compute a 2D Laplacian using central differences on a grid."""

    if u.dim() != 4:
        raise ValueError("u must be of shape (B, C, H, W)")

    if periodic:
        u_pad = _pad_periodic(u)
    else:
        u_pad = F.pad(u, (0, 0, 1, 1, 1, 1), mode="replicate")

    lap = (
        -4.0 * u_pad[..., 1:-1, 1:-1]
        + u_pad[..., 2:, 1:-1]
        + u_pad[..., :-2, 1:-1]
        + u_pad[..., 1:-1, 2:]
        + u_pad[..., 1:-1, :-2]
   ) / (spacing ** 2)

    # Ensure the Laplacian matches the original resolution (compensate for potential truncation
    # introduced by uneven padding when the grid is not perfectly divisible).
    target_h, target_w = u.shape[-2], u.shape[-1]
    pad_h = target_h - lap.shape[-2]
    pad_w = target_w - lap.shape[-1]
    if pad_h != 0 or pad_w != 0:
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        lap = F.pad(lap, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")

    return lap


def compute_rankine_hugoniot(u_left: Tensor, u_right: Tensor, flux_left: Tensor, flux_right: Tensor) -> Tensor:
    return (flux_right - flux_left) - (u_right - u_left)


@dataclass
class LossWeights:
    sup: float = 1.0
    pino: float = 1.0
    flux: float = 0.1
    rh: float = 0.1
    bc: float = 0.1


class HyperbolicLossBundle(nn.Module):
    """Compose supervised, PINO, and weak shock constraints."""

    def __init__(
        self,
        pde_operator: Optional[Callable[[Tensor, Dict[str, Tensor]], Tensor]] = None,
        flux_fn: Optional[Callable[[Tensor], Tensor]] = None,
        weights: LossWeights = LossWeights(),
   ) -> None:
        super().__init__()
        self.pde_operator = pde_operator
        self.flux_fn = flux_fn
        self.weights = weights

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        metadata: Dict[str, Tensor],
   ) -> Dict[str, Tensor]:
        losses: Dict[str, Tensor] = {}
        sup_loss = F.mse_loss(pred, target)
        losses["supervised"] = sup_loss

        if self.pde_operator is not None:
            residual = self.pde_operator(pred, metadata)
            losses["pde"] = torch.mean(residual ** 2)

        if self.flux_fn is not None and "shock_pairs" in metadata:
            idx_left = metadata["shock_pairs"]["left"]
            idx_right = metadata["shock_pairs"]["right"]
            u_left = pred[:, idx_left]
            u_right = pred[:, idx_right]
            flux_left = self.flux_fn(u_left)
            flux_right = self.flux_fn(u_right)
            rh_res = compute_rankine_hugoniot(u_left, u_right, flux_left, flux_right)
            losses["rankine_hugoniot"] = torch.mean(rh_res ** 2)

        if "boundary" in metadata:
            boundary = metadata["boundary"]
            bc_mask = boundary.get("mask")
            bc_value = boundary.get("value")
            if bc_mask is not None and bc_value is not None:
                bc_loss = F.mse_loss(pred[bc_mask], bc_value.expand_as(pred[bc_mask]))
                losses["boundary"] = bc_loss

        total = self.weights.sup * losses["supervised"]
        if "pde" in losses:
            total = total + self.weights.pino * losses["pde"]
        if "rankine_hugoniot" in losses:
            total = total + self.weights.rh * losses["rankine_hugoniot"]
        if "flux" in losses:
            total = total + self.weights.flux * losses["flux"]
        if "boundary" in losses:
            total = total + self.weights.bc * losses["boundary"]

        losses["total"] = total
        return losses


