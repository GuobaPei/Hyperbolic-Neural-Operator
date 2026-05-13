"""Simple block-structured hierarchy builder for PDEBench grids."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
from torch import Tensor


@dataclass
class StructuredHierarchy:
    H: int
    W: int
    leaf_parent: Tensor
    coarse_children: List[Tensor]
    root_children: Tensor
    neighbor_src: Tensor
    neighbor_dst: Tensor
    neighbor_degree: Tensor

    def to(self, device=None) -> "StructuredHierarchy":
        if device is None:
            return self
        leaf_parent = self.leaf_parent.to(device)
        coarse_children = [child.to(device) for child in self.coarse_children]
        root_children = self.root_children.to(device)
        neighbor_src = self.neighbor_src.to(device)
        neighbor_dst = self.neighbor_dst.to(device)
        neighbor_degree = self.neighbor_degree.to(device)
        return StructuredHierarchy(
            H=self.H,
            W=self.W,
            leaf_parent=leaf_parent,
            coarse_children=coarse_children,
            root_children=root_children,
            neighbor_src=neighbor_src,
            neighbor_dst=neighbor_dst,
            neighbor_degree=neighbor_degree,
       )


def build_block_hierarchy(H: int, W: int, *, patch: int = 8) -> StructuredHierarchy:
    """Construct a two-level quadtree-style hierarchy for a structured grid."""

    num_points = H * W
    leaf_parent = torch.empty(num_points, dtype=torch.long)
    coarse_children: List[Tensor] = []

    patch_h = max(1, min(patch, H))
    patch_w = max(1, min(patch, W))

    coarse_index = 0
    for h0 in range(0, H, patch_h):
        h1 = min(h0 + patch_h, H)
        for w0 in range(0, W, patch_w):
            w1 = min(w0 + patch_w, W)
            indices = []
            for i in range(h0, h1):
                base = i * W
                for j in range(w0, w1):
                    idx = base + j
                    indices.append(idx)
                    leaf_parent[idx] = coarse_index
            coarse_children.append(torch.tensor(indices, dtype=torch.long))
            coarse_index += 1

    root_children = torch.arange(coarse_index, dtype=torch.long)

    # Build four-neighbour connectivity for leaves (physical edges)
    src_list: List[int] = []
    dst_list: List[int] = []
    for idx in range(num_points):
        i = idx // W
        j = idx % W
        neighbours: Sequence[Tuple[int, int]] = (
            (i - 1, j),
            (i + 1, j),
            (i, j - 1),
            (i, j + 1),
       )
        for ni, nj in neighbours:
            if 0 <= ni < H and 0 <= nj < W:
                nidx = ni * W + nj
                src_list.append(nidx)
                dst_list.append(idx)
    if src_list:
        neighbor_src = torch.tensor(src_list, dtype=torch.long)
        neighbor_dst = torch.tensor(dst_list, dtype=torch.long)
        neighbor_degree = torch.bincount(neighbor_dst, minlength=num_points).float().clamp_min(1.0)
    else:
        neighbor_src = torch.empty(0, dtype=torch.long)
        neighbor_dst = torch.empty(0, dtype=torch.long)
        neighbor_degree = torch.ones(num_points, dtype=torch.float)

    return StructuredHierarchy(
        H=H,
        W=W,
        leaf_parent=leaf_parent,
        coarse_children=coarse_children,
        root_children=root_children,
        neighbor_src=neighbor_src,
        neighbor_dst=neighbor_dst,
        neighbor_degree=neighbor_degree,
   )


