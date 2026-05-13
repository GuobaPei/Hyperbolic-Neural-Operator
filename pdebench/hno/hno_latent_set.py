"""Compatibility exports for the latent-set Hyperbolic Neural Operator.

The implementation is split across smaller modules:
- ``lorentz.py``: Lorentz manifold utilities
- ``attention.py``: attention and interaction blocks
- ``perceiver.py``: latent-set HNO model and builder
- ``adaptive.py``: optional adaptive grid/mesh variant
"""

from pdebench.hno.adaptive import (
    AdaptiveHyperbolicNO,
    HyperbolicPatchAttention,
    HyperbolicPatchBlock,
    LocalFeatureExtractor,
    RelativePositionBias2D,
)
from pdebench.hno.attention import (
    CrossAttentionBlock,
    HyperbolicCrossAttention,
    HyperbolicSelfAttention,
    PerceiverBlock,
    SharedWeightBlock,
    SymmetricCrossAttention,
)
from pdebench.hno.lorentz import LorentzManifold
from pdebench.hno.perceiver import HNO, HyperbolicPerceiverNO, build_hno

__all__ = [
    "AdaptiveHyperbolicNO",
    "CrossAttentionBlock",
    "HNO",
    "HyperbolicCrossAttention",
    "HyperbolicPatchAttention",
    "HyperbolicPatchBlock",
    "HyperbolicPerceiverNO",
    "HyperbolicSelfAttention",
    "LocalFeatureExtractor",
    "LorentzManifold",
    "PerceiverBlock",
    "RelativePositionBias2D",
    "SharedWeightBlock",
    "SymmetricCrossAttention",
    "build_hno",
]
