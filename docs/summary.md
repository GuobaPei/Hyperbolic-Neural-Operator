# Hyperbolic Neural Operator (HNO)

ICML 2026 · PMLR 306

Authors: Jieyuan Pei, Zhuoxuan Li, Wei Li, Haobo Zhang, Jiawei Jiang, Jianwei Zheng

[Project](https://guobapei.github.io/Hyperbolic-Neural-Operator/) · [Paper PDF](https://guobapei.github.io/Hyperbolic-Neural-Operator/assets/paper.pdf) · [Full text](https://guobapei.github.io/Hyperbolic-Neural-Operator/fulltext.html) · [OpenReview](https://openreview.net/forum?id=CUQwYTTNu8) · [Code](https://github.com/GuobaPei/Hyperbolic-Neural-Operator)

## Abstract

Neural operators learn solution operators for parametric PDE families, mapping coefficients, forcing fields, or geometric inputs to full solution fields and thereby accelerating scientific computation. Transformer-based architectures offer strong flexibility on irregular domains, but dense dot-product attention often allocates pairwise scoring uniformly across token pairs, neglecting that far-field interactions in many discretized PDE kernels are numerically compressible. To address this mismatch, we draw inspiration from classical fast solvers that exploit hierarchical near–far organization. We further observe that embedding such tree-structured hierarchies in Euclidean space incurs inherent distortion, whereas hyperbolic space naturally accommodates exponential branching. Consequently, we propose Hyperbolic Neural Operator (HNO), which leverages intrinsic hyperbolic geometry to instantiate a continuous Gibbs kernel based on stabilized geodesic distances on the Lorentz hyperboloid. This design imposes a geometric inductive bias for learnable multi-scale near–far routing within a unified attention mechanism. Empirically, HNO achieves the lowest error among the evaluated methods on six PDE benchmarks and two large-scale unstructured CFD tasks, reducing the mean relative ℓ2 error by up to 40% in the best evaluated setting. Code is available in the GitHub repository.

## Research summary

Hyperbolic Neural Operator (HNO) is an ICML 2026 method for learning solution operators of parametric partial differential equations. It maps input fields or geometric descriptors to solution fields. HNO uses stabilized geodesic distances on the Lorentz hyperboloid to construct a Gibbs attention kernel. This geometry provides an inductive bias for hierarchical near–far interaction routing, inspired by the organization of classical fast solvers. Here, hyperbolic refers to the learned representation geometry; the evaluated tasks span multiple PDE families. Far-field compressibility refers to numerical or low-rank structure in interactions. The paper evaluates HNO on Elasticity, Navier–Stokes, Darcy, Plasticity, Airfoil and Pipe, plus the AirfRANS and ShapeNet Car CFD benchmarks with approximately 32,000 mesh nodes per sample. The released implementations use geometry-specific tokenization, including patch-based grid models and summary-token processing for point clouds. The paper also examines hierarchical tree-kernel fitting, attention locality, and Darcy ablations. HNO is relevant to research on efficient neural operators, non-Euclidean attention, multiscale physical interactions, and PDE surrogates on irregular meshes. Reported accuracy and efficiency values describe the paper’s evaluated protocols.

## Benchmarks

Table 2 of the paper reports mean relative ℓ2 error; lower is better. HNO results on the six standard PDE benchmarks are averaged over three runs unless otherwise noted. Baseline values follow official reports or authors’ implementations. HNO experiments use a single NVIDIA A6000 48GB GPU. Relative reduction is (second-best error − HNO error) / second-best error, using the displayed rounded values.

| Benchmark | Geometry | HNO | Second best | Relative reduction |
| --- | --- | ---: | ---: | ---: |
| Elasticity | Point cloud | 0.0037 | 0.0064 | 42.2% |
| Navier–Stokes | Regular grid | 0.0676 | 0.0892 | 24.2% |
| Darcy | Regular grid | 0.0045 | 0.0054 | 16.7% |
| Plasticity | Structured mesh | 0.0009 | 0.0012 | 25.0% |
| Airfoil | Structured mesh | 0.0048 | 0.0053 | 9.4% |
| Pipe | Structured mesh | 0.0027 | 0.0042 | 35.7% |

## Efficiency

Table 3 reports a Darcy configuration with 0.82M parameters, 0.227 GB VRAM, 0.73 h training time and 4.47 ms per batch inference time. Baselines in Table 3 use their official configurations and are not parameter-matched to HNO. Appendix J.1 / Table 12 provides a separate parameter-matched microbenchmark. These timing and memory values belong to the stated Darcy experiments.

## 中文简介

Hyperbolic Neural Operator（HNO，双曲神经算子）发表于 ICML 2026，利用 Lorentz 双曲空间中的测地距离构造 attention 核，学习 PDE 的多尺度近场与远场交互。这里的“双曲”指模型的表征几何；实验覆盖 Darcy、弹性力学、Navier–Stokes 等不同 PDE，以及 AirfRANS 和 ShapeNet Car 非结构网格 CFD 任务。

## Citation

```bibtex
@inproceedings{hno2026,
  title     = {Hyperbolic Neural Operator},
  author    = {Pei, Jieyuan and Li, Zhuoxuan and Li, Wei and Zhang, Haobo and Jiang, Jiawei and Zheng, Jianwei},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {306},
  publisher = {PMLR},
  year      = {2026},
  url       = {https://icml.cc/virtual/2026/poster/65554}
}
```
