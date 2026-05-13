# Hyperbolic Neural Operator

### Hyperbolic geometry for hierarchy-aware PDE operator learning

[![ICML 2026](https://img.shields.io/badge/ICML-2026-d45b3f?style=flat-square)](https://icml.cc/virtual/2026/poster/65554)
[![Project](https://img.shields.io/badge/Project-HNO-258f86?style=flat-square)](https://guobapei.github.io/Hyperbolic-Neural-Operator/)
[![Code](https://img.shields.io/badge/Code-GitHub-24292f?style=flat-square&logo=github)](https://github.com/GuobaPei/Hyperbolic-Neural-Operator)
[![License](https://img.shields.io/badge/License-MIT-7fbf3f?style=flat-square)](LICENSE)

🎉 **Hyperbolic Neural Operator (HNO) has been accepted to ICML 2026.**

We present **HNO**, a neural operator that learns near-far interaction routing
with stabilized Lorentz-hyperbolic distance kernels. HNO gives tokens a learned
scale coordinate, enabling compact hierarchical PDE surrogate modeling across
regular grids, structured meshes, point clouds, and large-scale CFD.

Key contributions include:

- **Hyperbolic routing kernel:** replaces dot-product token mixing with
  stabilized hyperbolic-distance attention.
- **Near-far physical organization:** learns FMM-inspired local-detail and
  far-field-summary structure without hand-built trees.
- **Broad PDE and CFD validation:** includes PDEBench tasks plus AirfRANS and
  ShapeNetCar large-scale unstructured meshes.

## 📊 Overview

<p align="center">
  <img src="figures/hno_motivation.png" alt="HNO overview" width="900">
</p>

## 🚀 Quick Start

```bash
python -m venv .venv_pdebench
source .venv_pdebench/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements_pdebench.txt
bash scripts/smoke_test.sh
```

Run one task:

```bash
python -m pdebench.scripts.train_darcy --data_path <DARCY_DATA_DIR>
```

## 📁 Repository

```text
pdebench/       HNO models, configs, and PDEBench training scripts
large_scale/    AirfRANS and ShapeNetCar code
scripts/        setup, smoke-test, and run wrappers
docs/           project website source
figures/        README figures
```

Datasets, checkpoints, logs, and generated caches are not included.

## 📝 Citation

```bibtex
@inproceedings{hno2026,
  title     = {Hyperbolic Neural Operator},
  author    = {Pei, Jieyuan and Li, Zhuoxuan and Li, Wei and Zhang, Haobo and Jiang, Jiawei and Zheng, Jianwei},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  year      = {2026},
  url       = {https://icml.cc/virtual/2026/poster/65554}
}
```

## License

MIT License. See [LICENSE](LICENSE).
