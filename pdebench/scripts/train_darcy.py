import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import scipy.io as scio
import torch
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from pdebench.hno.hno_patch import build_hno
from pdebench.utils.normalizer import UnitTransformer
from pdebench.utils.testloss import TestLoss


def central_diff(x: torch.Tensor, h: float, resolution: int) -> Tuple[torch.Tensor, torch.Tensor]:
    x = rearrange(x, "b (h w) c -> b h w c", h=resolution, w=resolution)
    x = F.pad(x, (0, 0, 1, 1, 1, 1), mode="constant", value=0.0)
    grad_x = (x[:, 1:-1, 2:, :] - x[:, 1:-1, :-2, :]) / (2 * h)
    grad_y = (x[:, 2:, 1:-1, :] - x[:, :-2, 1:-1, :]) / (2 * h)
    return grad_x, grad_y


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    default_data_path = repo_root / "data" / "pdebench" / "darcy"
    default_out_dir = repo_root / "outputs" / "pdebench" / "darcy"

    parser = argparse.ArgumentParser("Train HNO on PDEBench Darcy")
    parser.add_argument("--data_path", type=str, default=str(default_data_path))
    parser.add_argument("--out_dir", type=str, default=str(default_out_dir))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_grad_norm", type=float, default=None)

    parser.add_argument("--downsample", type=int, default=5)
    parser.add_argument("--ntrain", type=int, default=1000)

    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=5)
    parser.add_argument("--hyp_dim", type=int, default=16)
    parser.add_argument("--mlp_ratio", type=float, default=2.5)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--deriv_weight", type=float, default=0.1, help="Weight for derivative regularization.")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_path = Path(args.data_path) / "piececonst_r421_N1024_smooth1.mat"
    test_path = Path(args.data_path) / "piececonst_r421_N1024_smooth2.mat"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Darcy.mat files not found. Expected:\n"
            f"- {train_path}\n"
            f"- {test_path}\n"
            "Pass --data_path to the directory containing these files."
       )

    ntrain = int(args.ntrain)
    ntest = 200

    r = int(args.downsample)
    h = int(((421 - 1) / r) + 1)
    s = h
    dx = 1.0 / s

    train_data = scio.loadmat(str(train_path))
    x_train = train_data["coeff"][:ntrain, ::r, ::r][:, :s, :s].reshape(ntrain, -1)
    y_train = train_data["sol"][:ntrain, ::r, ::r][:, :s, :s].reshape(ntrain, -1)
    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train)

    test_data = scio.loadmat(str(test_path))
    x_test = test_data["coeff"][:ntest, ::r, ::r][:, :s, :s].reshape(ntest, -1)
    y_test = test_data["sol"][:ntest, ::r, ::r][:, :s, :s].reshape(ntest, -1)
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test)

    x_normalizer = UnitTransformer(x_train)
    y_normalizer = UnitTransformer(y_train)
    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)
    y_train = y_normalizer.encode(y_train)
    y_test = y_normalizer.encode(y_test)
    x_normalizer.to(device)
    y_normalizer.to(device)

    grid_x = np.linspace(0, 1, s)
    grid_y = np.linspace(0, 1, s)
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)
    pos = torch.tensor(np.c_[grid_x.ravel(), grid_y.ravel()], dtype=torch.float).unsqueeze(0)
    pos_train = pos.repeat(ntrain, 1, 1)
    pos_test = pos.repeat(ntest, 1, 1)

    train_loader = DataLoader(
        TensorDataset(pos_train, x_train, y_train),
        batch_size=int(args.batch_size),
        shuffle=True,
   )
    test_loader = DataLoader(
        TensorDataset(pos_test, x_test, y_test),
        batch_size=int(args.batch_size),
        shuffle=False,
   )

    model = build_hno(
        space_dim=2,
        fun_dim=1,
        out_dim=1,
        hidden_dim=int(args.hidden),
        num_layers=int(args.layers),
        num_heads=int(args.heads),
        patch_size=int(args.patch_size),
        hyp_dim=int(args.hyp_dim),
        mlp_ratio=float(args.mlp_ratio),
        dropout=float(args.dropout),
        H=s,
        W=s,
   ).to(device)

    print(f"[Darcy] params={count_parameters(model):,} ({count_parameters(model)/1e6:.2f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=float(args.lr),
        epochs=int(args.epochs),
        steps_per_epoch=len(train_loader),
        final_div_factor=10000.0,
   )

    rel_loss = TestLoss(size_average=False)
    der_x = TestLoss(size_average=False)
    der_y = TestLoss(size_average=False)

    best_err = float("inf")
    best_path = ckpt_dir / "hno_darcy_best.pt"

    for epoch in range(int(args.epochs)):
        model.train()
        train_loss = 0.0
        train_reg = 0.0

        for x, fx, y in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            x = x.to(device)
            fx = fx.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)

            out = model(x, fx=fx.unsqueeze(-1)).squeeze(-1)
            out = y_normalizer.decode(out)
            y_dec = y_normalizer.decode(y)

            l2 = rel_loss(out, y_dec)

            out_grid = rearrange(out.unsqueeze(-1), "b (h w) c -> b c h w", h=s)
            out_grid = out_grid[..., 1:-1, 1:-1].contiguous()
            out_grid = F.pad(out_grid, (1, 1, 1, 1), "constant", 0)
            out_grid = rearrange(out_grid, "b c h w -> b (h w) c")

            gt_grad_x, gt_grad_y = central_diff(y_dec.unsqueeze(-1), dx, s)
            pred_grad_x, pred_grad_y = central_diff(out_grid, dx, s)
            deriv = der_x(pred_grad_x, gt_grad_x) + der_y(pred_grad_y, gt_grad_y)

            loss = l2 + float(args.deriv_weight) * deriv
            loss.backward()
            if args.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
            optimizer.step()
            scheduler.step()

            train_loss += float(l2.item())
            train_reg += float(deriv.item())

        train_loss /= ntrain
        train_reg /= ntrain

        model.eval()
        rel_err = 0.0
        with torch.no_grad():
            for x, fx, y in test_loader:
                x = x.to(device)
                fx = fx.to(device)
                y = y.to(device)
                out = model(x, fx=fx.unsqueeze(-1)).squeeze(-1)
                out = y_normalizer.decode(out)
                y_dec = y_normalizer.decode(y)
                rel_err += float(rel_loss(out, y_dec).item())

        rel_err /= ntest
        print(f"epoch={epoch:04d} train_rel={train_loss:.6f} train_deriv={train_reg:.6f} test_rel={rel_err:.6f}")

        if rel_err < best_err:
            best_err = rel_err
            torch.save(model.state_dict(), best_path)

    print(f"[Darcy] best_test_rel={best_err:.6f} ckpt={best_path}")


if __name__ == "__main__":
    main()
