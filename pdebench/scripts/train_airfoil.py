import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from pdebench.hno.hno_patch import build_hno
from pdebench.utils.testloss import TestLoss


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    default_data_path = repo_root / "data" / "pdebench" / "airfoil" / "naca"
    default_out_dir = repo_root / "outputs" / "pdebench" / "airfoil"

    parser = argparse.ArgumentParser("Train HNO on PDEBench Airfoil")
    parser.add_argument("--data_path", type=str, default=str(default_data_path))
    parser.add_argument("--out_dir", type=str, default=str(default_out_dir))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)

    parser.add_argument("--downsamplex", type=int, default=1)
    parser.add_argument("--downsampley", type=int, default=1)
    parser.add_argument("--train_frac", type=float, default=1.0)
    parser.add_argument("--test_frac", type=float, default=1.0)
    parser.add_argument("--subset_seed", type=int, default=0)

    parser.add_argument("--hidden", type=int, default=56)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=5)
    parser.add_argument("--hyp_dim", type=int, default=16)
    parser.add_argument("--mlp_ratio", type=float, default=2.5)
    parser.add_argument("--dropout", type=float, default=0.0)
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data_path)
    x_path = data_path / "NACA_Cylinder_X.npy"
    y_path = data_path / "NACA_Cylinder_Y.npy"
    q_path = data_path / "NACA_Cylinder_Q.npy"
    if not x_path.exists() or not y_path.exists() or not q_path.exists():
        raise FileNotFoundError(
            "Airfoil.npy files not found. Expected:\n"
            f"- {x_path}\n"
            f"- {y_path}\n"
            f"- {q_path}\n"
            "Pass --data_path to the directory containing these files."
       )

    ntrain_full = 1000
    ntest_full = 200

    r1 = int(args.downsamplex)
    r2 = int(args.downsampley)
    s1 = int(((221 - 1) / r1) + 1)
    s2 = int(((51 - 1) / r2) + 1)

    input_x = torch.tensor(np.load(x_path), dtype=torch.float)
    input_y = torch.tensor(np.load(y_path), dtype=torch.float)
    coords = torch.stack([input_x, input_y], dim=-1)  # (N, 221, 51, 2)
    target = torch.tensor(np.load(q_path)[:, 4], dtype=torch.float)  # (N, 221, 51)

    x_train = coords[:ntrain_full, ::r1, ::r2][:, :s1, :s2].reshape(ntrain_full, -1, 2)
    y_train = target[:ntrain_full, ::r1, ::r2][:, :s1, :s2].reshape(ntrain_full, -1)
    x_test = coords[ntrain_full : ntrain_full + ntest_full, ::r1, ::r2][:, :s1, :s2].reshape(ntest_full, -1, 2)
    y_test = target[ntrain_full : ntrain_full + ntest_full, ::r1, ::r2][:, :s1, :s2].reshape(ntest_full, -1)

    # Normalize coordinates to [0, 1] using full training-set statistics.
    x_min = x_train.amin(dim=(0, 1), keepdim=True)
    x_max = x_train.amax(dim=(0, 1), keepdim=True)
    x_scale = (x_max - x_min).clamp_min(1e-8)
    x_train = (x_train - x_min) / x_scale
    x_test = (x_test - x_min) / x_scale

    def _subset(x: torch.Tensor, frac: float, seed: int) -> torch.Tensor:
        frac = float(frac)
        if frac >= 1.0:
            return torch.arange(x.shape[0])
        k = max(1, int(np.ceil(x.shape[0] * frac)))
        rng = np.random.RandomState(int(seed))
        return torch.from_numpy(rng.choice(x.shape[0], size=k, replace=False))

    train_idx = _subset(x_train, args.train_frac, args.subset_seed)
    test_idx = _subset(x_test, args.test_frac, args.subset_seed + 1)
    x_train = x_train[train_idx]
    y_train = y_train[train_idx]
    x_test = x_test[test_idx]
    y_test = y_test[test_idx]

    ntrain = x_train.shape[0]
    ntest = x_test.shape[0]

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=int(args.batch_size), shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=int(args.batch_size), shuffle=False)

    model = build_hno(
        space_dim=2,
        fun_dim=0,
        out_dim=1,
        hidden_dim=int(args.hidden),
        num_layers=int(args.layers),
        num_heads=int(args.heads),
        patch_size=int(args.patch_size),
        hyp_dim=int(args.hyp_dim),
        mlp_ratio=float(args.mlp_ratio),
        dropout=float(args.dropout),
        H=s1,
        W=s2,
   ).to(device)

    print(f"[Airfoil] params={count_parameters(model):,} ({count_parameters(model)/1e6:.2f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=float(args.lr),
        epochs=int(args.epochs),
        steps_per_epoch=len(train_loader),
        final_div_factor=10000.0,
   )

    rel_loss = TestLoss(size_average=False)
    best_err = float("inf")
    best_path = ckpt_dir / "hno_airfoil_best.pt"

    for epoch in range(int(args.epochs)):
        model.train()
        train_loss = 0.0
        for pos, y in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            pos = pos.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(pos, fx=None).squeeze(-1)
            loss = rel_loss(out, y)
            loss.backward()
            if args.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
            optimizer.step()
            scheduler.step()
            train_loss += float(loss.item())

        train_loss /= ntrain

        model.eval()
        rel_err = 0.0
        with torch.no_grad():
            for pos, y in test_loader:
                pos = pos.to(device)
                y = y.to(device)
                out = model(pos, fx=None).squeeze(-1)
                rel_err += float(rel_loss(out, y).item())
        rel_err /= ntest

        print(f"epoch={epoch:04d} train_rel={train_loss:.6f} test_rel={rel_err:.6f}")
        if rel_err < best_err:
            best_err = rel_err
            torch.save(model.state_dict(), best_path)

    print(f"[Airfoil] best_test_rel={best_err:.6f} ckpt={best_path}")


if __name__ == "__main__":
    main()

