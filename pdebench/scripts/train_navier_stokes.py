import argparse
from pathlib import Path

import numpy as np
import scipy.io as scio
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from pdebench.hno.hno_patch import build_hno
from pdebench.utils.testloss import TestLoss


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    default_data_path = repo_root / "data" / "pdebench" / "navier-stokes"
    default_out_dir = repo_root / "outputs" / "pdebench" / "navier-stokes"

    parser = argparse.ArgumentParser("Train HNO on PDEBench Navier-Stokes (2D, autoregressive)")
    parser.add_argument("--data_path", type=str, default=str(default_data_path))
    parser.add_argument("--out_dir", type=str, default=str(default_out_dir))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)

    parser.add_argument("--downsample", type=int, default=1)
    parser.add_argument("--train_frac", type=float, default=1.0)
    parser.add_argument("--test_frac", type=float, default=1.0)
    parser.add_argument("--subset_seed", type=int, default=0)

    parser.add_argument("--hidden", type=int, default=132)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--hyp_dim", type=int, default=16)
    parser.add_argument("--mlp_ratio", type=float, default=2.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use_temporal_conv", action="store_true", help="Enable temporal 1D conv in the lift.")

    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data_path) / "NavierStokes_V1e-5_N1200_T20.mat"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Navier-Stokes.mat not found: {data_path}\n"
            "Pass --data_path to the directory containing NavierStokes_V1e-5_N1200_T20.mat."
       )

    ntrain = 1000
    ntest = 200
    T_in = 10
    T = 10
    step = 1

    r = int(args.downsample)
    h = int(((64 - 1) / r) + 1)

    data = scio.loadmat(str(data_path))
    u = data["u"]  # (1200, 64, 64, 20)

    train_a = u[:ntrain, ::r, ::r, :T_in][:, :h, :h, :].reshape(ntrain, -1, T_in)
    train_u = u[:ntrain, ::r, ::r, T_in : (T + T_in)][:, :h, :h, :].reshape(ntrain, -1, T)
    test_a = u[-ntest:, ::r, ::r, :T_in][:, :h, :h, :].reshape(ntest, -1, T_in)
    test_u = u[-ntest:, ::r, ::r, T_in : (T + T_in)][:, :h, :h, :].reshape(ntest, -1, T)

    train_a = torch.from_numpy(train_a).float()
    train_u = torch.from_numpy(train_u).float()
    test_a = torch.from_numpy(test_a).float()
    test_u = torch.from_numpy(test_u).float()

    def _subset_indices(n: int, frac: float, seed: int) -> torch.Tensor:
        frac = float(frac)
        if frac >= 1.0:
            return torch.arange(n)
        k = max(1, int(np.ceil(n * frac)))
        rng = np.random.RandomState(int(seed))
        return torch.from_numpy(rng.choice(n, size=k, replace=False))

    train_idx = _subset_indices(ntrain, args.train_frac, args.subset_seed)
    test_idx = _subset_indices(ntest, args.test_frac, args.subset_seed + 1)
    train_a, train_u = train_a[train_idx], train_u[train_idx]
    test_a, test_u = test_a[test_idx], test_u[test_idx]
    ntrain_eff, ntest_eff = train_a.shape[0], test_a.shape[0]

    grid_x = np.linspace(0, 1, h)
    grid_y = np.linspace(0, 1, h)
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)
    pos = torch.tensor(np.c_[grid_x.ravel(), grid_y.ravel()], dtype=torch.float).unsqueeze(0)
    pos_train = pos.repeat(ntrain_eff, 1, 1)
    pos_test = pos.repeat(ntest_eff, 1, 1)

    train_loader = DataLoader(
        TensorDataset(pos_train, train_a, train_u),
        batch_size=int(args.batch_size),
        shuffle=True,
   )
    test_loader = DataLoader(
        TensorDataset(pos_test, test_a, test_u),
        batch_size=int(args.batch_size),
        shuffle=False,
   )

    model = build_hno(
        space_dim=2,
        fun_dim=T_in,
        out_dim=1,
        hidden_dim=int(args.hidden),
        num_layers=int(args.layers),
        num_heads=int(args.heads),
        patch_size=int(args.patch_size),
        hyp_dim=int(args.hyp_dim),
        mlp_ratio=float(args.mlp_ratio),
        dropout=float(args.dropout),
        H=h,
        W=h,
        use_temporal_conv=bool(args.use_temporal_conv),
   ).to(device)

    print(f"[Navier-Stokes] params={count_parameters(model):,} ({count_parameters(model)/1e6:.2f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=float(args.lr),
        epochs=int(args.epochs),
        steps_per_epoch=len(train_loader),
        final_div_factor=10000.0,
   )

    loss_fn = TestLoss(size_average=False)
    best_full = float("inf")
    best_path = ckpt_dir / "hno_navier_stokes_best.pt"

    for epoch in range(int(args.epochs)):
        model.train()
        train_step = 0.0
        train_full = 0.0

        for x, fx, yy in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            x = x.to(device)
            fx = fx.to(device)
            yy = yy.to(device)
            bsz = x.shape[0]

            optimizer.zero_grad(set_to_none=True)
            loss = 0.0
            for t in range(0, T, step):
                y = yy[..., t : t + step]
                im = model(x, fx=fx)  # (B, N, 1)
                loss = loss + loss_fn(im.reshape(bsz, -1), y.reshape(bsz, -1))
                fx = torch.cat((fx[..., step:], y), dim=-1)  # teacher forcing
                pred = im if t == 0 else torch.cat((pred, im), dim=-1)

            train_step += float(loss.item())
            train_full += float(loss_fn(pred.reshape(bsz, -1), yy.reshape(bsz, -1)).item())

            loss.backward()
            if args.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
            optimizer.step()
            scheduler.step()

        train_step /= (ntrain_eff * (T / step))
        train_full /= ntrain_eff

        model.eval()
        test_step = 0.0
        test_full = 0.0
        with torch.no_grad():
            for x, fx, yy in test_loader:
                x = x.to(device)
                fx = fx.to(device)
                yy = yy.to(device)
                bsz = x.shape[0]
                loss = 0.0
                for t in range(0, T, step):
                    y = yy[..., t : t + step]
                    im = model(x, fx=fx)
                    loss = loss + loss_fn(im.reshape(bsz, -1), y.reshape(bsz, -1))
                    fx = torch.cat((fx[..., step:], im), dim=-1)  # autoregressive
                    pred = im if t == 0 else torch.cat((pred, im), dim=-1)

                test_step += float(loss.item())
                test_full += float(loss_fn(pred.reshape(bsz, -1), yy.reshape(bsz, -1)).item())

        test_step /= (ntest_eff * (T / step))
        test_full /= ntest_eff
        print(
            f"epoch={epoch:04d} train_step={train_step:.5f} train_full={train_full:.5f} "
            f"test_step={test_step:.5f} test_full={test_full:.5f}"
       )

        if test_full < best_full:
            best_full = test_full
            torch.save(model.state_dict(), best_path)

    print(f"[Navier-Stokes] best_test_full={best_full:.5f} ckpt={best_path}")


if __name__ == "__main__":
    main()
