import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from pdebench.hno.hno_latent_set import build_hno
from pdebench.utils.normalizer import UnitTransformer
from pdebench.utils.testloss import TestLoss


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    default_data_path = repo_root / "data" / "pdebench" / "elasticity"
    default_out_dir = repo_root / "outputs" / "pdebench" / "elasticity"

    parser = argparse.ArgumentParser("Train HNO on PDEBench Elasticity (point cloud)")
    parser.add_argument("--data_path", type=str, default=str(default_data_path))
    parser.add_argument("--out_dir", type=str, default=str(default_out_dir))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--ntrain", type=int, default=1000)

    parser.add_argument("--hidden", type=int, default=384)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--num_latents", type=int, default=96)
    parser.add_argument("--hyp_dim", type=int, default=16)
    parser.add_argument("--mlp_ratio", type=float, default=2.5)
    parser.add_argument("--dropout", type=float, default=0.0)
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data_path)
    sigma_path = data_path / "Meshes" / "Random_UnitCell_sigma_10.npy"
    xy_path = data_path / "Meshes" / "Random_UnitCell_XY_10.npy"
    if not sigma_path.exists() or not xy_path.exists():
        # Legacy layout: data_path/elasticity/Meshes/...
        sigma_path = data_path / "elasticity" / "Meshes" / "Random_UnitCell_sigma_10.npy"
        xy_path = data_path / "elasticity" / "Meshes" / "Random_UnitCell_XY_10.npy"
    if not sigma_path.exists() or not xy_path.exists():
        raise FileNotFoundError(
            "Elasticity.npy files not found. Expected:\n"
            f"- {sigma_path}\n"
            f"- {xy_path}\n"
            "Pass --data_path to the directory containing Meshes/."
       )

    ntrain = int(args.ntrain)
    ntest = 200

    sigma = torch.tensor(np.load(sigma_path), dtype=torch.float).permute(1, 0)  # (1200, N)
    xy = torch.tensor(np.load(xy_path), dtype=torch.float).permute(2, 0, 1)  # (1200, N, 2)

    train_y = sigma[:ntrain]
    test_y = sigma[-ntest:]
    train_x = xy[:ntrain]
    test_x = xy[-ntest:]

    y_normalizer = UnitTransformer(train_y)
    train_y = y_normalizer.encode(train_y)
    y_normalizer.to(device)

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=int(args.batch_size), shuffle=True)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=int(args.batch_size), shuffle=False)

    model = build_hno(
        space_dim=2,
        fun_dim=0,
        out_dim=1,
        hidden_dim=int(args.hidden),
        num_layers=int(args.layers),
        num_heads=int(args.heads),
        num_latents=int(args.num_latents),
        hyp_dim=int(args.hyp_dim),
        mlp_ratio=float(args.mlp_ratio),
        dropout=float(args.dropout),
   ).to(device)

    print(f"[Elasticity] params={count_parameters(model):,} ({count_parameters(model)/1e6:.2f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(args.epochs))

    rel_loss = TestLoss(size_average=False)
    best_err = float("inf")
    best_path = ckpt_dir / "hno_elasticity_best.pt"

    for epoch in range(int(args.epochs)):
        model.train()
        train_loss = 0.0
        for pos, y in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            pos = pos.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(pos, fx=None).squeeze(-1)
            out = y_normalizer.decode(out)
            y_dec = y_normalizer.decode(y)
            loss = rel_loss(out, y_dec)
            loss.backward()
            if args.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
            optimizer.step()
            train_loss += float(loss.item())
        scheduler.step()

        train_loss /= ntrain

        model.eval()
        rel_err = 0.0
        with torch.no_grad():
            for pos, y in test_loader:
                pos = pos.to(device)
                y = y.to(device)
                out = model(pos, fx=None).squeeze(-1)
                out = y_normalizer.decode(out)
                rel_err += float(rel_loss(out, y).item())
        rel_err /= ntest

        print(f"epoch={epoch:04d} train_rel={train_loss:.6f} test_rel={rel_err:.6f}")
        if rel_err < best_err:
            best_err = rel_err
            torch.save(model.state_dict(), best_path)

    print(f"[Elasticity] best_test_rel={best_err:.6f} ckpt={best_path}")


if __name__ == "__main__":
    main()

