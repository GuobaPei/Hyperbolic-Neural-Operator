import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from pdebench.hno.hno_patch import build_hno
from pdebench.utils.normalizer import UnitTransformer
from pdebench.utils.testloss import TestLoss


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    default_data_path = repo_root / "data" / "pdebench" / "pipe"
    default_out_dir = repo_root / "outputs" / "pdebench" / "pipe"

    parser = argparse.ArgumentParser("Train HNO on PDEBench Pipe")
    parser.add_argument("--data_path", type=str, default=str(default_data_path))
    parser.add_argument("--out_dir", type=str, default=str(default_out_dir))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--downsamplex", type=int, default=1)
    parser.add_argument("--downsampley", type=int, default=1)
    parser.add_argument("--train_frac", type=float, default=1.0)
    parser.add_argument("--test_frac", type=float, default=1.0)
    parser.add_argument("--subset_seed", type=int, default=0)

    parser.add_argument("--hidden", type=int, default=132)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=3)
    parser.add_argument("--hyp_dim", type=int, default=16)
    parser.add_argument("--mlp_ratio", type=float, default=2.5)
    parser.add_argument("--dropout", type=float, default=0.0)
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data_path)
    x_path = data_path / "Pipe_X.npy"
    y_path = data_path / "Pipe_Y.npy"
    q_path = data_path / "Pipe_Q.npy"
    if not x_path.exists() or not y_path.exists() or not q_path.exists():
        raise FileNotFoundError(
            "Pipe.npy files not found. Expected:\n"
            f"- {x_path}\n"
            f"- {y_path}\n"
            f"- {q_path}\n"
            "Pass --data_path to the directory containing these files."
       )

    ntrain = 1000
    ntest = 200
    N = 1200

    r1 = int(args.downsamplex)
    r2 = int(args.downsampley)
    s1 = int(((129 - 1) / r1) + 1)
    s2 = int(((129 - 1) / r2) + 1)

    input_x = torch.tensor(np.load(x_path), dtype=torch.float)
    input_y = torch.tensor(np.load(y_path), dtype=torch.float)
    coords = torch.stack([input_x, input_y], dim=-1)  # (N, 129, 129, 2)
    target = torch.tensor(np.load(q_path)[:, 0], dtype=torch.float)  # (N, 129, 129)

    x_train = coords[:N][:ntrain, ::r1, ::r2][:, :s1, :s2].reshape(ntrain, -1, 2)
    y_train = target[:N][:ntrain, ::r1, ::r2][:, :s1, :s2].reshape(ntrain, -1)
    x_test = coords[:N][-ntest:, ::r1, ::r2][:, :s1, :s2].reshape(ntest, -1, 2)
    y_test = target[:N][-ntest:, ::r1, ::r2][:, :s1, :s2].reshape(ntest, -1)

    def _subset_indices(n: int, frac: float, seed: int) -> torch.Tensor:
        frac = float(frac)
        if frac >= 1.0:
            return torch.arange(n)
        k = max(1, int(np.ceil(n * frac)))
        rng = np.random.RandomState(int(seed))
        return torch.from_numpy(rng.choice(n, size=k, replace=False))

    train_idx = _subset_indices(ntrain, args.train_frac, args.subset_seed)
    test_idx = _subset_indices(ntest, args.test_frac, args.subset_seed + 1)
    x_train, y_train = x_train[train_idx], y_train[train_idx]
    x_test, y_test = x_test[test_idx], y_test[test_idx]
    ntrain, ntest = x_train.shape[0], x_test.shape[0]

    x_normalizer = UnitTransformer(x_train)
    y_normalizer = UnitTransformer(y_train)
    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)
    y_train = y_normalizer.encode(y_train)
    x_normalizer.to(device)
    y_normalizer.to(device)

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

    print(f"[Pipe] params={count_parameters(model):,} ({count_parameters(model)/1e6:.2f}M)")

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
    best_path = ckpt_dir / "hno_pipe_best.pt"

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
                out = y_normalizer.decode(out)
                rel_err += float(rel_loss(out, y).item())
        rel_err /= ntest

        print(f"epoch={epoch:04d} train_rel={train_loss:.6f} test_rel={rel_err:.6f}")
        if rel_err < best_err:
            best_err = rel_err
            torch.save(model.state_dict(), best_path)

    print(f"[Pipe] best_test_rel={best_err:.6f} ckpt={best_path}")


if __name__ == "__main__":
    main()

