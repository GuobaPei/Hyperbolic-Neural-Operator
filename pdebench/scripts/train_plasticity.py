import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import scipy.io as scio
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from pdebench.hno.hno_patch_time import build_hno
from pdebench.utils.normalizer import UnitTransformer
from pdebench.utils.testloss import TestLoss


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def random_time_shuffle_collate(batch):
    pos_list, t_list, a_list, u_list = [], [], [], []
    for pos, t, a, u in batch:
        # Shuffle time steps independently per sample.
        perm = torch.randperm(t.size(0))
        t = t[perm]
        u = u[..., perm]
        pos_list.append(pos)
        t_list.append(t)
        a_list.append(a)
        u_list.append(u)
    return (
        torch.stack(pos_list, dim=0),
        torch.stack(t_list, dim=0),
        torch.stack(a_list, dim=0),
        torch.stack(u_list, dim=0),
   )


def select_metric(pred: torch.Tensor, gt: torch.Tensor, metric: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if metric == "disp":
        return pred[..., 2:], gt[..., 2:]
    return pred, gt


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    default_data_path = repo_root / "data" / "pdebench" / "plasticity" / "plas_N987_T20.mat"
    default_out_dir = repo_root / "outputs" / "pdebench" / "plasticity"

    parser = argparse.ArgumentParser("Train HNO on PDEBench Plasticity")
    parser.add_argument("--data_path", type=str, default=str(default_data_path))
    parser.add_argument("--out_dir", type=str, default=str(default_out_dir))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_grad_norm", type=float, default=0.1)
    parser.add_argument("--update_per_t", type=int, default=1, help="1: optimizer step per time step (baseline). 0: accumulate over time then step once.")
    parser.add_argument("--scheduler_step_mode", type=str, default="per_batch", choices=["per_batch", "per_update"])

    parser.add_argument("--metric", type=str, default="all", choices=["all", "disp"])

    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--patch_size", type=int, default=3)
    parser.add_argument("--hyp_dim", type=int, default=16)
    parser.add_argument("--mlp_ratio", type=float, default=2.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--local_expansion", type=float, default=2.0)
    parser.add_argument("--use_attn_conv", type=int, default=1)
    parser.add_argument("--time_embed", type=str, default="scalar", choices=["scalar", "timestep", "none"])

    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Plasticity.mat not found: {data_path}")

    # Dataset constants.
    ntrain = 900
    ntest = 80
    s1 = 101
    s2 = 31
    T = 20
    deformation = 4

    data = scio.loadmat(str(data_path))
    inp = torch.tensor(data["input"], dtype=torch.float)  # (987, 101)
    out = torch.tensor(data["output"], dtype=torch.float).transpose(-2, -1)  # (987, 101, 31, 4, 20) -> ?

    x_train = inp[:ntrain, :s1].reshape(ntrain, s1, 1).repeat(1, 1, s2).reshape(ntrain, -1, 1)
    y_train = out[:ntrain, :s1, :s2].reshape(ntrain, -1, deformation, T)
    x_test = inp[-ntest:, :s1].reshape(ntest, s1, 1).repeat(1, 1, s2).reshape(ntest, -1, 1)
    y_test = out[-ntest:, :s1, :s2].reshape(ntest, -1, deformation, T)

    # Normalize input parameter field (x_train/x_test); output stays in raw scale.
    x_normalizer = UnitTransformer(x_train)
    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)
    x_normalizer.to(device)

    # Build 2D grid coordinates (must match the flatten order used above).
    gx = np.linspace(0, 1, s1)
    gy = np.linspace(0, 1, s2)
    xx, yy = np.meshgrid(gx, gy, indexing="ij")  # (s1, s2)
    pos = torch.tensor(np.c_[xx.reshape(-1), yy.reshape(-1)], dtype=torch.float).unsqueeze(0)  # (1, N, 2)
    pos_train = pos.repeat(ntrain, 1, 1)
    pos_test = pos.repeat(ntest, 1, 1)

    t = torch.tensor(np.linspace(0, 1, T), dtype=torch.float).unsqueeze(0)
    t_train = t.repeat(ntrain, 1)
    t_test = t.repeat(ntest, 1)

    train_loader = DataLoader(
        TensorDataset(pos_train, t_train, x_train, y_train),
        batch_size=int(args.batch_size),
        shuffle=True,
        collate_fn=random_time_shuffle_collate,
   )
    test_loader = DataLoader(
        TensorDataset(pos_test, t_test, x_test, y_test),
        batch_size=int(args.batch_size),
        shuffle=False,
   )

    model = build_hno(
        space_dim=2,
        fun_dim=1,
        out_dim=deformation,
        hidden_dim=int(args.hidden),
        num_layers=int(args.layers),
        num_heads=int(args.heads),
        patch_size=int(args.patch_size),
        hyp_dim=int(args.hyp_dim),
        mlp_ratio=float(args.mlp_ratio),
        dropout=float(args.dropout),
        H=s1,
        W=s2,
        local_expansion=float(args.local_expansion),
        use_attn_conv=bool(args.use_attn_conv),
        time_embed=str(args.time_embed),
   ).to(device)

    print(f"[Plasticity] params={count_parameters(model):,} ({count_parameters(model)/1e6:.2f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    steps_per_epoch = len(train_loader) * (T if (args.update_per_t and args.scheduler_step_mode == "per_update") else 1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=float(args.lr),
        epochs=int(args.epochs),
        steps_per_epoch=steps_per_epoch,
   )

    loss_fn = TestLoss(size_average=False)
    best_full = float("inf")
    best_path = ckpt_dir / "hno_plasticity_best.pt"

    for epoch in range(int(args.epochs)):
        model.train()
        train_step = 0.0

        for pos_b, t_b, fx_b, yy_b in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            pos_b = pos_b.to(device)
            t_b = t_b.to(device)
            fx_b = fx_b.to(device)
            yy_b = yy_b.to(device)
            bsz = pos_b.shape[0]

            if args.update_per_t:
                for ti in range(T):
                    target = yy_b[..., ti]
                    time_input = t_b[:, ti : ti + 1]
                    pred = model(pos_b, fx_b, T=time_input)
                    pred_m, target_m = select_metric(pred, target, args.metric)
                    loss = loss_fn(pred_m.reshape(bsz, -1), target_m.reshape(bsz, -1))
                    train_step += float(loss.item())

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if args.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
                    optimizer.step()
                    if args.scheduler_step_mode == "per_update":
                        scheduler.step()

                if args.scheduler_step_mode == "per_batch":
                    scheduler.step()
            else:
                optimizer.zero_grad(set_to_none=True)
                total_loss = 0.0
                for ti in range(T):
                    target = yy_b[..., ti]
                    time_input = t_b[:, ti : ti + 1]
                    pred = model(pos_b, fx_b, T=time_input)
                    pred_m, target_m = select_metric(pred, target, args.metric)
                    total_loss = total_loss + loss_fn(pred_m.reshape(bsz, -1), target_m.reshape(bsz, -1))
                total_loss = total_loss / T
                train_step += float(total_loss.item())
                total_loss.backward()
                if args.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
                optimizer.step()
                scheduler.step()

        model.eval()
        test_step = 0.0
        test_full = 0.0
        with torch.no_grad():
            for pos_b, t_b, fx_b, yy_b in test_loader:
                pos_b = pos_b.to(device)
                t_b = t_b.to(device)
                fx_b = fx_b.to(device)
                yy_b = yy_b.to(device)
                bsz = pos_b.shape[0]

                step_loss = 0.0
                for ti in range(T):
                    target = yy_b[..., ti]
                    time_input = t_b[:, ti : ti + 1]
                    pred = model(pos_b, fx_b, T=time_input)
                    pred_m, target_m = select_metric(pred, target, args.metric)
                    step_loss = step_loss + loss_fn(pred_m.reshape(bsz, -1), target_m.reshape(bsz, -1))
                    pred_all = pred.unsqueeze(-1) if ti == 0 else torch.cat((pred_all, pred.unsqueeze(-1)), dim=-1)

                test_step += float(step_loss.item())
                pred_m, yy_m = select_metric(pred_all, yy_b, args.metric)
                test_full += float(loss_fn(pred_m.reshape(bsz, -1), yy_m.reshape(bsz, -1)).item())

        train_norm = T if args.update_per_t else 1
        test_step_avg = test_step / (ntest * T)
        test_full_avg = test_full / ntest
        print(
            f"epoch={epoch:04d} train_step={train_step/(ntrain*train_norm):.5f} "
            f"test_step={test_step_avg:.5f} test_full={test_full_avg:.5f}"
       )

        if test_full_avg < best_full:
            best_full = test_full_avg
            torch.save(model.state_dict(), best_path)

    print(f"[Plasticity] best_test_full={best_full:.5f} ckpt={best_path}")


if __name__ == "__main__":
    main()
