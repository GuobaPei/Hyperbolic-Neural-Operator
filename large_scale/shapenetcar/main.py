import argparse
from pathlib import Path

import torch


def main() -> None:
    root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser("ShapeNetCar training (HNO)")
    parser.add_argument("--data_dir", type=str, required=True, help="Raw training data root.")
    parser.add_argument("--save_dir", type=str, required=True, help="Preprocessed data root (created if needed).")
    parser.add_argument("--output_dir", type=str, default=str(root / "outputs"))
    parser.add_argument("--fold_id", type=int, default=0)
    parser.add_argument("--preprocessed", type=int, default=1)

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--val_iter", type=int, default=10)
    parser.add_argument("--cfd_mesh", action="store_true")
    parser.add_argument("--r", type=float, default=0.2)

    parser.add_argument("--weight", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--nb_epochs", type=int, default=200)

    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--mlp_ratio", type=float, default=2.0)
    parser.add_argument("--slice_num", type=int, default=64)
    parser.add_argument("--hyp_dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.0)
    args = parser.parse_args()

    try:
        import train
        from dataset.load_dataset import load_train_val_fold
        from dataset.dataset import GraphDataset
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"{e}\n\n"
            "ShapeNetCar requires optional dependencies (e.g., torch_geometric, vtk). "
            "Install them with:\n"
            "  pip install -r requirements_large_scale.txt\n"
            "or run:\n"
            "  bash scripts/setup_large_scale_env.sh"
       ) from e

    hparams = {"lr": float(args.lr), "batch_size": int(args.batch_size), "nb_epochs": int(args.nb_epochs)}
    device = args.device

    train_data, val_data, coef_norm = load_train_val_fold(args, preprocessed=int(args.preprocessed))
    train_ds = GraphDataset(train_data, use_cfd_mesh=bool(args.cfd_mesh), r=float(args.r))
    val_ds = GraphDataset(val_data, use_cfd_mesh=bool(args.cfd_mesh), r=float(args.r))

    from models.hno import HNO

    model = HNO(
        n_hidden=int(args.hidden),
        n_layers=int(args.layers),
        space_dim=7,
        fun_dim=0,
        n_head=int(args.heads),
        mlp_ratio=float(args.mlp_ratio),
        out_dim=4,
        slice_num=int(args.slice_num),
        hyp_dim=int(args.hyp_dim),
        unified_pos=0,
        dropout=float(args.dropout),
   ).to(device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = out_dir / "hno" / f"fold_{int(args.fold_id)}" / f"{int(args.nb_epochs)}_{float(args.weight)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    _ = train.main(
        device,
        train_ds,
        val_ds,
        model,
        hparams,
        str(run_dir),
        val_iter=int(args.val_iter),
        reg=float(args.weight),
        coef_norm=coef_norm,
   )


if __name__ == "__main__":
    main()
