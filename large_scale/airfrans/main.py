import argparse
import json
import os.path as osp
from pathlib import Path

import torch


def main() -> None:
    root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser("AirfRANS training (HNO)")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing manifest.json and CFD data.")
    parser.add_argument("--save_dir", type=str, default=str(root / "outputs"))
    parser.add_argument("--task", type=str, default="full", choices=["full", "scarce", "reynolds", "aoa"])
    parser.add_argument("--num_runs", type=int, default=1)

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--nb_epochs", type=int, default=200)
    parser.add_argument("--val_iter", type=int, default=10)
    parser.add_argument("--weight", type=float, default=1.0, help="Weight on surface loss for MSE_weighted.")

    # Model architecture knobs (kept explicit for reproducibility).
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--mlp_ratio", type=float, default=2.0)
    parser.add_argument("--slice_num", type=int, default=64)
    parser.add_argument("--hyp_dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--unified_pos", type=int, default=1)
    args = parser.parse_args()

    try:
        import train
        from dataset.dataset import Dataset
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"{e}\n\n"
            "AirfRANS requires optional dependencies (e.g., torch_geometric, pyvista). "
            "Install them with:\n"
            "  pip install -r requirements_large_scale.txt\n"
            "or run:\n"
            "  bash scripts/setup_large_scale_env.sh"
       ) from e

    data_dir = Path(args.data_dir)
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found: {manifest_path}")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    train_ids = manifest[f"{args.task}_train"]
    n_val = int(0.1 * len(train_ids))
    train_split = train_ids[:-n_val]
    val_split = train_ids[-n_val:]

    print("Loading AirfRANS data...")
    train_dataset, coef_norm = Dataset(train_split, norm=True, sample=None, my_path=str(data_dir))
    val_dataset = Dataset(val_split, sample=None, coef_norm=coef_norm, my_path=str(data_dir))

    device = args.device
    hparams = {"lr": float(args.lr), "batch_size": int(args.batch_size), "nb_epochs": int(args.nb_epochs)}

    save_root = Path(args.save_dir)
    save_root.mkdir(parents=True, exist_ok=True)

    for run_idx in range(int(args.num_runs)):
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
            unified_pos=int(args.unified_pos),
            dropout=float(args.dropout),
       ).to(device)

        run_name = f"hno/run_{run_idx}"
        log_path = osp.join(str(save_root), args.task, run_name)
        print(f"Training HNO -> {log_path}")

        _ = train.main(
            device,
            train_dataset,
            val_dataset,
            model,
            hparams,
            log_path,
            criterion="MSE_weighted",
            reg=float(args.weight),
            val_iter=int(args.val_iter),
            name_mod="hno",
            val_sample=True,
       )


if __name__ == "__main__":
    main()
