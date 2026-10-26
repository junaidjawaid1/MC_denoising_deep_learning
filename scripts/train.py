"""Train the U-Net with the combined WMSE + differentiable gamma loss (Eq. 11).

Example:
    python scripts/train.py --data-root /path/to/dataset_2mm --output-dir runs
"""
import argparse
import csv
import datetime
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dose_denoising import (BetaScheduler, CombinedWMSEGammaLoss, DoseDataset, DoseDenoisingUNet,  # noqa: E402
                            GammaIndexLoss, WeightedMSE, max_normalize)
from dose_denoising.config import load_config, split_dir, voxel_size  # noqa: E402

LOG_FIELDS = ["epoch", "beta", "lr", "train_loss", "train_gpr", "train_wmse", "train_gamma",
              "val_loss", "val_gpr", "val_wmse", "val_gamma"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config.yaml"),
                        help="YAML config (default: config.yaml in the repository)")
    parser.add_argument("--data-root", help="dataset directory with train/ and validation/ (overrides data.root)")
    parser.add_argument("--output-dir", default="runs", help="a timestamped run directory is created here")
    parser.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    parser.add_argument("--wandb-project", default="proton-dose-denoising")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_grad_scaler():
    try:
        return torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):  # PyTorch < 2.3
        return torch.cuda.amp.GradScaler()


def make_loader(dataset, batch_size, shuffle, num_workers):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      prefetch_factor=4 if num_workers > 0 else None, pin_memory=True)


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, desc=""):
    """One pass over ``loader``; trains if ``optimizer`` is given. Returns sample-averaged metrics."""
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "gpr": 0.0, "wmse": 0.0, "gamma": 0.0}

    with torch.set_grad_enabled(training):
        for x, y in tqdm(loader, desc=desc, leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)  # already max-normalized to [0, 1]

            with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16):
                output = max_normalize(model(x))
                loss, parts = criterion(output, y)
                gpr = criterion.compute_pass_rate(output.detach(), y)  # hard GPR (%), no gradient

            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            n = x.size(0)
            totals["loss"] += loss.item() * n
            totals["gpr"] += gpr * n
            totals["wmse"] += parts["wmse"] * n
            totals["gamma"] += parts["gamma"] * n

    torch.cuda.empty_cache()
    n_samples = len(loader.dataset)
    return {k: v / n_samples for k, v in totals.items()}


def main():
    args = parse_args()
    cfg = load_config(args.config, args.data_root)
    train_cfg, loss_cfg = cfg["training"], cfg["loss"]
    if train_cfg.get("seed") is not None:
        seed_everything(train_cfg["seed"])

    if not torch.cuda.is_available():
        sys.exit("Training requires a CUDA GPU with bfloat16 support (the paper used an NVIDIA A100).")
    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    run_dir = Path(args.output_dir) / f"{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}"
    run_dir.mkdir(parents=True)
    shutil.copy(args.config, run_dir / "config.yaml")
    checkpoint_path = run_dir / "best_model.pth"
    log_path = run_dir / "training_log.csv"

    # Data
    train_set = DoseDataset(split_dir(cfg, "train"))
    val_set = DoseDataset(split_dir(cfg, "validation"))
    print(f"Train: {len(train_set.sample_ids)} volumes, validation: {len(val_set.sample_ids)} volumes "
          f"(x4 rotations each)")
    train_loader = make_loader(train_set, train_cfg["batch_size"], True, train_cfg["num_workers"])
    val_loader = make_loader(val_set, train_cfg["val_batch_size"], False, train_cfg["num_workers"])

    # Model, losses, optimization
    model = DoseDenoisingUNet().to(device)
    vs = voxel_size(cfg)
    gamma_loss = GammaIndexLoss(dose_percent=loss_cfg["dose_percent"], dta_mm=loss_cfg["dta_mm"],
                                voxel_size_mm=(vs, vs, vs), dose_cutoff=loss_cfg["dose_cutoff"],
                                beta_init=cfg["beta_schedule"]["beta_start"],
                                max_gamma=loss_cfg["max_gamma"]).to(device)
    criterion = CombinedWMSEGammaLoss(WeightedMSE(alpha=loss_cfg["wmse_alpha"]), gamma_loss,
                                      wmse_weight=loss_cfg["wmse_weight"], gamma_weight=loss_cfg["gamma_weight"],
                                      momentum=loss_cfg["ema_momentum"]).to(device)
    beta_scheduler = BetaScheduler(gamma_loss, **cfg["beta_schedule"])
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=train_cfg["lr_factor"], patience=train_cfg["lr_patience"],
        min_lr=train_cfg["min_learning_rate"])
    scaler = make_grad_scaler()

    wandb = None
    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=run_dir.name, config=cfg)

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(LOG_FIELDS)

    best_val_gpr = 0.0
    epochs = train_cfg["epochs"]
    for epoch in range(epochs):
        beta = beta_scheduler.step(epoch)
        train = run_epoch(model, train_loader, criterion, device, optimizer, scaler, desc=f"Epoch {epoch + 1}/{epochs}")
        val = run_epoch(model, val_loader, criterion, device, desc="Validation")

        lr = optimizer.param_groups[0]["lr"]
        row = {"epoch": epoch + 1, "beta": beta, "lr": lr,
               **{f"train_{k}": v for k, v in train.items()}, **{f"val_{k}": v for k, v in val.items()}}
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=LOG_FIELDS).writerow(row)
        if wandb:
            wandb.log({k.replace("_", "/", 1) if k.startswith(("train_", "val_")) else k: v for k, v in row.items()})

        print(f"Epoch {epoch + 1}/{epochs} | beta {beta:.3f} | lr {lr:.2e} | "
              f"train loss {train['loss']:.5f} GPR {train['gpr']:.2f}% | "
              f"val loss {val['loss']:.5f} GPR {val['gpr']:.2f}%")

        if val["gpr"] > best_val_gpr:
            best_val_gpr = val["gpr"]
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  saved best model (validation GPR {best_val_gpr:.2f}%) to {checkpoint_path}")

        lr_scheduler.step(val["gpr"])

    print(f"Training complete. Best validation GPR: {best_val_gpr:.2f}%. Run directory: {run_dir}")
    if wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
