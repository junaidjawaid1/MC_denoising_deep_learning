"""Gamma evaluation on absolute doses, direct and reverse (Table 2 of the paper).

Network prediction (every test volume under the four 90-degree rotations):
    python scripts/evaluate.py --data-root /path/to/dataset_2mm \
        --checkpoint runs/.../best_model.pth --output-dir results/2mm_prediction

5e3-history input baseline (one evaluation per volume; the passing rate does not depend on the rotation):
    python scripts/evaluate.py --data-root /path/to/dataset_2mm \
        --baseline --output-dir results/2mm_baseline

Per-sample results are appended to ``results.csv`` as they are computed, so an interrupted run resumes
where it stopped. ``summary.csv`` holds n / min / max / mean / median / std for each criterion and direction.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dose_denoising import DoseDataset, DoseDenoisingUNet, max_normalize  # noqa: E402
from dose_denoising.config import load_config, split_dir, voxel_size  # noqa: E402
from dose_denoising.data import N_ROTATIONS  # noqa: E402
from dose_denoising.evaluation import (absolute_dose, criterion_name, gamma_pass_rate,  # noqa: E402
                                       summary_stats)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config.yaml"),
                        help="YAML config (default: config.yaml in the repository)")
    parser.add_argument("--data-root", help="dataset directory with test/ (overrides data.root)")
    parser.add_argument("--split", default="test")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", help="trained weights (best_model.pth) to evaluate the prediction")
    group.add_argument("--baseline", action="store_true", help="evaluate the raw 5e3-history input instead")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--save-predictions", action="store_true",
                        help="also save each absolute predicted dose as <sample>_rot<k>_pred.npy")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_model(checkpoint, device):
    model = DoseDenoisingUNet().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model.eval()


@torch.no_grad()
def predict(model, x, device):
    """Forward pass in fp32 and max normalization; returns a [D, H, W] numpy array."""
    return max_normalize(model(x.to(device)).float())[0, 0].cpu().numpy()


def read_rows(path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def append_row(path, row):
    new_file = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def gamma_columns(evaluated, reference, cfg):
    """Passing rate for every configured criterion and direction."""
    ev, vs = cfg["evaluation"], voxel_size(cfg)
    ram = int(ev["ram_available_gb"] * 2 ** 30) if ev.get("ram_available_gb") else None
    row = {}
    for dose_percent, dta_mm in ev["criteria"]:
        for direction in ev["directions"]:
            row[f"gpr_{criterion_name(dose_percent, dta_mm)}_{direction}"] = gamma_pass_rate(
                evaluated, reference, vs, dose_percent, dta_mm, cutoff=ev["dose_cutoff"], direction=direction,
                max_gamma=ev["max_gamma"], interp_fraction=ev["interp_fraction"], ram_available=ram)
    return row


def write_summary(rows, output_dir, label):
    columns = [c for c in rows[0] if c.startswith("gpr_")]
    stats = [{"dose": label, "metric": c, **summary_stats([float(r[c]) for r in rows])} for c in columns]
    with open(output_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(stats[0]))
        writer.writeheader()
        writer.writerows(stats)
    print(f"\n{label}: gamma passing rates (%)")
    print(f"{'metric':<26}{'n':>6}{'min':>9}{'max':>9}{'mean':>9}{'median':>9}{'std':>8}")
    for s in stats:
        print(f"{s['metric']:<26}{s['n']:>6}{s['min']:>9.2f}{s['max']:>9.2f}{s['mean']:>9.2f}"
              f"{s['median']:>9.2f}{s['std']:>8.2f}")


def main():
    args = parse_args()
    cfg = load_config(args.config, args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.csv"

    dataset = DoseDataset(split_dir(cfg, args.split), return_raw=True)
    if args.baseline:
        indices = [i * N_ROTATIONS for i in range(len(dataset.sample_ids))]  # unrotated volumes only
    else:
        indices = list(range(len(dataset)))
    done = {int(r["index"]) for r in read_rows(results_path)}
    todo = [i for i in indices if i not in done]
    print(f"{len(indices)} evaluations, {len(done)} already in {results_path}, {len(todo)} to do")

    model = None if args.baseline else load_model(args.checkpoint, args.device)
    ev = cfg["evaluation"]
    loader = DataLoader(Subset(dataset, todo), batch_size=1, shuffle=False, num_workers=ev["num_workers"])

    for index, (x, y, raw) in tqdm(zip(todo, loader), total=len(todo)):
        dose_5k = raw["dose_5k"][0, 0].numpy().astype(np.float64)
        dose_1m = raw["dose_1m"][0, 0].numpy().astype(np.float64)
        row = {"index": index, "sample_id": raw["sample_id"][0], "rotation": int(raw["rotation"][0])}

        if args.baseline:
            evaluated = ev["history_ratio"] * dose_5k
        else:
            pred = predict(model, x, args.device)
            corr = float(np.corrcoef(pred.ravel(), y[0, 0].numpy().ravel())[0, 1])
            if corr < 0.5:
                tqdm.write(f"WARNING {row['sample_id']} rot {row['rotation']}: prediction/reference correlation "
                           f"{corr:.3f}; check the checkpoint and the data path")
            evaluated, scale = absolute_dose(pred, dose_5k, ev["history_ratio"], ev["abs_mask_fraction"])
            row["scale"] = scale
            if args.save_predictions:
                np.save(output_dir / f"{row['sample_id']}_rot{row['rotation']}_pred.npy",
                        evaluated.astype(np.float32))

        row.update(gamma_columns(evaluated, dose_1m, cfg))
        append_row(results_path, row)
        tqdm.write("  " + " | ".join(f"{k} {v:.2f}" for k, v in row.items() if k.startswith("gpr_")))

    rows = read_rows(results_path)
    if rows:
        write_summary(rows, output_dir, "5e3-history input" if args.baseline else "Prediction")


if __name__ == "__main__":
    main()
