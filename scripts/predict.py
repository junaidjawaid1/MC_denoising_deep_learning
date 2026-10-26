"""Denoise one 5e3-history dose and save it in absolute units (Eq. 12).

Example:
    python scripts/predict.py --checkpoint best_model.pth --ct CT_001.npy --dose-5k 001_5k.npy \
        --output 001_denoised.npy
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dose_denoising import DoseDenoisingUNet, max_normalize  # noqa: E402
from dose_denoising.data import min_max, normalize_ct  # noqa: E402
from dose_denoising.evaluation import absolute_dose  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ct", required=True, help="CT in Hounsfield units, [D, H, W] .npy")
    parser.add_argument("--dose-5k", required=True, help="5e3-history dose on the 1e6 scale, [D, H, W] .npy")
    parser.add_argument("--output", required=True, help="output .npy (absolute dose)")
    parser.add_argument("--history-ratio", type=float, default=1.0,
                        help="multiply by this if the 5e3-history dose was not already scaled by 200")
    parser.add_argument("--abs-mask-fraction", type=float, default=0.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ct = np.load(args.ct).astype(np.float32)
    dose_5k = np.load(args.dose_5k).astype(np.float32)
    if ct.shape != dose_5k.shape or any(n % 8 for n in ct.shape):
        sys.exit(f"CT {ct.shape} and dose {dose_5k.shape} must have the same shape, "
                 "with each dimension a multiple of 8")

    model = DoseDenoisingUNet().to(args.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model.eval()

    x = torch.from_numpy(np.stack((normalize_ct(ct), min_max(dose_5k)))[np.newaxis]).to(args.device)
    with torch.no_grad():
        pred = max_normalize(model(x).float())[0, 0].cpu().numpy()
    dose, scale = absolute_dose(pred, dose_5k.astype(np.float64), args.history_ratio, args.abs_mask_fraction)
    np.save(args.output, dose.astype(np.float32))
    print(f"Saved {args.output} (scale {scale:.4g}, max dose {dose.max():.4g})")


if __name__ == "__main__":
    main()
