"""Inference time and peak GPU memory per volume (Table 4 of the paper).

Forward pass + max normalization, batch size 1, bfloat16 autocast, timed with CUDA events after a warm-up.
Weights do not affect the timing, so the checkpoint is optional.

Example:
    python scripts/benchmark_inference.py
"""
import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dose_denoising import DoseDenoisingUNet, max_normalize  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config.yaml"),
                        help="YAML config (default: config.yaml in the repository)")
    parser.add_argument("--checkpoint")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()

    with open(args.config) as f:
        shape = tuple(yaml.safe_load(f)["data"]["input_shape"])
    if not torch.cuda.is_available():
        sys.exit("A CUDA GPU is required.")

    model = DoseDenoisingUNet().cuda().eval()
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cuda"))
    x = torch.rand(1, 2, *shape, device="cuda")

    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        max_normalize(model(x))
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    reserved_gb = torch.cuda.memory_reserved() / 1e9

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times_ms = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for _ in range(args.warmup):
            max_normalize(model(x))
        torch.cuda.synchronize()
        for _ in range(args.repeats):
            start.record()
            max_normalize(model(x))
            end.record()
            torch.cuda.synchronize()
            times_ms.append(start.elapsed_time(end))
    t = np.array(times_ms)

    print(f"GPU: {torch.cuda.get_device_name(0)}, PyTorch {torch.__version__}")
    print(f"Input: 1 x 2 x {shape}, bf16 autocast, {args.warmup} warm-up + {args.repeats} timed passes")
    print(f"Time per volume: mean {t.mean():.2f} ms, median {np.median(t):.2f}, min {t.min():.2f}, std {t.std():.2f}")
    print(f"Peak GPU memory: {peak_gb:.2f} GB allocated ({reserved_gb:.2f} GB reserved)")


if __name__ == "__main__":
    main()
