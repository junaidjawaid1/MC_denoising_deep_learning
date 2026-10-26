"""Dataset of CT / 5e3-history dose / 1e6-history dose volumes.

Expected layout of each split directory (train/, validation/, test/)::

    <split>/ct/CT_<id>.npy        CT in Hounsfield units
    <split>/dose_5k/<id>_5k.npy   5e3-history dose (already multiplied by 200, i.e. on the 1e6 scale)
    <split>/dose_1m/<id>_1M.npy   1e6-history dose

All arrays have the same shape [D, H, W] (256 x 32 x 256 at 2 mm).
"""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

N_ROTATIONS = 4  # 0, 90, 180, 270 degrees in the X-Z plane (array axes 0 and 2)
ROTATION_AXES = (0, 2)


def find_sample_ids(split_dir):
    """Sorted sample IDs (e.g. '001') that have a 1e6-history dose file in ``split_dir``."""
    files = sorted(Path(split_dir, "dose_1m").glob("*_1M.npy"))
    if not files:
        raise FileNotFoundError(f"No *_1M.npy files in {Path(split_dir, 'dose_1m')}")
    return [f.name[: -len("_1M.npy")] for f in files]


def normalize_ct(ct):
    """Eq. 1: (HU + 1000) / 3000, air -> 0, dense bone (~3000 HU) -> ~1."""
    return (ct + 1000.0) / 3000.0


def min_max(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def rotate(volume, k):
    return np.rot90(volume, k=k, axes=ROTATION_AXES).copy() if k else volume


class DoseDataset(Dataset):
    """Each volume appears 4 times, rotated by k * 90 degrees in the X-Z plane (k = index % 4).

    Returns ``X`` [2, D, H, W] (normalized CT, min-max normalized 5e3 dose) and ``Y`` [1, D, H, W]
    (min-max normalized 1e6 dose). With ``return_raw=True`` it also returns a dict with the
    unnormalized, identically rotated doses, the sample ID and the rotation index.
    """

    def __init__(self, split_dir, sample_ids=None, return_raw=False):
        self.split_dir = Path(split_dir)
        self.sample_ids = list(sample_ids) if sample_ids is not None else find_sample_ids(split_dir)
        self.return_raw = return_raw

    def __len__(self):
        return len(self.sample_ids) * N_ROTATIONS

    def load(self, sample_id):
        """Raw CT, 5e3 and 1e6 doses (float32) of one sample."""
        ct = np.load(self.split_dir / "ct" / f"CT_{sample_id}.npy").astype(np.float32)
        dose_5k = np.load(self.split_dir / "dose_5k" / f"{sample_id}_5k.npy").astype(np.float32)
        dose_1m = np.load(self.split_dir / "dose_1m" / f"{sample_id}_1M.npy").astype(np.float32)
        return ct, dose_5k, dose_1m

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx // N_ROTATIONS]
        k = idx % N_ROTATIONS
        ct, dose_5k, dose_1m = self.load(sample_id)

        x = np.stack((rotate(normalize_ct(ct), k), rotate(min_max(dose_5k), k)), axis=0)
        y = rotate(min_max(dose_1m), k)[np.newaxis]
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        if not self.return_raw:
            return x, y

        raw = {
            "dose_5k": torch.from_numpy(rotate(dose_5k, k)[np.newaxis]),
            "dose_1m": torch.from_numpy(rotate(dose_1m, k)[np.newaxis]),
            "sample_id": sample_id,
            "rotation": k,
        }
        return x, y, raw
