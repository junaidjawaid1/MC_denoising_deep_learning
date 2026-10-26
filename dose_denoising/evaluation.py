"""Test-time utilities: absolute dose recovery (Eq. 12) and direct / reverse gamma tests with PyMedPhys."""
import numpy as np

DIRECTIONS = ("direct", "reverse")


def absolute_dose(pred_norm, dose_5k, history_ratio=1.0, mask_fraction=0.10):
    """Rescale a max-normalized prediction to absolute dose using only the 5e3-history input (Eq. 12).

    s = history_ratio * sum(D_5k[m]) / sum(p[m]),  m = voxels where p >= mask_fraction * max(p).

    ``history_ratio`` is 1 because the 5e3-history doses were already multiplied by 200 when the
    dataset was generated. The sums run over the high-dose region of the prediction (``mask_fraction``
    = 0.1, as in the paper's evaluation); use ``mask_fraction=0`` to sum over the whole volume.
    """
    p = pred_norm.astype(np.float64)
    mask = p >= mask_fraction * p.max()
    scale = history_ratio * dose_5k[mask].sum() / p[mask].sum()
    return scale * p, scale


def gamma_pass_rate(evaluated, reference_1m, voxel_size_mm, dose_percent, dta_mm, cutoff=0.2,
                    direction="direct", max_gamma=2.0, interp_fraction=10, ram_available=None):
    """Global gamma passing rate (%) of an absolute dose against the 1e6-history dose.

    direct:  reference = 1e6-history dose, evaluated = ``evaluated``.
    reverse: reference = ``evaluated``, evaluated = 1e6-history dose.

    In both directions the global normalization is the 1e6-history maximum, and the passing rate is
    computed over the voxels where the 1e6-history dose is >= ``cutoff`` of its maximum (and, in the
    reverse test, where PyMedPhys also finds the reference ``evaluated`` above the same cutoff).
    ``max_gamma`` only bounds the search; it does not change the passing rate.
    """
    import pymedphys

    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    evaluated = np.asarray(evaluated, dtype=np.float32)
    reference_1m = np.asarray(reference_1m, dtype=np.float32)
    ref_max = float(reference_1m.max())
    valid_mask = reference_1m >= cutoff * ref_max
    if ref_max <= 0 or not valid_mask.any():
        return np.nan

    axes = tuple(np.arange(n) * voxel_size_mm for n in reference_1m.shape)
    reference, other = (reference_1m, evaluated) if direction == "direct" else (evaluated, reference_1m)
    kwargs = {"ram_available": ram_available} if ram_available else {}
    gamma = pymedphys.gamma(
        axes_reference=axes,
        dose_reference=reference,
        axes_evaluation=axes,
        dose_evaluation=other,
        dose_percent_threshold=dose_percent,
        distance_mm_threshold=dta_mm,
        lower_percent_dose_cutoff=cutoff * 100,
        interp_fraction=interp_fraction,
        max_gamma=max_gamma,
        local_gamma=False,
        global_normalisation=ref_max,
        quiet=True,
        **kwargs,
    )
    values = gamma[valid_mask]
    values = values[~np.isnan(values)]
    if values.size == 0:
        return np.nan
    return float((values < 1.0).sum() / len(values) * 100)


def criterion_name(dose_percent, dta_mm):
    return f"{dose_percent:g}pct_{dta_mm:g}mm"


def summary_stats(values):
    """n, min, max, mean, median and sample std (ddof=1) of the non-NaN values, as in Table 2."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return {"n": 0, "min": np.nan, "max": np.nan, "mean": np.nan, "median": np.nan, "std": np.nan}
    return {"n": int(v.size), "min": v.min(), "max": v.max(), "mean": v.mean(), "median": float(np.median(v)),
            "std": v.std(ddof=1) if v.size > 1 else 0.0}
