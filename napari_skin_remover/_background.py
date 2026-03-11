"""
_background.py — Corner-based background processing.

Samples two left-side corners of the stack to estimate the background
intensity ceiling (bg_max). All three modes use a single-sided threshold:

  threshold = bg_max + tolerance

  pixels <= threshold  →  treated as background
  pixels >  threshold  →  treated as signal (kept)

Corner regions sampled (both on left side, X=0..corner_xy):
  Top-left    : Z=0..corner_z,  Y=0..corner_xy,      X=0..corner_xy
  Bottom-left : Z=0..corner_z,  Y=H-corner_xy..H,    X=0..corner_xy

Three processing modes:
  1. remove_outside_brain  — inference-guided: zero background pixels
                             outside the brain mask only
  2. remove_global         — zero all background pixels in the full stack
  3. fill_random_background— replace background pixels with random samples
                             drawn from the corner pixel distribution
"""

import numpy as np


def _sample_corners(volume, corner_xy=50, corner_z=50):
    """
    Return flat array of all pixel values from the two left-side corners.

    Top-left    : Z=0..corner_z,  Y=0..corner_xy,      X=0..corner_xy
    Bottom-left : Z=0..corner_z,  Y=H-corner_xy..H,    X=0..corner_xy
    """
    Z, Y, X = volume.shape
    z_end       = min(corner_z, Z)
    y_end       = min(corner_xy, Y)
    x_end       = min(corner_xy, X)
    y_start_bot = max(0, Y - corner_xy)

    corner_tl = volume[:z_end, :y_end,       :x_end]
    corner_bl = volume[:z_end, y_start_bot:, :x_end]
    return np.concatenate([corner_tl.ravel(), corner_bl.ravel()])


def _threshold(volume, corner_xy=50, corner_z=50, tolerance_pct=0.05):
    """
    Compute the background threshold and background mask.

    threshold = bg_max + tolerance  (tolerance can be negative)
    background mask = pixels <= threshold

    Returns bg_values, bg_max, threshold, bg_mask
    """
    bg_values  = _sample_corners(volume, corner_xy, corner_z)
    bg_max     = float(bg_values.max())
    data_range = float(volume.max()) - float(volume.min())
    tol        = data_range * (tolerance_pct / 100.0)
    thresh     = bg_max + tol
    bg_mask    = volume <= thresh
    return bg_values, bg_max, thresh, bg_mask


# ------------------------------------------------------------------ #
# Mode 1 — inference-guided removal (outside brain only)
# ------------------------------------------------------------------ #

def remove_outside_brain(
    volume: np.ndarray,
    brain_mask: np.ndarray,
    corner_xy: int = 50,
    corner_z: int = 101,
    tolerance_pct: float = 0.05,
):
    """
    Zero background pixels OUTSIDE the brain mask.
    Pixels inside the brain are always preserved.
    """
    bg_values, bg_max, thresh, bg_mask = _threshold(
        volume, corner_xy, corner_z, tolerance_pct
    )
    outside  = ~brain_mask.astype(bool)
    to_zero  = outside & bg_mask
    n_removed = int(to_zero.sum())

    print(f"   Background ceiling (corners): {bg_max:.1f}"
          f"  tol={tolerance_pct:+.3f}%  →  threshold={thresh:.1f}")
    print(f"   Removed {n_removed:,} outside-brain background voxels"
          f"  ({100.*n_removed/volume.size:.1f}% of stack)")

    result = volume.copy()
    result[to_zero] = 0
    return result, bg_max, thresh, n_removed


# ------------------------------------------------------------------ #
# Mode 2 — global threshold removal (whole stack)
# ------------------------------------------------------------------ #

def remove_global(
    volume: np.ndarray,
    corner_xy: int = 50,
    corner_z: int = 101,
    tolerance_pct: float = 0.05,
):
    """
    Zero ALL pixels in the full stack at or below the background threshold.
    Negative tolerance shrinks the threshold (removes less),
    positive tolerance raises it (removes more).
    """
    bg_values, bg_max, thresh, bg_mask = _threshold(
        volume, corner_xy, corner_z, tolerance_pct
    )
    n_removed = int(bg_mask.sum())

    print(f"   Background ceiling (corners): {bg_max:.1f}"
          f"  tol={tolerance_pct:+.3f}%  →  threshold={thresh:.1f}")
    print(f"   Removed {n_removed:,} voxels globally"
          f"  ({100.*n_removed/volume.size:.1f}% of stack)")

    result = volume.copy()
    result[bg_mask] = 0
    return result, bg_max, thresh, n_removed


# ------------------------------------------------------------------ #
# Mode 3 — random background fill (whole stack)
# ------------------------------------------------------------------ #

def fill_outside_brain_random(
    volume: np.ndarray,
    brain_mask: np.ndarray,
    corner_xy: int = 50,
    corner_z: int = 50,
):
    """
    Fill all pixels zeroed by skin removal (outside the brain mask) with
    random samples drawn from the corner pixel distribution.

    Inside brain  → original pixel values preserved
    Outside brain → random samples from corner background distribution

    No intensity threshold is used — the brain mask defines what gets filled.
    """
    bg_values = _sample_corners(volume, corner_xy, corner_z)

    outside  = ~brain_mask.astype(bool)
    n_filled = int(outside.sum())

    print(f"   Corner samples: {len(bg_values):,}"
          f"  (min={bg_values.min():.1f}  max={bg_values.max():.1f}"
          f"  mean={bg_values.mean():.1f})")
    print(f"   Filling {n_filled:,} outside-brain voxels with random noise"
          f"  ({100.*n_filled/volume.size:.1f}% of stack)")

    # Fit Gaussian to corner values, remove outliers beyond ±2σ
    mu    = float(bg_values.mean())
    sigma = float(bg_values.std())
    bg_pool = bg_values[np.abs(bg_values - mu) <= 2.0 * sigma]

    print(f"   Corner Gaussian: μ={mu:.2f}  σ={sigma:.2f}"
          f"  →  pool [{mu-2*sigma:.1f}, {mu+2*sigma:.1f}]"
          f"  ({len(bg_pool):,} / {len(bg_values):,} samples kept)")

    result = volume.copy()
    if n_filled > 0:
        random_fill = np.random.choice(bg_pool, size=n_filled, replace=True)
        result[outside] = random_fill.astype(volume.dtype)
    return result, n_filled
