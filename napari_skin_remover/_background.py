"""
_background.py — Corner-based background processing.

Samples two top corners of the stack (top-left and top-right) to estimate
the background intensity distribution.

Two modes:
  remove  — zero all pixels whose value falls within the background range
             ± a user-defined tolerance (outside brain mask only)
  fill    — fill all zero-valued pixels with the mean background value
             (restores scanner noise floor to black/empty regions)

Corner regions sampled:
  Top-left : Z=0..corner_z,  Y=0..corner_xy,  X=0..corner_xy
  Top-right: Z=0..corner_z,  Y=0..corner_xy,  X=(W-corner_xy)..W
"""

import numpy as np


def _sample_corners(volume, corner_xy=50, corner_z=101):
    """Return flat array of all pixel values from the two top corners."""
    Z, Y, X = volume.shape
    z_end         = min(corner_z, Z)
    y_end         = min(corner_xy, Y)
    x_end         = min(corner_xy, X)
    x_start_right = max(0, X - corner_xy)

    corner_tl = volume[:z_end, :y_end, :x_end]
    corner_tr = volume[:z_end, :y_end, x_start_right:]
    return np.concatenate([corner_tl.ravel(), corner_tr.ravel()])


def remove_background(
    volume: np.ndarray,
    corner_xy: int = 50,
    corner_z: int = 101,
    tolerance_pct: float = 0.05,
):
    """
    Zero out background pixels estimated from top-corner regions.

    Parameters
    ----------
    volume        : (Z, Y, X) ndarray
    corner_xy     : pixels in X and Y to sample from each corner
    corner_z      : Z slices to sample from the top
    tolerance_pct : tolerance as % of full data range (default 0.05 = 0.05%)

    Returns
    -------
    volume_clean  : copy of volume with background pixels set to 0
    bg_min, bg_max: background intensity bounds (before tolerance expansion)
    n_removed     : number of voxels zeroed
    """
    bg_values = _sample_corners(volume, corner_xy, corner_z)
    bg_min = float(bg_values.min())
    bg_max = float(bg_values.max())

    data_range = float(volume.max()) - float(volume.min())
    tol  = data_range * (tolerance_pct / 100.0)
    low  = bg_min - tol
    high = bg_max + tol

    print(f"   Background range (corners): [{bg_min:.1f}, {bg_max:.1f}]")
    print(f"   Tolerance: ±{tolerance_pct:.3f}%  ({tol:.2f} intensity units)")
    print(f"   Zeroing pixels in [{low:.1f}, {high:.1f}]")

    mask = (volume >= low) & (volume <= high)
    n_removed = int(mask.sum())
    print(f"   Removed {n_removed:,} background voxels ({100.*n_removed/volume.size:.1f}% of stack)")

    result = volume.copy()
    result[mask] = 0
    return result, bg_min, bg_max, n_removed


def fill_zeros_with_background(
    volume: np.ndarray,
    corner_xy: int = 50,
    corner_z: int = 101,
):
    """
    Fill zero-valued pixels with the mean background value from corner regions.

    Useful when the stack contains black gaps or padded regions that would
    create artificial boundaries during processing.

    Parameters
    ----------
    volume    : (Z, Y, X) ndarray
    corner_xy : pixels in X and Y to sample from each corner
    corner_z  : Z slices to sample from the top

    Returns
    -------
    volume_filled : copy of volume with zero pixels replaced by mean background
    bg_mean       : mean background intensity used for filling
    n_filled      : number of voxels filled
    """
    bg_values = _sample_corners(volume, corner_xy, corner_z)
    bg_mean = float(bg_values.mean())

    print(f"   Background mean (corners): {bg_mean:.2f}")

    zero_mask = volume == 0
    n_filled  = int(zero_mask.sum())
    print(f"   Filling {n_filled:,} zero voxels ({100.*n_filled/volume.size:.1f}% of stack)")

    result = volume.copy()
    result[zero_mask] = np.array(bg_mean, dtype=volume.dtype)
    return result, bg_mean, n_filled
