"""
_background.py — Corner-based background removal.

Samples two top corners of the stack (top-left and top-right) to estimate
the background intensity range, then zeros all pixels in the full volume
whose value falls within that range ± a user-defined tolerance.

Corner regions sampled:
  Top-left : Z=0..z_depth,  Y=0..corner_xy,  X=0..corner_xy
  Top-right: Z=0..z_depth,  Y=0..corner_xy,  X=(W-corner_xy)..W
"""

import numpy as np


def remove_background(
    volume: np.ndarray,
    corner_xy: int = 50,
    corner_z: int = 101,
    tolerance_pct: float = 0.05,
) -> np.ndarray:
    """
    Zero out background pixels estimated from top-corner regions.

    Parameters
    ----------
    volume        : (Z, Y, X) ndarray
    corner_xy     : number of pixels in X and Y to sample from each corner
    corner_z      : number of Z slices to sample from the top
    tolerance_pct : tolerance as % of full data range (default 0.05 = 0.05%)

    Returns
    -------
    volume_clean  : copy of volume with background pixels set to 0
    bg_min, bg_max: background intensity bounds (before tolerance expansion)
    n_removed     : number of voxels zeroed
    """
    Z, Y, X = volume.shape

    z_end = min(corner_z, Z)
    y_end = min(corner_xy, Y)
    x_end = min(corner_xy, X)
    x_start_right = max(0, X - corner_xy)

    corner_tl = volume[:z_end, :y_end, :x_end]
    corner_tr = volume[:z_end, :y_end, x_start_right:]

    bg_values = np.concatenate([corner_tl.ravel(), corner_tr.ravel()])
    bg_min = float(bg_values.min())
    bg_max = float(bg_values.max())

    data_range = float(volume.max()) - float(volume.min())
    tol = data_range * (tolerance_pct / 100.0)

    low  = bg_min - tol
    high = bg_max + tol

    print(f"   Background range (corners): [{bg_min:.1f}, {bg_max:.1f}]")
    print(f"   Tolerance: ±{tolerance_pct:.3f}%  ({tol:.2f} intensity units)")
    print(f"   Zeroing pixels in [{low:.1f}, {high:.1f}]")

    mask = (volume >= low) & (volume <= high)
    n_removed = int(mask.sum())
    pct = 100.0 * n_removed / volume.size

    result = volume.copy()
    result[mask] = 0

    print(f"   Removed {n_removed:,} background voxels ({pct:.1f}% of stack)")

    return result, bg_min, bg_max, n_removed
