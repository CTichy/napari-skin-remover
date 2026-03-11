"""
_labeling.py — 3D connected component labeling with overlap-based slice linking.

Workflow:
  1. Binary mask from brain_only > 0  (already done by option 2 background removal)
  2. Gaussian smoothing (separate sigma for XY and Z) to soften contours
  3. Re-threshold at 0.5 → smooth solid blobs
  4. Per-slice 2D labeling, globally unique IDs, small blobs removed
  5. Overlap graph between adjacent slices:
       overlap_ratio = intersection / min(area_A, area_B)
       if overlap_ratio >= min_overlap_pct → same 3D object
  6. Union-Find → connected components across all slices
  7. Sequential renumbering → napari Labels layer
"""

import numpy as np
from scipy.ndimage import gaussian_filter, label as nd_label, binary_fill_holes


class _UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[ry] = rx


def create_labels(
    volume: np.ndarray,
    sigma_xy: float = 1.0,
    sigma_z: float = 0.5,
    min_overlap_pct: float = 10.0,
    min_volume: int = 7500,
) -> np.ndarray:
    """
    Create 3D labels from brain_only volume using overlap-based slice linking.

    Parameters
    ----------
    volume          : (Z, Y, X) ndarray — brain_only output (background already removed)
    sigma_xy        : Gaussian smoothing sigma in XY (voxels)
    sigma_z         : Gaussian smoothing sigma in Z (voxels)
    min_overlap_pct : minimum 2D overlap % to link blobs across slices
    min_volume      : minimum 3D blob size in voxels — smaller blobs removed after linking

    Returns
    -------
    output : (Z, Y, X) int32 ndarray — labeled volume, 0=background, 1..N=objects
    """
    Z, Y, X = volume.shape

    # ------------------------------------------------------------------ #
    # Steps 1–3: binary → Gaussian smooth → re-threshold
    # ------------------------------------------------------------------ #
    binary  = (volume > 0).astype(np.float32)
    blurred = gaussian_filter(binary, sigma=(sigma_z, sigma_xy, sigma_xy))
    smooth_mask = blurred > 0.5

    # Fill holes per 2D slice — no pores inside contours
    for z in range(smooth_mask.shape[0]):
        smooth_mask[z] = binary_fill_holes(smooth_mask[z])

    print(f"   Smoothing: σ_xy={sigma_xy:.1f}  σ_z={sigma_z:.1f}  (holes filled per slice)")
    print(f"   Signal voxels after smoothing: {smooth_mask.sum():,}")

    # ------------------------------------------------------------------ #
    # Step 4: per-slice 2D labeling with globally unique IDs
    # ------------------------------------------------------------------ #
    slice_labels = np.zeros((Z, Y, X), dtype=np.int32)
    offset = 0

    for z in range(Z):
        lbl, n = nd_label(smooth_mask[z])
        if n == 0:
            continue
        # Assign globally unique IDs (no per-slice area filter — 3D filter applied later)
        lbl[lbl > 0] += offset
        slice_labels[z] = lbl
        offset += n

    total_blobs = offset
    print(f"   2D blobs: {total_blobs}")

    if total_blobs == 0:
        print("   No blobs found — try lowering min blob area or smoothing sigma.")
        return slice_labels

    # ------------------------------------------------------------------ #
    # Steps 5–6: build overlap graph + Union-Find
    # ------------------------------------------------------------------ #
    uf = _UnionFind()
    min_overlap = min_overlap_pct / 100.0
    n_links = 0

    for z in range(Z - 1):
        lz  = slice_labels[z]
        lz1 = slice_labels[z + 1]

        overlap_pixels = (lz > 0) & (lz1 > 0)
        if not overlap_pixels.any():
            continue

        # All (a, b) pairs that share at least one pixel
        pairs = np.unique(
            np.stack([lz[overlap_pixels], lz1[overlap_pixels]], axis=1), axis=0
        )

        for a, b in pairs:
            area_a = int((lz  == a).sum())
            area_b = int((lz1 == b).sum())
            inter  = int(((lz == a) & (lz1 == b)).sum())
            if inter / min(area_a, area_b) >= min_overlap:
                uf.union(int(a), int(b))
                n_links += 1

    print(f"   Cross-slice links: {n_links}  (min overlap={min_overlap_pct:.1f}%)")

    # ------------------------------------------------------------------ #
    # Step 7: renumber sequentially using lookup table
    # ------------------------------------------------------------------ #
    all_labels   = np.unique(slice_labels[slice_labels > 0])
    roots        = {int(lbl): uf.find(int(lbl)) for lbl in all_labels}
    unique_roots = sorted(set(roots.values()))
    root_to_new  = {root: i + 1 for i, root in enumerate(unique_roots)}

    max_lbl = int(slice_labels.max())
    lut = np.zeros(max_lbl + 1, dtype=np.int32)
    for old_lbl in all_labels:
        lut[int(old_lbl)] = root_to_new[roots[int(old_lbl)]]

    output  = lut[slice_labels]

    # 3D volume filter — remove blobs smaller than min_volume voxels
    final_labels = np.unique(output[output > 0])
    removed = 0
    for lbl in final_labels:
        vol = int((output == lbl).sum())
        if vol < min_volume:
            output[output == lbl] = 0
            removed += 1

    # Re-number sequentially after volume filtering
    remaining = np.unique(output[output > 0])
    lut2 = np.zeros(int(output.max()) + 1, dtype=np.int32)
    for new_id, old_id in enumerate(remaining, start=1):
        lut2[int(old_id)] = new_id
    output = lut2[output]

    n_final = int(output.max())
    print(f"   3D blobs removed (< {min_volume} vox): {removed}")
    print(f"   Final 3D labels: {n_final}")

    return output
