"""
_labeling.py — 3D connected component labeling with overlap-based slice linking.

Backend priority
----------------
1. CUDA  — CuPy + cupyx.scipy.ndimage  (full GPU path, vectorised overlap)
2. MPS   — Apple Silicon Metal          (threaded CPU; MPS lacks ndimage ops)
3. CPU   — scipy.ndimage + ThreadPool   (multithreaded, portable fallback)

Workflow
--------
1. Binary mask  : volume > 0
2. Gaussian smooth (σ_xy, σ_z) → re-threshold at 0.5
3. Fill holes per Z slice
4. Per-slice 2D labeling with globally unique IDs
5. Overlap graph between adjacent slices:
       overlap_ratio = intersection / min(area_A, area_B)
       if ratio >= min_overlap_pct → same 3D object
6. Union-Find → 3D connected components  (always CPU — data is tiny)
7. Remove blobs < min_volume voxels
8. Renumber 1…N by descending volume  (label 1 = largest)
"""

from __future__ import annotations

import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from scipy.ndimage import (
    gaussian_filter as cpu_gaussian,
    label        as cpu_label,
    binary_fill_holes as cpu_fill_holes,
)


# ─────────────────────────────────────────────────────────────────────────────
# Backend detection  (run once at import time)
# ─────────────────────────────────────────────────────────────────────────────

def _detect_backend() -> tuple[str, object, object]:
    """Return (backend_name, cupy_module, cupyx_ndimage_module)."""

    # ── CUDA via CuPy ──────────────────────────────────────────────────────
    try:
        import cupy as cp
        import cupyx.scipy.ndimage as cpnd
        # Exercise NVRTC/JIT so a broken install fails here, not mid-run
        _t = cp.zeros((4, 4), dtype=cp.float32)
        cpnd.gaussian_filter(_t, sigma=1.0)
        return "cuda", cp, cpnd
    except Exception:
        pass

    # ── Apple Silicon MPS ──────────────────────────────────────────────────
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps", None, None
    except Exception:
        pass

    return "cpu", None, None


_BACKEND, _CP, _CPND = _detect_backend()
_N_THREADS = max(1, (os.cpu_count() or 4) // 2)


# ─────────────────────────────────────────────────────────────────────────────
# Union-Find  (CPU, always — pairs array is tiny after GPU extraction)
# ─────────────────────────────────────────────────────────────────────────────

class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[ry] = rx


# ─────────────────────────────────────────────────────────────────────────────
# CUDA path
# ─────────────────────────────────────────────────────────────────────────────

def _create_labels_cuda(
    volume: np.ndarray,
    sigma_xy: float,
    sigma_z: float,
    min_overlap_pct: float,
    min_volume: int,
) -> np.ndarray:
    cp   = _CP
    cpnd = _CPND
    Z, Y, X = volume.shape

    # ── Steps 1–2: binary → Gaussian smooth → re-threshold ────────────────
    vol_gpu     = cp.asarray(volume, dtype=cp.float32)
    binary_gpu  = (vol_gpu > 0).astype(cp.float32)
    blurred_gpu = cpnd.gaussian_filter(binary_gpu, sigma=(sigma_z, sigma_xy, sigma_xy))
    smooth_gpu  = blurred_gpu > 0.5
    del vol_gpu, binary_gpu, blurred_gpu

    print(f"   σ_xy={sigma_xy:.1f}  σ_z={sigma_z:.1f}  "
          f"signal voxels: {int(smooth_gpu.sum()):,}")

    # ── Step 3: fill holes per slice (GPU loop) ────────────────────────────
    for z in range(Z):
        smooth_gpu[z] = cpnd.binary_fill_holes(smooth_gpu[z])

    # ── Step 4: per-slice 2D labeling with globally unique IDs ────────────
    slice_labels_gpu = cp.zeros((Z, Y, X), dtype=cp.int32)
    offset = 0
    for z in range(Z):
        lbl_gpu, n = cpnd.label(smooth_gpu[z])
        n = int(n)
        if n == 0:
            continue
        lbl_gpu = cp.where(lbl_gpu > 0, lbl_gpu + offset, cp.int32(0))
        slice_labels_gpu[z] = lbl_gpu
        offset += n
    del smooth_gpu

    print(f"   2D blobs: {offset}")
    if offset == 0:
        return slice_labels_gpu.get().astype(np.int32)

    # ── Steps 5–6: vectorised overlap graph + Union-Find ──────────────────
    # Pairs are encoded as  a * encode_base + b  (int64, no overflow)
    encode_base = int(offset) + 1
    min_overlap  = min_overlap_pct / 100.0
    uf      = _UnionFind()
    n_links = 0

    for z in range(Z - 1):
        lz_flat  = slice_labels_gpu[z].ravel().astype(cp.int64)
        lz1_flat = slice_labels_gpu[z + 1].ravel().astype(cp.int64)

        overlap_mask = (lz_flat > 0) & (lz1_flat > 0)
        if not int(overlap_mask.sum()):
            continue

        a_vals = lz_flat[overlap_mask]
        b_vals = lz1_flat[overlap_mask]

        # Count intersections via unique encoding
        encoded           = a_vals * encode_base + b_vals
        unique_enc, inter = cp.unique(encoded, return_counts=True)

        a_u = (unique_enc // encode_base).astype(cp.int32)
        b_u = (unique_enc  % encode_base).astype(cp.int32)

        # Per-label areas (full slice, not just overlap) via bincount
        area_z  = cp.bincount(lz_flat,  minlength=encode_base)
        area_z1 = cp.bincount(lz1_flat, minlength=encode_base)

        min_area = cp.minimum(area_z[a_u], area_z1[b_u]).astype(cp.float32)
        valid    = inter.astype(cp.float32) / min_area >= min_overlap

        if not int(valid.sum()):
            continue

        # Transfer only the tiny valid-pairs array to CPU for Union-Find
        for a, b in zip(a_u[valid].get().tolist(), b_u[valid].get().tolist()):
            uf.union(a, b)
            n_links += 1

    print(f"   Cross-slice links: {n_links}  (min overlap={min_overlap_pct:.1f}%)")

    # ── Step 7a: build root→new_id LUT on CPU, apply on GPU ───────────────
    all_labels   = cp.unique(slice_labels_gpu[slice_labels_gpu > 0]).get().tolist()
    roots        = {lbl: uf.find(lbl) for lbl in all_labels}
    unique_roots = sorted(set(roots.values()))
    root_to_new  = {root: i + 1 for i, root in enumerate(unique_roots)}

    max_lbl = int(slice_labels_gpu.max())
    lut     = np.zeros(max_lbl + 1, dtype=np.int32)
    for old_lbl in all_labels:
        lut[old_lbl] = root_to_new[roots[old_lbl]]

    output_gpu = cp.asarray(lut)[slice_labels_gpu]
    del slice_labels_gpu

    # ── Step 7b: remove small blobs — vectorised on GPU ───────────────────
    max_out = int(output_gpu.max())
    counts  = cp.bincount(output_gpu.ravel().astype(cp.int64), minlength=max_out + 1)

    keep_lut    = counts >= min_volume
    keep_lut[0] = True                          # always keep background
    output_gpu  = cp.where(keep_lut[output_gpu], output_gpu, cp.int32(0))
    removed     = int(((counts[1:] > 0) & (counts[1:] < min_volume)).sum())

    # ── Step 7c: renumber 1…N by descending volume (GPU-assisted) ─────────
    remaining      = cp.unique(output_gpu[output_gpu > 0]).get().tolist()
    counts_cpu     = counts.get()
    volumes_sorted = sorted(
        [(int(counts_cpu[lbl]), int(lbl)) for lbl in remaining], reverse=True
    )
    max_out2 = int(output_gpu.max())
    lut2     = np.zeros(max_out2 + 1, dtype=np.int32)
    for new_id, (_vol, old_id) in enumerate(volumes_sorted, start=1):
        lut2[old_id] = new_id

    output  = cp.asarray(lut2)[output_gpu].get()
    n_final = int(output.max())
    print(f"   3D blobs removed (< {min_volume} vox): {removed}")
    print(f"   Final 3D labels: {n_final}  (label 1 = largest)")
    return output.astype(np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# Threaded CPU path  (also used for Apple MPS — MPS lacks ndimage ops)
# ─────────────────────────────────────────────────────────────────────────────

def _fill_and_label_slice(args: tuple) -> tuple:
    """Worker: fill holes + 2D label for one slice. Returns (z, labels, n)."""
    z, mask_slice = args
    filled = cpu_fill_holes(mask_slice)
    lbl, n = cpu_label(filled)
    return z, lbl.astype(np.int32), n


def _compute_slice_pair_links(args: tuple) -> list[tuple[int, int]]:
    """Worker: return all (a, b) label pairs that pass the overlap threshold."""
    lz, lz1, min_overlap, encode_base = args
    lz_flat  = lz.ravel().astype(np.int64)
    lz1_flat = lz1.ravel().astype(np.int64)

    overlap_mask = (lz_flat > 0) & (lz1_flat > 0)
    if not overlap_mask.any():
        return []

    a_vals = lz_flat[overlap_mask]
    b_vals = lz1_flat[overlap_mask]

    encoded           = a_vals * encode_base + b_vals
    unique_enc, inter = np.unique(encoded, return_counts=True)

    a_u = (unique_enc // encode_base).astype(np.int32)
    b_u = (unique_enc  % encode_base).astype(np.int32)

    area_z  = np.bincount(lz_flat,  minlength=encode_base)
    area_z1 = np.bincount(lz1_flat, minlength=encode_base)

    min_area = np.minimum(area_z[a_u], area_z1[b_u]).astype(np.float32)
    valid    = inter.astype(np.float32) / min_area >= min_overlap

    return list(zip(a_u[valid].tolist(), b_u[valid].tolist()))


def _create_labels_threaded(
    volume: np.ndarray,
    sigma_xy: float,
    sigma_z: float,
    min_overlap_pct: float,
    min_volume: int,
) -> np.ndarray:
    Z, Y, X = volume.shape

    # ── Steps 1–2: Gaussian smooth (scipy already multi-threaded internally)
    binary      = (volume > 0).astype(np.float32)
    blurred     = cpu_gaussian(binary, sigma=(sigma_z, sigma_xy, sigma_xy))
    smooth_mask = blurred > 0.5
    del binary, blurred

    print(f"   σ_xy={sigma_xy:.1f}  σ_z={sigma_z:.1f}  "
          f"signal voxels: {int(smooth_mask.sum()):,}")

    # ── Steps 3+4: fill holes + 2D label in parallel across slices ────────
    with ThreadPoolExecutor(max_workers=_N_THREADS) as pool:
        results = list(pool.map(
            _fill_and_label_slice,
            [(z, smooth_mask[z]) for z in range(Z)],
        ))
    del smooth_mask

    results.sort(key=lambda r: r[0])
    slice_labels = np.zeros((Z, Y, X), dtype=np.int32)
    offset = 0
    for z, lbl, n in results:
        if n == 0:
            continue
        lbl[lbl > 0] += offset
        slice_labels[z] = lbl
        offset += n

    print(f"   2D blobs: {offset}")
    if offset == 0:
        return slice_labels

    # ── Steps 5–6: parallel overlap collection + sequential Union-Find ─────
    encode_base = int(offset) + 1
    min_overlap  = min_overlap_pct / 100.0

    with ThreadPoolExecutor(max_workers=_N_THREADS) as pool:
        pair_lists = list(pool.map(
            _compute_slice_pair_links,
            [
                (slice_labels[z], slice_labels[z + 1], min_overlap, encode_base)
                for z in range(Z - 1)
            ],
        ))

    uf      = _UnionFind()
    n_links = 0
    for pairs in pair_lists:
        for a, b in pairs:
            uf.union(a, b)
            n_links += 1

    print(f"   Cross-slice links: {n_links}  (min overlap={min_overlap_pct:.1f}%)")

    # ── Step 7a: build LUT and apply ──────────────────────────────────────
    all_labels   = np.unique(slice_labels[slice_labels > 0]).tolist()
    roots        = {lbl: uf.find(lbl) for lbl in all_labels}
    unique_roots = sorted(set(roots.values()))
    root_to_new  = {root: i + 1 for i, root in enumerate(unique_roots)}

    max_lbl = int(slice_labels.max())
    lut     = np.zeros(max_lbl + 1, dtype=np.int32)
    for old_lbl in all_labels:
        lut[old_lbl] = root_to_new[roots[old_lbl]]

    output = lut[slice_labels]
    del slice_labels

    # ── Step 7b: remove small blobs ───────────────────────────────────────
    max_out = int(output.max())
    counts  = np.bincount(output.ravel().astype(np.int64), minlength=max_out + 1)

    keep_lut    = counts >= min_volume
    keep_lut[0] = True
    output      = np.where(keep_lut[output], output, 0).astype(np.int32)
    removed     = int(((counts[1:] > 0) & (counts[1:] < min_volume)).sum())

    # ── Step 7c: renumber 1…N by descending volume ────────────────────────
    remaining      = np.unique(output[output > 0]).tolist()
    volumes_sorted = sorted(
        [(int(counts[lbl]), int(lbl)) for lbl in remaining], reverse=True
    )
    max_out2 = int(output.max())
    lut2     = np.zeros(max_out2 + 1, dtype=np.int32)
    for new_id, (_vol, old_id) in enumerate(volumes_sorted, start=1):
        lut2[old_id] = new_id

    output  = lut2[output]
    n_final = int(output.max())
    print(f"   3D blobs removed (< {min_volume} vox): {removed}")
    print(f"   Final 3D labels: {n_final}  (label 1 = largest)")
    return output.astype(np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def resort_labels(
    labels: np.ndarray,
    sort_by: str = "size",
    reverse: bool = False,
) -> np.ndarray:
    """
    Renumber labels 1…N by the chosen criterion.

    Parameters
    ----------
    labels   : (Z, Y, X) int32 ndarray — existing label volume (0 = background)
    sort_by  : "size" | "centroid_z" | "centroid_y" | "centroid_x"
    reverse  : reverse the natural sort order
                 size      — natural = descending (largest = label 1)
                 centroid  — natural = ascending  (smallest coord = label 1)

    Returns
    -------
    (Z, Y, X) int32 ndarray — same objects, renumbered 1…N
    """
    from scipy.ndimage import center_of_mass as _com

    unique = np.unique(labels)
    unique = unique[unique > 0]
    if unique.size == 0:
        return labels.copy()

    label_list = unique.tolist()
    max_lbl    = int(unique.max())

    if sort_by == "size":
        counts = np.bincount(labels.ravel().astype(np.int64), minlength=max_lbl + 1)
        keyed  = [(int(counts[lbl]), int(lbl)) for lbl in label_list]
        # natural: descending (largest first → label 1)
        keyed.sort(key=lambda t: t[0], reverse=not reverse)
    else:
        axis_map = {"centroid_z": 0, "centroid_y": 1, "centroid_x": 2}
        axis     = axis_map[sort_by]
        raw      = _com(labels > 0, labels, label_list)
        if unique.size == 1:
            raw = [raw]   # scipy returns a single tuple when only one label
        keyed = [(float(c[axis]), int(lbl)) for lbl, c in zip(label_list, raw)]
        # natural: ascending (smallest coordinate first → label 1)
        keyed.sort(key=lambda t: t[0], reverse=reverse)

    lut = np.zeros(max_lbl + 1, dtype=np.int32)
    for new_id, (_key, old_id) in enumerate(keyed, start=1):
        lut[old_id] = new_id

    return lut[labels].astype(np.int32)


def create_labels(
    volume: np.ndarray,
    sigma_xy: float = 1.0,
    sigma_z: float = 0.5,
    min_overlap_pct: float = 10.0,
    min_volume: int = 7500,
) -> np.ndarray:
    """
    Create 3D labels from brain_only volume using overlap-based slice linking.

    Dispatches to the fastest available backend:
      CUDA (CuPy)  →  Apple MPS (threaded CPU)  →  CPU threaded

    Parameters
    ----------
    volume          : (Z, Y, X) ndarray — brain_only output
    sigma_xy        : Gaussian smoothing sigma in XY (voxels)
    sigma_z         : Gaussian smoothing sigma in Z (voxels)
    min_overlap_pct : minimum 2D overlap % to link blobs across slices
    min_volume      : minimum 3D blob size in voxels

    Returns
    -------
    (Z, Y, X) int32 ndarray — 0=background, 1..N=objects (1=largest)
    """
    backend_label = {
        "cuda": "CUDA (CuPy)",
        "mps":  f"Apple MPS → threaded CPU  (threads={_N_THREADS})",
        "cpu":  f"CPU threaded  (threads={_N_THREADS})",
    }[_BACKEND]
    print(f"   Backend: {backend_label}")

    if _BACKEND == "cuda":
        try:
            return _create_labels_cuda(
                volume, sigma_xy, sigma_z, min_overlap_pct, min_volume
            )
        except Exception as exc:
            # e.g. out-of-memory — fall back gracefully
            print(f"   CUDA error ({exc}), falling back to CPU.")

    return _create_labels_threaded(
        volume, sigma_xy, sigma_z, min_overlap_pct, min_volume
    )
