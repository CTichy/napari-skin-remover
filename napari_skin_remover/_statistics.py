"""
_statistics.py — Per-label morphological statistics + natural-language descriptions.

Statistics computed per label
------------------------------
  label               : label ID
  volume_vox          : voxel count
  volume_um3          : physical volume (µm³)
  centroid_z/y/x_vox  : centroid in voxels
  centroid_z/y/x_um   : centroid in µm
  bbox_dz/dy/dx_um    : bounding box size (µm)
  eq_diam_um          : equivalent sphere diameter (µm)
  axis1/2/3_um        : principal axis lengths (µm), axis1 = longest
  elongation          : axis1 / axis3  (1 = sphere, > 1 = elongated)
  principal_axis_dir  : dominant axis of elongation ('Z', 'Y', or 'X')
  solidity            : volume / convex_hull_volume  (1 = convex, < 1 = lobulated)
  extent              : volume / bounding_box_volume
  surface_area_um2    : mesh surface area via marching cubes (µm²)
  sphericity          : π^(1/3) × (6V)^(2/3) / A  (1 = perfect sphere)
  n_branches          : skeleton branch count  (requires skan)
  n_endpoints         : number of free-end endpoints
  mean_branch_len_um  : mean branch length (µm)
  description         : natural-language description

GPU acceleration
----------------
  Phase 1 — batch regionprops: cuCIM (GPU) → skimage regionprops_table (CPU)
  Phase 2 — per-label marching cubes + skeleton: ThreadPoolExecutor (parallel CPU)
  Phase 3 — description generation: online API or rule-based (CPU)

Description backends
--------------------
  rule   – built-in rule-based templates (always available, fully offline)
  ollama – local Ollama LLM (free, no API key; install from https://ollama.com)
  openai – OpenAI API (paid; requires API key from https://platform.openai.com)
  claude – Anthropic Claude API (paid; requires API key from https://console.anthropic.com)

Ollama setup (quick guide)
--------------------------
  1. Install Ollama:  https://ollama.com/download
  2. Pull a model:    ollama pull llama3
  3. It starts automatically; default endpoint is http://localhost:11434
  4. Select "Ollama (local)" in the plugin, enter the model name (e.g. llama3)

OpenAI setup
------------
  1. Create account at https://platform.openai.com
  2. Generate an API key under API Keys
  3. Select "OpenAI API", paste the key and choose a model
     (e.g. gpt-4o-mini for low cost, gpt-4o for best quality)

Anthropic Claude setup
----------------------
  1. Create account at https://console.anthropic.com
  2. Generate an API key under API Keys
  3. Select "Claude API", paste the key and choose a model
     (e.g. claude-haiku-4-5-20251001 for low cost, claude-sonnet-4-6 for quality)
"""

from __future__ import annotations

import json
import math
import os
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

_N_THREADS = max(1, (os.cpu_count() or 4) // 2)


# ── GPU backend detection ─────────────────────────────────────────────────────

def _detect_stats_backend():
    """Return (cupy, cucim_measure) or (None, None)."""
    import io, sys
    # cupy / cucim print a spurious cuVS/pylibraft warning to stdout on import;
    # suppress it — it doesn't affect regionprops functionality.
    _saved, sys.stdout = sys.stdout, io.StringIO()
    try:
        import cupy as cp
        import cucim.skimage.measure as cucim_measure
        # Quick smoke-test
        t = cp.zeros((4, 4, 4), dtype=cp.int32)
        t[1:3, 1:3, 1:3] = 1
        cucim_measure.regionprops_table(t, properties=["label", "area"])
        return cp, cucim_measure
    except Exception:
        return None, None
    finally:
        sys.stdout = _saved


_CP_STATS, _CUCIM = _detect_stats_backend()
_STATS_BACKEND = "cuda" if _CP_STATS is not None else "cpu"

# Properties requested from regionprops_table (both skimage and cuCIM)
_RPROPS = [
    "label", "area", "centroid", "bbox",
    "inertia_tensor", "inertia_tensor_eigvals",
    "axis_major_length", "axis_minor_length",
    "solidity", "extent",
]


# ── Phase 1: batch regionprops (GPU or CPU) ──────────────────────────────────

def _batch_regionprops(labels: np.ndarray) -> dict:
    """
    Compute regionprops for ALL labels in one vectorised pass.
    Returns a dict of flat numpy arrays keyed by property name.
    Uses cuCIM (GPU) when available, falls back to skimage (CPU).
    """
    from skimage.measure import regionprops_table as _cpu_rpt

    if _STATS_BACKEND == "cuda":
        try:
            labels_gpu = _CP_STATS.asarray(labels)
            table = _CUCIM.regionprops_table(labels_gpu, properties=_RPROPS)
            # Move all arrays to CPU (.get() for CuPy, asarray for anything else)
            result = {}
            for k, v in table.items():
                result[k] = v.get() if hasattr(v, "get") else np.asarray(v)
            del labels_gpu
            _CP_STATS.get_default_memory_pool().free_all_blocks()
            print(f"   regionprops: cuCIM GPU ({len(result['label'])} labels)")
            return result
        except Exception as exc:
            print(f"   regionprops: cuCIM failed ({exc}), using CPU")

    table = _cpu_rpt(labels, properties=_RPROPS)
    print(f"   regionprops: skimage CPU ({len(table['label'])} labels)")
    return {k: np.asarray(v) for k, v in table.items()}


# ── Phase 2: per-label slow ops (parallelised CPU) ────────────────────────────

def _surface_area(binary: np.ndarray, scale_zyx: tuple) -> float:
    """Mesh surface area in µm² via marching cubes. Returns 0.0 on failure."""
    try:
        from skimage.measure import marching_cubes, mesh_surface_area
        padded = np.pad(binary.astype(np.uint8), 1)
        verts, faces, _, _ = marching_cubes(padded, level=0.5, spacing=scale_zyx)
        return float(mesh_surface_area(verts, faces))
    except Exception:
        return 0.0


def _skeleton_stats(binary: np.ndarray, scale_zyx: tuple) -> tuple[int, int, float]:
    """
    Return (n_branches, n_endpoints, mean_branch_len_um).
    Uses skan if available, otherwise (0, 0, 0.0).
    """
    try:
        from skimage.morphology import skeletonize
        import skan
        skeleton = skeletonize(binary)
        if not skeleton.any():
            return 0, 0, 0.0
        sk = skan.Skeleton(
            skeleton, spacing=scale_zyx,
            source_image=binary.astype(np.float32),
        )
        branch_data = skan.summarize(sk, separator="-")
        if len(branch_data) == 0:
            return 0, 0, 0.0
        n_branches  = len(branch_data)
        n_endpoints = int((branch_data["branch-type"] == 1).sum())
        mean_len    = float(branch_data["euclidean-distance"].mean())
        return n_branches, n_endpoints, mean_len
    except Exception:
        return 0, 0, 0.0


def _slow_stats_worker(args: tuple) -> tuple:
    """Worker for ThreadPoolExecutor: returns (lbl, sa_um2, n_br, n_ep, br_len)."""
    lbl, labels, bbox, scale_zyx = args
    z0, y0, x0, z1, y1, x1 = bbox
    binary = (labels[z0:z1, y0:y1, x0:x1] == lbl)
    sa  = _surface_area(binary, scale_zyx)
    n_br, n_ep, br_len = _skeleton_stats(binary, scale_zyx)
    return lbl, sa, n_br, n_ep, br_len


# ── Description backends ──────────────────────────────────────────────────────

def _rule_based_description(row: dict) -> str:
    lbl     = row["label"]
    vol     = row["volume_um3"]
    elong   = row["elongation"]
    adir    = row["principal_axis_dir"]
    spher   = row["sphericity"]
    solid   = row["solidity"]
    n_br    = row["n_branches"]
    n_ep    = row["n_endpoints"]
    br_len  = row["mean_branch_len_um"]
    cz, cy, cx = row["centroid_z_um"], row["centroid_y_um"], row["centroid_x_um"]

    if spher > 0.85:
        shape = "spherical"
    elif spher > 0.70:
        shape = f"rounded, elongated along {adir}-axis ({elong:.1f}:1)" if elong > 1.8 else "rounded"
    elif elong > 2.5:
        shape = f"elongated along {adir}-axis ({elong:.1f}:1)"
    elif elong > 1.5:
        shape = f"moderately elongated along {adir}-axis"
    else:
        shape = "compact, irregular"

    surface = (
        "smooth surface" if solid > 0.90 else
        "slightly lobulated surface" if solid > 0.75 else
        "lobulated/irregular surface"
    )

    if n_br == 0:
        branch_str = "no branching detected"
    elif n_br <= 2:
        branch_str = f"{n_ep} protrusion(s)" + (f" (mean {br_len:.1f} µm)" if br_len > 0 else "")
    else:
        branch_str = f"{n_br} branches, {n_ep} endpoints" + (f" (mean {br_len:.1f} µm)" if br_len > 0 else "")

    return (
        f"Label {lbl}: {shape.capitalize()}, volume {vol:,.0f} µm³, "
        f"centroid Z={cz:.1f} Y={cy:.1f} X={cx:.1f} µm. "
        f"{surface.capitalize()}, sphericity {spher:.2f}, solidity {solid:.2f}. "
        f"{branch_str.capitalize()}."
    )


_STATS_PROMPT = """\
You are analyzing a single 3D cell from a zebrafish brain confocal microscopy image.
Given the morphological statistics below, write ONE concise sentence (max 40 words) \
describing the cell's shape, size, and notable features.

Label: {label}
Volume: {volume_um3:.0f} µm³
Elongation (longest/shortest axis): {elongation:.2f}
Dominant axis: {principal_axis_dir}
Sphericity (1=sphere): {sphericity:.3f}
Solidity (1=convex): {solidity:.3f}
Branches: {n_branches}  Endpoints: {n_endpoints}  Mean branch length: {mean_branch_len_um:.1f} µm
Centroid: Z={centroid_z_um:.1f} Y={centroid_y_um:.1f} X={centroid_x_um:.1f} µm

Description:"""


def _ollama_description(row: dict, endpoint: str, model: str) -> str:
    prompt  = _STATS_PROMPT.format(**row)
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    req = urllib.request.Request(
        f"{endpoint.rstrip('/')}/api/generate", data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read()).get("response", "").strip()
    except Exception as exc:
        return f"[Ollama error: {exc}]"


def _openai_description(row: dict, api_key: str, model: str, api_url: str) -> str:
    url     = (api_url or "https://api.openai.com").rstrip("/") + "/v1/chat/completions"
    payload = json.dumps({
        "model": model or "gpt-4o-mini",
        "messages": [{"role": "user", "content": _STATS_PROMPT.format(**row)}],
        "max_tokens": 80,
    }).encode()
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        return f"[OpenAI error: {exc}]"


def _claude_description(row: dict, api_key: str, model: str) -> str:
    payload = json.dumps({
        "model": model or "claude-haiku-4-5-20251001",
        "max_tokens": 80,
        "messages": [{"role": "user", "content": _STATS_PROMPT.format(**row)}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())["content"][0]["text"].strip()
    except Exception as exc:
        return f"[Claude error: {exc}]"


def _make_desc_fn(backend_config: dict | None):
    if not backend_config or backend_config.get("backend", "rule") == "rule":
        return _rule_based_description
    b = backend_config["backend"]
    if b == "ollama":
        ep = backend_config.get("ollama_endpoint", "http://localhost:11434")
        mo = backend_config.get("ollama_model", "llama3")
        return lambda row: _ollama_description(row, ep, mo)
    if b == "openai":
        return lambda row: _openai_description(
            row,
            backend_config.get("api_key", ""),
            backend_config.get("api_model", "gpt-4o-mini"),
            backend_config.get("api_url", ""),
        )
    if b == "claude":
        return lambda row: _claude_description(
            row,
            backend_config.get("api_key", ""),
            backend_config.get("api_model", "claude-haiku-4-5-20251001"),
        )
    return _rule_based_description


# ── Main entry point ──────────────────────────────────────────────────────────

def compute_stats(
    labels: np.ndarray,
    scale_zyx: tuple,
    backend_config: dict | None = None,
) -> "Any":  # returns pd.DataFrame
    """
    Compute per-label morphological statistics and generate descriptions.

    Parameters
    ----------
    labels         : (Z, Y, X) int32 ndarray, 0 = background
    scale_zyx      : (z_um, y_um, x_um) — physical voxel size in µm,
                     read from the napari layer .scale — never hardcoded
    backend_config : description backend config dict (see module docstring)

    Returns
    -------
    pd.DataFrame — one row per label, columns as documented at top of module
    """
    import pandas as pd

    sz, sy, sx = float(scale_zyx[0]), float(scale_zyx[1]), float(scale_zyx[2])
    vox_vol    = sz * sy * sx
    desc_fn    = _make_desc_fn(backend_config)

    print(f"\n{'='*60}")
    print(f"GENERATE STATISTICS  (GPU={_STATS_BACKEND}  threads={_N_THREADS})")
    print(f"Scale: Z={sz:.4f}  Y={sy:.4f}  X={sx:.4f} µm/vox")
    print(f"{'='*60}")

    # ── Phase 1: batch regionprops ────────────────────────────────────────
    table = _batch_regionprops(labels)
    N     = len(table["label"])

    # Reconstruct N×3×3 inertia tensors and batch-diagonalise
    IT = np.stack([
        [table["inertia_tensor-0-0"], table["inertia_tensor-0-1"], table["inertia_tensor-0-2"]],
        [table["inertia_tensor-1-0"], table["inertia_tensor-1-1"], table["inertia_tensor-1-2"]],
        [table["inertia_tensor-2-0"], table["inertia_tensor-2-1"], table["inertia_tensor-2-2"]],
    ], axis=0).T                                 # shape (N, 3, 3)
    _, eigvecs = np.linalg.eigh(IT)             # eigvecs[:, :, k] for k-th vector
    longest_vecs = eigvecs[:, :, 0]             # smallest eigval → longest axis
    axis_dirs    = [
        ["Z", "Y", "X"][int(np.argmax(np.abs(v)))]
        for v in longest_vecs
    ]

    # Axis scale along dominant direction
    axis_scale_map = {"Z": sz, "Y": sy, "X": sx}
    axis1_scales   = np.array([axis_scale_map[d] for d in axis_dirs])
    scale_mean     = (sz + sy + sx) / 3.0

    # Principal axis lengths (voxel → µm)
    a1_um = table["axis_major_length"] * axis1_scales
    a3_um = table["axis_minor_length"] * scale_mean
    a2_um = (a1_um + a3_um) / 2.0              # approximation for middle axis
    elongation = a1_um / np.maximum(a3_um, 1e-10)

    # Bounding box size in µm
    dz_um = (table["bbox-3"] - table["bbox-0"]) * sz
    dy_um = (table["bbox-4"] - table["bbox-1"]) * sy
    dx_um = (table["bbox-5"] - table["bbox-2"]) * sx

    vol_um3    = table["area"] * vox_vol
    eq_diam_um = (6.0 * vol_um3 / math.pi) ** (1.0 / 3.0)

    # ── Phase 2: per-label marching cubes + skeleton (parallelised) ───────
    print(f"   Surface area + skeleton ({_N_THREADS} threads)...")
    bbox_list = list(zip(
        table["bbox-0"].astype(int), table["bbox-1"].astype(int),
        table["bbox-2"].astype(int), table["bbox-3"].astype(int),
        table["bbox-4"].astype(int), table["bbox-5"].astype(int),
    ))
    worker_args = [
        (int(lbl), labels, bbox, (sz, sy, sx))
        for lbl, bbox in zip(table["label"], bbox_list)
    ]
    with ThreadPoolExecutor(max_workers=_N_THREADS) as pool:
        slow = list(pool.map(_slow_stats_worker, worker_args))

    # Index slow results by label
    slow_by_lbl = {lbl: (sa, n_br, n_ep, br_len) for lbl, sa, n_br, n_ep, br_len in slow}

    # ── Phase 3: assemble rows + descriptions ─────────────────────────────
    print(f"   Assembling rows and generating descriptions...")
    rows: list[dict] = []
    for i in range(N):
        lbl   = int(table["label"][i])
        sa    = slow_by_lbl[lbl][0]
        n_br  = slow_by_lbl[lbl][1]
        n_ep  = slow_by_lbl[lbl][2]
        br_l  = slow_by_lbl[lbl][3]

        vol   = float(vol_um3[i])
        spher = 0.0
        if sa > 0:
            spher = min(
                float((math.pi ** (1.0 / 3.0)) * ((6.0 * vol) ** (2.0 / 3.0)) / sa),
                1.0,
            )

        try:
            solid = float(table["solidity"][i])
        except Exception:
            solid = 1.0

        row = {
            "label":              lbl,
            "volume_vox":         int(table["area"][i]),
            "volume_um3":         round(vol, 2),
            "centroid_z_vox":     round(float(table["centroid-0"][i]), 2),
            "centroid_y_vox":     round(float(table["centroid-1"][i]), 2),
            "centroid_x_vox":     round(float(table["centroid-2"][i]), 2),
            "centroid_z_um":      round(float(table["centroid-0"][i]) * sz, 2),
            "centroid_y_um":      round(float(table["centroid-1"][i]) * sy, 2),
            "centroid_x_um":      round(float(table["centroid-2"][i]) * sx, 2),
            "bbox_dz_um":         round(float(dz_um[i]), 2),
            "bbox_dy_um":         round(float(dy_um[i]), 2),
            "bbox_dx_um":         round(float(dx_um[i]), 2),
            "eq_diam_um":         round(float(eq_diam_um[i]), 2),
            "axis1_um":           round(float(a1_um[i]), 2),
            "axis2_um":           round(float(a2_um[i]), 2),
            "axis3_um":           round(float(a3_um[i]), 2),
            "elongation":         round(float(elongation[i]), 3),
            "principal_axis_dir": axis_dirs[i],
            "solidity":           round(solid, 4),
            "extent":             round(float(table["extent"][i]), 4),
            "surface_area_um2":   round(sa, 2),
            "sphericity":         round(spher, 4),
            "n_branches":         n_br,
            "n_endpoints":        n_ep,
            "mean_branch_len_um": round(br_l, 2),
        }
        row["description"] = desc_fn(row)
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"   Done — {len(df)} labels.")
    print(f"{'='*60}\n")
    return df
