# MONAI Skin-Remover — Complete User Guide

**For zebrafish confocal microscopy — step by step, from zero to microglia labels.**

---

## Table of Contents

1. [What this plugin does](#1-what-this-plugin-does)
2. [Installation](#2-installation)
3. [Getting the model file](#3-getting-the-model-file)
4. [Opening the plugin in napari](#4-opening-the-plugin-in-napari)
5. [Tab 1 — Skin Remover: every option explained](#5-tab-1--skin-remover-every-option-explained)
6. [Tab 2 — Create Labels: every option explained](#6-tab-2--create-labels-every-option-explained)
7. [Full workflow: from raw stack to microglia labels](#7-full-workflow-from-raw-stack-to-microglia-labels)
8. [Reinstalling after an update](#8-reinstalling-after-an-update)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. What this plugin does

You have a confocal microscopy stack of a zebrafish brain. The image contains:
- The brain you care about
- Skin, tissue, and background surrounding the brain

This plugin does two things:

**Step A — Skin Removal:** Uses a trained AI model (MONAI U-Net) to automatically detect and remove everything outside the brain, leaving you with a clean `brain_only` image.

**Step B — Create Labels:** From the cleaned image, automatically finds and labels each individual microglia (or other stained cell type) as a separate numbered region — ready for analysis.

---

## 2. Installation

You need Python with napari already installed. Open a terminal and run:

```bash
pip install git+https://github.com/CTichy/napari-skin-remover.git
```

That's it. All dependencies (PyTorch, MONAI, etc.) are installed automatically.

> **Mac users with Apple Silicon (M1/M2/M3):** The plugin automatically uses your GPU via Metal (MPS). No extra steps needed — it will be much faster than CPU.

> **Windows/Linux with NVIDIA GPU:** Also automatic — CUDA is detected and used.

> **No GPU:** Works on CPU too, just slower (~30–60 minutes per stack).

---

## 3. Getting the model file

The AI model is a large file (~220 MB) that is **not included in the plugin**. You need to download it separately.

**Download the model here:**
👉 [best_model_fullstack.pth](https://cloud.technikum-wien.at/s/kYQ4qq3Jsn4xEyY)

```
https://cloud.technikum-wien.at/s/kYQ4qq3Jsn4xEyY
```

**Where to save it:**

Create a folder called `models` anywhere on your computer that is easy to find. For example:

```
Documents/
└── brain-peel-model/
    └── best_model_fullstack.pth
```

You will point the plugin to this file the first time you use it. **The plugin remembers the path** — you only need to do this once.

---

## 4. Opening the plugin in napari

1. Open a terminal and type `napari` to launch it
2. In the napari menu bar, click **Plugins**
3. Click **MONAI Skin-Remover**
4. A panel appears on the right side of napari with two tabs: **Skin Remover** and **Create Labels**

---

## 5. Tab 1 — Skin Remover: every option explained

### Open TIF / IMS file

Click this button to open your microscopy file (`.tif`, `.tiff`, or `.ims` format).

- All channels in the file are loaded as **separate layers** in napari, each with a different color:
  - Channel 0 → gray
  - Channel 1 → green
  - Channel 2 → magenta
  - Channel 3 → cyan
- Voxel size (scale) is read automatically from the file metadata

> **Important:** After loading, **click on the channel you want to process** in the Layers panel on the left. This tells the plugin which channel to run the AI on. For microglia, this is usually the green channel (ch1).

---

### Model (.pth) — Browse button `[...]`

The plugin shows the path to the AI model. If it says "— no model selected —":

1. Click the `[...]` button
2. Navigate to where you saved the model file
3. Select `best_model_fullstack.pth`
4. Click Open

The path is saved automatically. Next time you open the plugin, it will already be there.

---

### Threshold slider

**Range:** 0.01 to 0.99 — **Default: 0.30**

The AI model outputs a probability map (0 = definitely not brain, 1 = definitely brain). This slider sets the cutoff:

- **Lower values (e.g. 0.20):** More generous — includes uncertain areas, might keep some skin
- **Higher values (e.g. 0.50):** More strict — might cut into brain edges
- **Recommended:** Start at 0.30, adjust only if results look wrong

> Most of the time, **0.30 works well** because the post-processing (largest component + fill holes) cleans up the result anyway.

---

### Erosion slider

**Range:** 0 to 15 voxels — **Default: 0**

After the brain mask is computed, erosion "shrinks" the mask inward by the specified number of voxels. This strips a rim of skin from the edge of `brain_only`.

- **0:** No erosion — use the raw mask as-is
- **2–3:** Typical for zebrafish — removes a thin skin rim
  - 1 voxel in Z ≈ 1 µm
  - 1 voxel in XY ≈ 0.17 µm

> **Note:** The `brain_mask.tif` saved to disk is **always the un-eroded mask**. Erosion only affects `brain_only.tif`.

---

### Background (brain mode) — three options

This section removes or replaces the background signal inside the brain area. The background level is estimated automatically using the **mode** (most frequent intensity) of pixels inside the brain, computed after inference.

> **What is "mode"?** The mode is the intensity value that appears most often. Since background pixels are the majority of pixels inside the brain, the most common value is the background level.

#### Option: Off
No background processing. `brain_only` is simply the original volume multiplied by the brain mask — everything outside the brain is zero, everything inside is the original signal.

#### Option 1 — Remove background outside brain (inference)
Removes background-level pixels only **outside** the brain. Useful for cleaning up the outer region while leaving the brain completely untouched.

- Uses the same threshold logic as option 2
- Brain interior is fully protected — nothing inside changes

#### Option 2 — Remove background globally (full stack) ⭐ **Recommended for labels**
Removes all pixels in the **entire stack** (including inside the brain) whose intensity is at or below the background threshold.

**Result:** Only the actual signal (bright microglia, stained cells) survives. Everything at background level becomes zero, leaving clean isolated blobs with empty space between them.

**This is the option to use before creating labels.**

#### Option 3 — Fill removed with random background
After skin removal, fills the empty region outside the brain with **random noise** sampled from the background pixel distribution. The result looks like the original stack but with the skin replaced by natural scanner noise — no hard black boundary.

- Uses Gaussian-filtered sampling (±2σ outlier removal) so the noise matches the real scanner texture
- No tolerance needed for this mode

---

### Tolerance slider

**Range:** -1.00% to +1.00% — **Default: +0.50%**

*(Only active for options 1 and 2)*

Fine-tunes the background threshold:

```
threshold = background_mode + tolerance
pixels ≤ threshold → removed (background)
pixels > threshold → kept (signal)
```

| Tolerance | Effect |
|-----------|--------|
| **Negative (e.g. -0.50%)** | Threshold drops → only the very darkest pixels removed → more signal preserved |
| **Zero (0.00%)** | Threshold = exactly the mode → remove background, keep all signal above it |
| **Positive (e.g. +0.60%)** | Threshold rises → removes background + some dim signal → cleaner separation |

> **For microglia labeling, +0.60% works very well** — it produces clean isolated blobs with clear gaps between cells that the label algorithm can separate properly.

---

### Save checkboxes

- **Save brain_only.tif** — saves the processed brain volume (checked by default)
- **Save brain_mask.tif** — saves the binary mask as 0/255 uint8 (checked by default)

Both files are saved in the **same folder as your input file**, with `_brain_only` and `_brain_mask` added to the filename.

---

### Run Skin-Remover button

Click to start processing. The status bar below the button shows progress. Results appear as new layers in napari:

- `*_brain_mask` — the binary mask (shown in cyan, semi-transparent)
- `*_brain_only` — the cleaned brain volume

Processing time:
- NVIDIA GPU: ~30 seconds
- Apple Silicon (MPS): ~5–10 minutes
- CPU only: ~30–60 minutes

---

## 6. Tab 2 — Create Labels: every option explained

> **Before using this tab:** Run Tab 1 with **Option 2 — Remove background globally** first. The Create Labels function reads the resulting `brain_only` layer.

Click the **brain_only** layer in the Layers panel to select it, then switch to this tab.

---

### Smooth σ XY slider

**Range:** 0.0 to 5.0 — **Default: 1.0** — **Recommended: 1.5**

Controls the **smoothness of blob contours within each 2D slice** (XY plane).

- Rounds jagged edges and fills tiny pixel-level gaps within the same slice
- **This is the knob to adjust for softer, rounder blob outlines**
- Keep this low — high values risk merging neighbouring cells within a slice

| Value | Effect |
|-------|--------|
| 0.0 | No smoothing — raw jagged edges |
| 1.0 | Light smoothing — fine details preserved |
| **1.5** | **Recommended — smooth solid blobs, cells stay separated** |
| 3.0+ | Heavy smoothing — risk of merging nearby cells within the same slice |

---

### Smooth σ Z slider

**Range:** 0.0 to 5.0 — **Default: 0.5** — **Recommended: 3.0**

Controls **cross-slice connectivity** — how easily blobs in neighbouring Z slices are recognised as part of the same 3D cell.

- A cell that disappears for 1–2 slices and reappears will be correctly linked with a higher σ Z
- **σ Z and σ XY are NOT comparable values** — they serve completely different purposes and operate at different physical scales

> **Why is σ Z = 3.0 while σ XY = 1.5?**
> Zebrafish confocal stacks are highly anisotropic: each Z slice is ~1 µm thick while each XY pixel is only ~0.17 µm. A σ Z of 3.0 voxels spans ~3 µm physically. A σ XY of 1.5 voxels spans only ~0.26 µm.
>
> Both sigmas carry a merging risk — but at different physical scales:
> - σ XY = 1.5 risks merging cells that are ~0.26 µm apart in XY (very tight — happens easily in dense tissue)
> - σ Z = 3.0 risks merging cells that are ~3 µm apart in Z (much less likely for microglia, which are typically 10–20 µm in diameter)
>
> **Conclusion:** σ Z = 3.0 is a validated safe choice for zebrafish microglia, but increasing it beyond 4–5 may start merging cells that lie close to each other in the Z direction.

| Value | Effect |
|-------|--------|
| 0.0 | No cross-slice smoothing — each slice fully independent |
| 0.5 | Minimal — only immediately adjacent slices connected |
| **3.0** | **Recommended for zebrafish — bridges 1–3 slice gaps within a cell** |
| 5.0+ | Very aggressive — may link cells at different Z depths |

---

### Min overlap (%) slider

**Range:** 1 to 100 — **Default: 10%**

Two blobs in adjacent slices are considered the **same 3D cell** if they overlap by at least this percentage.

```
overlap_ratio = shared_pixels / smaller_blob_area
if overlap_ratio ≥ min_overlap% → same object
```

- **Lower values (e.g. 5%):** More permissive — blobs with small overlap still linked
- **Higher values (e.g. 30%):** More strict — only heavily overlapping blobs linked
- **Start at 10%** and adjust if cells are being split or merged incorrectly

---

### Min volume (vox) slider

**Range:** 5000 to 10000 — **Default: 7500**

After all blobs are linked into 3D objects, any object smaller than this number of voxels is **deleted** as noise/fragment.

- **5000 vox:** Keeps smaller objects — might include noise fragments
- **7500 vox:** Good default for adult zebrafish microglia
- **10000 vox:** Strict — only keeps large objects

> Microglia in zebrafish 4dpf typically occupy ~15,000–50,000 voxels at standard resolution.

---

### Save labels.tif checkbox

When checked (default), saves the labels as an int32 TIF file named `*_labels.tif` in the same folder as your input.

Each label number (1, 2, 3...) corresponds to one 3D object. The file can be opened in napari or Fiji/ImageJ for further analysis.

---

### Create Labels button

Click to run. The console output shows:

```
2D blobs: 4823                          ← blobs found per slice before linking
3D blobs removed (< 7500 vox): 4756    ← noise removed
Final 3D labels: 67                     ← microglia found
```

A new `*_labels` layer appears in napari, each cell shown in a different color.

---

## 7. Full workflow: from raw stack to microglia labels

This is the complete recommended workflow for creating microglia labels. Settings shown are those that have been validated and work well.

### Step 1 — Open your file

1. In the plugin (Tab 1), click **Open TIF / IMS file**
2. Select your confocal stack
3. All channels appear as layers
4. **Click the microglia channel** (usually ch1, green) in the Layers panel

### Step 2 — Run skin removal

Set these values in Tab 1:

| Setting | Value |
|---------|-------|
| Threshold | 0.30 (default) |
| Erosion | 0 (default) |
| Background | **Option 2 — Remove globally** |
| Tolerance | **+0.60%** |

Click **Run Skin-Remover** and wait.

**What you should see:** A `brain_only` layer where microglia appear as bright isolated blobs on a black background, with clear space between cells. If blobs look hollow or have halos, adjust the tolerance slightly.

> **Tolerance tuning tips:**
> - Too many dim pixels remaining → increase tolerance (e.g. +0.80%)
> - Microglia missing pieces → decrease tolerance (e.g. +0.40%)
> - The console prints the exact threshold used so you can compare runs

### Step 3 — Create labels

1. **Click the `brain_only` layer** in the Layers panel to select it
2. Switch to the **Create Labels** tab
3. Set these values:

| Setting | Value |
|---------|-------|
| Smooth σ XY | **1.5** |
| Smooth σ Z | **3.0** |
| Min overlap | 10% (default) |
| Min volume | 7500 (default) |

4. Click **Create Labels**

**What you should see:** A labeled layer where each microglia is a different color. The console shows how many were found.

> **Tuning tips:**
> - Too many fragments → increase σ XY or σ Z
> - Cells merging together → decrease σ XY
> - Many very small objects → increase Min volume
> - Large cells being cut into pieces → decrease Min overlap %

### Step 4 — Check results in napari

- Toggle the labels layer on/off to compare with the original
- Zoom into slices to verify cells are correctly separated
- The label number of each cell is shown when you hover over it

---

## 8. Reinstalling after an update

When a new version of the plugin is released, reinstall with:

```bash
pip uninstall napari-skin-remover -y
pip install git+https://github.com/CTichy/napari-skin-remover.git
```

Then **fully close and reopen napari** — important! If napari is still running when you reinstall, it will keep using the old version until restarted.

> **Your model path is remembered** across reinstalls — you do not need to browse to the `.pth` file again.

---

## 9. Troubleshooting

### The plugin doesn't appear in Plugins menu
- Make sure napari is fully closed and reopened after installation
- Try running: `pip show napari-skin-remover` to confirm it installed

### "No model selected" after reinstalling
- Click `[...]` and browse to your `.pth` file again
- The config is stored in `~/.config/napari-skin-remover/config.json`

### Processing runs on CPU (very slow) on Mac
- Check your PyTorch version supports MPS: `python -c "import torch; print(torch.backends.mps.is_available())"`
- Should print `True` on Apple Silicon

### brain_only looks mostly empty / all black
- Tolerance is too high — lower it (e.g. from +0.60% to +0.30%)
- Or background mode probe got confused — try running on a different channel first

### Create Labels finds 0 or too few objects
- Lower the Min volume slider (e.g. to 5000)
- Increase σ XY and σ Z slightly
- Make sure you selected the `brain_only` layer (not the original channel) before clicking Create Labels

### Create Labels finds hundreds of tiny fragments
- Increase Min volume to 10000
- Make sure Option 2 with a sufficient tolerance was run first — the brain_only layer must have clean gaps between cells

### The labels layer shows cells merging that shouldn't merge
- Lower σ XY (e.g. from 1.5 to 1.0)
- Lower σ Z (e.g. from 3.0 to 1.5)
- Increase Min overlap % (e.g. from 10% to 20%)

---

## Quick Reference Card

### Tab 1 — Skin Remover

| Control | Recommended | What it does |
|---------|------------|--------------|
| Threshold | 0.30 | AI confidence cutoff for brain detection |
| Erosion | 0 | Shrinks mask inward (strips skin rim) |
| Background | Option 2 | Removes background globally |
| Tolerance | +0.60% | Fine-tunes background threshold |

### Tab 2 — Create Labels

| Control | Recommended | What it does |
|---------|------------|--------------|
| σ XY | 1.5 | Contour softness in each slice |
| σ Z | 3.0 | Cross-slice blob connection |
| Min overlap | 10% | Overlap needed to link blobs across slices |
| Min volume | 7500 | Minimum voxels to keep a 3D object |

---

*Plugin developed at FH Technikum Wien — Artificial Intelligence & Data Science*
*Contact: carlos.tichy@gmail.com*
