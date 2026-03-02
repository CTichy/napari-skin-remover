# napari-skin-remover

A [napari](https://napari.org) plugin for automated 3D brain extraction (skin removal) from zebrafish confocal stacks using a MONAI 3D U-Net.

Developed at **FH Technikum Wien** — Artificial Intelligence & Data Science.

---

## What it does

Given a 3D confocal volume (TIF or IMS), the plugin:

1. Runs a trained MONAI U-Net to predict the brain mask
2. Post-processes the mask (largest connected component + fill holes)
3. Optionally erodes the mask to strip residual skin from the brain surface
4. Outputs `brain_mask.tif` and `brain_only.tif` alongside the source file

---

## Installation

```bash
pip install git+https://github.com/CTichy/napari-skin-remover.git
```

Or clone and install in editable mode:

```bash
git clone https://github.com/CTichy/napari-skin-remover.git
cd napari-skin-remover
pip install -e .
```

**Dependencies** (installed automatically): `napari`, `torch`, `monai`, `tifffile`, `numpy`, `qtpy`

**Optional** (for `.ims` Imaris file support):

```bash
pip install imaris_ims_file_reader
```

---

## Model file

The plugin requires a trained `.pth` checkpoint — **not included in this repo** (file size ~220 MB).

**Download:**
[best_model_fullstack_v1_epoch460_dice9573.pth](https://cloud.technikum-wien.at/s/kYQ4qq3Jsn4xEyY)

**Where to put it:**

Create a `models` folder anywhere convenient, for example:

```
Documents/
└── skin-remover-model/
    └── best_model_fullstack_v1_epoch460_dice9573.pth
```

Then point the plugin to it using the **Browse (...)** button in the widget. The path is saved automatically and remembered across sessions — you only need to do this once.

---

## Usage

### From napari (Plugins menu)

```bash
napari
```

Then: **Plugins → MONAI Skin-Remover**

### CLI

```bash
skin-remover path/to/stack.tif
python -m napari_skin_remover path/to/stack.ims
```

### Workflow

1. **Open a file** — click "Open TIF / IMS file" or drag & drop into napari.
   All channels are loaded as separate layers (ch0=gray, ch1=green, ch2=magenta, ch3=cyan).
2. **Select the channel** to process by clicking its layer in the Layers panel.
3. **Browse to the model** `.pth` file if not auto-detected.
4. **Adjust threshold** (default 0.30 — keep low, post-processing cleans the rest).
5. **Adjust erosion** (0–15 voxels, optional — strips skin rim from `brain_only`; `brain_mask` is always saved un-eroded).
6. Click **Run Brain Peel**.

Results are added as new layers and saved next to the source file.

---

## Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| Threshold | 0.30 | Sigmoid probability cutoff. Keep low — post-processing handles cleanup. |
| Erosion | 0 vox | Erodes the mask applied to `brain_only`. 2–3 vox typical for zebrafish. |

### Typical voxel dimensions (zebrafish 4 dpf, 25× objective)

| Axis | Size |
|------|------|
| Z | 1.0 µm |
| X, Y | 0.174 µm |
| Anisotropy | ~5.75:1 |

---

## File format support

| Format | Read | Channels | Metadata |
|--------|------|----------|----------|
| `.tif` / `.tiff` | Yes | single or multi-channel (C,Z,Y,X) | ImageJ tags or `*_metadata.txt` |
| `.ims` (Imaris) | Yes (requires `imaris_ims_file_reader`) | all channels | embedded or `*_metadata.txt` |

Voxel metadata is extracted automatically (priority: external `*_metadata.txt` → TIF ImageJ tags → IMS embedded → default 1,1,1).

---

## Output files

Saved in the same directory as the input:

| File | Content |
|------|---------|
| `*_brain_only.tif` | Input volume with everything outside the brain zeroed |
| `*_brain_mask.tif` | Binary mask (0/255 uint8), un-eroded |

---

## Troubleshooting

**"no default model found"** — expected. Use the Browse button to select your `.pth` file.

**CUDA out of memory** — the plugin will fall back to CPU automatically if no GPU is available. On CPU, inference takes several minutes per stack.

**`.ims` files fail to open** — install `imaris_ims_file_reader`: `pip install imaris_ims_file_reader`

---

## Contact

Carlos Tichy — ai24m016@technikum-wien.at
FH Technikum Wien — Artificial Intelligence & Data Science
