"""
_widget.py — SkinRemoverWidget napari dock panel.
"""

import json
import threading
import traceback
from pathlib import Path

import numpy as np
import tifffile
import torch
import napari

from qtpy.QtWidgets import (
    QPushButton, QLabel, QWidget, QVBoxLayout, QHBoxLayout,
    QSlider, QCheckBox, QFileDialog, QSizePolicy, QButtonGroup, QRadioButton,
    QTabWidget, QComboBox,
)
from qtpy.QtCore import Qt, QTimer

from ._io import load_file
from ._inference import DEFAULT_MODEL, _SKIN_SEG_DIR, run_inference
from ._background import remove_outside_brain, remove_global, fill_outside_brain_random
from ._labeling import create_labels, resort_labels

_CONFIG_PATH = Path.home() / ".config" / "napari-skin-remover" / "config.json"


def _load_saved_model_path():
    """Return the last-used model path from config, or None."""
    try:
        if _CONFIG_PATH.exists():
            data = json.loads(_CONFIG_PATH.read_text())
            p = Path(data.get("model_path", ""))
            if p.exists():
                return p
    except Exception:
        pass
    return None


def _save_model_path(path: Path):
    """Persist model path to config file."""
    try:
        _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CONFIG_PATH.write_text(json.dumps({"model_path": str(path)}))
    except Exception:
        pass


def _sep():
    """Thin horizontal separator line."""
    w = QWidget()
    w.setFixedHeight(1)
    w.setStyleSheet("background-color: #666;")
    return w


class SkinRemoverWidget(QWidget):
    """
    Napari dock panel for MONAI skin removal.

    Layout
    ------
    [Open TIF / IMS file]
    ─────────────────────
    Model (.pth):
    [path…]  [...]
    ─────────────────────
    Input: bottom layer (auto)
      "{name}"  (Z×Y×X  dtype)
    ─────────────────────
      Z=1.0000  Y=0.1740  X=0.1740 µm
      Anisotropy 5.75:1  |  TIF ImageJ metadata
    ─────────────────────
    Threshold: [────●──]  0.30
    ─────────────────────
    [x] Save brain_only.tif
    [x] Save brain_mask.tif
    ─────────────────────
    [     Run Skin-Remover     ]
    Status: Ready
    """

    def __init__(self, napari_viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = napari_viewer
        # Model path priority: saved config > hardcoded default > None
        saved = _load_saved_model_path()
        if saved:
            initial_model = saved
        elif DEFAULT_MODEL.exists():
            initial_model = DEFAULT_MODEL
        else:
            initial_model = None
        self._state = {
            "model_path":     initial_model,
            "last_file_path": None,
            "metadata":       None,
        }
        self._build_ui()
        self._connect_signals()
        self._refresh_layer_info()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #

    def _build_ui(self):
        tabs = QTabWidget()

        # ============================================================ #
        # TAB 1 — Skin Remover
        # ============================================================ #
        tab1 = QWidget()
        t1 = QVBoxLayout()
        t1.setSpacing(6)

        self._open_btn = QPushButton("Open TIF / IMS file")
        t1.addWidget(self._open_btn)

        t1.addWidget(_sep())

        t1.addWidget(QLabel("Model (.pth):"))
        model_row = QHBoxLayout()
        self._model_lbl = QLabel(
            str(self._state["model_path"]) if self._state["model_path"] else "— no model selected —"
        )
        self._model_lbl.setWordWrap(True)
        self._model_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._model_browse_btn = QPushButton("...")
        self._model_browse_btn.setFixedWidth(32)
        model_row.addWidget(self._model_lbl)
        model_row.addWidget(self._model_browse_btn)
        t1.addLayout(model_row)

        t1.addWidget(_sep())

        t1.addWidget(QLabel("Input: active (selected) layer"))
        self._layer_info = QLabel("  — no layers yet —")
        self._layer_info.setWordWrap(True)
        t1.addWidget(self._layer_info)

        t1.addWidget(_sep())

        self._meta_lbl = QLabel("  — voxel info unavailable —")
        self._meta_lbl.setWordWrap(True)
        self._meta_lbl.setStyleSheet("color: #aaa; font-size: 10px;")
        t1.addWidget(self._meta_lbl)

        t1.addWidget(_sep())

        thresh_row = QHBoxLayout()
        thresh_row.addWidget(QLabel("Threshold:"))
        self._thresh_slider = QSlider(Qt.Horizontal)
        self._thresh_slider.setMinimum(1)
        self._thresh_slider.setMaximum(99)
        self._thresh_slider.setValue(30)
        self._thresh_val = QLabel("0.30")
        self._thresh_val.setFixedWidth(36)
        thresh_row.addWidget(self._thresh_slider)
        thresh_row.addWidget(self._thresh_val)
        t1.addLayout(thresh_row)

        erosion_row = QHBoxLayout()
        erosion_row.addWidget(QLabel("Erosion (vox):"))
        self._erosion_slider = QSlider(Qt.Horizontal)
        self._erosion_slider.setMinimum(0)
        self._erosion_slider.setMaximum(15)
        self._erosion_slider.setValue(0)
        self._erosion_val = QLabel("0")
        self._erosion_val.setFixedWidth(24)
        erosion_row.addWidget(self._erosion_slider)
        erosion_row.addWidget(self._erosion_val)
        t1.addLayout(erosion_row)
        erosion_note = QLabel(
            "  Erodes mask before applying to brain_only\n"
            "  (raw brain_mask is always saved un-eroded)"
        )
        erosion_note.setStyleSheet("color: #aaa; font-size: 10px;")
        t1.addWidget(erosion_note)

        t1.addWidget(_sep())

        t1.addWidget(QLabel("Background (brain mode):"))
        self._bg_group = QButtonGroup(self)
        self._bg_off_rb    = QRadioButton("Off")
        self._bg_mode1_rb  = QRadioButton("1 — Remove background outside brain (inference)")
        self._bg_mode2_rb  = QRadioButton("2 — Remove background globally (full stack)")
        self._bg_mode3_rb  = QRadioButton("3 — Fill removed with random background")
        self._bg_group.addButton(self._bg_off_rb,   0)
        self._bg_group.addButton(self._bg_mode1_rb, 1)
        self._bg_group.addButton(self._bg_mode2_rb, 2)
        self._bg_group.addButton(self._bg_mode3_rb, 3)
        self._bg_off_rb.setChecked(True)
        t1.addWidget(self._bg_off_rb)
        t1.addWidget(self._bg_mode1_rb)
        t1.addWidget(self._bg_mode2_rb)
        t1.addWidget(self._bg_mode3_rb)

        tol_row = QHBoxLayout()
        self._tol_lbl = QLabel("  Tolerance (%):")
        tol_row.addWidget(self._tol_lbl)
        self._tol_slider = QSlider(Qt.Horizontal)
        self._tol_slider.setMinimum(-100)
        self._tol_slider.setMaximum(100)
        self._tol_slider.setValue(50)
        self._tol_val = QLabel("+0.50")
        self._tol_val.setFixedWidth(42)
        tol_row.addWidget(self._tol_slider)
        tol_row.addWidget(self._tol_val)
        t1.addLayout(tol_row)

        bg_note = QLabel(
            "  Probe: inside-brain mode (post-inference)\n"
            "  Mode 1 & 2 use tolerance  |  Mode 3: no tolerance"
        )
        bg_note.setStyleSheet("color: #aaa; font-size: 10px;")
        t1.addWidget(bg_note)

        t1.addWidget(_sep())

        self._save_only_cb = QCheckBox("Save brain_only.tif")
        self._save_only_cb.setChecked(True)
        self._save_mask_cb = QCheckBox("Save brain_mask.tif")
        self._save_mask_cb.setChecked(True)
        t1.addWidget(self._save_only_cb)
        t1.addWidget(self._save_mask_cb)

        t1.addWidget(_sep())

        self._run_btn = QPushButton("Run Skin-Remover")
        self._run_btn.setStyleSheet("QPushButton { font-weight: bold; padding: 6px; }")
        t1.addWidget(self._run_btn)

        self._status_lbl = QLabel("Status: Ready")
        self._status_lbl.setWordWrap(True)
        t1.addWidget(self._status_lbl)

        t1.addStretch()
        tab1.setLayout(t1)
        tabs.addTab(tab1, "Skin Remover")

        # ============================================================ #
        # TAB 2 — Create Labels
        # ============================================================ #
        tab2 = QWidget()
        t2 = QVBoxLayout()
        t2.setSpacing(6)

        lbl_note = QLabel(
            "Run option 2 (Remove globally) first to get a\n"
            "brain_only layer, then select it and click below."
        )
        lbl_note.setWordWrap(True)
        lbl_note.setStyleSheet("color: #aaa; font-size: 10px;")
        t2.addWidget(lbl_note)

        t2.addWidget(_sep())

        sxy_row = QHBoxLayout()
        sxy_row.addWidget(QLabel("Smooth σ XY:"))
        self._sxy_slider = QSlider(Qt.Horizontal)
        self._sxy_slider.setMinimum(0)
        self._sxy_slider.setMaximum(50)
        self._sxy_slider.setValue(10)
        self._sxy_val = QLabel("1.0")
        self._sxy_val.setFixedWidth(28)
        sxy_row.addWidget(self._sxy_slider)
        sxy_row.addWidget(self._sxy_val)
        t2.addLayout(sxy_row)

        sz_row = QHBoxLayout()
        sz_row.addWidget(QLabel("Smooth σ Z:"))
        self._sz_slider = QSlider(Qt.Horizontal)
        self._sz_slider.setMinimum(0)
        self._sz_slider.setMaximum(50)
        self._sz_slider.setValue(5)
        self._sz_val = QLabel("0.5")
        self._sz_val.setFixedWidth(28)
        sz_row.addWidget(self._sz_slider)
        sz_row.addWidget(self._sz_val)
        t2.addLayout(sz_row)

        ovlp_row = QHBoxLayout()
        ovlp_row.addWidget(QLabel("Min overlap (%):"))
        self._ovlp_slider = QSlider(Qt.Horizontal)
        self._ovlp_slider.setMinimum(1)
        self._ovlp_slider.setMaximum(100)
        self._ovlp_slider.setValue(10)
        self._ovlp_val = QLabel("10")
        self._ovlp_val.setFixedWidth(28)
        ovlp_row.addWidget(self._ovlp_slider)
        ovlp_row.addWidget(self._ovlp_val)
        t2.addLayout(ovlp_row)

        area_row = QHBoxLayout()
        area_row.addWidget(QLabel("Min volume (vox):"))
        self._area_slider = QSlider(Qt.Horizontal)
        self._area_slider.setMinimum(5000)
        self._area_slider.setMaximum(10000)
        self._area_slider.setValue(7500)
        self._area_val = QLabel("7500")
        self._area_val.setFixedWidth(40)
        area_row.addWidget(self._area_slider)
        area_row.addWidget(self._area_val)
        t2.addLayout(area_row)

        t2.addWidget(_sep())

        self._save_labels_cb = QCheckBox("Save labels.tif")
        self._save_labels_cb.setChecked(True)
        t2.addWidget(self._save_labels_cb)

        self._labels_btn = QPushButton("Create Labels")
        self._labels_btn.setStyleSheet("QPushButton { font-weight: bold; padding: 6px; }")
        t2.addWidget(self._labels_btn)

        self._labels_status_lbl = QLabel("")
        self._labels_status_lbl.setWordWrap(True)
        t2.addWidget(self._labels_status_lbl)

        t2.addWidget(_sep())

        sort_row = QHBoxLayout()
        sort_row.addWidget(QLabel("Sort by:"))
        self._sort_combo = QComboBox()
        self._sort_combo.addItem("Size",       "size")
        self._sort_combo.addItem("Centroid Z", "centroid_z")
        self._sort_combo.addItem("Centroid Y", "centroid_y")
        self._sort_combo.addItem("Centroid X", "centroid_x")
        sort_row.addWidget(self._sort_combo)
        t2.addLayout(sort_row)

        self._sort_reverse_cb = QCheckBox("Reverse order")
        t2.addWidget(self._sort_reverse_cb)

        self._resort_btn = QPushButton("Resort Labels")
        self._resort_btn.setStyleSheet("QPushButton { padding: 5px; }")
        t2.addWidget(self._resort_btn)

        self._resort_status_lbl = QLabel("")
        self._resort_status_lbl.setWordWrap(True)
        t2.addWidget(self._resort_status_lbl)

        t2.addStretch()
        tab2.setLayout(t2)
        tabs.addTab(tab2, "Create Labels")

        # ── outer layout ────────────────────────────────────────────── #
        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(tabs)
        self.setLayout(outer)

    # ------------------------------------------------------------------ #
    # Signal connections
    # ------------------------------------------------------------------ #

    def _connect_signals(self):
        self._open_btn.clicked.connect(self._on_open)
        self._model_browse_btn.clicked.connect(self._on_browse_model)
        self._thresh_slider.valueChanged.connect(
            lambda v: self._thresh_val.setText(f"{v / 100:.2f}")
        )
        self._erosion_slider.valueChanged.connect(
            lambda v: self._erosion_val.setText(str(v))
        )
        self._tol_slider.valueChanged.connect(
            lambda v: self._tol_val.setText(f"{v/100:+.2f}")
        )
        self._bg_group.buttonClicked.connect(self._on_bg_mode_changed)
        self._run_btn.clicked.connect(self._on_run)
        self._sxy_slider.valueChanged.connect(
            lambda v: self._sxy_val.setText(f"{v/10:.1f}")
        )
        self._sz_slider.valueChanged.connect(
            lambda v: self._sz_val.setText(f"{v/10:.1f}")
        )
        self._ovlp_slider.valueChanged.connect(
            lambda v: self._ovlp_val.setText(str(v))
        )
        self._area_slider.valueChanged.connect(
            lambda v: self._area_val.setText(f"{v:,}")
        )
        self._labels_btn.clicked.connect(self._on_create_labels)
        self._resort_btn.clicked.connect(self._on_resort_labels)
        self._viewer.layers.events.inserted.connect(self._refresh_layer_info)
        self._viewer.layers.events.removed.connect(self._refresh_layer_info)
        self._viewer.layers.selection.events.changed.connect(self._refresh_layer_info)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _status(self, msg):
        self._status_lbl.setText(f"Status: {msg}")

    def _on_bg_mode_changed(self, btn):
        """Enable/disable tolerance slider depending on mode."""
        mode = self._bg_group.checkedId()
        has_tol = mode in (1, 2)
        self._tol_slider.setEnabled(has_tol)
        self._tol_val.setEnabled(has_tol)
        self._tol_lbl.setEnabled(has_tol)

    def _get_layer_scale(self):
        """
        Return (z, y, x) scale in µm.

        Priority:
          1. metadata from a file loaded via Open button
          2. scale of the active layer (respects per-layer scale set by reader)
          3. default (1.0, 1.0, 1.0)
        """
        meta = self._state.get("metadata")
        if meta:
            return meta["scale"]
        lyr = self._active_layer()
        if lyr is not None:
            sc = lyr.scale
            if len(sc) == 3:
                return tuple(float(v) for v in sc)
        return (1.0, 1.0, 1.0)

    def _refresh_meta_lbl(self):
        scale = self._get_layer_scale()
        z, y, x = scale
        meta = self._state.get("metadata")
        if meta:
            source     = meta.get("source", "Unknown")
            anisotropy = meta.get("anisotropy", 1.0)
        else:
            xy         = (x + y) / 2.0
            anisotropy = z / xy if xy > 0 else 1.0
            source     = (
                "from layer scale"
                if scale != (1.0, 1.0, 1.0)
                else "default (1, 1, 1)"
            )
        line1 = f"Z={z:.4f}  Y={y:.4f}  X={x:.4f} \u00b5m"
        line2 = f"Anisotropy {anisotropy:.2f}:1  |  {source}"
        self._meta_lbl.setText(f"{line1}\n{line2}")

    def _active_layer(self):
        """Return the active (selected) Image layer, or None."""
        active = self._viewer.layers.selection.active
        if active is not None and isinstance(active, napari.layers.Image):
            return active
        # fall back to topmost Image layer
        for lyr in reversed(self._viewer.layers):
            if isinstance(lyr, napari.layers.Image):
                return lyr
        return None

    def _refresh_layer_info(self, *_):
        lyr = self._active_layer()
        if lyr is None:
            self._layer_info.setText("  — no image layers yet —")
            self._meta_lbl.setText("  — voxel info unavailable —")
            return
        d = lyr.data
        self._layer_info.setText(f'  "{lyr.name}"\n  {d.shape}  {d.dtype}')
        self._refresh_meta_lbl()

    # ------------------------------------------------------------------ #
    # Public helper (used by __main__.py for CLI pre-loading)
    # ------------------------------------------------------------------ #

    def _add_channels(self, path, channels):
        """Add a list of (volume, name, metadata) channel tuples as image layers."""
        colormaps = ["gray", "green", "magenta", "cyan"]
        self._state["last_file_path"] = path
        self._state["metadata"]       = channels[0][2]   # shared metadata
        for i, (volume, name, metadata) in enumerate(channels):
            cmap = colormaps[i % len(colormaps)]
            self._viewer.add_image(
                volume, name=name, colormap=cmap, scale=metadata["scale"]
            )
        self._refresh_layer_info()

    def preload(self, path):
        """Load a file programmatically (CLI / __main__.py use)."""
        path = Path(path)
        self._status(f"Loading {path.name}...")
        try:
            channels = load_file(path)
            self._add_channels(path, channels)
            n = len(channels)
            shape = channels[0][0].shape
            self._status(f"Loaded: {path.name}  {n} ch  {shape}")
        except Exception as exc:
            self._status(f"ERROR: {exc}")
            print(f"ERROR loading {path.name}: {exc}")

    # ------------------------------------------------------------------ #
    # Slots
    # ------------------------------------------------------------------ #

    def _on_open(self):
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Open confocal stack",
            "",
            "Confocal stacks (*.tif *.tiff *.ims);;All files (*)",
        )
        if not path_str:
            return
        path = Path(path_str)
        self._status(f"Loading {path.name}...")
        try:
            channels = load_file(path)
            self._add_channels(path, channels)
            n = len(channels)
            shape = channels[0][0].shape
            self._status(f"Loaded: {path.name}  {n} ch  {shape}")
        except Exception as exc:
            self._status(f"ERROR: {exc}")
            print(f"ERROR loading {path.name}: {exc}")

    def _on_browse_model(self):
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Select model checkpoint",
            str(_SKIN_SEG_DIR / "models"),
            "PyTorch checkpoints (*.pth)",
        )
        if not path_str:
            return
        p = Path(path_str)
        self._state["model_path"] = p
        self._model_lbl.setText(path_str)
        _save_model_path(p)
        self._status(f"Model: {p.name}")

    def _on_run(self):
        if not self._state["model_path"] or not Path(self._state["model_path"]).exists():
            self._status("ERROR: model file not found — browse to a .pth file.")
            return
        target = self._active_layer()
        if target is None:
            self._status("ERROR: no image layer selected — open a file and click a layer.")
            return
        volume = np.asarray(target.data)
        if volume.ndim != 3:
            self._status(f"ERROR: 3D volume required, got {volume.ndim}D {volume.shape}.")
            return

        threshold        = self._thresh_slider.value() / 100.0
        erosion_voxels   = self._erosion_slider.value()
        bg_mode          = self._bg_group.checkedId()  # 0=off, 1=remove, 2=fill
        bg_tolerance_pct = self._tol_slider.value() / 100.0
        model_path       = Path(self._state["model_path"])
        stem             = target.name
        file_path        = self._state.get("last_file_path")
        # Prefer scale directly from the target layer (set by reader or Open btn)
        sc = target.scale
        scale = tuple(float(v) for v in sc) if len(sc) == 3 else self._get_layer_scale()
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        self._run_btn.setEnabled(False)
        self._status(f"Running on {device} (threshold={threshold:.2f})...")

        print(f"\n{'='*70}")
        print(f"SKIN-REMOVER — {stem}  shape={volume.shape}")
        print(f"Model    : {model_path.name}")
        print(f"Threshold: {threshold}   Device: {device}")
        print(f"Erosion  : {erosion_voxels} voxel(s)")
        bg_mode_str = {
            0: "Off",
            1: f"Remove outside-brain (tol={bg_tolerance_pct:+.2f}%)",
            2: f"Remove globally (tol={bg_tolerance_pct:+.2f}%)",
            3: "Fill sub-background with random noise",
        }[bg_mode]
        print(f"BG mode  : {bg_mode_str}")
        print(f"Scale    : Z={scale[0]:.4f}  Y={scale[1]:.4f}  X={scale[2]:.4f} µm")
        print(f"{'='*70}")

        result = {}

        def _worker():
            try:
                # Step 1: inference on original volume (never bg-removed)
                brain_mask, brain_only = run_inference(
                    volume, model_path, threshold, device, erosion_voxels
                )

                # Step 2: optional background processing
                if bg_mode == 1:
                    vol_proc, *_ = remove_outside_brain(
                        volume, brain_mask, tolerance_pct=bg_tolerance_pct
                    )
                    brain_only = (vol_proc * brain_mask).astype(volume.dtype)
                elif bg_mode == 2:
                    vol_proc, *_ = remove_global(
                        volume, brain_mask, tolerance_pct=bg_tolerance_pct
                    )
                    brain_only = (vol_proc * brain_mask).astype(volume.dtype)
                elif bg_mode == 3:
                    brain_only, _ = fill_outside_brain_random(
                        volume, brain_mask
                    )

                result["brain_mask"] = brain_mask
                result["brain_only"] = brain_only
            except Exception as exc:
                traceback.print_exc()
                result["error"] = str(exc)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        timer = QTimer(self)

        def _poll():
            if thread.is_alive():
                return

            timer.stop()

            if "error" in result:
                self._status(f"ERROR: {result['error']}")
                self._run_btn.setEnabled(True)
                return

            brain_mask  = result["brain_mask"]
            brain_only  = result["brain_only"]
            nonzero_pct = 100.0 * brain_mask.sum() / brain_mask.size
            print(f"Brain mask: {brain_mask.sum():,} voxels ({nonzero_pct:.1f}%)")

            # Replace stale output layers if present
            for lname in (f"{stem}_brain_mask", f"{stem}_brain_only"):
                if lname in self._viewer.layers:
                    self._viewer.layers.remove(lname)

            mask_layer = self._viewer.add_labels(
                brain_mask,
                name=f"{stem}_brain_mask",
                opacity=0.4,
                scale=scale,
            )
            try:
                mask_layer.color = {1: "cyan"}
            except Exception:
                pass  # older napari — default label color is fine
            self._viewer.add_image(
                brain_only,
                name=f"{stem}_brain_only",
                colormap="gray",
                scale=scale,
            )
            self._refresh_layer_info()

            out_dir = file_path.parent if file_path else Path(".")
            if self._save_only_cb.isChecked():
                out = out_dir / f"{stem}_brain_only.tif"
                tifffile.imwrite(str(out), brain_only, compression="zlib")
                print(f"Saved: {out}")
            if self._save_mask_cb.isChecked():
                out = out_dir / f"{stem}_brain_mask.tif"
                tifffile.imwrite(
                    str(out),
                    (brain_mask * 255).astype(np.uint8),
                    compression="zlib",
                )
                print(f"Saved: {out}")

            self._status(f"Done — brain={nonzero_pct:.1f}% of volume.")
            self._run_btn.setEnabled(True)

            print(f"{'='*70}")
            print("SKIN-REMOVER COMPLETE")
            print(f"{'='*70}\n")

        timer.timeout.connect(_poll)
        timer.start(500)

    def _active_labels_layer(self):
        """Return the active Labels layer, or the topmost one, or None."""
        active = self._viewer.layers.selection.active
        if active is not None and isinstance(active, napari.layers.Labels):
            return active
        for lyr in reversed(self._viewer.layers):
            if isinstance(lyr, napari.layers.Labels):
                return lyr
        return None

    def _on_resort_labels(self):
        lyr = self._active_labels_layer()
        if lyr is None:
            self._resort_status_lbl.setText("No Labels layer selected.")
            return

        sort_by = self._sort_combo.currentData()
        reverse = self._sort_reverse_cb.isChecked()

        self._resort_btn.setEnabled(False)
        self._resort_status_lbl.setText("Resorting...")

        import numpy as np
        labels = np.asarray(lyr.data)
        result = {}

        def _worker():
            try:
                result["labels"] = resort_labels(labels, sort_by=sort_by, reverse=reverse)
            except Exception as exc:
                traceback.print_exc()
                result["error"] = str(exc)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        timer = QTimer(self)

        def _poll():
            if thread.is_alive():
                return
            timer.stop()
            if "error" in result:
                self._resort_status_lbl.setText(f"ERROR: {result['error']}")
                self._resort_btn.setEnabled(True)
                return
            lyr.data = result["labels"]
            n = int(result["labels"].max())
            sort_label = self._sort_combo.currentText()
            rev_str    = " (reversed)" if reverse else ""
            self._resort_status_lbl.setText(
                f"Done — {n} labels, sorted by {sort_label}{rev_str}."
            )
            self._resort_btn.setEnabled(True)

        timer.timeout.connect(_poll)
        timer.start(200)

    def _on_create_labels(self):
        # Read active layer
        target = self._active_layer()
        if target is None:
            self._labels_status_lbl.setText("Select a brain_only layer first.")
            return
        volume = np.asarray(target.data)
        if volume.ndim != 3:
            self._labels_status_lbl.setText(
                f"ERROR: 3D volume required, got {volume.ndim}D."
            )
            return

        sigma_xy        = self._sxy_slider.value()  / 10.0
        sigma_z         = self._sz_slider.value()   / 10.0
        min_overlap_pct = float(self._ovlp_slider.value())
        min_volume      = self._area_slider.value()
        stem            = target.name
        scale           = tuple(float(v) for v in target.scale) if len(target.scale) == 3 else (1., 1., 1.)

        self._labels_btn.setEnabled(False)
        self._labels_status_lbl.setText("Running...")

        print(f"\n{'='*70}")
        print(f"CREATE LABELS — {stem}  shape={volume.shape}")
        print(f"σ_xy={sigma_xy}  σ_z={sigma_z}  min_overlap={min_overlap_pct}%  min_volume={min_volume} vox")
        print(f"{'='*70}")

        result = {}

        def _worker():
            try:
                labels = create_labels(
                    volume,
                    sigma_xy=sigma_xy,
                    sigma_z=sigma_z,
                    min_overlap_pct=min_overlap_pct,
                    min_volume=min_volume,
                )
                result["labels"] = labels
            except Exception as exc:
                traceback.print_exc()
                result["error"] = str(exc)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        timer = QTimer(self)

        def _poll():
            if thread.is_alive():
                return
            timer.stop()

            if "error" in result:
                self._labels_status_lbl.setText(f"ERROR: {result['error']}")
                self._labels_btn.setEnabled(True)
                return

            labels     = result["labels"]
            n_labels   = int(labels.max())
            lname      = f"{stem}_labels"

            if lname in self._viewer.layers:
                self._viewer.layers.remove(lname)

            self._viewer.add_labels(labels, name=lname, scale=scale)

            if self._save_labels_cb.isChecked():
                file_path = self._state.get("last_file_path")
                out_dir   = file_path.parent if file_path else Path(".")
                out       = out_dir / f"{stem}_labels.tif"
                tifffile.imwrite(str(out), labels.astype(np.int32), compression="zlib")
                print(f"Saved: {out}")

            self._labels_status_lbl.setText(f"Done — {n_labels} labels.")
            self._labels_btn.setEnabled(True)

            print(f"{'='*70}")
            print(f"CREATE LABELS COMPLETE — {n_labels} objects")
            print(f"{'='*70}\n")

        timer.timeout.connect(_poll)
        timer.start(500)
