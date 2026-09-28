"""Live Scan Viewer tab.

Watches a directory for mosaic tile results as they arrive from the beamline
and fills a grid canvas in real time.  No beamline connection needed — it
just polls the result directory on a 1-second timer.

Tile-ready signal: all per-element TIFFs for a scan_id are present:
    {watch_dir}/automap_{scan_id}/scan_{scan_id}_{element}.tiff
for every element in elem_list (read from the JSON config).

The per-element grayscale TIFFs are composited to RGB (R=elem0, G=elem1, B=elem2)
for display, matching the real beamline pipeline file layout.

Grid dimensions come from mosaic_params in the JSON config (same math as
CoarseScanWidget._calc_tile_info).
"""

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QGraphicsView, QGraphicsScene, QSizePolicy,
    QGraphicsRectItem, QGraphicsPixmapItem, QGraphicsTextItem,
    QScrollArea,
)
from qtpy.QtCore import Qt, QTimer, QRectF
from qtpy.QtGui import QPen, QColor, QPixmap, QImage, QFont, QPainter, QBrush


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CELL_SIZE = 220       # pixels — display size of each tile cell
CELL_GAP = 14         # pixels — gap between cells
PLACEHOLDER_COLOR = QColor(55, 55, 60)
BORDER_COLOR = QColor(90, 90, 100)
BOX_COLOR = QColor(255, 255, 255)   # union box overlay color


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _grid_dims_from_config(config: dict) -> tuple[int, int]:
    """Return (n_cols, n_rows) from mosaic_params, or (0, 0) on bad input."""
    mp = config.get("mosaic_params", {})
    try:
        mot1_s = float(mp["mot1_s"])
        mot1_e = float(mp["mot1_e"])
        xlen = float(mp["xlen"])
        ylen = float(mp["ylen"])
        overlap_per = float(mp.get("overlap_per", 0))
    except (KeyError, TypeError, ValueError):
        return 0, 0
    scan_range = abs(mot1_e - mot1_s)
    if scan_range <= 0:
        return 0, 0
    grid_step = scan_range * (1 - overlap_per * 0.01)
    if grid_step <= 0:
        return 0, 0
    start = grid_step / 2
    n_cols = max(1, math.floor((xlen - start) / grid_step) + 1) if xlen >= start else 1
    n_rows = max(1, math.floor((ylen - start) / grid_step) + 1) if ylen >= start else 1
    return n_cols, n_rows


def _elem_list_from_config(config: dict) -> list[str]:
    """Return the first element group from export_params.elem_list, e.g. ['Ca','Fe','S']."""
    try:
        groups = config["export_params"]["elem_list"]
        return list(groups[0]) if groups else []
    except (KeyError, IndexError, TypeError):
        return []


def _composite_to_pixmap(elem_tiff_paths: list[Path], target: int):
    """Load per-element grayscale TIFFs and composite to an RGB QPixmap.

    Up to 3 elements are mapped R→G→B.  Returns (pixmap, tiff_width, tiff_height)
    or (None, 0, 0) on failure.
    """
    try:
        channels = []
        for p in elem_tiff_paths[:3]:
            arr = np.array(Image.open(p), dtype=np.float32)
            if arr.max() > 0:
                arr = arr / arr.max()
            channels.append(arr)

        # Pad to 3 channels if fewer than 3 elements
        while len(channels) < 3:
            channels.append(np.zeros_like(channels[0]))

        h, w = channels[0].shape
        rgb = np.stack(channels, axis=-1)
        rgb = (rgb * 255).astype(np.uint8)

        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        pm = QPixmap.fromImage(qimg)
        return pm.scaled(target, target, Qt.KeepAspectRatio, Qt.SmoothTransformation), w, h
    except Exception as exc:
        print(f"[LiveScan] Could not composite {elem_tiff_paths}: {exc}")
        return None, 0, 0


def _load_union_boxes(results_dir: Path) -> list[dict]:
    """Return list of {cx_px, cy_px, side_px} from unions_output_*.json files."""
    boxes = []
    for jf in results_dir.glob("unions_output_*.json"):
        try:
            data = json.loads(jf.read_text())
            for entry in data.values():
                ic = entry.get("image_center")
                il = entry.get("image_length")
                if ic and il:
                    boxes.append({"cx_px": ic[0], "cy_px": ic[1], "side_px": il})
        except Exception as exc:
            print(f"[LiveScan] Could not read {jf}: {exc}")
    return boxes


# ---------------------------------------------------------------------------
# Simple non-zoomable QGraphicsView for the grid canvas
# ---------------------------------------------------------------------------

class _GridView(QGraphicsView):
    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setBackgroundBrush(QBrush(QColor(30, 30, 35)))

    def wheelEvent(self, event):
        factor = 1.20 if event.angleDelta().y() > 0 else 1 / 1.20
        self.scale(factor, factor)


# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------

class LiveScanViewerWidget(QWidget):
    """Live mosaic tile viewer tab."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._config: dict | None = None
        self._watch_dir: Path | None = None
        self._n_cols: int = 0
        self._n_rows: int = 0
        self._elements: list[str] = []          # e.g. ['Ca', 'Fe', 'S']
        self._seen_tiles: set[str] = set()      # scan_ids already drawn
        self._tile_order: list[str] = []        # arrival order → grid index
        # scan_id → (grid_idx, tiff_w, results_dir) for tiles drawn without boxes yet
        self._pending_boxes: dict[str, tuple[int, float, Path]] = {}

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(1000)
        self._poll_timer.timeout.connect(self._poll)

        self._scene = QGraphicsScene(self)
        self._placeholder_items: dict[int, list] = {}  # idx → [rect, text_label]

        self._setup_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # --- Control bar ---
        ctrl = QHBoxLayout()
        ctrl.setSpacing(8)

        self._config_btn = QPushButton("Load Config")
        self._config_btn.setFixedWidth(110)
        self._config_btn.clicked.connect(self._on_load_config)
        ctrl.addWidget(self._config_btn)

        self._config_lbl = QLabel("No config loaded")
        self._config_lbl.setStyleSheet("color: #888;")
        ctrl.addWidget(self._config_lbl, 1)

        self._dir_btn = QPushButton("Select Watch Dir")
        self._dir_btn.setFixedWidth(130)
        self._dir_btn.clicked.connect(self._on_select_dir)
        ctrl.addWidget(self._dir_btn)

        self._dir_lbl = QLabel("No directory selected")
        self._dir_lbl.setStyleSheet("color: #888;")
        ctrl.addWidget(self._dir_lbl, 1)

        self._toggle_btn = QPushButton("Start Watching")
        self._toggle_btn.setFixedWidth(130)
        self._toggle_btn.setEnabled(False)
        self._toggle_btn.clicked.connect(self._on_toggle)
        ctrl.addWidget(self._toggle_btn)

        root.addLayout(ctrl)

        # --- Status bar ---
        self._status_lbl = QLabel("Load a config and select a directory to begin.")
        self._status_lbl.setStyleSheet("color: #aaa; font-size: 12px;")
        root.addWidget(self._status_lbl)

        # --- Canvas ---
        self._view = _GridView(self._scene, self)
        self._view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root.addWidget(self._view, 1)

    # ------------------------------------------------------------------
    # Slot: load config
    # ------------------------------------------------------------------

    def _on_load_config(self):
        configs_dir = Path(__file__).parents[2] / "configs"
        start_dir = str(configs_dir) if configs_dir.exists() else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select JSON Config", start_dir, "JSON files (*.json)"
        )
        if not path:
            return
        try:
            config = json.loads(Path(path).read_text())
        except Exception as exc:
            self._status_lbl.setText(f"Error reading config: {exc}")
            return

        n_cols, n_rows = _grid_dims_from_config(config)
        if n_cols == 0 or n_rows == 0:
            self._status_lbl.setText(
                "Could not compute grid size — check mosaic_params in the config."
            )
            return

        self._config = config
        self._n_cols = n_cols
        self._n_rows = n_rows
        self._elements = _elem_list_from_config(config)
        elem_str = ", ".join(self._elements) if self._elements else "unknown"
        self._config_lbl.setText(
            f"{Path(path).name}  ({n_cols}×{n_rows} grid  |  {elem_str})"
        )
        self._config_lbl.setStyleSheet("")

        # Auto-set watch directory from data_wd in the config
        data_wd = config.get("export_params", {}).get("data_wd", "")
        if data_wd:
            data_wd_path = Path(data_wd)
            data_wd_path.mkdir(parents=True, exist_ok=True)
            self._watch_dir = data_wd_path
            self._dir_lbl.setText(str(data_wd_path))
            self._dir_lbl.setStyleSheet("")

        self._reset_grid()
        self._update_toggle_state()

    # ------------------------------------------------------------------
    # Slot: select directory
    # ------------------------------------------------------------------

    def _on_select_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select Watch Directory")
        if not path:
            return
        self._watch_dir = Path(path)
        self._dir_lbl.setText(str(self._watch_dir))
        self._dir_lbl.setStyleSheet("")
        self._update_toggle_state()

    # ------------------------------------------------------------------
    # Slot: start / stop
    # ------------------------------------------------------------------

    def _on_toggle(self):
        if self._poll_timer.isActive():
            self._poll_timer.stop()
            self._toggle_btn.setText("Start Watching")
            self._status_lbl.setText(
                f"Stopped. {len(self._tile_order)} / "
                f"{self._n_cols * self._n_rows} tiles."
            )
        else:
            self._reset_grid()
            self._poll_timer.start()
            self._toggle_btn.setText("Stop")
            self._status_lbl.setText("Watching…")

    def _update_toggle_state(self):
        ready = self._config is not None and self._watch_dir is not None
        self._toggle_btn.setEnabled(ready)

    # ------------------------------------------------------------------
    # Grid management
    # ------------------------------------------------------------------

    def _reset_grid(self):
        self._poll_timer.stop()
        self._toggle_btn.setText("Start Watching")
        self._seen_tiles.clear()
        self._tile_order.clear()
        self._placeholder_items.clear()
        self._pending_boxes.clear()
        self._scene.clear()

        n_cols, n_rows = self._n_cols, self._n_rows
        stride = CELL_SIZE + CELL_GAP
        scene_w = n_cols * stride - CELL_GAP
        scene_h = n_rows * stride - CELL_GAP
        self._scene.setSceneRect(0, 0, scene_w, scene_h)

        lbl_font = QFont()
        lbl_font.setPointSize(20)
        lbl_font.setBold(True)

        for idx in range(n_cols * n_rows):
            col = idx % n_cols
            row = idx // n_cols
            x = col * stride
            y = row * stride

            rect = self._scene.addRect(
                x, y, CELL_SIZE, CELL_SIZE,
                QPen(BORDER_COLOR, 1),
                QBrush(PLACEHOLDER_COLOR),
            )

            lbl = QGraphicsTextItem(str(idx + 1))
            lbl.setFont(lbl_font)
            lbl.setDefaultTextColor(QColor(120, 120, 130))
            lbl.setPos(x + CELL_SIZE / 2 - lbl.boundingRect().width() / 2,
                       y + CELL_SIZE / 2 - lbl.boundingRect().height() / 2)
            self._scene.addItem(lbl)

            self._placeholder_items[idx] = [rect, lbl]

        self._view.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    def _poll(self):
        if self._watch_dir is None:
            return

        total = self._n_cols * self._n_rows
        elements = self._elements or ["Ca", "Fe", "S"]

        # Find all automap_* tile directories
        tile_dirs = sorted(
            (d for d in self._watch_dir.glob("automap_*") if d.is_dir()),
            key=lambda d: d.stat().st_mtime,
        )

        for tile_dir in tile_dirs:
            scan_id = tile_dir.name.replace("automap_", "", 1)
            if scan_id in self._seen_tiles:
                continue
            if len(self._tile_order) >= total:
                break

            # Tile is ready when all per-element TIFFs exist
            elem_paths = [
                tile_dir / f"scan_{scan_id}_{elem}.tiff" for elem in elements
            ]
            if not all(p.exists() for p in elem_paths):
                continue

            results_dir = tile_dir / f"automap_{scan_id}_results"
            idx = len(self._tile_order)
            tiff_w = self._draw_tile(idx, elem_paths, results_dir)
            self._seen_tiles.add(scan_id)
            self._tile_order.append(scan_id)
            # If boxes weren't drawn yet (results not ready), queue for later
            if tiff_w > 0 and not _load_union_boxes(results_dir):
                self._pending_boxes[scan_id] = (idx, tiff_w, results_dir)

        # Check tiles that were drawn without boxes — fill them in once results land
        for scan_id, (idx, tiff_w, results_dir) in list(self._pending_boxes.items()):
            boxes = _load_union_boxes(results_dir)
            if boxes:
                self._draw_boxes(idx, tiff_w, boxes)
                del self._pending_boxes[scan_id]

        done = len(self._tile_order)
        self._status_lbl.setText(f"Watching…  {done} / {total} tiles complete")
        if done >= total and not self._pending_boxes:
            self._poll_timer.stop()
            self._toggle_btn.setText("Start Watching")
            self._status_lbl.setText(f"Scan complete — all {total} tiles received.")

    # ------------------------------------------------------------------
    # Drawing a tile into the grid
    # ------------------------------------------------------------------

    def _draw_tile(self, idx: int, elem_paths: list[Path], results_dir: Path) -> float:
        """Draw image + boxes for a tile. Returns tiff_w (0 on failure)."""
        n_cols = self._n_cols
        col = idx % n_cols
        row = idx // n_cols
        stride = CELL_SIZE + CELL_GAP
        x0 = col * stride
        y0 = row * stride

        # Remove placeholder rect and number label
        if idx in self._placeholder_items:
            for item in self._placeholder_items.pop(idx):
                self._scene.removeItem(item)

        # Composite per-element TIFFs to RGB pixmap
        pm, tiff_w, tiff_h = _composite_to_pixmap(elem_paths, CELL_SIZE)
        if pm is None:
            return 0

        pix_item = QGraphicsPixmapItem(pm)
        pix_item.setPos(x0, y0)
        self._scene.addItem(pix_item)

        # Border around the tile
        self._scene.addRect(
            x0, y0, pm.width(), pm.height(),
            QPen(BORDER_COLOR, 1),
            QBrush(Qt.transparent),
        )

        # Draw boxes if results are already available
        boxes = _load_union_boxes(results_dir)
        if boxes:
            self._draw_boxes(idx, tiff_w, boxes)

        return tiff_w

    def _draw_boxes(self, idx: int, tiff_w: float, boxes: list[dict]):
        """Overlay union boxes onto an already-drawn tile cell."""
        col = idx % self._n_cols
        row = idx // self._n_cols
        stride = CELL_SIZE + CELL_GAP
        x0 = col * stride
        y0 = row * stride
        scale = CELL_SIZE / tiff_w
        pen = QPen(BOX_COLOR, 2, Qt.SolidLine)
        pen.setCosmetic(True)
        for box in boxes:
            bx = (box["cx_px"] - box["side_px"] / 2) * scale + x0
            by = (box["cy_px"] - box["side_px"] / 2) * scale + y0
            bw = box["side_px"] * scale
            self._scene.addRect(bx, by, bw, bw, pen)
