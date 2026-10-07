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
    QSplitter, QTreeWidget, QTreeWidgetItem, QCheckBox, QGroupBox,
    QComboBox,
)
from qtpy.QtCore import Qt, QTimer, QRectF, QPoint
from qtpy.QtGui import QPen, QColor, QPixmap, QImage, QFont, QPainter, QBrush


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CELL_SIZE = 220
CELL_GAP = 14
PLACEHOLDER_COLOR = QColor(55, 55, 60)
BORDER_COLOR = QColor(90, 90, 100)
BOX_COLOR = QColor(255, 255, 255)
HIGHLIGHT_COLOR = QColor(255, 220, 0)

ELEM_COLORS = [QColor(255, 80, 80), QColor(80, 220, 80), QColor(80, 150, 255)]


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
    """Return the first element group from export_params.elem_list."""
    try:
        groups = config["export_params"]["elem_list"]
        return list(groups[0]) if groups else []
    except (KeyError, IndexError, TypeError):
        return []


def _composite_to_pixmap(elem_tiff_paths: list[Path], target: int):
    """Load per-element grayscale TIFFs and composite to an RGB QPixmap.

    Returns (pixmap, tiff_width, tiff_height) or (None, 0, 0) on failure.
    """
    try:
        channels = []
        for p in elem_tiff_paths[:3]:
            arr = np.array(Image.open(p), dtype=np.float32)
            if arr.max() > 0:
                arr = arr / arr.max()
            channels.append(arr)

        while len(channels) < 3:
            channels.append(np.zeros_like(channels[0]))

        h, w = channels[0].shape
        channels = [
            ch if ch.shape == (h, w)
            else np.array(Image.fromarray(ch).resize((w, h), Image.BILINEAR))
            for ch in channels
        ]
        rgb = np.stack(channels, axis=-1)
        rgb = (rgb * 255).astype(np.uint8)

        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        pm = QPixmap.fromImage(qimg)
        return pm.scaled(target, target, Qt.KeepAspectRatio, Qt.SmoothTransformation), w, h
    except Exception as exc:
        print(f"[LiveScan] Could not composite {elem_tiff_paths}: {exc}")
        return None, 0, 0


def _load_union_boxes(results_dir: Path, group_name: str | None = None) -> list[dict]:
    """Return list of box dicts from the group's unions_output JSON.

    Each dict has cx_px, cy_px, side_px, label, and mean_intensity (None for
    geometric union boxes which have no single intensity value).
    """
    boxes = []

    if group_name:
        candidates = [
            results_dir / f"unions_output_{group_name}.json",
            results_dir / "unions_output.json",
        ]
        files = [p for p in candidates if p.exists()]
    else:
        files = list(results_dir.glob("unions_output_*.json"))

    for jf in files:
        try:
            data = json.loads(jf.read_text())
            for entry in data.values():
                ic = entry.get("image_center")
                il = entry.get("image_length") or (entry.get("image_radius", 0) * 2) or None
                if ic and il:
                    boxes.append({
                        "cx_px": ic[0],
                        "cy_px": ic[1],
                        "side_px": il,
                        "label": entry.get("text"),
                        "mean_intensity": entry.get("mean_intensity"),  # None for union boxes
                    })
        except Exception as exc:
            print(f"[LiveScan] Could not read {jf}: {exc}")
    return boxes


def _load_all_boxes(results_dir: Path, group_name: str | None = None) -> list[dict]:
    """Load all_boxes JSON — returns list of {element, image_center, mean_intensity}.

    Used to show per-element contributing intensities on union/merged box tooltips.
    """
    import re
    if group_name:
        files = [p for p in [
            results_dir / f"all_boxes_{group_name}.json",
            results_dir / "all_boxes.json",
        ] if p.exists()]
    else:
        files = list(results_dir.glob("all_boxes_*.json"))

    entries = []
    for jf in files:
        try:
            data = json.loads(jf.read_text())
            for entry in data.values():
                ic = entry.get("image_center")
                if not ic:
                    continue
                text = entry.get("text", "")
                m = re.match(r"All Box (\S+) #\d+", text)
                elem = m.group(1) if m else "?"
                entries.append({
                    "element": elem,
                    "image_center": ic,
                    "mean_intensity": entry.get("mean_intensity") or 0,
                })
        except Exception as exc:
            print(f"[LiveScan] Could not read {jf}: {exc}")
    return entries


# ---------------------------------------------------------------------------
# QGraphicsView — forwards mouse events to the parent viewer
# ---------------------------------------------------------------------------

class _GridView(QGraphicsView):
    def __init__(self, scene, live_viewer, parent=None):
        super().__init__(scene, parent)
        self._live_viewer = live_viewer
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setBackgroundBrush(QBrush(QColor(30, 30, 35)))
        self.setMouseTracking(True)

    def wheelEvent(self, event):
        factor = 1.20 if event.angleDelta().y() > 0 else 1 / 1.20
        self.scale(factor, factor)

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        self._live_viewer.handle_hover(event, self.mapToScene(event.pos()))

    def leaveEvent(self, event):
        super().leaveEvent(event)
        self._live_viewer._hover_label.hide()


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
        self._elements: list[str] = []
        self._all_groups: list[list[str]] = []
        self._seen_tiles: set[str] = set()
        self._tile_order: list[str] = []
        self._pending_boxes: dict[str, tuple[int, float, Path]] = {}
        self._tile_all_boxes: dict[str, list[dict]] = {}  # scan_id -> all_boxes entries

        # Box and tile tracking
        self._box_meta: list[dict] = []          # per-box: coords + scene_item ref
        self._box_scene_items: list = []          # QGraphicsRectItem refs for toggle
        self._tile_scene_pos: dict[int, tuple[float, float]] = {}
        self._tile_highlight: QGraphicsRectItem | None = None
        self._tile_list_items: dict[int, QTreeWidgetItem] = {}

        # Selection state
        self._selected_tile_idx: int | None = None
        self._selected_box_meta_idx: int | None = None

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(1000)
        self._poll_timer.timeout.connect(self._poll)

        self._scene = QGraphicsScene(self)
        self._placeholder_items: dict[int, list] = {}

        self._setup_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # Control bar
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

        # Status bar
        self._status_lbl = QLabel("Load a config and select a directory to begin.")
        self._status_lbl.setStyleSheet("color: #aaa; font-size: 12px;")
        root.addWidget(self._status_lbl)

        # Floating hover tooltip
        self._hover_label = QLabel(self)
        self._hover_label.setWindowFlags(Qt.ToolTip)
        self._hover_label.setStyleSheet(
            "QLabel { background: #2a2a2e; color: #eee; border: 1px solid #555; "
            "border-radius: 4px; padding: 6px; font-size: 12px; }"
        )
        self._hover_label.hide()

        # Splitter: canvas (left) | panel (right)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        self._view = _GridView(self._scene, self)
        self._view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        splitter.addWidget(self._view)

        right_panel = self._build_right_panel()
        splitter.addWidget(right_panel)
        splitter.setSizes([800, 220])

        root.addWidget(splitter, 1)

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(180)
        panel.setMaximumWidth(280)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 4, 8, 8)
        layout.setSpacing(10)

        # Group selector (hidden when config has only one group)
        group_row = QHBoxLayout()
        self._group_label = QLabel("Display Group:")
        self._group_label.setStyleSheet("color: #ccc; font-size: 12px;")
        self._group_combo = QComboBox()
        self._group_combo.setStyleSheet(
            "QComboBox { background: white; color: black; border: 1px solid #aaa; "
            "border-radius: 3px; padding: 2px 6px; font-size: 12px; }"
            "QComboBox QAbstractItemView { background: white; color: black; "
            "selection-background-color: #1a6fdb; selection-color: white; }"
        )
        self._group_combo.currentIndexChanged.connect(self._on_group_changed)
        group_row.addWidget(self._group_label)
        group_row.addWidget(self._group_combo, 1)
        layout.addLayout(group_row)
        self._group_label.setVisible(False)
        self._group_combo.setVisible(False)

        # Element legend
        legend_group = QGroupBox("Elements")
        legend_group.setStyleSheet(
            "QGroupBox { font-weight: bold; color: #ccc; "
            "margin-top: 14px; padding-top: 8px; }"
        )
        legend_layout = QVBoxLayout(legend_group)
        legend_layout.setSpacing(4)
        self._legend_labels: list[QLabel] = []
        for color in ELEM_COLORS:
            lbl = QLabel("● —")
            lbl.setStyleSheet(
                f"color: rgb({color.red()},{color.green()},{color.blue()}); "
                "font-size: 12px;"
            )
            legend_layout.addWidget(lbl)
            self._legend_labels.append(lbl)
        layout.addWidget(legend_group)

        # Toggle: union boxes
        self._boxes_checkbox = QCheckBox("Union Boxes")
        self._boxes_checkbox.setChecked(True)
        self._boxes_checkbox.setStyleSheet("color: #ccc; font-size: 12px;")
        self._boxes_checkbox.stateChanged.connect(self._on_toggle_boxes)
        layout.addWidget(self._boxes_checkbox)

        # Tile tree
        tiles_group = QGroupBox("Tiles")
        tiles_group.setStyleSheet(
            "QGroupBox { font-weight: bold; color: #ccc; "
            "margin-top: 14px; padding-top: 8px; }"
        )
        tiles_layout = QVBoxLayout(tiles_group)
        tiles_layout.setContentsMargins(4, 4, 4, 4)
        self._tile_tree = QTreeWidget()
        self._tile_tree.setHeaderHidden(True)
        self._tile_tree.setRootIsDecorated(True)
        self._tile_tree.setStyleSheet(
            "QTreeWidget { background: #1e1e22; color: #ccc; border: none; "
            "font-size: 11px; }"
            "QTreeWidget::item { padding: 2px 2px; }"
            "QTreeWidget::item:selected { background: #3a3a50; color: #fff; }"
        )
        self._tile_tree.itemClicked.connect(self._on_tree_item_clicked)
        tiles_layout.addWidget(self._tile_tree)
        layout.addWidget(tiles_group, 1)

        # Stats
        self._stats_lbl = QLabel("0 / 0 tiles\n0 boxes total")
        self._stats_lbl.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self._stats_lbl)

        return panel

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
        self.load_config_from_path(path)

    def load_config_from_path(self, path: str) -> bool:
        """Load a JSON config by path. Returns True on success. Safe to call programmatically."""
        try:
            config = json.loads(Path(path).read_text())
        except Exception as exc:
            self._status_lbl.setText(f"Error reading config: {exc}")
            return False

        n_cols, n_rows = _grid_dims_from_config(config)
        if n_cols == 0 or n_rows == 0:
            self._status_lbl.setText(
                "Could not compute grid size — check mosaic_params in the config."
            )
            return False

        self._config = config
        self._n_cols = n_cols
        self._n_rows = n_rows

        # Extract all element groups — supports both elem_list and intensity-filter groups format
        ep = config.get("export_params", {})
        raw = ep.get("elem_list", [])
        if raw and isinstance(raw[0], list):
            self._all_groups = [list(g) for g in raw]
        elif raw and isinstance(raw[0], str):
            self._all_groups = [list(raw)]
        elif ep.get("use_intensity_filter") and ep.get("groups"):
            self._all_groups = [[e["element"] for e in g["elements"]] for g in ep["groups"]]
        else:
            self._all_groups = []

        self._group_combo.blockSignals(True)
        self._group_combo.clear()
        for i, g in enumerate(self._all_groups):
            self._group_combo.addItem(f"Group {i + 1}: {', '.join(g)}", i)
        self._group_combo.setCurrentIndex(0)
        self._group_combo.blockSignals(False)

        multi = len(self._all_groups) > 1
        self._group_label.setVisible(multi)
        self._group_combo.setVisible(multi)

        self._elements = list(self._all_groups[0]) if self._all_groups else []
        elem_str = ", ".join(self._elements) if self._elements else "unknown"
        self._config_lbl.setText(
            f"{Path(path).name}  ({n_cols}×{n_rows} grid  |  {elem_str})"
        )
        self._config_lbl.setStyleSheet("")

        self._update_legend()

        data_wd = config.get("export_params", {}).get("data_wd", "")
        if data_wd:
            data_wd_path = Path(data_wd)
            data_wd_path.mkdir(parents=True, exist_ok=True)
            self._watch_dir = data_wd_path
            self._dir_lbl.setText(str(data_wd_path))
            self._dir_lbl.setStyleSheet("")

        self._reset_grid()
        self._update_toggle_state()
        return True

    def start_watching(self):
        """Start the poll timer if config and directory are ready."""
        if self._config is None or self._watch_dir is None:
            return
        if self._poll_timer.isActive():
            return
        self._reset_grid()
        self._poll_timer.start()
        self._toggle_btn.setText("Stop")
        self._status_lbl.setText("Watching…")

    def _update_legend(self):
        for i, lbl in enumerate(self._legend_labels):
            if i < len(self._elements):
                color = ELEM_COLORS[i]
                lbl.setText(f"● {self._elements[i]}")
                lbl.setStyleSheet(
                    f"color: rgb({color.red()},{color.green()},{color.blue()}); "
                    "font-size: 12px;"
                )
            else:
                lbl.setText("● —")
                lbl.setStyleSheet("color: #444; font-size: 12px;")

    def _on_group_changed(self, index: int):
        if not self._all_groups or index < 0 or index >= len(self._all_groups):
            return
        self._elements = list(self._all_groups[index])
        self._update_legend()
        self._redraw_all_tiles()

    def _redraw_all_tiles(self):
        """Clear and redraw all seen tiles using the current self._elements."""
        if not self._watch_dir:
            return
        was_watching = self._poll_timer.isActive()
        saved_order = list(self._tile_order)
        self._reset_grid()
        gname = "".join(self._elements)
        for scan_id in saved_order:
            tile_dir = self._watch_dir / f"automap_{scan_id}"
            elem_paths = [
                tile_dir / f"scan_{scan_id}_{elem}.tiff" for elem in self._elements
            ]
            results_dir = tile_dir / f"automap_{scan_id}_results"
            idx = len(self._tile_order)
            tiff_w = self._draw_tile(idx, scan_id, elem_paths, results_dir, gname)
            self._seen_tiles.add(scan_id)
            self._tile_order.append(scan_id)
            if tiff_w > 0 and not _load_union_boxes(results_dir, gname):
                self._pending_boxes[scan_id] = (idx, tiff_w, results_dir, gname)
        if was_watching:
            self._poll_timer.start()
            self._toggle_btn.setText("Stop")
            self._status_lbl.setText("Watching…")

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
    # Slot: toggle union box visibility
    # ------------------------------------------------------------------

    def _on_toggle_boxes(self, state):
        visible = bool(state)
        for item in self._box_scene_items:
            item.setVisible(visible)

    # ------------------------------------------------------------------
    # Slot: tree item clicked
    # ------------------------------------------------------------------

    def _on_tree_item_clicked(self, item: QTreeWidgetItem, _column: int):
        if item.parent() is None:
            # Top-level item = tile
            tile_idx = item.data(0, Qt.UserRole)
            if tile_idx == self._selected_tile_idx:
                self._deselect_tile()
            else:
                self._select_tile(tile_idx, item)
        else:
            # Child item = box
            meta_idx = item.data(0, Qt.UserRole)
            self._select_box(meta_idx)

    def _select_tile(self, tile_idx: int, tree_item: QTreeWidgetItem):
        # Collapse and restore highlight on previously selected tile
        if self._selected_tile_idx is not None and self._selected_tile_idx != tile_idx:
            prev_item = self._tile_list_items.get(self._selected_tile_idx)
            if prev_item:
                prev_item.setExpanded(False)
        self._restore_box_highlight()
        self._remove_tile_highlight()

        self._selected_tile_idx = tile_idx
        tree_item.setExpanded(True)

        if tile_idx in self._tile_scene_pos:
            x0, y0 = self._tile_scene_pos[tile_idx]
            self._tile_highlight = self._scene.addRect(
                x0 - 3, y0 - 3, CELL_SIZE + 6, CELL_SIZE + 6,
                QPen(HIGHLIGHT_COLOR, 3),
                QBrush(Qt.transparent),
            )
            self._view.centerOn(x0 + CELL_SIZE / 2, y0 + CELL_SIZE / 2)

    def _deselect_tile(self):
        self._restore_box_highlight()
        self._remove_tile_highlight()
        if self._selected_tile_idx is not None:
            item = self._tile_list_items.get(self._selected_tile_idx)
            if item:
                item.setExpanded(False)
        self._selected_tile_idx = None
        self._tile_tree.clearSelection()

    def _select_box(self, meta_idx: int):
        self._restore_box_highlight()
        self._selected_box_meta_idx = meta_idx
        scene_item = self._box_meta[meta_idx].get("scene_item")
        if scene_item:
            pen = QPen(HIGHLIGHT_COLOR, 2, Qt.SolidLine)
            pen.setCosmetic(True)
            scene_item.setPen(pen)

    def _restore_box_highlight(self):
        if self._selected_box_meta_idx is not None:
            scene_item = self._box_meta[self._selected_box_meta_idx].get("scene_item")
            if scene_item:
                pen = QPen(BOX_COLOR, 2, Qt.SolidLine)
                pen.setCosmetic(True)
                scene_item.setPen(pen)
        self._selected_box_meta_idx = None

    def _remove_tile_highlight(self):
        if self._tile_highlight is not None:
            self._scene.removeItem(self._tile_highlight)
            self._tile_highlight = None

    # ------------------------------------------------------------------
    # Hover / tooltip
    # ------------------------------------------------------------------

    def handle_hover(self, event, scene_pos):
        if not self._boxes_checkbox.isChecked() or not self._box_meta:
            self._hover_label.hide()
            return

        for meta in self._box_meta:
            rect = QRectF(meta["scene_x"], meta["scene_y"], meta["scene_w"], meta["scene_w"])
            if rect.contains(scene_pos):
                self._show_tooltip(event, self._format_box_tooltip(meta))
                return

        self._hover_label.hide()

    def _show_tooltip(self, event, html: str):
        self._hover_label.setText(html)
        self._hover_label.adjustSize()
        mouse_pos = self._view.mapTo(self, event.pos())
        new_pos = QPoint(
            mouse_pos.x() + 16,
            mouse_pos.y() - self._hover_label.height() - 8,
        )
        self._hover_label.move(new_pos)
        self._hover_label.show()

    def _format_box_tooltip(self, meta: dict) -> str:
        tile_idx = meta["tile_idx"]
        col = tile_idx % self._n_cols
        row = tile_idx // self._n_cols
        box_idx = meta["box_idx"]
        cx, cy = meta["cx_px"], meta["cy_px"]
        side = meta["side_px"]
        area_px = side ** 2

        box_label = meta.get("label") or f"Box #{box_idx + 1}"
        lines = [
            f"<b>Tile {tile_idx + 1} (row {row + 1}, col {col + 1})</b><br>",
            f"automap_{meta['scan_id']}<br>",
            f"{box_label}<br><br>",
            f"Center: ({cx}, {cy}) px<br>",
            f"Size: {side} × {side} px<br>",
            f"Area: {area_px} px²",
        ]

        cal = (self._config or {}).get("calibration_params", {})
        mpp_x = cal.get("microns_per_pixel_x")
        mpp_y = cal.get("microns_per_pixel_y")
        ox = cal.get("true_origin_x", 0)
        oy = cal.get("true_origin_y", 0)
        if mpp_x and mpp_y:
            real_cx = cx * mpp_x + ox
            real_cy = cy * mpp_y + oy
            real_w = side * mpp_x
            real_h = side * mpp_y
            real_area = real_w * real_h
            lines += [
                f"<br><br>Real center: ({real_cx:.2f}, {real_cy:.2f}) µm<br>",
                f"Real size: {real_w:.2f} × {real_h:.2f} µm<br>",
                f"Real area: {real_area:.2f} µm²",
            ]

        # Intensity info
        mi = meta.get("mean_intensity")
        label = meta.get("label") or ""
        is_multi = (mi is None) or label.startswith("Union Box") or label.startswith("Cross-element")

        if not is_multi and mi is not None:
            # Individual blob — show its own mean intensity directly
            lines.append(f"<br><br>Mean intensity: {mi:.1f}")
        else:
            # Union box, cross-element merge, or any multi-blob box —
            # look up contributing blobs spatially from all_boxes JSON
            scan_id = meta["scan_id"]
            all_boxes = self._tile_all_boxes.get(scan_id, [])
            if all_boxes:
                half = side / 2
                lo_x, hi_x = cx - half, cx + half
                lo_y, hi_y = cy - half, cy + half
                elem_intensities: dict[str, list[float]] = {}
                for entry in all_boxes:
                    ic = entry["image_center"]
                    if lo_x <= ic[0] <= hi_x and lo_y <= ic[1] <= hi_y:
                        elem = entry["element"]
                        elem_intensities.setdefault(elem, []).append(entry["mean_intensity"])
                if elem_intensities:
                    lines.append("<br><br><b>Contributing blobs:</b>")
                    for elem, vals in sorted(elem_intensities.items()):
                        avg = sum(vals) / len(vals)
                        lines.append(f"<br>{elem}: {avg:.1f} (n={len(vals)})")
            elif mi is not None:
                # Fallback: all_boxes not available, show averaged intensity
                lines.append(f"<br><br>Mean intensity: {mi:.1f}")

        return "".join(lines)

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
        self._tile_all_boxes.clear()
        self._box_meta.clear()
        self._box_scene_items.clear()
        self._tile_scene_pos.clear()
        self._tile_list_items.clear()
        self._tile_highlight = None
        self._selected_tile_idx = None
        self._selected_box_meta_idx = None
        self._scene.clear()
        self._tile_tree.clear()

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
        self._update_stats()

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    def _poll(self):
        if self._watch_dir is None:
            return

        total = self._n_cols * self._n_rows
        elements = self._elements or ["Ca", "Fe", "S"]

        # Sort by numeric scan_id so tiles arrive in grid order (1000, 1001, 1002…)
        def _scan_id_key(d: Path) -> int:
            try:
                return int(d.name.replace("automap_", ""))
            except ValueError:
                return 0

        tile_dirs = sorted(
            (d for d in self._watch_dir.glob("automap_*") if d.is_dir()),
            key=_scan_id_key,
        )

        for tile_dir in tile_dirs:
            scan_id = tile_dir.name.replace("automap_", "", 1)
            if scan_id in self._seen_tiles:
                continue
            if len(self._tile_order) >= total:
                break

            elem_paths = [
                tile_dir / f"scan_{scan_id}_{elem}.tiff" for elem in elements
            ]
            if not all(p.exists() for p in elem_paths):
                continue

            results_dir = tile_dir / f"automap_{scan_id}_results"
            idx = len(self._tile_order)
            gname = "".join(self._elements)
            tiff_w = self._draw_tile(idx, scan_id, elem_paths, results_dir, gname)
            self._seen_tiles.add(scan_id)
            self._tile_order.append(scan_id)
            if tiff_w > 0 and not _load_union_boxes(results_dir, gname):
                self._pending_boxes[scan_id] = (idx, tiff_w, results_dir, gname)

        for scan_id, (idx, tiff_w, results_dir, gname) in list(self._pending_boxes.items()):
            boxes = _load_union_boxes(results_dir, gname)
            if boxes:
                self._draw_boxes(idx, scan_id, tiff_w, boxes)
                del self._pending_boxes[scan_id]

        done = len(self._tile_order)
        self._status_lbl.setText(f"Watching…  {done} / {total} tiles complete")
        self._update_stats()

        if done >= total and not self._pending_boxes:
            self._poll_timer.stop()
            self._toggle_btn.setText("Start Watching")
            self._status_lbl.setText(f"Scan complete — all {total} tiles received.")

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw_tile(self, idx: int, scan_id: str, elem_paths: list[Path], results_dir: Path, group_name: str = "") -> float:
        """Draw image + boxes for a tile. Returns tiff_w (0 on failure)."""
        n_cols = self._n_cols
        col = idx % n_cols
        row = idx // n_cols
        stride = CELL_SIZE + CELL_GAP
        x0 = col * stride
        y0 = row * stride

        self._tile_scene_pos[idx] = (x0, y0)

        if idx in self._placeholder_items:
            for item in self._placeholder_items.pop(idx):
                self._scene.removeItem(item)

        pm, tiff_w, tiff_h = _composite_to_pixmap(elem_paths, CELL_SIZE)
        if pm is None:
            return 0

        pix_item = QGraphicsPixmapItem(pm)
        pix_item.setPos(x0, y0)
        self._scene.addItem(pix_item)

        self._scene.addRect(
            x0, y0, pm.width(), pm.height(),
            QPen(BORDER_COLOR, 1),
            QBrush(Qt.transparent),
        )

        # Add top-level tree item (collapsed, no children yet)
        tree_item = QTreeWidgetItem(self._tile_tree)
        tree_item.setText(0, f"Tile {idx + 1}  (r{row + 1}, c{col + 1})")
        tree_item.setData(0, Qt.UserRole, idx)
        self._tile_list_items[idx] = tree_item

        self._tile_all_boxes[scan_id] = _load_all_boxes(results_dir, group_name or None)

        boxes = _load_union_boxes(results_dir, group_name or None)
        if boxes:
            self._draw_boxes(idx, scan_id, tiff_w, boxes)

        return tiff_w

    def _draw_boxes(self, idx: int, scan_id: str, tiff_w: float, boxes: list[dict]):
        """Overlay union boxes onto an already-drawn tile cell."""
        col = idx % self._n_cols
        row = idx // self._n_cols
        stride = CELL_SIZE + CELL_GAP
        x0 = col * stride
        y0 = row * stride
        scale = CELL_SIZE / tiff_w
        show = self._boxes_checkbox.isChecked()

        pen = QPen(BOX_COLOR, 2, Qt.SolidLine)
        pen.setCosmetic(True)

        tree_item = self._tile_list_items.get(idx)

        for box_idx, box in enumerate(boxes):
            cx_px = box["cx_px"]
            cy_px = box["cy_px"]
            side_px = box["side_px"]

            bx = (cx_px - side_px / 2) * scale + x0
            by = (cy_px - side_px / 2) * scale + y0
            bw = side_px * scale

            rect_item = self._scene.addRect(bx, by, bw, bw, pen)
            rect_item.setVisible(show)
            self._box_scene_items.append(rect_item)

            meta_idx = len(self._box_meta)
            self._box_meta.append({
                "tile_idx": idx,
                "box_idx": box_idx,
                "scan_id": scan_id,
                "label": box.get("label"),
                "cx_px": cx_px,
                "cy_px": cy_px,
                "side_px": side_px,
                "mean_intensity": box.get("mean_intensity"),
                "scene_x": bx,
                "scene_y": by,
                "scene_w": bw,
                "scene_item": rect_item,
            })

            # Add child item to tile's tree entry
            if tree_item is not None:
                child = QTreeWidgetItem(tree_item)
                child.setText(0, f"Box #{box_idx + 1}  ({cx_px}, {cy_px}) px")
                child.setData(0, Qt.UserRole, meta_idx)

        # Update tile label with box count
        if tree_item is not None:
            n = len(boxes)
            tree_item.setText(
                0,
                f"Tile {idx + 1}  (r{row + 1}, c{col + 1}) — {n} box{'es' if n != 1 else ''}"
            )
            # If this tile is currently selected, expand to show new children
            if self._selected_tile_idx == idx:
                tree_item.setExpanded(True)

        self._update_stats()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def _update_stats(self):
        total = self._n_cols * self._n_rows
        done = len(self._tile_order)
        boxes = len(self._box_meta)
        self._stats_lbl.setText(f"{done} / {total} tiles\n{boxes} boxes total")
