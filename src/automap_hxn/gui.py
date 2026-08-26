import os
import re
import json
import pickle
import threading
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import tifffile as tiff

from qtpy.QtWidgets import (
    QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QCheckBox, QSlider, QFileDialog, QListWidget, QListWidgetItem,
    QMessageBox, QDoubleSpinBox, QProgressBar, QGridLayout, QGraphicsEllipseItem,
    QTabWidget, QComboBox, QLineEdit, QSpinBox, QScrollArea, QStackedWidget,
    QGroupBox, QFormLayout, QSizePolicy
)
from qtpy.QtGui import QPixmap, QImage, QPainter, QColor, QPen
from qtpy.QtCore import Qt, QRect, QTimer, QPoint, QEvent
from qtpy.QtWidgets import QStyle, QStyleOptionButton

from .app_state import AppState
from .queue import (
    build_fine_scan_requests,
    submit_fine_scan_requests,
)
from .blobs.detection import (
    CELLPOSE_AVAILABLE,
    CELLPOSE_IMPORT_ERROR,
    YOLO_AVAILABLE,
    YOLO_IMPORT_ERROR,
    STARDIST_AVAILABLE,
    STARDIST_IMPORT_ERROR,
    detect_blobs,
)

# TODO: look up these functions and import them from submodules
from automap_hxn.utils import (
    resize_if_needed, normalize_and_dilate,
    make_json_serializable
)

from .json_maker import JSONMakerWidget
from .coarse_scan_widget import CoarseScanWidget


class ZoomableGraphicsView(QGraphicsView):
    def __init__(self, scene, parent_window):
        super().__init__(scene)
        self.parent_window = parent_window
        self.app_state = parent_window.app_state
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setMouseTracking(True)
        self.setDragMode(QGraphicsView.NoDrag)
        self.blobs = []
        self.visible_colors = set()
        self.highlight_items = []
        self.union_dict = {}
        self.current_qimage = None

    def wheelEvent(self, event):
        cursor_pos = event.pos()
        scene_pos = self.mapToScene(cursor_pos)
        zoom_factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(zoom_factor, zoom_factor)
        mouse_centered = self.mapFromScene(scene_pos)
        delta = cursor_pos - mouse_centered
        self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
        self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.setDragMode(QGraphicsView.NoDrag)
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        pos = self.mapToScene(event.pos())
        self.parent_window.update_mouse_coordinates(pos)
        self.parent_window.handle_hover(event, pos)

    def update_blobs(self, blobs, visible_colors):
        self.blobs = blobs
        self.visible_colors = visible_colors

    def highlight_selected_boxes(self, selected_items):
        for item in self.highlight_items:
            self.scene().removeItem(item)
        self.highlight_items.clear()
    
        for item in selected_items:
            text = item.toolTip()
            center_match = re.search(r"Center: \((\d+), (\d+)\)", text)
            length_match = re.search(r"Length: (\d+)\s*px", text)
    
            if center_match and length_match:
                x, y, length = int(center_match.group(1)), int(center_match.group(2)), int(length_match.group(1))
                radius = length / 2 + 5
                circle = QGraphicsEllipseItem(x - radius, y - radius, radius * 2, radius * 2)
                circle.setPen(QPen(QColor("yellow"), 2, Qt.SolidLine))
                circle.setZValue(100)
                self.scene().addItem(circle)
                self.highlight_items.append(circle)


class ChannelCheckBox(QCheckBox):
    """Checkbox with a durable channel-colored outline and checkmark."""

    def __init__(self, text, color, parent=None):
        super().__init__(text, parent)
        self._channel_color = QColor(color)
        self.setStyleSheet(
            "QCheckBox::indicator {"
            " width: 14px; height: 14px;"
            f" border: 2px solid {color};"
            " background: #ffffff;"
            "}"
        )

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.isChecked():
            return

        option = QStyleOptionButton()
        self.initStyleOption(option)
        indicator = self.style().subElementRect(
            QStyle.SE_CheckBoxIndicator, option, self
        )
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(self._channel_color, 2))
        painter.drawLine(indicator.left() + 3, indicator.center().y(),
                         indicator.center().x() - 1, indicator.bottom() - 3)
        painter.drawLine(indicator.center().x() - 1, indicator.bottom() - 3,
                         indicator.right() - 2, indicator.top() + 3)
        painter.end()


class MainWindow(QWidget):
    """AutoMap's embeddable Qt widget.

    The historical name is retained so existing standalone launchers keep
    working.  Hosts should pass their parent widget and reuse their existing
    QApplication instance rather than create another one.
    """

    def __init__(self, app_state, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self.setWindowTitle("X-AutoMap")
        # Claim available space when embedded in a host layout; the default
        # Preferred policy loses vertical room to Expanding siblings.
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        if parent is None:
            # A standalone window needs an initial screen size. Embedded hosts
            # should instead use AutoMap's natural layout size.
            self.resize(1900, 1000)
        self._init_ui_elements()
        self._init_ui()
        self.blob_items = []
        self.union_box_items = []
        # A blob calculation continues in a background thread.  Keep a
        # generation number so events/results from a discarded analysis view
        # cannot update the next one.
        self._analysis_generation = 0
        self._analysis_lock = threading.Lock()
        self.source_images = []
        self.norm_dilated = []

    def _init_ui_elements(self):
        self.file_list_widget = QListWidget()
        self.graphics_view = None
        self.graphics_scene = None
        self.pixmap_item = None
        self.checkboxes = {}
        self.sliders = {}
        self.area_sliders = {}
        self.slider_labels = {}
        self.area_slider_labels = {}
        self.union_list_widget = QListWidget()
        self.queue_server_list = QListWidget()
        self.union_checkbox = QCheckBox("Union Boxes")
        self.hover_label = QLabel(self)
        self.hover_label.setWindowFlags(Qt.ToolTip)
        self.custom_box_number = 1
        self.x_label = QLabel("X: 0")
        self.y_label = QLabel("Y: 0")
        self.x_micron_label = QLabel("X Real: 0")
        self.y_micron_label = QLabel("Y Real: 0")
        self.progress_bar = QProgressBar()
        self.analysis_widget = None
        self.analysis_widgets = []  # every stacked analysis box, oldest first
        self.detection_config_path = Path(__file__).resolve().parents[2] / "configs" / "initial_scan_opencv.json"
        self.detection_method = "simple"
        self.detection_settings = {}
        self.cellpose_cellprob_values = []
        self.cellpose_min_size_values = []
        self.fixed_threshold_values = []
        self.fixed_min_area_values = []
        self.simple_uses_fixed_values = False
        # Setup screen state (config-driven)
        self._setup_config = None
        self._setup_elements = []
        self._setup_element_files = {}

    def _uses_fixed_values(self):
        """True when sliders and the compute sweep use configured value lists."""
        if self.detection_method in {"connected_components", "watershed"}:
            return True
        return self.detection_method == "simple" and self.simple_uses_fixed_values

    def _init_ui(self):
        root_layout = QVBoxLayout(self)
        self.tab_widget = QTabWidget()
        root_layout.addWidget(self.tab_widget)

        self.automap_tab = QWidget()
        self.outer_layout = QVBoxLayout(self.automap_tab)
        self.setup_widget = self._create_setup_screen()
        self.outer_layout.addWidget(self.setup_widget)
        self.tab_widget.addTab(self.automap_tab, "AutoMap")

        self.json_maker_tab = self._create_json_maker_tab()
        self.tab_widget.addTab(self.json_maker_tab, "JSON Maker")

        self.coarse_scan_tab = CoarseScanWidget()
        self.tab_widget.addTab(self.coarse_scan_tab, "Mosaic Scan")


    def _create_json_maker_tab(self):
        return JSONMakerWidget()

    def _create_setup_screen(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        # --- Config section (mandatory) ---
        config_group = QGroupBox("Configuration")
        config_group.setStyleSheet("QGroupBox { font-weight: bold; margin-top: 18px; padding-top: 12px; } QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 8px; top: 2px; }")
        config_layout = QVBoxLayout(config_group)
        config_layout.setContentsMargins(10, 20, 10, 10)

        config_row = QHBoxLayout()
        self.setup_config_btn = QPushButton("Select JSON Config")
        self.setup_config_btn.clicked.connect(self.on_setup_config_selected)
        config_row.addWidget(self.setup_config_btn)
        self.setup_config_label = QLabel("No config loaded — select a JSON to continue.")
        self.setup_config_label.setStyleSheet("color: #888;")
        self.setup_config_label.setWordWrap(True)
        config_row.addWidget(self.setup_config_label, 1)
        config_layout.addLayout(config_row)
        layout.addWidget(config_group)

        # --- Element mapping row ---
        elem_group = QGroupBox("Elements → TIFF Files")
        elem_group.setStyleSheet("QGroupBox { font-weight: bold; margin-top: 18px; padding-top: 12px; } QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 8px; top: 2px; }")
        elem_layout = QVBoxLayout(elem_group)
        elem_layout.setContentsMargins(10, 20, 10, 10)
        self.elem_mapping_label = QLabel("Load a config to see expected elements.")
        self.elem_mapping_label.setStyleSheet("color: #888; font-size: 12px;")
        self.elem_mapping_label.setWordWrap(True)
        elem_layout.addWidget(self.elem_mapping_label)
        layout.addWidget(elem_group)

        # --- Directory + file list ---
        dir_group = QGroupBox("TIFF Directory")
        dir_group.setStyleSheet("QGroupBox { font-weight: bold; margin-top: 18px; padding-top: 12px; } QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 8px; top: 2px; }")
        dir_layout = QVBoxLayout(dir_group)
        dir_layout.setContentsMargins(10, 20, 10, 10)

        dir_row = QHBoxLayout()
        self.setup_dir_btn = QPushButton("Select Directory")
        self.setup_dir_btn.clicked.connect(self.on_dir_selected)
        self.setup_dir_btn.setEnabled(False)
        dir_row.addWidget(self.setup_dir_btn)
        self.dir_label = QLabel("No directory selected.")
        self.dir_label.setStyleSheet("color: #888;")
        dir_row.addWidget(self.dir_label, 1)
        dir_layout.addLayout(dir_row)
        dir_layout.addWidget(self.file_list_widget)
        layout.addWidget(dir_group, 1)

        # --- Bottom buttons ---
        bottom_row = QHBoxLayout()
        load_backup_btn = QPushButton("Load Backup (.pkl)")
        load_backup_btn.clicked.connect(self.on_load_backup_clicked)
        bottom_row.addWidget(load_backup_btn)

        bottom_row.addStretch()

        self.setup_confirm_btn = QPushButton("Confirm and Load Images")
        self.setup_confirm_btn.clicked.connect(self.on_confirm_clicked)
        self.setup_confirm_btn.setEnabled(False)
        self.setup_confirm_btn.setStyleSheet("font-weight: bold; padding: 6px 14px;")
        bottom_row.addWidget(self.setup_confirm_btn)
        layout.addLayout(bottom_row)

        # State
        self._setup_elements = []       # ordered unique element list from config
        self._setup_element_files = {}  # element -> abs path (or None)

        return widget

    @staticmethod
    def _try_auto_install(*pip_packages):
        """Attempt to pip-install one or more packages into the running interpreter.

        Returns (success: bool, message: str).
        Runs synchronously — caller should notify the user before calling.
        """
        import subprocess, sys
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", *pip_packages],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode == 0:
                return True, f"Installed {', '.join(pip_packages)} successfully."
            return False, result.stderr.strip() or result.stdout.strip()
        except subprocess.TimeoutExpired:
            return False, "Installation timed out after 5 minutes."
        except Exception as error:
            return False, str(error)

    def _load_detection_config(self):
        """Load the GUI's blob-detector selection and method-specific settings."""
        try:
            with self.detection_config_path.open() as stream:
                params = json.load(stream)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Could not read {self.detection_config_path.name}: {error}") from error

        segmentation = params.get("segmentation_params", {})
        method = segmentation.get("blob_detection_method", "simple").lower()
        methods = params.get("detection_methods", {})
        if method not in {"simple", "cellpose", "connected_components", "watershed", "yolo", "stardist"}:
            raise ValueError(
                "The GUI currently supports 'simple' (OpenCV), 'cellpose', "
                "'connected_components', 'watershed', 'yolo', and 'stardist' "
                f"detection methods, not {method!r}."
            )
        if method == "cellpose" and not CELLPOSE_AVAILABLE:
            model_name = methods.get("cellpose", {}).get("model_type", "Cellpose")
            QMessageBox.information(
                self, "Installing Cellpose",
                "Cellpose is not installed in this environment.\n\n"
                "Attempting automatic installation — this may take a minute..."
            )
            ok, msg = self._try_auto_install("cellpose")
            if ok:
                try:
                    from cellpose import models as _cp_models
                    import automap_hxn.blobs.detection as _det
                    _det.models = _cp_models
                    _det.CELLPOSE_AVAILABLE = True
                    _det.CELLPOSE_IMPORT_ERROR = None
                except Exception as re_err:
                    raise ValueError(
                        f"Cellpose was installed but could not be imported: {re_err}\n\n"
                        "Try restarting the application."
                    )
            else:
                raise ValueError(
                    f"Model '{model_name}' is not available — Cellpose is not installed.\n\n"
                    f"Attempted automatic installation but it failed:\n{msg}\n\n"
                    "Select another configuration or install Cellpose manually."
                )
        if method == "yolo" and not YOLO_AVAILABLE:
            model_name = methods.get("yolo", {}).get("model", "yolo26s-seg.pt")
            QMessageBox.information(
                self, "Installing Ultralytics (YOLO)",
                "The Ultralytics library is not installed in this environment.\n\n"
                "Attempting automatic installation — this should only take a moment..."
            )
            ok, msg = self._try_auto_install("ultralytics")
            if ok:
                try:
                    from ultralytics import YOLO as _YOLO
                    import automap_hxn.blobs.detection as _det
                    _det.YOLO = _YOLO
                    _det.YOLO_AVAILABLE = True
                    _det.YOLO_IMPORT_ERROR = None
                except Exception as re_err:
                    raise ValueError(
                        f"Ultralytics was installed but could not be imported: {re_err}\n\n"
                        "To fix, run in the hxn-gui directory:\n\n"
                        "    pixi add ultralytics==8.3.253\n\n"
                        "Then reinstall automap:\n\n"
                        "    pixi run pip install -e . --no-deps"
                    )
            else:
                raise ValueError(
                    f"Model '{model_name}' is not available — Ultralytics is not installed.\n\n"
                    f"Attempted automatic installation but it failed:\n{msg}\n\n"
                    "To install, run the following in the hxn-gui directory:\n\n"
                    "    pixi add ultralytics==8.3.253\n\n"
                    "Or select another detection model (simple, watershed, StarDist) "
                    "which are already available."
                )
        if method == "stardist" and not STARDIST_AVAILABLE:
            model_name = methods.get("stardist", {}).get("model_name", "2D_versatile_fluo")
            QMessageBox.information(
                self, "Installing StarDist",
                "StarDist is not installed in this environment.\n\n"
                "Attempting automatic installation — this should only take a moment..."
            )
            ok, msg = self._try_auto_install("stardist")
            if ok:
                try:
                    from stardist.models import StarDist2D as _SD
                    import automap_hxn.blobs.detection as _det
                    _det.StarDist2D = _SD
                    _det.STARDIST_AVAILABLE = True
                    _det.STARDIST_IMPORT_ERROR = None
                except Exception as re_err:
                    # StarDist installed but can't load — likely TensorFlow missing
                    raise ValueError(
                        f"StarDist installed but could not be imported: {re_err}\n\n"
                        "StarDist requires TensorFlow 2.19 which is not installed "
                        "in this environment.\n\n"
                        "To fix, run the following in the hxn-gui environment:\n\n"
                        "    pip install \"tensorflow>=2.19,<2.20\"\n\n"
                        "Then restart the application. This is the version used by "
                        "the AutoMap standalone environment.\n\n"
                        "Alternatively, select another detection model (simple, YOLO, "
                        "watershed) which are already available."
                    )
            else:
                raise ValueError(
                    f"Model '{model_name}' is not available — StarDist is not installed.\n\n"
                    f"Attempted automatic installation but it failed:\n{msg}\n\n"
                    "To install manually, run:\n\n"
                    "    pip install stardist \"tensorflow>=2.19,<2.20\"\n\n"
                    "Or select another detection model."
                )

        if method == "cellpose":
            cellpose_settings = methods.get("cellpose", {})
            cellprob_default = float(cellpose_settings.get("cellprob_threshold", 0.0))
            min_size_default = int(cellpose_settings.get("min_size", segmentation.get("min_threshold_area", 10)))
            self.cellpose_cellprob_values = list(
                cellpose_settings.get(
                    "gui_cellprob_threshold_values",
                    [cellprob_default - 1.0, cellprob_default, cellprob_default + 1.0],
                )
            )
            self.cellpose_min_size_values = list(
                cellpose_settings.get(
                    "gui_min_size_values",
                    [max(0, min_size_default // 2), min_size_default, min_size_default * 2],
                )
            )
            if not self.cellpose_cellprob_values or not self.cellpose_min_size_values:
                raise ValueError(
                    "Cellpose GUI settings must provide at least one "
                    "gui_cellprob_threshold_values and one gui_min_size_values."
                )
            # The GUI's existing threshold/area state stores the two Cellpose
            # parameters exposed as interactive sliders. Threshold is scaled
            # by ten because QSlider operates on integers.
            initial_cellprob = min(
                self.cellpose_cellprob_values,
                key=lambda value: abs(value - cellprob_default),
            )
            initial_min_size = min(
                self.cellpose_min_size_values,
                key=lambda value: abs(value - min_size_default),
            )
            min_threshold = int(round(initial_cellprob * 10))
            min_area = initial_min_size
        elif method in {"yolo", "stardist"}:
            min_threshold = 0
            min_area = 0
        else:
            min_threshold = segmentation.get("min_threshold_intensity", 100)
            min_area = segmentation.get("min_threshold_area", 200)

        self.simple_uses_fixed_values = False
        if method in {"connected_components", "watershed"}:
            method_settings = methods.get(method, {})
            self.fixed_threshold_values = sorted(
                method_settings.get("gui_threshold_values", [min_threshold])
            )
            self.fixed_min_area_values = sorted(
                method_settings.get("gui_min_area_values", [min_area])
            )
            if not self.fixed_threshold_values or not self.fixed_min_area_values:
                raise ValueError(
                    f"{method} GUI settings must provide at least one "
                    "gui_threshold_values and one gui_min_area_values."
                )
            # Sliders start on the middle value of each sorted list.
            min_threshold = self.fixed_threshold_values[len(self.fixed_threshold_values) // 2]
            min_area = self.fixed_min_area_values[len(self.fixed_min_area_values) // 2]
        elif method == "simple":
            # Optional for simple: configs may pin the sweep to explicit value
            # lists (a single value each = one detection run). Configs without
            # them keep the legacy full sweep.
            method_settings = methods.get(method, {})
            threshold_values = method_settings.get("gui_threshold_values")
            min_area_values = method_settings.get("gui_min_area_values")
            if threshold_values or min_area_values:
                self.simple_uses_fixed_values = True
                self.fixed_threshold_values = sorted(threshold_values or [min_threshold])
                self.fixed_min_area_values = sorted(min_area_values or [min_area])
                # Sliders start on the middle value of each sorted list.
                min_threshold = self.fixed_threshold_values[len(self.fixed_threshold_values) // 2]
                min_area = self.fixed_min_area_values[len(self.fixed_min_area_values) // 2]

        self.detection_method = method
        self.detection_settings = {
            "min_threshold": min_threshold,
            "min_area": min_area,
            **methods.get(method, {}),
        }

    def on_setup_config_selected(self):
        """Load a JSON config; unlocks directory selection and populates element row."""
        config_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select JSON Configuration",
            str(Path(__file__).resolve().parents[2] / "configs"),
            "JSON files (*.json)",
        )
        if not config_path:
            return

        config_path = Path(config_path)
        try:
            with config_path.open() as f:
                config = json.load(f)
        except (OSError, json.JSONDecodeError) as error:
            QMessageBox.warning(self, "Invalid Configuration",
                                f"Could not read {config_path.name}:\n{error}")
            return

        elem_list_nested = config.get("export_params", {}).get("elem_list", [])
        if not elem_list_nested:
            QMessageBox.warning(self, "Invalid Configuration",
                                f"{config_path.name} has no elem_list in export_params.")
            return
        if isinstance(elem_list_nested[0], str):
            elem_list_nested = [elem_list_nested]
        unique_elements = list(dict.fromkeys(
            e for group in elem_list_nested for e in group
        ))
        if not unique_elements or len(unique_elements) > 3:
            QMessageBox.warning(self, "Invalid Configuration",
                                f"elem_list must have 1–3 unique elements, got {unique_elements}")
            return

        # Load detection settings from the config
        self.detection_config_path = config_path
        try:
            self._load_detection_config()
        except ValueError as error:
            QMessageBox.warning(self, "Detection Configuration", str(error))
            return

        method = config.get("segmentation_params", {}).get("blob_detection_method", "simple")
        self.setup_config_label.setText(
            f"{config_path.name}   |   method: {method}   |   elements: {', '.join(unique_elements)}"
        )
        self.setup_config_label.setStyleSheet("")

        # Store calibration for use at confirm time
        self._setup_config = config
        self._setup_elements = unique_elements
        self._setup_element_files = {e: None for e in unique_elements}

        # Unlock directory selection
        self.setup_dir_btn.setEnabled(True)

        # Update element mapping row; try auto-match if a directory is already chosen
        self._refresh_element_mapping()
        if self.app_state.selected_directory:
            self._auto_match_elements()

    def _refresh_element_mapping(self):
        """Rebuild the element→file label and update the Confirm button state."""
        if not self._setup_elements:
            self.elem_mapping_label.setText("Load a config to see expected elements.")
            self.elem_mapping_label.setStyleSheet("color: #888; font-size: 12px;")
            self.setup_confirm_btn.setEnabled(False)
            return

        parts = []
        all_matched = True
        for elem in self._setup_elements:
            fpath = self._setup_element_files.get(elem)
            if fpath:
                parts.append(f"{elem}: {os.path.basename(fpath)}")
            else:
                parts.append(f"{elem}: —")
                all_matched = False

        self.elem_mapping_label.setText("     ".join(parts))
        self.elem_mapping_label.setStyleSheet("font-size: 12px;")
        self.setup_confirm_btn.setEnabled(all_matched)

    def _auto_match_elements(self):
        """Scan the selected directory and match filenames to element symbols."""
        directory = self.app_state.selected_directory
        if not directory or not self._setup_elements:
            return
        tiffs = [f for f in sorted(os.listdir(directory))
                 if f.lower().endswith(('.tif', '.tiff'))]
        for elem in self._setup_elements:
            if self._setup_element_files.get(elem):
                continue  # already matched; don't override a user pick
            sym = elem.lower()
            # Score: 3 = exact stem, 2 = stem starts with sym followed by
            # a non-alpha char or end (e.g. Fe_map matches Fe), 1 = sym
            # appears at a word boundary within the stem. Plain substring is
            # not used — "sulphur" contains "s" but should not match "S".
            import re as _re
            boundary_re = _re.compile(
                r'(?<![a-z])' + _re.escape(sym) + r'(?![a-z])'
            )
            best, best_score = None, 0
            for fname in tiffs:
                stem = Path(fname).stem.lower()
                if stem == sym:
                    score = 3
                elif stem.startswith(sym) and (len(stem) == len(sym) or not stem[len(sym)].isalpha()):
                    score = 2
                elif boundary_re.search(stem):
                    score = 1
                else:
                    score = 0
                if score > best_score:
                    best, best_score = fname, score
            if best:
                self._setup_element_files[elem] = os.path.join(directory, best)
        self._refresh_element_mapping()
        self._sync_file_list_checkboxes()

    def _sync_file_list_checkboxes(self):
        """Check list items whose paths were auto-matched; uncheck the rest."""
        matched_paths = {
            os.path.normpath(p)
            for p in self._setup_element_files.values() if p
        }
        self.app_state.selected_files_order = []
        for row in range(self.file_list_widget.count()):
            item = self.file_list_widget.item(row)
            cb = self.file_list_widget.itemWidget(item)
            path = os.path.normpath(self.app_state.file_paths[row])
            cb.blockSignals(True)
            cb.setChecked(path in matched_paths)
            cb.blockSignals(False)
            if path in matched_paths:
                self.app_state.selected_files_order.append(row)

    def on_load_backup_clicked(self):
        if not getattr(self, '_setup_config', None):
            QMessageBox.warning(self, "No Config", "Select a JSON config first.")
            return

        n = len(self._setup_elements)
        tiff_paths, _ = QFileDialog.getOpenFileNames(
            self, f"Select {n} TIFF file(s)", "", "TIFF Files (*.tif *.tiff)"
        )
        if len(tiff_paths) != n:
            QMessageBox.warning(self, "Invalid Selection",
                                f"Please select exactly {n} TIFF file(s).")
            return

        pkl_path, _ = QFileDialog.getOpenFileName(
            self, "Select precomputed_blobs.pkl file", "", "Pickle Files (*.pkl)"
        )
        if not pkl_path:
            return

        calib = self._setup_config.get("calibration_params", {})
        self.app_state.microns_per_pixel_x = calib.get("microns_per_pixel_x", 1.0)
        self.app_state.microns_per_pixel_y = calib.get("microns_per_pixel_y", 1.0)
        self.app_state.true_origin_x = calib.get("true_origin_x", 0.0)
        self.app_state.true_origin_y = calib.get("true_origin_y", 0.0)

        self.app_state.img_paths = tiff_paths
        self.app_state.file_names = [os.path.basename(p) for p in tiff_paths]
        color_map = ['red', 'green', 'blue']
        self.app_state.element_colors = color_map[:n]
        self.app_state.thresholds = {c: self.detection_settings.get("min_threshold", 100)
                                     for c in self.app_state.element_colors}
        self.app_state.area_thresholds = {c: self.detection_settings.get("min_area", 200)
                                          for c in self.app_state.element_colors}
        try:
            with open(pkl_path, 'rb') as f:
                self.app_state.precomputed_blobs = pickle.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Error Loading File", f"Could not load pickle file: {e}")
            return

        self._init_analysis_gui(from_backup=True)

    def on_dir_selected(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if not directory:
            return
        self.app_state.selected_directory = directory
        self.dir_label.setText(directory)
        self.dir_label.setStyleSheet("")
        self.app_state.file_paths = []
        self.app_state.selected_files_order = []
        # Reset any previous auto-match so we re-run against the new directory
        self._setup_element_files = {e: None for e in self._setup_elements}
        self.file_list_widget.clear()
        for fname in sorted(os.listdir(directory)):
            if fname.lower().endswith(('.tif', '.tiff')):
                item = QListWidgetItem()
                checkbox = ChannelCheckBox(fname, "#6b7280")
                item.setSizeHint(checkbox.sizeHint())
                self.file_list_widget.addItem(item)
                self.file_list_widget.setItemWidget(item, checkbox)
                checkbox.stateChanged.connect(
                    lambda state, selected_item=item, selected_checkbox=checkbox:
                    self.update_file_selection(selected_item, selected_checkbox, state)
                )
                self.app_state.file_paths.append(os.path.join(directory, fname))
        self._auto_match_elements()

    def update_file_selection(self, item, checkbox, state):
        """Track checked TIFFs and assign them to unmatched elements in order."""
        n = len(self._setup_elements)
        checked_indices = [
            row for row in range(self.file_list_widget.count())
            if self.file_list_widget.itemWidget(self.file_list_widget.item(row)).isChecked()
        ]

        # Reject the check if it would exceed the number of elements the config expects
        if len(checked_indices) > n:
            checkbox.blockSignals(True)
            checkbox.setChecked(False)
            checkbox.blockSignals(False)
            elem_names = ", ".join(self._setup_elements)
            QMessageBox.warning(
                self, "Too Many Files Selected",
                f"The loaded config expects exactly {n} element(s): {elem_names}.\n\n"
                "Deselect a file before selecting another."
            )
            return

        selected = [
            row for row in (self.app_state.selected_files_order or [])
            if row in checked_indices
        ]
        for row in checked_indices:
            if row not in selected:
                selected.append(row)
        self.app_state.selected_files_order = selected
        # Map checked files to elements in order
        for i, elem in enumerate(self._setup_elements):
            if i < len(selected):
                self._setup_element_files[elem] = self.app_state.file_paths[selected[i]]
            else:
                self._setup_element_files[elem] = None
        self._refresh_element_mapping()

    def on_confirm_clicked(self):
        # Guard: all elements must be matched
        missing = [e for e in self._setup_elements
                   if not self._setup_element_files.get(e)]
        if missing:
            QMessageBox.warning(self, "Missing Files",
                                f"No TIFF assigned for: {', '.join(missing)}")
            return

        # Calibration comes from the JSON, not spinboxes
        calib = self._setup_config.get("calibration_params", {})
        self.app_state.microns_per_pixel_x = calib.get("microns_per_pixel_x", 1.0)
        self.app_state.microns_per_pixel_y = calib.get("microns_per_pixel_y", 1.0)
        self.app_state.true_origin_x = calib.get("true_origin_x", 0.0)
        self.app_state.true_origin_y = calib.get("true_origin_y", 0.0)

        # Build img_paths in element order
        self.app_state.img_paths = [self._setup_element_files[e]
                                    for e in self._setup_elements]
        self.app_state.file_names = [os.path.basename(p)
                                     for p in self.app_state.img_paths]
        color_map = ['red', 'green', 'blue']
        self.app_state.element_colors = color_map[:len(self._setup_elements)]
        self.app_state.thresholds = {
            color: self.detection_settings["min_threshold"]
            for color in self.app_state.element_colors
        }
        self.app_state.area_thresholds = {
            color: self.detection_settings["min_area"]
            for color in self.app_state.element_colors
        }
        self._init_analysis_gui()

    def _init_analysis_gui(self, from_backup=False):
        with self._analysis_lock:
            self._analysis_generation += 1
            generation = self._analysis_generation

        # Only one analysis box at a time: loading again replaces the current
        # view instead of stacking a second box below it.
        if self.analysis_widgets:
            with self._analysis_lock:
                self.blob_items.clear()
                self.union_box_items.clear()
            for box in self.analysis_widgets:
                self.outer_layout.removeWidget(box)
                box.deleteLater()
            self.analysis_widgets = []
            if self.graphics_scene is not None:
                self.graphics_scene.deleteLater()
            self.graphics_scene = None
            self.graphics_view = None
            self.pixmap_item = None

        self.setup_widget.setParent(None)
        self.analysis_widget = QWidget()
        self.analysis_widgets.append(self.analysis_widget)
        self.main_layout = QHBoxLayout(self.analysis_widget)
        
        left_panel = QVBoxLayout()
        self._create_image_view_panel()
        left_panel.addWidget(self.graphics_view)
        
        self.main_layout.addLayout(left_panel)

        self._create_controls_panel()
        self.outer_layout.addWidget(self.analysis_widget)
        
        self.progress_bar.setParent(None)
        self.outer_layout.addWidget(self.progress_bar)

        if from_backup:
            self.progress_bar.hide()
            self.update_boxes()
        else:
            QTimer.singleShot(
                100, lambda: self._start_blob_computation(generation)
            )

    def return_to_selection(self):
        """Discard the current analysis screen and rebuild the TIFF selector."""
        # The scene owns its graphics items.  Clear Python references before
        # destroying that scene, otherwise the next analysis run tries to
        # remove wrappers for C++ items that Qt has already deleted.
        with self._analysis_lock:
            self._analysis_generation += 1
            self.blob_items.clear()
            self.union_box_items.clear()

        for box in self.analysis_widgets:
            self.outer_layout.removeWidget(box)
            box.deleteLater()
        self.analysis_widgets = []
        self.analysis_widget = None
        self.outer_layout.removeWidget(self.progress_bar)
        self.progress_bar.deleteLater()
        self.setup_widget.deleteLater()
        if self.graphics_scene is not None:
            self.graphics_scene.deleteLater()
        self.graphics_scene = None
        self.graphics_view = None
        self.pixmap_item = None

        # Keep the same AppState instance for embedded hosts, but reset it to
        # the original selection-screen defaults before rebuilding the widgets.
        self.app_state.__init__()
        self._init_ui_elements()
        self.setup_widget = self._create_setup_screen()
        self.outer_layout.addWidget(self.setup_widget)


    def _create_image_view_panel(self):
        # Load 1-3 TIFFs; pad missing channels with zeros of the same shape.
        loaded_images = [tiff.imread(p).astype(np.float32) for p in self.app_state.img_paths]
        shapes = [img.shape for img in loaded_images]
        self.app_state.target_shape = Counter(shapes).most_common(1)[0][0]

        # Resize loaded images to target shape
        for i, img in enumerate(loaded_images):
            fname = self.app_state.file_names[i] if i < len(self.app_state.file_names) else f"image_{i}"
            loaded_images[i] = resize_if_needed(img, fname, self.app_state.target_shape)

        # Pad to 3 channels (RGB) with zeros
        while len(loaded_images) < 3:
            loaded_images.append(np.zeros(self.app_state.target_shape, dtype=np.float32))

        img_r, img_g, img_b = loaded_images[:3]
        self.source_images = [img_r, img_g, img_b]
        _morph = getattr(self, '_setup_config', {}).get('morphology_params', {})
        _kernel = tuple(_morph.get('normalize_kernel_size', [3, 3]))
        _iters  = _morph.get('dilate_iterations', 2)
        self.norm_dilated = [normalize_and_dilate(im, kernel_size=_kernel, iterations=_iters) for im in self.source_images]
        
        merged_rgb = cv2.merge([nd[0] for nd in self.norm_dilated])
        # Keep an owned reference: PySide6 may destroy a scene that is only a
        # local variable, leaving the view blank once setup returns.
        self.graphics_scene = QGraphicsScene(self)
        q_img = QImage(merged_rgb.data, merged_rgb.shape[1], merged_rgb.shape[0], merged_rgb.shape[1] * 3, QImage.Format_RGB888)
        
        self.graphics_view = ZoomableGraphicsView(self.graphics_scene, self)
        self.graphics_view.current_qimage = q_img
        self.pixmap_item = QGraphicsPixmapItem(QPixmap.fromImage(q_img))
        self.graphics_scene.addItem(self.pixmap_item)

    def _create_controls_panel(self):
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)

        return_btn = QPushButton("Return to Selection")
        return_btn.clicked.connect(self.return_to_selection)
        reset_btn = QPushButton("Reset View")
        reset_btn.clicked.connect(
            lambda: self.graphics_view.resetTransform() if self.graphics_view else None
        )

        button_layout = QHBoxLayout()
        button_layout.addWidget(return_btn)
        button_layout.addWidget(reset_btn)
        controls_layout.addLayout(button_layout)

        # Lists and buttons
        self.union_list_widget.setSelectionMode(QListWidget.MultiSelection)
        self.union_list_widget.itemSelectionChanged.connect(self.on_union_item_selected)
        
        send_to_list_btn = QPushButton("Add to list")
        send_to_list_btn.clicked.connect(self.send_to_list)
        get_elements_btn = QPushButton("Get all elements")
        get_elements_btn.clicked.connect(self.get_elements_list)
        union_btn = QPushButton("Get unions")
        union_btn.clicked.connect(self.union_function)
        add_box_btn = QPushButton("Add Box")
        add_box_btn.clicked.connect(self.add_box)

        union_list_layout = QVBoxLayout()
        union_list_layout.addWidget(self.union_list_widget)
        union_list_layout.addWidget(send_to_list_btn)
        union_list_layout.addWidget(get_elements_btn)
        union_list_layout.addWidget(union_btn)
        union_list_layout.addWidget(add_box_btn)

        send_to_queue_btn = QPushButton("Queue Selected Plans")
        send_to_queue_btn.clicked.connect(self.send_to_queue_server)
        clear_queue_btn = QPushButton("Clear")
        clear_queue_btn.clicked.connect(self.clear_queue_server_list)

        queue_list_layout = QVBoxLayout()
        queue_list_layout.addWidget(self.queue_server_list)
        queue_list_layout.addWidget(send_to_queue_btn)
        queue_list_layout.addWidget(clear_queue_btn)

        dual_list_layout = QHBoxLayout()
        dual_list_layout.addLayout(union_list_layout)
        dual_list_layout.addLayout(queue_list_layout)
        controls_layout.addLayout(dual_list_layout)

        # Coordinates
        coord_layout = QHBoxLayout()
        coord_layout.addWidget(self.x_label)
        coord_layout.addWidget(self.y_label)
        coord_layout.addWidget(self.x_micron_label)
        coord_layout.addWidget(self.y_micron_label)
        controls_layout.addLayout(coord_layout)

        # Legend
        legend_layout = QHBoxLayout()
        legend_label = QLabel("Legend")
        legend_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        legend_layout.addWidget(legend_label)
        for i, color in enumerate(self.app_state.element_colors):
            cb = ChannelCheckBox(self.app_state.file_names[i], color)
            cb.setChecked(True)
            cb.setStyleSheet(
                f"QCheckBox {{ color: {color}; }}"
                "QCheckBox::indicator {"
                " width: 14px; height: 14px;"
                f" border: 2px solid {color};"
                " background: #ffffff;"
                "}"
            )
            cb.stateChanged.connect(self.update_boxes)
            self.checkboxes[color] = cb
            legend_layout.addWidget(cb)

        self.union_checkbox = ChannelCheckBox("Union Boxes", "black")
        self.union_checkbox.setChecked(True)
        self.union_checkbox.setStyleSheet(
            "QCheckBox { color: black; }"
            "QCheckBox::indicator {"
            " width: 14px; height: 14px;"
            " border: 2px solid #000000;"
            " background: #ffffff;"
            "}"
        )
        self.union_checkbox.stateChanged.connect(self.update_boxes)
        legend_layout.addWidget(self.union_checkbox)
        self.checkboxes['union'] = self.union_checkbox
        legend_layout.addStretch()
        controls_layout.addLayout(legend_layout)

        if self.detection_method in {"yolo", "stardist"}:
            controls_layout.addWidget(
                QLabel(
                    "YOLO26s instance segmentation: default settings (one cached inference per element)."
                    if self.detection_method == "yolo"
                    else "StarDist instance segmentation: default settings (one cached inference per element)."
                )
            )
            controls_widget.setLayout(controls_layout)
            self.main_layout.addWidget(controls_widget)
            return

        # Sliders
        slider_layout = QHBoxLayout()
        for color in self.app_state.element_colors:
            i = self.app_state.element_colors.index(color)
            vbox = QVBoxLayout()
            if self.detection_method == "cellpose":
                label = QLabel(
                    f"{self.app_state.file_names[i]}_cellprob threshold: "
                    f"{self.app_state.thresholds[color] / 10:.1f}"
                )
            else:
                label = QLabel(f"{self.app_state.file_names[i]}_threshold: {self.app_state.thresholds[color]}")
            slider = QSlider(Qt.Horizontal)
            if self.detection_method == "cellpose":
                slider.setRange(0, len(self.cellpose_cellprob_values) - 1)
                slider.setValue(min(
                    range(len(self.cellpose_cellprob_values)),
                    key=lambda index: abs(self.cellpose_cellprob_values[index] - self.app_state.thresholds[color] / 10),
                ))
            elif self._uses_fixed_values():
                slider.setRange(0, len(self.fixed_threshold_values) - 1)
                slider.setValue(min(
                    range(len(self.fixed_threshold_values)),
                    key=lambda index: abs(self.fixed_threshold_values[index] - self.app_state.thresholds[color]),
                ))
            else:
                slider.setRange(0, 255)
                slider.setValue(self.app_state.thresholds[color])
            slider.setTickInterval(1 if self.detection_method == "cellpose" else 10)
            slider.setTickPosition(QSlider.TicksBelow)
            slider.valueChanged.connect(lambda val, c=color: self.on_slider_change(val, c))
            if self.detection_method == "cellpose":
                slider.setToolTip("Choose one of the precomputed Cellpose cellprob_threshold values.")
            elif self._uses_fixed_values():
                slider.setToolTip("Choose one of the configured threshold values.")
            self.sliders[color] = slider
            self.slider_labels[color] = label
            vbox.addWidget(label)
            vbox.addWidget(slider)
            slider_layout.addLayout(vbox)
        controls_layout.addLayout(slider_layout)

        # Area Sliders
        area_slider_layout = QHBoxLayout()
        for color in self.app_state.element_colors:
            i = self.app_state.element_colors.index(color)
            vbox = QVBoxLayout()
            if self.detection_method == "cellpose":
                label = QLabel(f"{self.app_state.file_names[i]}_min size: {self.app_state.area_thresholds[color]}")
            else:
                label = QLabel(f"{self.app_state.file_names[i]}_min_area: {self.app_state.area_thresholds[color]}")
            slider = QSlider(Qt.Horizontal)
            if self.detection_method == "cellpose":
                slider.setRange(0, len(self.cellpose_min_size_values) - 1)
                slider.setValue(min(
                    range(len(self.cellpose_min_size_values)),
                    key=lambda index: abs(self.cellpose_min_size_values[index] - self.app_state.area_thresholds[color]),
                ))
            elif self._uses_fixed_values():
                slider.setRange(0, len(self.fixed_min_area_values) - 1)
                slider.setValue(min(
                    range(len(self.fixed_min_area_values)),
                    key=lambda index: abs(self.fixed_min_area_values[index] - self.app_state.area_thresholds[color]),
                ))
            else:
                slider.setRange(10, 400)
                slider.setValue(self.app_state.area_thresholds[color])
            slider.setTickInterval(1 if self.detection_method == "cellpose" else 10)
            slider.setTickPosition(QSlider.TicksBelow)
            slider.valueChanged.connect(lambda val, c=color: self.on_area_slider_change(val, c))
            if self.detection_method == "cellpose":
                slider.setToolTip("Choose one of the precomputed Cellpose min_size values.")
            elif self._uses_fixed_values():
                slider.setToolTip("Choose one of the configured minimum-area values.")
            self.area_sliders[color] = slider
            self.area_slider_labels[color] = label
            vbox.addWidget(label)
            vbox.addWidget(slider)
            area_slider_layout.addLayout(vbox)
        controls_layout.addLayout(area_slider_layout)

        controls_widget.setLayout(controls_layout)
        self.main_layout.addWidget(controls_widget)

    def _start_blob_computation(self, expected_generation=None):
        with self._analysis_lock:
            if (
                expected_generation is not None
                and expected_generation != self._analysis_generation
            ):
                return
            generation = self._analysis_generation
            self.app_state.precomputed_blobs = {color: {} for color in self.app_state.element_colors}
            self.app_state.current_iteration = 0
        
        if self.detection_method == "cellpose":
            thresholds_range = [int(round(value * 10)) for value in self.cellpose_cellprob_values]
            area_range = self.cellpose_min_size_values
        elif self.detection_method in {"yolo", "stardist"}:
            thresholds_range = [0]
            area_range = [0]
        elif self._uses_fixed_values():
            thresholds_range = self.fixed_threshold_values
            area_range = self.fixed_min_area_values
        else:
            # Legacy full sweep for simple configs without gui_*_values lists.
            # Clamp to max_threshold: SimpleBlobDetector rejects
            # minThreshold > maxThreshold.
            max_t = int(self.detection_settings.get("max_threshold", 255))
            thresholds_range = [t for t in range(0, 256, 10) if t <= max_t] or [0]
            area_range = list(range(10, 501, 10))
        total_iterations = len(self.app_state.element_colors) * len(thresholds_range) * len(area_range)
        
        self.progress_bar.setRange(0, total_iterations)
        self.progress_bar.setValue(0)
        if self.detection_method == "cellpose":
            self.progress_bar.setFormat("Cellpose: %v of %m model runs complete")
        elif self.detection_method == "yolo":
            self.progress_bar.setFormat("YOLO: %v of %m inference runs complete")
        elif self.detection_method == "stardist":
            self.progress_bar.setFormat("StarDist: %v of %m inference runs complete")
        else:
            self.progress_bar.setFormat("Computing blobs... %p%")
        self.progress_bar.show()

        threading.Thread(
            target=self._blob_computation_thread,
            args=(
                generation,
                thresholds_range,
                area_range,
                list(self.app_state.element_colors),
                list(self.app_state.file_names),
                list(self.source_images),
                [nd[1] for nd in self.norm_dilated],
            ),
            daemon=True,
        ).start()

    def _blob_computation_thread(
        self, generation, thresholds_range, area_range, colors, file_names,
        source_images, dilated_imgs,
    ):
        total_runs = len(colors) * len(thresholds_range) * len(area_range)
        
        for i, color in enumerate(colors):
            for t_val in thresholds_range:
                for a_val in area_range:
                    with self._analysis_lock:
                        run_number = self.app_state.current_iteration + 1
                    heartbeat_done = threading.Event()
                    if self.detection_method == "cellpose":
                        print(
                            f"[GUI] Cellpose starting run {run_number}/{total_runs}: "
                            f"{file_names[i]} ({color}); cellprob_threshold={t_val / 10:.1f}, "
                            f"min_size={a_val}."
                        )

                        def report_elapsed(
                            run_number=run_number,
                            total_runs=total_runs,
                            heartbeat_done=heartbeat_done,
                        ):
                            started = time.monotonic()
                            while not heartbeat_done.wait(15):
                                elapsed = int(time.monotonic() - started)
                                print(
                                    f"[GUI] Cellpose run {run_number}/{total_runs} is still running "
                                    f"({elapsed}s elapsed)."
                                )

                        threading.Thread(target=report_elapsed, daemon=True).start()
                    try:
                        blobs = self._detect_blobs(
                            dilated_imgs[i],
                            source_images[i],
                            t_val, a_val, color,
                            file_names[i],
                        )
                    finally:
                        heartbeat_done.set()
                    if self.detection_method == "cellpose":
                        print(
                            f"[GUI] Cellpose found {len(blobs)} boxes in "
                            f"{file_names[i]} ({color}); "
                            f"cellprob_threshold={t_val / 10:.1f}, min_size={a_val}."
                        )
                    with self._analysis_lock:
                        if generation != self._analysis_generation:
                            return
                        self.app_state.precomputed_blobs[color][(t_val, a_val)] = blobs
                        self.app_state.current_iteration += 1
                        iteration = self.app_state.current_iteration
                    if self.detection_method == "cellpose":
                        print(f"[GUI] Cellpose progress: {iteration}/{total_runs} model runs complete.")
                    QApplication.instance().postEvent(
                        self, UpdateProgressEvent(iteration, generation)
                    )
        
        QApplication.instance().postEvent(self, ComputationFinishedEvent(generation))

    def customEvent(self, event):
        # PySide6 wraps posted custom events as QEvent, so checking the Python
        # subclass with isinstance is unreliable across Qt bindings.
        if event.type() == UpdateProgressEvent.EVENT_TYPE:
            if event.generation == self._analysis_generation:
                self.progress_bar.setValue(event.value)
        elif event.type() == ComputationFinishedEvent.EVENT_TYPE:
            if event.generation != self._analysis_generation:
                return
            self.progress_bar.hide()
            if self.app_state.selected_directory and self.app_state.precomputed_blobs:
                output_path = Path(self.app_state.selected_directory) / "precomputed_blobs.pkl"
                with open(output_path, "wb") as f:
                    pickle.dump(self.app_state.precomputed_blobs, f)
            self.update_boxes()


    def _detect_blobs(self, img_norm, img_orig, min_thresh, min_area, color, file_name):
        if self.detection_method == "cellpose":
            cellpose_params = {
                key: value for key, value in self.detection_settings.items()
                if key not in {"min_threshold", "min_area", "gui_cellprob_threshold_values", "gui_min_size_values"}
            }
            cellpose_params["cellprob_threshold"] = min_thresh / 10
            cellpose_params["min_size"] = min_area
            # The shared detector owns the Cellpose model cache and uses the
            # original image, without the GUI's OpenCV dilation, for inference.
            return detect_blobs(
                img_orig, img_orig, min_thresh, min_area, color, file_name,
                method="cellpose", **cellpose_params,
            )

        if self.detection_method == "yolo":
            yolo_params = {
                key: value for key, value in self.detection_settings.items()
                if key not in {"min_threshold", "min_area"}
            }
            return detect_blobs(
                img_orig, img_orig, min_thresh, min_area, color, file_name,
                method="yolo", **yolo_params,
            )

        if self.detection_method == "stardist":
            stardist_params = {
                key: value for key, value in self.detection_settings.items()
                if key not in {"min_threshold", "min_area"}
            }
            return detect_blobs(
                img_orig, img_orig, min_thresh, min_area, color, file_name,
                method="stardist", **stardist_params,
            )

        if self.detection_method in {"connected_components", "watershed"}:
            method_params = {
                key: value for key, value in self.detection_settings.items()
                if key not in {
                    "min_threshold", "min_area", "gui_threshold_values", "gui_min_area_values",
                }
            }
            return detect_blobs(
                img_norm, img_orig, min_thresh, min_area, color, file_name,
                method=self.detection_method, **method_params,
            )

        max_threshold = self.detection_settings.get("max_threshold", 255)
        if not 0 <= min_thresh <= max_threshold:
            print(
                f"[GUI] Skipping simple detection for {file_name} ({color}): "
                f"threshold {min_thresh} outside [0, {max_threshold}]."
            )
            return []

        # SimpleBlobDetector requires strictly 0 < minArea <= maxArea.
        # A configured 0 means "no minimum", so clamp it to the smallest
        # accepted value instead of crashing.
        max_area = self.detection_settings.get("max_area", 50000)
        effective_min_area = max(1, min_area)
        if effective_min_area > max_area:
            print(
                f"[GUI] Skipping simple detection for {file_name} ({color}): "
                f"min_area {min_area} exceeds max_area {max_area}."
            )
            return []

        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = min_thresh
        params.maxThreshold = max_threshold
        params.filterByArea = True
        params.minArea = effective_min_area
        params.maxArea = max_area
        params.thresholdStep = self.detection_settings.get("threshold_step", 5)
        params.filterByColor = self.detection_settings.get("filter_by_color", False)
        params.filterByCircularity = self.detection_settings.get("filter_by_circularity", False)
        params.filterByInertia = False
        params.filterByConvexity = False
        params.minRepeatability = 1
        
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(img_norm)
        blobs = []

        for idx, kp in enumerate(keypoints, start=1):
            x, y = int(kp.pt[0]), int(kp.pt[1])
            radius = int(kp.size / 2)
            box_size = 2 * radius
            box_x, box_y = x - radius, y - radius

            x1, y1 = max(0, box_x), max(0, box_y)
            x2, y2 = min(img_orig.shape[1], x + radius), min(img_orig.shape[0], y + radius)
            roi_orig = img_orig[y1:y2, x1:x2]
            roi_dilated = img_norm[y1:y2, x1:x2]

            if roi_orig.size > 0:
                blobs.append({
                    'Box': f"{file_name} Box #{idx}",
                    'center': (x, y), 'radius': radius, 'color': color, 'file': file_name,
                    'max_intensity': roi_orig.max(), 'mean_intensity': roi_orig.mean(),
                    'mean_dilation': float(roi_dilated.mean()),
                    'box_x': box_x, 'box_y': box_y, 'box_size': box_size
                })
        return blobs

    def update_boxes(self):
        if self.graphics_view is None:
            return
        selected_colors = {c for c, cb in self.checkboxes.items() if cb.isChecked() and c != 'union'}
        blobs = self.get_current_blobs()
        self.graphics_view.update_blobs(blobs, selected_colors)
        self.redraw_boxes(blobs, selected_colors)

    def get_current_blobs(self):
        blobs = []
        if not self.app_state.precomputed_blobs:
            return blobs

        for color in self.app_state.element_colors:
            if color in self.app_state.thresholds and color in self.app_state.area_thresholds:
                threshold = self.app_state.thresholds[color]
                area = self.app_state.area_thresholds[color]
                key = (threshold, area)
                if color in self.app_state.precomputed_blobs and key in self.app_state.precomputed_blobs[color]:
                    blobs.extend(self.app_state.precomputed_blobs[color][key])
        return blobs

    def redraw_boxes(self, blobs, selected_colors):
        for item in self.blob_items:
            self.graphics_view.scene().removeItem(item)
        self.blob_items.clear()

        for blob in blobs:
            if blob['color'] in selected_colors:
                pen = QPen(QColor(blob['color']), 2)
                rect_item = self.graphics_view.scene().addRect(blob['box_x'], blob['box_y'], blob['box_size'], blob['box_size'], pen)
                self.blob_items.append(rect_item)

        for item in self.union_box_items:
            self.graphics_view.scene().removeItem(item)
        self.union_box_items.clear()

        if self.union_checkbox.isChecked():
            for idx, ub in self.graphics_view.union_dict.items():
                cx, cy = ub['center']
                length = ub['length']
                pen = QPen(QColor("white"), 2, Qt.DashLine)
                rect_item = self.graphics_view.scene().addRect(cx - length / 2, cy - length / 2, length, length, pen)
                self.union_box_items.append(rect_item)

    def update_mouse_coordinates(self, pos):
        x, y = int(pos.x()), int(pos.y())
        self.x_label.setText(f"X: {x}")
        self.y_label.setText(f"Y: {y}")
        self.x_micron_label.setText(f"X Real (µm): {(x * self.app_state.microns_per_pixel_x) + self.app_state.true_origin_x:.2f}")
        self.y_micron_label.setText(f"Y Real (µm): {(y * self.app_state.microns_per_pixel_y) + self.app_state.true_origin_y:.2f}")

    def handle_hover(self, event, scene_pos):
        x, y = int(scene_pos.x()), int(scene_pos.y())
        
        for blob in self.graphics_view.blobs:
            if blob['color'] not in self.graphics_view.visible_colors: continue
            cx, cy = blob['center']
            r = blob['radius']
            if abs(x - cx) <= r and abs(y - cy) <= r:
                self._show_tooltip(event, self._format_blob_tooltip(blob))
                return
        
        if self.union_checkbox.isChecked():
            for idx, ub in self.graphics_view.union_dict.items():
                cx, cy = ub['center']
                length = ub['length']
                rect = QRect(
                    int(cx - length / 2),
                    int(cy - length / 2),
                    int(length),
                    int(length)
                )
                if rect.contains(scene_pos.toPoint()):
                    self._show_tooltip(event, self._format_union_tooltip(ub, idx))
                    return

        self.hover_label.hide()

    def _show_tooltip(self, event, html):
        self.hover_label.setText(html)
        self.hover_label.adjustSize()
        mouse_pos = self.graphics_view.mapTo(self, event.pos())
        new_pos = QPoint(mouse_pos.x() + 20, mouse_pos.y() - self.hover_label.height() - 10)
        self.hover_label.move(new_pos)
        self.hover_label.show()

    def _format_blob_tooltip(self, blob):
        cx, cy = blob['center']
        real_cx = (cx * self.app_state.microns_per_pixel_x) + self.app_state.true_origin_x
        real_cy = (cy * self.app_state.microns_per_pixel_y) + self.app_state.true_origin_y
        real_w = blob['box_size'] * self.app_state.microns_per_pixel_x
        real_h = blob['box_size'] * self.app_state.microns_per_pixel_y
        return (
            f"<b>{blob['Box']}</b><br>"
            f"Center: ({cx}, {cy})<br>"
            f"Length: {blob['box_size']} px<br>"
            f"Area: {blob['box_size']**2} px²<br><br>"
            f"Real Center: ({real_cx:.2f}, {real_cy:.2f}) µm<br>"
            f"Real Size: {real_w:.2f} × {real_h:.2f} µm<br>"
            f"Real Area: {real_w * real_h:.2f} µm²<br><br>"
            f"Max intensity: {blob['max_intensity']:.3f}"
        )

    def _format_union_tooltip(self, ub, idx):
        return (
            f"<b>Union Box #{idx}</b><br>"
            f"Center: ({ub['center'][0]}, {ub['center'][1]})<br>"
f"Length: {ub['length']} px<br>"
            f"Area: {ub['area']} px²<br><br>"
            f"Real Center: ({ub['real_center'][0]:.2f}, {ub['real_center'][1]:.2f}) µm<br>"
            f"Real Size: {ub['real_size'][0]:.2f} × {ub['real_size'][1]:.2f} µm<br>"
            f"Real Area: {ub['real_area']:.2f} µm²"
        )

    def on_slider_change(self, value, color):
        if self.detection_method == "cellpose":
            cellprob_threshold = self.cellpose_cellprob_values[value]
            self.app_state.thresholds[color] = int(round(cellprob_threshold * 10))
            self.slider_labels[color].setText(
                f"{self.checkboxes[color].text()}_cellprob threshold: {cellprob_threshold:.1f}"
            )
            self.update_boxes()
            return
        if self._uses_fixed_values():
            threshold = self.fixed_threshold_values[value]
            self.app_state.thresholds[color] = threshold
            self.slider_labels[color].setText(
                f"{self.checkboxes[color].text()}_threshold: {threshold}"
            )
            self.update_boxes()
            return
        snapped = round(value / 10) * 10
        if self.app_state.thresholds[color] != snapped:
            self.app_state.thresholds[color] = snapped
            self.sliders[color].blockSignals(True)
            self.sliders[color].setValue(snapped)
            self.sliders[color].blockSignals(False)
            self.slider_labels[color].setText(f"{self.checkboxes[color].text()}_threshold: {snapped}")
            self.update_boxes()

    def on_area_slider_change(self, value, color):
        if self.detection_method == "cellpose":
            min_size = self.cellpose_min_size_values[value]
            self.app_state.area_thresholds[color] = min_size
            self.area_slider_labels[color].setText(
                f"{self.checkboxes[color].text()}_min size: {min_size}"
            )
            self.update_boxes()
            return
        if self._uses_fixed_values():
            min_area = self.fixed_min_area_values[value]
            self.app_state.area_thresholds[color] = min_area
            self.area_slider_labels[color].setText(
                f"{self.checkboxes[color].text()}_min_area: {min_area}"
            )
            self.update_boxes()
            return
        snapped = round(value / 10) * 10
        if self.app_state.area_thresholds[color] != snapped:
            self.app_state.area_thresholds[color] = snapped
            self.area_sliders[color].blockSignals(True)
            self.area_sliders[color].setValue(snapped)
            self.area_sliders[color].blockSignals(False)
            self.area_slider_labels[color].setText(f"{self.checkboxes[color].text()}_min_area: {snapped}")
            self.update_boxes()

    def on_union_item_selected(self):
        selected_items = self.union_list_widget.selectedItems()
        self.graphics_view.highlight_selected_boxes(selected_items)

    def add_box(self):
        QMessageBox.information(self, "Add Union Box", "Click and drag to define a new union box.")

        self.original_mouse_press_event = self.graphics_view.mousePressEvent
        self.original_mouse_release_event = self.graphics_view.mouseReleaseEvent
        
        temp_state = {'start': None}

        def on_press(event):
            if event.button() == Qt.LeftButton:
                temp_state['start'] = self.graphics_view.mapToScene(event.pos()).toPoint()
            else:
                self.original_mouse_press_event(event)

        def on_release(event):
            if event.button() != Qt.LeftButton or temp_state['start'] is None:
                self.original_mouse_release_event(event)
                return

            end = self.graphics_view.mapToScene(event.pos()).toPoint()
            start = temp_state['start']
            temp_state['start'] = None

            self.graphics_view.mousePressEvent = self.original_mouse_press_event
            self.graphics_view.mouseReleaseEvent = self.original_mouse_release_event

            x1, y1 = start.x(), start.y()
            x2, y2 = end.x(), end.y()
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            length = max(abs(x2 - x1), abs(y2 - y1))
            area = length * length

            real_cx = (cx * self.app_state.microns_per_pixel_x) + self.app_state.true_origin_x
            real_cy = (cy * self.app_state.microns_per_pixel_y) + self.app_state.true_origin_y
            real_length_x = length * self.app_state.microns_per_pixel_x
            real_length_y = length * self.app_state.microns_per_pixel_y
            real_area = real_length_x * real_length_y
            
            new_union = {
                'center': (cx, cy),
                'length': length,
                'area': area,
                'real_center': (real_cx, real_cy),
                'real_size': (real_length_x, real_length_y),
                'real_area': real_area,
            }

            next_index = max(self.graphics_view.union_dict.keys(), default=0) + 1
            self.graphics_view.union_dict[next_index] = new_union

            item_text = f"Custom Box #{self.custom_box_number}"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, self.custom_box_number)
            item.setData(Qt.UserRole + 1, new_union)
            item.setToolTip(self._format_union_tooltip(new_union, self.custom_box_number))
            self.union_list_widget.addItem(item)
            
            self.custom_box_number += 1

            if self.app_state.selected_directory:
                output_path = Path(self.app_state.selected_directory) / "union_blobs.json"
                serializable_dict = make_json_serializable(self.graphics_view.union_dict)
                with open(output_path, "w") as f:
                    json.dump(serializable_dict, f, indent=4)

            self.update_boxes()

        self.graphics_view.mousePressEvent = on_press
        self.graphics_view.mouseReleaseEvent = on_release

    def union_function(self):
        blobs = self.get_current_blobs()
        blobs_by_color = {color: [] for color in self.app_state.element_colors}
        for blob in blobs:
            blobs_by_color[blob['color']].append(blob)

        union_objects = {}
        union_index = 1
        
        reds = blobs_by_color.get('red', [])
        greens = blobs_by_color.get('green', [])
        blues = blobs_by_color.get('blue', [])
    
        for r in reds:
            for g in greens:
                if not self._boxes_intersect(r, g): continue
                for b in blues:
                    if self._boxes_intersect(r, b) and self._boxes_intersect(g, b):
                        cx, cy = self._union_center(r, g, b)
                        length, area = self._union_box_dimensions(r, g, b)
                        
                        real_cx = (cx * self.app_state.microns_per_pixel_x) + self.app_state.true_origin_x
                        real_cy = (cy * self.app_state.microns_per_pixel_y) + self.app_state.true_origin_y
                        real_length_x = length * self.app_state.microns_per_pixel_x
                        real_length_y = length * self.app_state.microns_per_pixel_y
                        
                        union_objects[union_index] = {
                            'center': (cx, cy), 'length': length, 'area': area,
                            'real_center': (real_cx, real_cy),
                            'real_size': (real_length_x, real_length_y),
                            'real_area': real_length_x * real_length_y
                        }
                        union_index += 1
        
        self.graphics_view.union_dict = union_objects
        self.union_list_widget.clear()
        for idx, ub in union_objects.items():
            item = QListWidgetItem(f"Union Box #{idx}")
            item.setToolTip(self._format_union_tooltip(ub, idx))
            item.setData(Qt.UserRole + 1, ub)
            self.union_list_widget.addItem(item)

        if self.app_state.selected_directory:
            output_path = Path(self.app_state.selected_directory) / "union_blobs.json"
            serializable_dict = make_json_serializable(self.graphics_view.union_dict)
            with open(output_path, "w") as f:
                json.dump(serializable_dict, f, indent=4)
        
        self.update_boxes()

    def _boxes_intersect(self, b1, b2):
        x1_min, y1_min = b1['box_x'], b1['box_y']
        x1_max = x1_min + b1['box_size']
        y1_max = y1_min + b1['box_size']
        x2_min, y2_min = b2['box_x'], b2['box_y']
        x2_max = x2_min + b2['box_size']
        y2_max = y2_min + b2['box_size']
        return not (x1_max < x2_min or x1_min > x2_max or y1_max < y2_min or y1_min > y2_max)

    def _union_center(self, b1, b2, b3):
        x_vals = [b1['center'][0], b2['center'][0], b3['center'][0]]
        y_vals = [b1['center'][1], b2['center'][1], b3['center'][1]]
        return (sum(x_vals) // 3, sum(y_vals) // 3)

    def _union_box_dimensions(self, b1, b2, b3):
        xs = [b['box_x'] for b in [b1, b2, b3]]
        ys = [b['box_y'] for b in [b1, b2, b3]]
        sizes = [b['box_size'] for b in [b1, b2, b3]]
        min_x = min(xs)
        min_y = min(ys)
        max_x = max(x + s for x, s in zip(xs, sizes))
        max_y = max(y + s for y, s in zip(ys, sizes))
        length = max(max_x - min_x, max_y - min_y)
        return length, length**2

    def send_to_list(self):
        existing_texts = {self.queue_server_list.item(i).text() for i in range(self.queue_server_list.count())}
        for item in self.union_list_widget.selectedItems():
            if item.text() not in existing_texts:
                new_item = QListWidgetItem(item.text())
                new_item.setToolTip(item.toolTip())
                new_item.setData(Qt.UserRole + 1, item.data(Qt.UserRole + 1))
                self.queue_server_list.addItem(new_item)

    def get_elements_list(self):
        self.union_list_widget.clear()
        all_blobs = self.get_current_blobs()
        for blob in all_blobs:
            item = QListWidgetItem(blob['Box'])
            item.setToolTip(self._format_blob_tooltip(blob))
            item.setData(Qt.UserRole + 1, self._scan_geometry_from_blob(blob))
            self.union_list_widget.addItem(item)

    def _scan_geometry_from_blob(self, blob):
        """Return a blob in the same real-space format used by union boxes."""
        center_x, center_y = blob['center']
        size = blob['box_size']
        size_x = size * self.app_state.microns_per_pixel_x
        size_y = size * self.app_state.microns_per_pixel_y
        return {
            'real_center': (
                (center_x * self.app_state.microns_per_pixel_x) + self.app_state.true_origin_x,
                (center_y * self.app_state.microns_per_pixel_y) + self.app_state.true_origin_y,
            ),
            'real_size': (size_x, size_y),
            'real_area': size_x * size_y,
        }

    def send_to_queue_server(self):
        fine_scan_rows = self._staged_fine_scan_rows()
        if not fine_scan_rows:
            QMessageBox.information(self, "Queue Preview", "Add one or more union boxes to the list first.")
            return

        default_config = self.detection_config_path
        config_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select fine-scan configuration",
            str(default_config),
            "JSON files (*.json)",
        )
        if not config_path:
            return

        import pandas as pd
        try:
            mode, requests = build_fine_scan_requests(config_path, pd.DataFrame(fine_scan_rows))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            QMessageBox.warning(self, "Queue Preview", f"Could not build scan requests:\n{error}")
            return

        if mode != "real":
            QMessageBox.warning(
                self,
                "Queue submission disabled",
                "This configuration is not in real mode, so no plans were sent.\n\n"
                "For the local HXN QueueServer simulator, use a configuration with "
                "execution_params.mode set to 'real'.",
            )
            return

        first = requests[0]
        summary = (
            "The following plans will be sent to the configured local QueueServer.\n"
            "The queue will start immediately after all plans are accepted.\n\n"
            "Request builder:\n"
            "build_fine_scan_requests(config_path, fine_scans_table)\n\n"
            f"Planned scans: {len(requests)}\n\n"
            f"First scan: {first['label']}\n"
            "Box row passed to the function:\n"
            f"{json.dumps(first['input_row'])}\n\n"
            f"Center: X={first['center']['x']:.2f} µm, Y={first['center']['y']:.2f} µm\n"
            f"Requested size: {first['requested_size']['x']:.2f} × {first['requested_size']['y']:.2f} µm\n"
            f"Padded size: {first['padded_size']['x']:.2f} × {first['padded_size']['y']:.2f} µm\n"
            f"Motors: {first['motors']['x']} / {first['motors']['y']}\n"
            f"Points: {first['points']['x']} × {first['points']['y']}\n"
            f"Dwell: {first['dwell']:.4f} s\n"
            f"Plan: {first['plan_name']}\n\n"
            "Submit all staged boxes to QueueServer?"
        )
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Confirm Queue Submission")
        dialog.setIcon(QMessageBox.Warning)
        dialog.setText(summary)
        dialog.setDetailedText(json.dumps(requests, indent=2))
        dialog.setStandardButtons(QMessageBox.Cancel | QMessageBox.Ok)
        dialog.button(QMessageBox.Ok).setText("Submit and Start Queue")
        dialog.button(QMessageBox.Cancel).setText("Cancel")
        if dialog.exec_() != QMessageBox.Ok:
            return

        try:
            result = submit_fine_scan_requests(requests)
        except Exception as error:
            QMessageBox.critical(
                self,
                "Queue submission failed",
                f"No further plans were sent.\n\n{error}",
            )
            return

        submitted_labels = "\n".join(f"• {item['label']}" for item in result["submitted"])
        status = result["status"]
        QMessageBox.information(
            self,
            "Plans submitted",
            f"QueueServer accepted and started {len(result['submitted'])} plan(s):\n"
            f"{submitted_labels}\n\n"
            f"Queue state: {status.get('manager_state', 'unknown')}\n"
            "For the local simulator, receipt files are written in hxn-qserver-sim/output/.",
        )

    def _staged_fine_scan_rows(self):
        """Convert selected GUI union boxes into fine-scan table rows in microns."""
        rows = []
        for index in range(self.queue_server_list.count()):
            item = self.queue_server_list.item(index)
            union = item.data(Qt.UserRole + 1)
            if not union:
                continue
            real_center = union['real_center']
            real_size = union['real_size']
            rows.append({
                'label': item.text(),
                'cx': real_center[0],
                'cy': real_center[1],
                'num_x': real_size[0],
                'num_y': real_size[1],
            })
        return rows

    def clear_queue_server_list(self):
        self.queue_server_list.clear()

class UpdateProgressEvent(QEvent):
    EVENT_TYPE = QEvent.Type(QEvent.User + 1)
    def __init__(self, value, generation):
        super().__init__(self.EVENT_TYPE)
        self.value = value
        self.generation = generation

class ComputationFinishedEvent(QEvent):
    EVENT_TYPE = QEvent.Type(QEvent.User + 2)
    def __init__(self, generation):
        super().__init__(self.EVENT_TYPE)
        self.generation = generation


def create_automap_widget(parent=None, app_state=None):
    """Create AutoMap for embedding in another Qt application's layout.

    No QApplication is created here; this is safe to call from the shared HXN
    GUI, which already owns the process-wide application instance.
    """
    return MainWindow(app_state or AppState(), parent=parent)
