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
from .blobs.processing import find_union_blobs_by_category

from .json_maker import JSONMakerWidget
from .coarse_scan_widget import CoarseScanWidget
from .llm_json_maker import LLMJsonMakerWidget
from .live_scan_viewer import LiveScanViewerWidget


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
            center_match = re.search(r"Center: \((\d+\.?\d*), (\d+\.?\d*)\)", text)
            length_match = re.search(r"Length: (\d+\.?\d*)\s*px", text)

            if center_match and length_match:
                x, y, length = int(float(center_match.group(1))), int(float(center_match.group(2))), int(float(length_match.group(1)))
                radius = length / 2 + 5
                circle = QGraphicsEllipseItem(x - radius, y - radius, radius * 2, radius * 2)
                circle.setPen(QPen(QColor("yellow"), 2, Qt.SolidLine))
                circle.setZValue(100)
                self.scene().addItem(circle)
                self.highlight_items.append(circle)


class ChannelCheckBox(QCheckBox):
    """Tri-state checkbox with colored outline, checkmark, and X mark.

    States:
    - Unchecked: Element not shown
    - Checked: Show boxes containing this element
    - PartiallyChecked (X): Exclude boxes overlapping with this element
    """

    def __init__(self, text, color, parent=None):
        super().__init__(text, parent)
        self._channel_color = QColor(color)
        self.setTristate(True)
        self.setStyleSheet(
            "QCheckBox::indicator {"
            " width: 14px; height: 14px;"
            f" border: 2px solid {color};"
            " background: #ffffff;"
            "}"
        )

    def paintEvent(self, event):
        super().paintEvent(event)

        state = self.checkState()
        if state == Qt.Unchecked:
            return

        option = QStyleOptionButton()
        self.initStyleOption(option)
        indicator = self.style().subElementRect(
            QStyle.SE_CheckBoxIndicator, option, self
        )
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(self._channel_color, 2))

        if state == Qt.Checked:
            # Draw checkmark
            painter.drawLine(indicator.left() + 3, indicator.center().y(),
                             indicator.center().x() - 1, indicator.bottom() - 3)
            painter.drawLine(indicator.center().x() - 1, indicator.bottom() - 3,
                             indicator.right() - 2, indicator.top() + 3)
        elif state == Qt.PartiallyChecked:
            # Draw X mark
            painter.drawLine(indicator.left() + 3, indicator.top() + 3,
                             indicator.right() - 3, indicator.bottom() - 3)
            painter.drawLine(indicator.right() - 3, indicator.top() + 3,
                             indicator.left() + 3, indicator.bottom() - 3)

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
        self.union_categories = {
            '3-element': {},
            '2-element': {},
            '1-element': {},
        }
        self.blob_union_membership = {}  # Maps blob_id -> set of colors it overlaps with

        # Confidence scoring parameters - ADJUST THESE after testing
        self.scoring_weights = {
            '1-element': {
                'brightness': 0.5,    # Weight for intensity score
                'size': 0.3,          # Weight for size score
                'compactness': 0.2,   # Weight for shape quality
            },
            '2-element': {
                'brightness': 0.4,    # Both elements should be bright
                'size': 0.2,          # Reasonable size
                'overlap': 0.25,      # Good spatial overlap
                'balance': 0.15,      # Balanced intensity ratio
            },
            '3-element': {
                'brightness': 0.35,   # All three elements bright
                'size': 0.15,         # Reasonable size
                'overlap': 0.3,       # Tight co-localization
                'balance': 0.2,       # Balanced across all three
            }
        }
        # Size preferences (in pixels) - ADJUST based on your typical particles
        self.optimal_size_range = (50, 500)  # Min and max "good" size in pixels

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
        root_layout.setSpacing(0)

        # Persistent top bar — always visible across all tabs
        _top_bar = QHBoxLayout()
        _top_bar.setContentsMargins(6, 4, 6, 4)
        _top_bar.addStretch(1)
        _hxn_lbl = QLabel("<b>HXN</b>")
        _hxn_lbl.setStyleSheet(
            "font-size: 16px; font-weight: bold; padding: 4px 12px;"
            " border: 2px solid palette(mid); border-radius: 4px;"
        )
        _hxn_lbl.setAlignment(Qt.AlignCenter)
        _top_bar.addWidget(_hxn_lbl)
        root_layout.addLayout(_top_bar)

        self.tab_widget = QTabWidget()
        root_layout.addWidget(self.tab_widget)

        self.automap_tab = QWidget()
        self.outer_layout = QVBoxLayout(self.automap_tab)

        # AutoMap tab header
        _header_row = QHBoxLayout()
        _title_lbl = QLabel("<b>AutoMap</b>")
        _title_lbl.setStyleSheet("font-size: 14px; padding: 5px;")
        _header_row.addWidget(_title_lbl)
        self.outer_layout.addLayout(_header_row)

        self.tab_widget.addTab(self.automap_tab, "AutoMap")
        self.setup_widget = self._create_setup_screen()
        self.outer_layout.addWidget(self.setup_widget)

        self.json_maker_tab = self._create_json_maker_tab()
        self.tab_widget.addTab(self.json_maker_tab, "JSON Maker")

        self.coarse_scan_tab = CoarseScanWidget()
        self.tab_widget.addTab(self.coarse_scan_tab, "Area Scan")

        self.live_scan_tab = LiveScanViewerWidget()
        self.tab_widget.addTab(self.live_scan_tab, "Live View")

        self.coarse_scan_tab.scan_started.connect(self._on_area_scan_started)

        self.llm_json_maker_tab = LLMJsonMakerWidget()
        self.tab_widget.addTab(self.llm_json_maker_tab, "LLM JSON Maker")


    def _on_area_scan_started(self, json_path: str):
        """Switch to Live View, load the same config, and start watching."""
        live_view_idx = self.tab_widget.indexOf(self.live_scan_tab)
        self.tab_widget.setCurrentIndex(live_view_idx)
        if self.live_scan_tab.load_config_from_path(json_path):
            self.live_scan_tab.start_watching()

    def _create_json_maker_tab(self):
        return JSONMakerWidget()

    def _create_setup_screen(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        # --- Config section (mandatory) ---
        config_group = QGroupBox("Configuration")
        config_group.setStyleSheet("QGroupBox { font-weight: bold; margin-top: 18px; padding-top: 12px; }")
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
        elem_group.setStyleSheet("QGroupBox { font-weight: bold; margin-top: 18px; padding-top: 12px; }")
        elem_layout = QVBoxLayout(elem_group)
        elem_layout.setContentsMargins(10, 20, 10, 10)
        self.elem_mapping_label = QLabel("Load a config to see expected elements.")
        self.elem_mapping_label.setStyleSheet("color: #888; font-size: 12px;")
        self.elem_mapping_label.setWordWrap(True)
        elem_layout.addWidget(self.elem_mapping_label)
        layout.addWidget(elem_group)

        # --- Directory + file list ---
        dir_group = QGroupBox("TIFF Directory")
        dir_group.setStyleSheet("QGroupBox { font-weight: bold; margin-top: 18px; padding-top: 12px; }")
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

        ep = config.get("export_params", {})
        use_intensity_filter = ep.get("use_intensity_filter", False)

        if use_intensity_filter:
            groups = ep.get("groups", [])
            has_elements = groups and all(g.get("elements") for g in groups)
            if not has_elements:
                QMessageBox.warning(self, "Invalid Configuration",
                                    f"{config_path.name}: use_intensity_filter is true but "
                                    f"'groups' with elements is missing or empty.")
                return
            unique_elements = list(dict.fromkeys(
                e["element"] for g in groups for e in g.get("elements", [])
            ))
        else:
            elem_list_nested = ep.get("elem_list", [])
            if not elem_list_nested:
                QMessageBox.warning(self, "Invalid Configuration",
                                    f"{config_path.name}: use_intensity_filter is false but "
                                    f"'elem_list' is missing or empty in export_params.")
                return
            if isinstance(elem_list_nested[0], str):
                elem_list_nested = [elem_list_nested]
            unique_elements = list(dict.fromkeys(
                e for group in elem_list_nested for e in group
            ))

        if not unique_elements:
            QMessageBox.warning(self, "Invalid Configuration",
                                f"{config_path.name}: no elements found in config.")
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
        default_dir = getattr(self, "_setup_config", {}).get("export_params", {}).get("data_wd", "") or str(Path.home())
        directory = QFileDialog.getExistingDirectory(self, "Select Directory", default_dir)
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
                checkbox.setTristate(False)  # File selection is binary
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

        config_name = getattr(self, "detection_config_path", None)
        config_name = Path(config_name).name if config_name else "No config loaded"
        self.active_config_label = QLabel(f"Config: <b>{config_name}</b>")
        self.active_config_label.setStyleSheet("font-size: 11px; color: #555; padding: 2px 0;")
        self.active_config_label.setWordWrap(True)
        controls_layout.addWidget(self.active_config_label)

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

        self.queue_server_list.setSelectionMode(QListWidget.MultiSelection)
        self.queue_server_list.itemSelectionChanged.connect(self.on_queue_item_selected)

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
        self.union_checkbox.setTristate(False)  # Union checkbox is binary on/off
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
                cache_dir = Path(self.app_state.selected_directory) / "automap_cache"
                cache_dir.mkdir(exist_ok=True)
                with open(cache_dir / "precomputed_blobs.pkl", "wb") as f:
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
                img_norm, img_orig, min_thresh, min_area, color, file_name,
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

        method_params = {
            key: value for key, value in self.detection_settings.items()
            if key not in {"min_threshold", "min_area", "gui_threshold_values", "gui_min_area_values"}
        }
        return detect_blobs(
            img_norm, img_orig, min_thresh, min_area, color, file_name,
            method="simple", **method_params,
        )

    def update_boxes(self):
        if self.graphics_view is None:
            return

        # Separate checked (included) and partially checked (excluded) colors
        included_colors = set()
        excluded_colors = set()
        for c, cb in self.checkboxes.items():
            if c == 'union':
                continue
            state = cb.checkState()
            if state == Qt.Checked:
                included_colors.add(c)
            elif state == Qt.PartiallyChecked:
                excluded_colors.add(c)

        blobs = self.get_current_blobs()
        self.graphics_view.update_blobs(blobs, included_colors)
        self.redraw_boxes(blobs, included_colors, excluded_colors)

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

    def redraw_boxes(self, blobs, included_colors, excluded_colors):
        for item in self.blob_items:
            self.graphics_view.scene().removeItem(item)
        self.blob_items.clear()

        for blob in blobs:
            # First filter: blob's primary color must be in included colors
            if blob['color'] not in included_colors:
                continue

            # Get the set of all colors this blob contains (for union membership)
            blob_key = (blob['color'], tuple(blob['center']))
            if hasattr(self, 'blob_union_membership') and blob_key in self.blob_union_membership:
                blob_colors = self.blob_union_membership[blob_key]
            else:
                blob_colors = {blob['color']}

            # Second filter: when multiple colors included, blob must contain AT LEAST all of them
            if len(included_colors) > 1:
                if not included_colors.issubset(blob_colors):
                    continue

            # Third filter: blob must NOT overlap with any excluded colors
            if excluded_colors and blob_colors.intersection(excluded_colors):
                continue

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

                # Color code by category
                category = ub.get('category', 'unknown')
                if category == '3-element':
                    color = QColor("#e74c3c")  # Red - exact match
                    width = 3
                elif category == '2-element':
                    color = QColor("#f39c12")  # Orange - partial match
                    width = 2
                elif category == '1-element':
                    color = QColor("#95a5a6")  # Gray - solo
                    width = 2
                else:
                    color = QColor("white")  # Fallback
                    width = 2

                pen = QPen(color, width, Qt.DashLine)
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
        # Get color to element mapping
        color_to_element = self._get_color_to_element_map()

        # Convert colors to element symbols
        colors = ub.get('colors', [])
        element_symbols = [color_to_element.get(c, c) for c in colors]
        elements_display = ', '.join(element_symbols)

        tooltip = (
            f"<b>Union Box #{idx}</b><br>"
            f"Category: {ub.get('category', 'unknown')}<br>"
            f"Elements: {elements_display}<br>"
            f"<b>Confidence: {ub.get('confidence', 0):.1f}%</b><br><br>"
            f"Center: ({ub['center'][0]}, {ub['center'][1]})<br>"
            f"Length: {ub['length']} px<br>"
            f"Area: {ub['area']} px²<br><br>"
            f"Real Center: ({ub['real_center'][0]:.2f}, {ub['real_center'][1]:.2f}) µm<br>"
            f"Real Size: {ub['real_size'][0]:.2f} × {ub['real_size'][1]:.2f} µm<br>"
            f"Real Area: {ub['real_area']:.2f} µm²<br>"
        )

        # Add intensity information if source blobs available
        source_blobs = ub.get('source_blobs', [])
        if source_blobs:
            tooltip += "<br><b>Intensities:</b><br>"
            for blob in source_blobs:
                color = blob.get('color', 'unknown')
                element = color_to_element.get(color, color)
                mean_int = blob.get('mean_intensity', 0)
                max_int = blob.get('max_intensity', 0)
                tooltip += f"{element}: mean={mean_int:.3f}, max={max_int:.3f}<br>"

        return tooltip

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

    def on_queue_item_selected(self):
        """Handle selection in queue_server_list (second text box) - highlight on map."""
        selected_items = self.queue_server_list.selectedItems()
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
                cache_dir = Path(self.app_state.selected_directory) / "automap_cache"
                cache_dir.mkdir(exist_ok=True)
                serializable_dict = make_json_serializable(self.graphics_view.union_dict)
                with open(cache_dir / "union_blobs.json", "w") as f:
                    json.dump(serializable_dict, f, indent=4)

            self.update_boxes()

        self.graphics_view.mousePressEvent = on_press
        self.graphics_view.mouseReleaseEvent = on_release

    def _get_color_to_element_map(self):
        """Create mapping from color to element symbol."""
        color_to_element = {}

        # Common element symbols (1-2 characters, capitalized)
        common_elements = {
            'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
            'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
            'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
            'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
            'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
            'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
            'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
            'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
            'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
            'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf',
        }

        for i, color in enumerate(self.app_state.element_colors):
            if i < len(self.app_state.file_names):
                # Extract element symbol from filename (e.g., "detsum_Ca.tif" -> "Ca")
                filename = self.app_state.file_names[i]
                # Remove file extension
                name = filename.replace('.tif', '').replace('.tiff', '').replace('.TIF', '').replace('.TIFF', '')

                # Split by common delimiters and find element symbol
                parts = name.replace('_', ' ').replace('-', ' ').split()
                element = None

                # Look for a known element symbol in the parts
                for part in parts:
                    if part in common_elements:
                        element = part
                        break

                # If no known element found, try to extract capitalized 1-2 char strings
                if element is None:
                    for part in parts:
                        if len(part) <= 2 and part[0].isupper():
                            element = part
                            break

                # Final fallback: take the last non-numeric part
                if element is None:
                    for part in reversed(parts):
                        if not part.isdigit() and part:
                            element = part
                            break

                color_to_element[color] = element if element else color
            else:
                color_to_element[color] = color  # Fallback to color name
        return color_to_element

    def _score_unions_in_category(self, unions, category):
        """
        Score all unions in a category using percentile-based ranking.
        This ensures scores are relative to other particles in the same category.
        Modifies unions in-place to add 'confidence' field.
        """
        if not unions:
            return

        weights = self.scoring_weights[category]

        # Extract raw feature values for all unions
        all_brightness = []
        all_sizes = []
        all_compactness = []  # For 1-element
        all_overlap = []      # For 2/3-element
        all_balance = []      # For 2/3-element

        for union in unions:
            source_blobs = union.get('source_blobs', [])
            intensities = [blob.get('mean_intensity', 0) for blob in source_blobs]

            # Brightness: mean intensity across all blobs
            brightness = np.mean(intensities) if intensities else 0
            all_brightness.append(brightness)

            # Size: area
            all_sizes.append(union.get('area', 0))

            # Category-specific features
            if category == '1-element':
                # Compactness
                area = union.get('area', 1)
                length = union.get('length', 1)
                ideal_area = 0.785 * (length ** 2)
                compactness = min(area / ideal_area, 1.0) if ideal_area > 0 else 0
                all_compactness.append(compactness)

            elif category == '2-element':
                # Overlap quality
                if len(source_blobs) >= 2:
                    centers = [blob['center'] for blob in source_blobs[:2]]
                    radii = [blob.get('radius', 10) for blob in source_blobs[:2]]
                    dist = np.sqrt((centers[0][0] - centers[1][0])**2 + (centers[0][1] - centers[1][1])**2)
                    max_dist = sum(radii)
                    overlap = max(0, 1.0 - dist / max_dist) if max_dist > 0 else 0
                else:
                    overlap = 0.5
                all_overlap.append(overlap)

                # Balance
                if len(intensities) >= 2:
                    balance = min(intensities[0], intensities[1]) / max(intensities[0], intensities[1], 1e-6)
                else:
                    balance = 0.5
                all_balance.append(balance)

            elif category == '3-element':
                # Overlap quality (tight co-localization)
                if len(source_blobs) >= 3:
                    centers = [blob['center'] for blob in source_blobs[:3]]
                    dists = []
                    for i in range(len(centers)):
                        for j in range(i+1, len(centers)):
                            d = np.sqrt((centers[i][0] - centers[j][0])**2 + (centers[i][1] - centers[j][1])**2)
                            dists.append(d)
                    avg_dist = np.mean(dists)
                    union_size = union.get('length', 100)
                    overlap = max(0, 1.0 - avg_dist / union_size) if union_size > 0 else 0
                else:
                    overlap = 0.5
                all_overlap.append(overlap)

                # Balance
                if len(intensities) >= 3:
                    min_int = min(intensities[:3])
                    max_int = max(intensities[:3])
                    balance = min_int / max(max_int, 1e-6)
                else:
                    balance = 0.5
                all_balance.append(balance)

        # Calculate percentile ranks (0-1) for each feature
        # Higher values get higher percentiles
        def percentile_rank(values):
            """Return percentile rank (0-1) for each value."""
            if len(values) == 1:
                return [1.0]
            # Use numpy's rankdata equivalent
            sorted_indices = np.argsort(values)
            ranks = np.empty_like(sorted_indices)
            ranks[sorted_indices] = np.arange(len(values))
            return ranks / max(len(values) - 1, 1)

        brightness_scores = percentile_rank(all_brightness)
        size_scores = percentile_rank(all_sizes)

        # For size, we might want to penalize extremes - but for now keep simple ranking
        # Could add Gaussian-style scoring centered on optimal range later

        # Calculate final confidence scores
        for i, union in enumerate(unions):
            scores = {
                'brightness': brightness_scores[i],
                'size': size_scores[i],
            }

            if category == '1-element':
                compactness_scores = percentile_rank(all_compactness)
                scores['compactness'] = compactness_scores[i]

            elif category in ['2-element', '3-element']:
                overlap_scores = percentile_rank(all_overlap)
                balance_scores = percentile_rank(all_balance)
                scores['overlap'] = overlap_scores[i]
                scores['balance'] = balance_scores[i]

            # Weighted combination
            final_score = sum(scores[feat] * weights[feat] for feat in weights)
            union['confidence'] = final_score * 100.0  # Convert to percentage

    def union_function(self):
        blobs = self.get_current_blobs()
        blobs_by_color = {color: [] for color in self.app_state.element_colors}
        for blob in blobs:
            blobs_by_color[blob['color']].append(blob)

        # Wrap in the {color: {key: [blobs]}} format expected by find_union_blobs_by_category
        blobs_for_union = {color: {'gui': blob_list} for color, blob_list in blobs_by_color.items()}

        overlap_thresh = getattr(self, "_setup_config", {}).get("segmentation_params", {}).get("overlap_thresh", 0.5)

        # Get categorized unions
        categorized_raw = find_union_blobs_by_category(
            blobs_for_union,
            self.app_state.microns_per_pixel_x,
            self.app_state.microns_per_pixel_y,
            self.app_state.true_origin_x,
            self.app_state.true_origin_y,
            overlap_thresh=overlap_thresh,
        )

        # Store categorized unions with metadata AND build blob membership map
        self.union_categories = {
            '3-element': {},
            '2-element': {},
            '1-element': {},
        }

        # Map blob ID to the colors it overlaps with
        self.blob_union_membership = {}  # blob_id -> set of colors it's in union with

        # Build union categories and track blob membership using source_blobs
        # TWO-PASS APPROACH: First collect all unions, then score them by COLOR COMBINATION
        global_idx = 1
        all_unions_for_display = {}
        self.blob_union_membership = {}  # Reset blob membership
        unions_by_color_combo = {}  # Group by specific color combination (e.g., 'Ca+Ce')

        # FIRST PASS: Collect all union data grouped by color combination
        for category in ['3-element', '2-element', '1-element']:
            for _local_idx, u in categorized_raw[category].items():
                real_cx, real_cy = u['real_center_um']
                real_w, real_h = u['real_size_um']
                source_blobs = u.get('source_blobs', [])

                # Build union data structure
                union_data = {
                    'center': tuple(u['center']),
                    'length': u['length'],
                    'area': u['area'],
                    'real_center': (real_cx, real_cy),
                    'real_size': (real_w, real_h),
                    'real_area': real_w * real_h,
                    'colors': u.get('colors', []),
                    'element_count': u.get('element_count', 1),
                    'category': category,
                    'source_blobs': source_blobs,
                    'global_idx': global_idx,
                }

                # Create color combination key (sorted for consistency)
                colors = sorted(union_data.get('colors', []))
                colors_key = '+'.join(colors)
                union_data['colors_key'] = colors_key

                if colors_key not in unions_by_color_combo:
                    unions_by_color_combo[colors_key] = []
                unions_by_color_combo[colors_key].append(union_data)

                # Track blob membership using source_blobs from union detection
                union_colors = set(u.get('colors', []))
                for source_blob in source_blobs:
                    blob_key = (source_blob['color'], tuple(source_blob['center']))
                    if blob_key not in self.blob_union_membership:
                        self.blob_union_membership[blob_key] = set()
                    self.blob_union_membership[blob_key].update(union_colors)

                global_idx += 1

        # SECOND PASS: Score each COLOR COMBINATION separately
        for colors_key, combo_unions in unions_by_color_combo.items():
            if not combo_unions:
                continue

            # Determine category (1, 2, or 3 element) from number of colors
            num_colors = len(colors_key.split('+'))
            category = f'{num_colors}-element'

            # Score this color combination using appropriate category weights
            self._score_unions_in_category(combo_unions, category)

            # Store in display dictionaries
            for union_data in combo_unions:
                idx = union_data['global_idx']
                original_category = union_data['category']
                self.union_categories[original_category][idx] = union_data
                all_unions_for_display[idx] = union_data

        # Update union display boxes
        self.graphics_view.union_dict = all_unions_for_display
        self.union_list_widget.clear()

        # Get color to element mapping
        color_to_element = self._get_color_to_element_map()

        # Display unions grouped by category, then by color combination, sorted by confidence
        for category in ['3-element', '2-element', '1-element']:
            category_unions = [(idx, ub) for idx, ub in all_unions_for_display.items()
                              if ub.get('category') == category]

            if not category_unions:
                continue

            # Category header removed per user request - unions grouped only by element combination

            # Group by color combination within this category
            color_combos = {}
            for idx, ub in category_unions:
                colors_key = ub.get('colors_key', '+'.join(sorted(ub.get('colors', []))))
                if colors_key not in color_combos:
                    color_combos[colors_key] = []
                color_combos[colors_key].append((idx, ub))

            # Sort color combinations alphabetically
            for colors_key in sorted(color_combos.keys()):
                combo_unions = color_combos[colors_key]

                # Sort by confidence within color combination (highest first)
                combo_unions.sort(key=lambda x: x[1].get('confidence', 0), reverse=True)

                # Convert color names to element symbols for display
                color_names = colors_key.split('+')
                element_symbols = [color_to_element.get(c, c) for c in color_names]
                elements_display = '+'.join(element_symbols)

                # Add color combination sub-header
                subheader = QListWidgetItem(f"── {elements_display} ({len(combo_unions)}) ──")
                subheader.setFlags(Qt.NoItemFlags)  # Make it non-selectable
                subheader.setBackground(QColor("#34495e"))
                subheader.setForeground(QColor("#bdc3c7"))
                self.union_list_widget.addItem(subheader)

                # Add unions in this color combination
                for idx, ub in combo_unions:
                    confidence = ub.get('confidence', 0)
                    item = QListWidgetItem(f"    #{idx} - {confidence:.1f}%")
                    item.setToolTip(self._format_union_tooltip(ub, idx))
                    item.setData(Qt.UserRole + 1, ub)

                    # Color-code by confidence level
                    if confidence >= 75:
                        item.setForeground(QColor("#27ae60"))  # Green - high confidence
                    elif confidence >= 50:
                        item.setForeground(QColor("#f39c12"))  # Orange - medium confidence
                    else:
                        item.setForeground(QColor("#95a5a6"))  # Gray - low confidence

                    self.union_list_widget.addItem(item)

        # Apply fine_scan_selection filtering and populate queue_server_list
        fine_scan_selection = getattr(self, "_setup_config", {}).get("fine_scan_selection", [])
        if fine_scan_selection:
            self._apply_fine_scan_selection(fine_scan_selection, all_unions_for_display, color_to_element)

        if self.app_state.selected_directory:
            cache_dir = Path(self.app_state.selected_directory) / "automap_cache"
            cache_dir.mkdir(exist_ok=True)
            serializable_dict = make_json_serializable(self.union_categories)
            with open(cache_dir / "union_blobs_categorized.json", "w") as f:
                json.dump(serializable_dict, f, indent=4)

        self.update_boxes()

    def _apply_fine_scan_selection(self, fine_scan_selection, all_unions, color_to_element):
        """
        Apply fine_scan_selection filtering to select top N unions per relationship.
        Automatically adds selected unions to queue_server_list.
        """
        # Build reverse map: element -> color
        element_to_color = {elem: color for color, elem in color_to_element.items()}

        # Clear queue_server_list before populating
        self.queue_server_list.clear()

        total_selected = 0

        for spec in fine_scan_selection:
            elements_str = spec.get("elements", "")
            count = spec.get("count", 0)

            if count <= 0:
                continue

            # Parse element string (e.g., "Ca-Ce-P", "Ce")
            requested_elements = [e.strip() for e in elements_str.split("-")]

            # Convert elements to colors
            try:
                target_colors = sorted([element_to_color[elem] for elem in requested_elements])
            except KeyError as e:
                print(f"[GUI] Element {e} not found in loaded files - skipping {elements_str}")
                continue

            # Create colors_key for matching
            colors_key = '+'.join(target_colors)

            # Find matching unions
            matching_unions = []
            for idx, union in all_unions.items():
                union_colors = sorted(union.get('colors', []))
                union_key = '+'.join(union_colors)
                if union_key == colors_key:
                    matching_unions.append((idx, union))

            # Sort by confidence (highest first) and take top N
            matching_unions.sort(key=lambda x: x[1].get('confidence', 0), reverse=True)
            selected = matching_unions[:count]

            # Add to queue_server_list
            for idx, ub in selected:
                confidence = ub.get('confidence', 0)
                # Convert colors to element symbols for display
                color_names = colors_key.split('+')
                element_symbols = [color_to_element.get(c, c) for c in color_names]
                elements_display = '+'.join(element_symbols)

                item_text = f"#{idx} {elements_display} - {confidence:.1f}%"
                item = QListWidgetItem(item_text)
                item.setToolTip(self._format_union_tooltip(ub, idx))
                item.setData(Qt.UserRole + 1, ub)

                # Color-code by confidence
                if confidence >= 75:
                    item.setForeground(QColor("#27ae60"))
                elif confidence >= 50:
                    item.setForeground(QColor("#f39c12"))
                else:
                    item.setForeground(QColor("#95a5a6"))

                self.queue_server_list.addItem(item)
                total_selected += 1

            print(f"[FINE_SCAN_SELECTION] {elements_str}: selected {len(selected)} of {len(matching_unions)} (requested {count})")

        print(f"[FINE_SCAN_SELECTION] Total: {total_selected} unions added to fine scan queue")


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
