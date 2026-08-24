"""Standalone JSON Maker widget for building AutoMap initial-scan configs.

Pure Qt + stdlib — no cv2, numpy, bluesky, or model library imports.
Safe to embed in HXN GUI without pulling in any heavy dependencies.
"""

import os
import json
from pathlib import Path

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QCheckBox, QDoubleSpinBox, QSpinBox, QLineEdit,
    QScrollArea, QGroupBox, QFormLayout, QTabWidget, QMessageBox,
    QFileDialog,
)
from qtpy.QtCore import Qt


# ---------------------------------------------------------------------------
# Constants — mirror the reference configs in configs/initial_scan_*.json
# ---------------------------------------------------------------------------

JSON_MAKER_UNIVERSAL = "universal"

DETECTION_METHOD_BLOCKS = {
    "simple": {
        # max_threshold is derived automatically at create time: it matches
        # the largest gui_threshold_values entry.
        "max_area": 50000,
        "threshold_step": 5,
        "filter_by_color": False,
        "filter_by_circularity": False,
        "gui_threshold_values": [0, 10, 100],
        "gui_min_area_values": [100, 200, 300],
    },
    "hough": {
        "max_radius": 40,
        "dp": 1,
        "min_dist": 20,
        "param1": 50,
        "param2": 30,
    },
    "watershed": {
        "min_distance": 10,
        "threshold_abs": 0.3,
        "gui_threshold_values": [0, 10, 100],
        "gui_min_area_values": [100, 200, 300],
    },
    "cellpose": {
        "diameter": 30,
        "model_type": "cpsam",
        "gpu": False,
        "flow_threshold": 1.0,
        "cellprob_threshold": 0.0,
        "gui_cellprob_threshold_values": [0.0],
        "channels": [0, 0],
        "resample": False,
        "tile_overlap": 0.05,
        "bsize": 256,
        "min_diameter": 20,
        "max_diameter": 80,
        "min_size": 10,
        "gui_min_size_values": [10],
    },
    "connected_components": {
        "connectivity": 8,
        "gui_threshold_values": [0, 10, 100],
        "gui_min_area_values": [100, 200, 300],
    },
    "contours": {
        "mode": "external",
        "method": "simple",
    },
    "yolo": {
        "model": "yolo26s-seg.pt",
        "conf": 0.0001,
        "imgsz": 640,
        "tile_size": 640,
        "tile_overlap": 128,
        "max_box_fraction": 0.15,
    },
    "stardist": {
        "model_name": "2D_versatile_fluo",
        "prob_thresh": 0.5,
        "nms_thresh": 0.4,
        "min_size": 0,
    },
}

# Methods selectable in the JSON Maker dropdown; must stay in sync with the
# supported set enforced by MainWindow._load_detection_config.
JSON_MAKER_METHODS = [
    "simple", "cellpose", "connected_components", "watershed", "yolo", "stardist",
]

# Display names for the dropdowns; the JSON always uses the plain method key.
JSON_MAKER_METHOD_LABELS = {"simple": "simple (OpenCV)"}

# Per-method starting thresholds for segmentation_params. YOLO and StarDist
# ignore these in the GUI, so they stay at zero.
JSON_MAKER_SEGMENTATION_THRESHOLDS = {
    "simple": (100, 200),
    "cellpose": (25, 9),
    "connected_components": (100, 200),
    "watershed": (100, 200),
    "yolo": (0, 0),
    "stardist": (0, 0),
}

# Common (non-detection) sections of an initial-scan config. The JSON Maker
# form is generated from this schema; widget types are inferred from each
# default's Python type (bool -> checkbox, int/float -> spinbox, list -> JSON
# text field, str/None -> text field).
JSON_MAKER_COMMON_SECTIONS = {
    "execution_params": {"mode": "real", "proceed_with_fine_scan": True},
    "scan_params": {
        "label": "universal_test", "scan_id": None,
        "dets": "dets_fast", "det_names": ["fs", "eiger2", "xspress3"],
        "mot1": "zpssx", "mot1_s": -12.5, "mot1_e": 12.5,
        "mot2": "zpssy", "mot2_s": -12.5, "mot2_e": 12.5,
        "exp_t": 0.005, "step_size": 0.25, "zp_move_flag": 0,
        "smar_move_flag": 0, "roi_positions_file": None,
    },
    "fine_scan_params": {
        "step_size_fine": 0.1, "exp_t_fine": 0.005, "fine_scan_pad_ratio": 0.3,
    },
    "export_params": {
        "elem_list": [["Ca", "Fe", "S"]], "export_norm": "sclr1_ch4",
        "data_wd": "/nsls2/data/hxn/legacy/users/2026Q1/synaps_demo_2_2026Q1",
        "tiled_reconstructions": "tst/sandbox/eugene/synaps/reconstructions",
        "tiled_segmentations": "tst/sandbox/eugene/synaps/segmentations",
        "tiled_uri": None,
    },
    "calibration_params": {
        "microns_per_pixel_x": 1.5, "microns_per_pixel_y": 1.5,
        "true_origin_x": 10, "true_origin_y": 10,
    },
    "analysis_params": {"analysis": True, "execute": True},
    "segmentation_params": {
        "min_threshold_intensity": 100, "min_threshold_area": 200,
        "overlap_thresh": 0.5,
    },
    "morphology_params": {
        "normalize_kernel_size": [3, 3], "dilate_iterations": 2,
        "blur_kernel": [3, 3],
    },
    "mosaic_params": {
        "xlen": 100,
        "ylen": 100,
        "overlap_per": 0,
        "step_size": 250,
        "dwell": 0.01,
        "mll": False,
        "remote_seg": False,
        "followup_fine_scan": False,
        "ref_scan_id": None,
    },
}

JSON_MAKER_MODES = ["real", "simulation", "offline", "analysis-only"]

# Known model choices for the JSON Maker's model dropdowns, keyed by
# (detection method, field name). Fields listed here render as a dropdown
# with an "Other..." entry that reveals a free-text input.
JSON_MAKER_MODEL_OPTIONS = {
    ("cellpose", "model_type"): ["cpsam", "cyto3", "cyto2", "cyto", "nuclei"],
    ("yolo", "model"): [
        "yolo26n-seg.pt", "yolo26s-seg.pt", "yolo26m-seg.pt",
        "yolo26l-seg.pt", "yolo26x-seg.pt",
    ],
    ("stardist", "model_name"): [
        "2D_versatile_fluo", "2D_versatile_he", "2D_paper_dsb2018",
    ],
}


# ---------------------------------------------------------------------------
# Helper widgets
# ---------------------------------------------------------------------------

class ModelSelectField(QWidget):
    """Dropdown of known model names with an 'Other...' free-text fallback."""

    OTHER = "Other..."

    def __init__(self, options, default, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.combo = QComboBox()
        self.combo.addItems(list(options))
        self.combo.addItem(self.OTHER)
        self.custom_input = QLineEdit()
        self.custom_input.setPlaceholderText("custom model name")
        layout.addWidget(self.combo, 1)
        layout.addWidget(self.custom_input, 1)
        if default in options:
            self.combo.setCurrentText(default)
        else:
            self.combo.setCurrentText(self.OTHER)
            self.custom_input.setText(str(default))
        self.custom_input.setVisible(self.combo.currentText() == self.OTHER)
        self.combo.currentTextChanged.connect(
            lambda text: self.custom_input.setVisible(text == self.OTHER)
        )

    def value(self):
        if self.combo.currentText() == self.OTHER:
            return self.custom_input.text().strip()
        return self.combo.currentText()


class SweepValuesField(QWidget):
    """Editor for gui_*_values sweep lists: explicit custom values or an
    auto-generated min/max/step range. Either way value() returns the plain
    list that the analysis sliders and compute sweep consume."""

    CUSTOM = "Custom set"
    RANGE = "Range"

    def __init__(self, default, parent=None):
        super().__init__(parent)
        self._is_float = any(isinstance(v, float) for v in default)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems([self.CUSTOM, self.RANGE])
        layout.addWidget(self.mode_combo)

        # Custom mode: one JSON list field, e.g. [100] or [50, 80, 100]
        self.custom_input = QLineEdit()
        self.custom_input.setText(json.dumps(default))
        self.custom_input.setPlaceholderText("e.g. [100] or [50, 80, 100]")
        layout.addWidget(self.custom_input, 1)

        # Range mode: min / max / step spinboxes
        def _spin(value):
            spin = QDoubleSpinBox() if self._is_float else QSpinBox()
            if self._is_float:
                spin.setDecimals(3)
                spin.setRange(-1e6, 1e6)
            else:
                spin.setRange(-1_000_000, 1_000_000)
            spin.setValue(value)
            return spin

        first = default[0] if default else 0
        last = default[-1] if default else 0
        self.min_spin = _spin(first)
        self.max_spin = _spin(last)
        self.step_spin = _spin(1.0 if self._is_float else 10)
        self.range_widgets = QWidget()
        range_layout = QHBoxLayout(self.range_widgets)
        range_layout.setContentsMargins(0, 0, 0, 0)
        range_layout.setSpacing(14)
        for text, spin in (("min", self.min_spin), ("max", self.max_spin), ("step", self.step_spin)):
            pair = QHBoxLayout()
            pair.setContentsMargins(0, 0, 0, 0)
            pair.setSpacing(4)
            pair.addWidget(QLabel(text))
            pair.addWidget(spin, 1)
            range_layout.addLayout(pair, 1)
        layout.addWidget(self.range_widgets, 1)

        self.range_widgets.setVisible(False)
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)

    def _on_mode_changed(self, mode):
        self.custom_input.setVisible(mode == self.CUSTOM)
        self.range_widgets.setVisible(mode == self.RANGE)

    def value(self):
        if self.mode_combo.currentText() == self.RANGE:
            start, stop = self.min_spin.value(), self.max_spin.value()
            step = self.step_spin.value()
            if step <= 0:
                raise ValueError("Range step must be greater than zero.")
            if stop < start:
                raise ValueError("Range max must not be less than min.")
            values, current = [], start
            while current <= stop + (1e-9 if self._is_float else 0):
                values.append(round(current, 6) if self._is_float else int(round(current)))
                current += step
            return values
        text = self.custom_input.text().strip()
        try:
            values = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError(f"Custom values must be a JSON list, e.g. [100] ({error}).")
        if not isinstance(values, list) or not values:
            raise ValueError("Custom values must be a non-empty JSON list, e.g. [100].")
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            raise ValueError("Custom values must contain only numbers.")
        return sorted(values)


# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------

class JSONMakerWidget(QWidget):
    """Self-contained JSON configuration builder widget.

    Importable without any heavy dependencies (cv2, numpy, torch, etc.).
    Embed in HXN GUI or run inside the full AutoMap window — same code.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        self.setStyleSheet(
            "QGroupBox {"
            "  font-weight: bold;"
            "  margin-top: 18px;"
            "  padding-top: 12px;"
            "}"
            "QGroupBox::title {"
            "  subcontrol-origin: margin;"
            "  subcontrol-position: top left;"
            "  left: 8px;"
            "  top: 2px;"
            "}"
        )
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        # === HEADER ===
        header_label = QLabel("<b>JSON Configuration Builder</b>")
        header_label.setStyleSheet("font-size: 14px; padding: 5px;")
        layout.addWidget(header_label)

        # === MODEL SELECTION ===
        model_group = QGroupBox("Detection Method Configuration")
        model_layout = QVBoxLayout(model_group)
        model_layout.setSpacing(8)
        model_layout.setContentsMargins(10, 24, 10, 10)

        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Configuration Type:"))
        self.json_maker_method_combo = QComboBox()
        self.json_maker_method_combo.addItem("Universal (all methods)", JSON_MAKER_UNIVERSAL)
        for method in JSON_MAKER_METHODS:
            self.json_maker_method_combo.addItem(
                JSON_MAKER_METHOD_LABELS.get(method, method), method
            )
        self.json_maker_method_combo.setCurrentIndex(0)
        self.json_maker_method_combo.currentIndexChanged.connect(self._on_json_maker_method_changed)
        method_row.addWidget(self.json_maker_method_combo, 1)
        method_row.addStretch()
        model_layout.addLayout(method_row)

        active_row = QHBoxLayout()
        self.json_maker_active_label = QLabel("Default Active Model:")
        active_row.addWidget(self.json_maker_active_label)
        self.json_maker_active_combo = QComboBox()
        for method in JSON_MAKER_METHODS:
            self.json_maker_active_combo.addItem(
                JSON_MAKER_METHOD_LABELS.get(method, method), method
            )
        self.json_maker_active_combo.currentIndexChanged.connect(self._on_json_maker_method_changed)
        active_row.addWidget(self.json_maker_active_combo, 1)
        active_row.addStretch()
        model_layout.addLayout(active_row)

        self.json_maker_hint = QLabel(
            "ℹ️ Universal creates a config with all detection methods. "
            "Model-specific creates a config for only one method."
        )
        self.json_maker_hint.setWordWrap(True)
        self.json_maker_hint.setStyleSheet("color: #666; padding: 5px; font-size: 11px;")
        model_layout.addWidget(self.json_maker_hint)

        layout.addWidget(model_group)

        # === TABBED CONFIGURATION SECTIONS ===
        self.json_maker_fields = {}
        config_tabs = QTabWidget()
        config_tabs.setStyleSheet("QTabWidget::pane { border: 1px solid #ccc; }")

        config_tabs.addTab(self._create_json_maker_core_tab(), "Model Settings")
        config_tabs.addTab(self._create_json_maker_scan_export_tab(), "Scan & Export")
        config_tabs.addTab(self._create_json_maker_advanced_tab(), "Advanced")

        layout.addWidget(config_tabs, 1)

        # === BOTTOM BUTTON ===
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        reset_btn = QPushButton("↺ Reset to Defaults")
        reset_btn.setStyleSheet("padding: 8px 16px;")
        reset_btn.clicked.connect(self._reset_to_defaults)
        button_layout.addWidget(reset_btn)
        create_btn = QPushButton("💾 Create JSON Configuration")
        create_btn.setStyleSheet("padding: 8px 16px; font-weight: bold;")
        create_btn.clicked.connect(self.on_create_json_clicked)
        button_layout.addWidget(create_btn)
        button_layout.addStretch()
        layout.addLayout(button_layout)

        self._on_json_maker_method_changed()

    def _create_json_maker_core_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(12)
        layout.setContentsMargins(10, 10, 10, 10)

        seg_group = QGroupBox("segmentation_params")
        seg_form = QFormLayout(seg_group)
        seg_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        seg_form.setContentsMargins(10, 22, 10, 10)
        self.json_maker_fields["segmentation_params"] = {}
        derived_seg_keys = {
            "blob_detection_method", "min_threshold_intensity", "min_threshold_area",
        }
        for key, default in JSON_MAKER_COMMON_SECTIONS["segmentation_params"].items():
            if key not in derived_seg_keys:
                field_widget = self._make_json_maker_field(key, default)
                seg_form.addRow(key, field_widget)
                self.json_maker_fields["segmentation_params"][key] = (field_widget, default)
        seg_note = QLabel(
            "min_threshold_intensity / min_threshold_area are set automatically "
            "from the smallest values in the model's slider lists below."
        )
        seg_note.setWordWrap(True)
        seg_note.setStyleSheet("color: #666; font-size: 11px;")
        seg_form.addRow(seg_note)
        layout.addWidget(seg_group)

        self.json_maker_method_groups = {}
        self.json_maker_method_fields = {}
        for method in JSON_MAKER_METHODS:
            group = QGroupBox(f"detection_methods: {method}")
            group_form = QFormLayout(group)
            group_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
            group_form.setContentsMargins(10, 22, 10, 10)
            self.json_maker_method_fields[method] = {}
            for key, default in DETECTION_METHOD_BLOCKS[method].items():
                field_widget = self._make_json_maker_field(
                    key, default, options=JSON_MAKER_MODEL_OPTIONS.get((method, key))
                )
                group_form.addRow(key, field_widget)
                self.json_maker_method_fields[method][key] = (field_widget, default)
            self.json_maker_method_groups[method] = group
            layout.addWidget(group)

        layout.addStretch()
        scroll.setWidget(container)

        wrapper = QWidget()
        wrapper_layout = QVBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        wrapper_layout.addWidget(scroll)
        return wrapper

    def _create_json_maker_scan_export_tab(self):
        """Scan Parameters + Export & Calibration in one scrollable tab."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(12)
        layout.setContentsMargins(10, 10, 10, 10)

        scan_group = QGroupBox("scan_params")
        scan_form = QFormLayout(scan_group)
        scan_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        scan_form.setContentsMargins(10, 22, 10, 10)
        self.json_maker_fields["scan_params"] = {}
        for key, default in JSON_MAKER_COMMON_SECTIONS["scan_params"].items():
            field_widget = self._make_json_maker_field(key, default)
            scan_form.addRow(key, field_widget)
            self.json_maker_fields["scan_params"][key] = (field_widget, default)
        layout.addWidget(scan_group)

        mosaic_group = QGroupBox("mosaic_params")
        mosaic_form = QFormLayout(mosaic_group)
        mosaic_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        mosaic_form.setContentsMargins(10, 22, 10, 10)
        self.json_maker_fields["mosaic_params"] = {}
        for key, default in JSON_MAKER_COMMON_SECTIONS["mosaic_params"].items():
            field_widget = self._make_json_maker_field(key, default)
            mosaic_form.addRow(key, field_widget)
            self.json_maker_fields["mosaic_params"][key] = (field_widget, default)
        layout.addWidget(mosaic_group)

        fine_group = QGroupBox("fine_scan_params")
        fine_form = QFormLayout(fine_group)
        fine_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        fine_form.setContentsMargins(10, 22, 10, 10)
        self.json_maker_fields["fine_scan_params"] = {}
        for key, default in JSON_MAKER_COMMON_SECTIONS["fine_scan_params"].items():
            field_widget = self._make_json_maker_field(key, default)
            fine_form.addRow(key, field_widget)
            self.json_maker_fields["fine_scan_params"][key] = (field_widget, default)
        layout.addWidget(fine_group)

        export_group = QGroupBox("export_params")
        export_form = QFormLayout(export_group)
        export_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        export_form.setContentsMargins(10, 22, 10, 10)
        self.json_maker_fields["export_params"] = {}
        for key, default in JSON_MAKER_COMMON_SECTIONS["export_params"].items():
            field_widget = self._make_json_maker_field(key, default)
            export_form.addRow(key, field_widget)
            self.json_maker_fields["export_params"][key] = (field_widget, default)
        layout.addWidget(export_group)

        calib_group = QGroupBox("calibration_params")
        calib_form = QFormLayout(calib_group)
        calib_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        calib_form.setContentsMargins(10, 22, 10, 10)
        self.json_maker_fields["calibration_params"] = {}
        for key, default in JSON_MAKER_COMMON_SECTIONS["calibration_params"].items():
            field_widget = self._make_json_maker_field(key, default)
            calib_form.addRow(key, field_widget)
            self.json_maker_fields["calibration_params"][key] = (field_widget, default)
        layout.addWidget(calib_group)

        layout.addStretch()
        scroll.setWidget(container)

        wrapper = QWidget()
        wrapper_layout = QVBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        wrapper_layout.addWidget(scroll)
        return wrapper

    def _create_json_maker_advanced_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(12)
        layout.setContentsMargins(10, 10, 10, 10)

        exec_group = QGroupBox("execution_params")
        exec_form = QFormLayout(exec_group)
        exec_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        exec_form.setContentsMargins(10, 22, 10, 10)
        self.json_maker_fields["execution_params"] = {}
        for key, default in JSON_MAKER_COMMON_SECTIONS["execution_params"].items():
            field_widget = self._make_json_maker_field(key, default)
            exec_form.addRow(key, field_widget)
            self.json_maker_fields["execution_params"][key] = (field_widget, default)
        layout.addWidget(exec_group)

        morph_group = QGroupBox("morphology_params")
        morph_form = QFormLayout(morph_group)
        morph_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        morph_form.setContentsMargins(10, 22, 10, 10)
        self.json_maker_fields["morphology_params"] = {}
        for key, default in JSON_MAKER_COMMON_SECTIONS["morphology_params"].items():
            field_widget = self._make_json_maker_field(key, default)
            morph_form.addRow(key, field_widget)
            self.json_maker_fields["morphology_params"][key] = (field_widget, default)
        layout.addWidget(morph_group)

        analysis_group = QGroupBox("analysis_params")
        analysis_form = QFormLayout(analysis_group)
        analysis_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        analysis_form.setContentsMargins(10, 22, 10, 10)
        self.json_maker_fields["analysis_params"] = {}
        for key, default in JSON_MAKER_COMMON_SECTIONS["analysis_params"].items():
            field_widget = self._make_json_maker_field(key, default)
            analysis_form.addRow(key, field_widget)
            self.json_maker_fields["analysis_params"][key] = (field_widget, default)
        layout.addWidget(analysis_group)

        layout.addStretch()
        return widget

    def _make_json_maker_field(self, key, default, options=None):
        """Build an input widget for one config value based on its default's type."""
        if options:
            return ModelSelectField(options, default)
        if key in {
            "gui_threshold_values", "gui_min_area_values",
            "gui_cellprob_threshold_values", "gui_min_size_values",
        }:
            return SweepValuesField(default)
        if key == "mode":
            combo = QComboBox()
            combo.addItems(JSON_MAKER_MODES)
            combo.setCurrentText(default)
            return combo
        if isinstance(default, bool):
            checkbox = QCheckBox()
            checkbox.setChecked(default)
            return checkbox
        if isinstance(default, int):
            spin = QSpinBox()
            spin.setRange(-1_000_000_000, 1_000_000_000)
            spin.setValue(default)
            return spin
        if isinstance(default, float):
            spin = QDoubleSpinBox()
            spin.setDecimals(6)
            spin.setRange(-1e9, 1e9)
            spin.setValue(default)
            return spin
        # str, list, and None all edit as text; lists round-trip through JSON.
        line = QLineEdit()
        if isinstance(default, list):
            line.setText(json.dumps(default))
        elif default is not None:
            line.setText(str(default))
        else:
            line.setPlaceholderText("null")
        return line

    @staticmethod
    def _read_json_maker_field(name, field_widget, default):
        """Read one config value back from its input widget."""
        if isinstance(field_widget, ModelSelectField):
            value = field_widget.value()
            if not value:
                raise ValueError(
                    f"'{name}' is set to '{ModelSelectField.OTHER}' but no custom model name was entered."
                )
            return value
        if isinstance(field_widget, SweepValuesField):
            try:
                return field_widget.value()
            except ValueError as error:
                raise ValueError(f"'{name}': {error}") from error
        if isinstance(field_widget, QComboBox):
            return field_widget.currentText()
        if isinstance(field_widget, QCheckBox):
            return field_widget.isChecked()
        if isinstance(field_widget, (QSpinBox, QDoubleSpinBox)):
            return field_widget.value()
        text = field_widget.text().strip()
        if isinstance(default, list):
            try:
                value = json.loads(text)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"'{name}' must be valid JSON, e.g. {json.dumps(default)} ({error})."
                ) from error
            if not isinstance(value, list):
                raise ValueError(f"'{name}' must be a JSON list, e.g. {json.dumps(default)}.")
            return value
        if default is None and not text:
            return None
        return text

    def _json_maker_selection(self):
        """Return (dropdown selection, active blob_detection_method)."""
        selection = self.json_maker_method_combo.currentData()
        if selection == JSON_MAKER_UNIVERSAL:
            return selection, self.json_maker_active_combo.currentData()
        return selection, selection

    def _on_json_maker_method_changed(self, *_args):
        """Sync the form with the dropdown: method panel, thresholds, and label."""
        selection, active_method = self._json_maker_selection()
        is_universal = selection == JSON_MAKER_UNIVERSAL
        self.json_maker_active_label.setVisible(is_universal)
        self.json_maker_active_combo.setVisible(is_universal)

        for method, group in self.json_maker_method_groups.items():
            group.setVisible(is_universal or method == active_method)
            if method == active_method:
                group.setTitle(f"detection_methods: {method}  (active)")
            else:
                group.setTitle(f"detection_methods: {method}")

        label = "universal_test" if is_universal else f"{active_method}_test"
        self.json_maker_fields["scan_params"]["label"][0].setText(label)

    def _build_json_maker_config(self, selection, active_method):
        """Assemble a config dict from the form for the chosen method selection."""
        config = {}
        for section, fields in self.json_maker_fields.items():
            config[section] = {
                key: self._read_json_maker_field(f"{section}.{key}", field_widget, default)
                for key, (field_widget, default) in fields.items()
            }

        if "segmentation_params" not in config:
            config["segmentation_params"] = {}
        config["segmentation_params"]["blob_detection_method"] = active_method

        if selection == JSON_MAKER_UNIVERSAL:
            method_names = list(JSON_MAKER_METHODS) + ["hough", "contours"]
        else:
            method_names = [selection]
        detection_methods = {}
        for method in method_names:
            if method in self.json_maker_method_fields:
                detection_methods[method] = {
                    key: self._read_json_maker_field(f"{method}.{key}", field_widget, default)
                    for key, (field_widget, default) in self.json_maker_method_fields[method].items()
                }
            else:
                detection_methods[method] = dict(DETECTION_METHOD_BLOCKS[method])
        config["detection_methods"] = detection_methods

        if "simple" in detection_methods:
            simple_block = detection_methods["simple"]
            threshold_values = simple_block.get("gui_threshold_values", [100])
            simple_block["max_threshold"] = max(threshold_values)
            area_values = simple_block.get("gui_min_area_values", [200])
            max_area = simple_block.get("max_area", 50000)
            oversized = [a for a in area_values if a > max_area]
            if oversized:
                raise ValueError(
                    f"simple.gui_min_area_values contains {oversized}, which "
                    f"exceed max_area={max_area}. Raise max_area or remove "
                    "those values — blobs larger than max_area are rejected."
                )

        seg = config["segmentation_params"]
        active_block = detection_methods.get(active_method, {})
        if active_method in {"simple", "watershed", "connected_components"}:
            seg["min_threshold_intensity"] = min(active_block.get("gui_threshold_values", [100]))
            seg["min_threshold_area"] = min(active_block.get("gui_min_area_values", [200]))
        elif active_method == "cellpose":
            seg["min_threshold_intensity"] = int(round(
                min(active_block.get("gui_cellprob_threshold_values", [0.0])) * 10
            ))
            seg["min_threshold_area"] = min(active_block.get("gui_min_size_values", [10]))
        else:
            seg["min_threshold_intensity"] = 0
            seg["min_threshold_area"] = 0
        return config

    def _reset_to_defaults(self):
        """Reset every form field back to its schema default."""
        # Common section fields
        for section, fields in self.json_maker_fields.items():
            for key, (widget, default) in fields.items():
                self._reset_field(widget, default)

        # Per-method detection fields
        for method, fields in self.json_maker_method_fields.items():
            for key, (widget, default) in fields.items():
                self._reset_field(widget, default)

        # Re-sync the method dropdowns / visibility
        self._on_json_maker_method_changed()

    @staticmethod
    def _reset_field(widget, default):
        """Set one widget back to its default value."""
        if isinstance(widget, ModelSelectField):
            options = [widget.combo.itemText(i) for i in range(widget.combo.count())
                       if widget.combo.itemText(i) != ModelSelectField.OTHER]
            if str(default) in options:
                widget.combo.setCurrentText(str(default))
            else:
                widget.combo.setCurrentText(ModelSelectField.OTHER)
                widget.custom_input.setText(str(default))
        elif isinstance(widget, SweepValuesField):
            widget.mode_combo.setCurrentText(SweepValuesField.CUSTOM)
            widget.custom_input.setText(json.dumps(default))
        elif isinstance(widget, QComboBox):
            widget.setCurrentText(str(default))
        elif isinstance(widget, QCheckBox):
            widget.setChecked(bool(default))
        elif isinstance(widget, QSpinBox):
            widget.setValue(int(default))
        elif isinstance(widget, QDoubleSpinBox):
            widget.setValue(float(default))
        elif isinstance(widget, QLineEdit):
            if isinstance(default, list):
                widget.setText(json.dumps(default))
            elif default is not None:
                widget.setText(str(default))
            else:
                widget.clear()

    def on_create_json_clicked(self):
        """Write a new initial-scan JSON from the form values."""
        selection, active_method = self._json_maker_selection()
        try:
            config = self._build_json_maker_config(selection, active_method)
        except ValueError as error:
            QMessageBox.warning(self, "Invalid Value", str(error))
            return

        configs_dir = Path(__file__).resolve().parents[2] / "configs"
        default_name = f"initial_scan_{selection}.json"
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save JSON Configuration",
            str(configs_dir / default_name),
            "JSON files (*.json)",
        )
        if not output_path:
            return
        if not output_path.lower().endswith(".json"):
            output_path += ".json"

        try:
            with open(output_path, "w") as stream:
                json.dump(config, stream, indent=2)
                stream.write("\n")
        except OSError as error:
            QMessageBox.critical(self, "Error Saving File", f"Could not write JSON: {error}")
            return
        QMessageBox.information(
            self,
            "JSON Created",
            f"Saved {os.path.basename(output_path)} with "
            f"blob_detection_method='{active_method}'.",
        )
