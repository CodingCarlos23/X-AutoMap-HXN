"""Coarse / Mosaic Scan Sender widget.

Reads ALL scan geometry from the JSON config (mosaic_params block) and submits
mosaic_overlap_scan_auto_relative in a background QThread. Configuration is done
in the JSON Maker — this widget is the executor only.

Pure Qt + stdlib at import time. Heavy deps (bluesky) are lazy-imported inside
the thread worker only.
"""

import json
import math
from pathlib import Path

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFileDialog, QGroupBox, QTextEdit, QMessageBox,
    QScrollArea, QSizePolicy,
)
from qtpy.QtCore import Qt, QThread, Signal
from qtpy.QtGui import QFont
import datetime


# ---------------------------------------------------------------------------
# Background thread
# ---------------------------------------------------------------------------

class MosaicScanThread(QThread):
    """Runs mosaic_overlap_scan_auto_relative without blocking the UI."""

    finished = Signal(str)
    error = Signal(str)

    def __init__(self, json_path, params, tiled_uri=None, parent=None):
        super().__init__(parent)
        self._json_path = json_path
        self._params = params  # kwargs forwarded to mosaic_overlap_scan_auto_relative
        self._tiled_uri = tiled_uri

    def run(self):
        try:
            from automap_hxn.workflows import mosaic_overlap_scan_auto_relative

            params = dict(self._params)

            # Auto-create Tiled client from URI if remote_seg is enabled
            if params.get("remote_seg") and self._tiled_uri:
                try:
                    from tiled.client import from_uri
                    params["tiled_client"] = from_uri(self._tiled_uri)
                except Exception as err:
                    self.error.emit(
                        f"Could not connect to Tiled at '{self._tiled_uri}':\n{err}\n\n"
                        "Check tiled_uri in export_params or disable remote_seg."
                    )
                    return

            mosaic_overlap_scan_auto_relative(
                beamline_params=self._json_path,
                initial_scan_path=self._json_path,
                **params,
            )
            self.finished.emit("Mosaic scan completed successfully.")
        except ImportError as err:
            self.error.emit(
                f"Could not import workflows module: {err}\n\n"
                "Install bluesky-queueserver-api and its dependencies."
            )
        except Exception as err:
            self.error.emit(str(err))


# ---------------------------------------------------------------------------
# Helper: tile count + time estimate (pure Python, no numpy needed)
# ---------------------------------------------------------------------------

def _calc_tile_info(mot1_s, mot1_e, xlen, ylen, overlap_per, step_size, dwell):
    """Return (x_tiles, y_tiles, est_minutes) or (0, 0, 0) on bad inputs."""
    scan_range = abs(mot1_e - mot1_s)
    if scan_range <= 0 or step_size <= 0 or dwell <= 0:
        return 0, 0, 0.0
    grid_step = scan_range * (1 - overlap_per * 0.01)
    if grid_step <= 0:
        return 0, 0, 0.0
    start = grid_step / 2
    x_tiles = max(0, math.floor((xlen - start) / grid_step) + 1) if xlen >= start else 0
    y_tiles = max(0, math.floor((ylen - start) / grid_step) + 1) if ylen >= start else 0
    num_steps_fly = round(25_000 / step_size)
    fly_time = (num_steps_fly ** 2) * dwell * 2
    est_minutes = (fly_time * x_tiles * y_tiles) / 60
    return x_tiles, y_tiles, est_minutes


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

class CoarseScanWidget(QWidget):
    """Load a JSON config and send the mosaic scan — no editable parameters.

    All scan geometry lives in the JSON's mosaic_params block (configured in
    the JSON Maker). This widget reads it, shows a summary, and fires the scan.

    Emits log_message(str) for each key event — connect to a log widget to
    build a persistent history across scans.
    """

    log_message = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._json_path = None
        self._json_params = {}
        self._scan_thread = None
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

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
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QScrollArea.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        header = QLabel("<b>Mosaic Scan Sender</b>")
        header.setStyleSheet("font-size: 14px; padding: 5px;")
        layout.addWidget(header)

        layout.addWidget(self._build_json_section())
        layout.addWidget(self._build_summary_section())
        layout.addWidget(self._build_preview_section())
        layout.addLayout(self._build_button_row())
        layout.addStretch()

        scroll.setWidget(container)
        outer.addWidget(scroll)

    def _build_json_section(self):
        group = QGroupBox("JSON Configuration")
        outer = QVBoxLayout(group)
        outer.setContentsMargins(10, 22, 10, 10)
        outer.setSpacing(6)

        path_row = QHBoxLayout()
        self._json_path_edit = QLineEdit()
        self._json_path_edit.setPlaceholderText("Select an initial_scan_*.json config file…")
        self._json_path_edit.setReadOnly(True)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_json)
        path_row.addWidget(self._json_path_edit, 1)
        path_row.addWidget(browse_btn)
        outer.addLayout(path_row)

        return group

    def _build_summary_section(self):
        group = QGroupBox("Scan Summary")
        outer = QVBoxLayout(group)
        outer.setContentsMargins(10, 22, 10, 10)

        self._summary_label = QLabel("Load a JSON config to see the scan summary.")
        self._summary_label.setWordWrap(True)
        self._summary_label.setStyleSheet("font-size: 11px; padding: 2px 4px;")
        outer.addWidget(self._summary_label)

        self._tile_label = QLabel("")
        self._tile_label.setWordWrap(True)
        self._tile_label.setStyleSheet("font-size: 12px; font-weight: bold; padding: 4px;")
        outer.addWidget(self._tile_label)

        return group

    def _build_preview_section(self):
        group = QGroupBox("Preview")
        outer = QVBoxLayout(group)
        outer.setContentsMargins(10, 22, 10, 10)

        self._preview_text = QTextEdit()
        self._preview_text.setReadOnly(True)
        mono = QFont("Monospace")
        mono.setStyleHint(QFont.TypeWriter)
        self._preview_text.setFont(mono)
        self._preview_text.setStyleSheet("background: #f8f8f8; font-size: 11px;")
        self._preview_text.setPlaceholderText("Load a JSON config to preview its contents here.")
        self._preview_text.setMinimumHeight(120)
        self._preview_text.setMaximumHeight(220)
        self._preview_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        outer.addWidget(self._preview_text)

        return group

    def _build_button_row(self):
        row = QHBoxLayout()
        row.addStretch()

        self._preview_json_btn = QPushButton("Preview JSON")
        self._preview_json_btn.setStyleSheet("padding: 8px 14px;")
        self._preview_json_btn.clicked.connect(self._on_preview_json_clicked)
        row.addWidget(self._preview_json_btn)

        self._preview_btn = QPushButton("Preview Plans")
        self._preview_btn.setStyleSheet("padding: 8px 14px;")
        self._preview_btn.clicked.connect(self._on_preview_plans_clicked)
        row.addWidget(self._preview_btn)

        self._send_btn = QPushButton("Send Mosaic Scan")
        self._send_btn.setStyleSheet(
            "padding: 8px 14px; font-weight: bold; background: #2a6ebb; color: white;"
        )
        self._send_btn.clicked.connect(self._on_send_clicked)
        row.addWidget(self._send_btn)

        row.addStretch()
        return row

    # ------------------------------------------------------------------
    # JSON loading
    # ------------------------------------------------------------------

    def _browse_json(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select JSON Config", "", "JSON files (*.json)"
        )
        if path:
            self._load_json(path)

    def _load_json(self, path):
        try:
            with open(path) as f:
                params = json.load(f)
        except (OSError, json.JSONDecodeError) as err:
            QMessageBox.critical(self, "Error Loading JSON", str(err))
            return

        self._json_path = path
        self._json_params = params
        self._json_path_edit.setText(path)
        self._update_summary()
        self._preview_text.setPlainText(json.dumps(params, indent=2))

    def _update_summary(self):
        if not self._json_params:
            return
        sp = self._json_params.get("scan_params", {})
        ep = self._json_params.get("execution_params", {})
        mp = self._json_params.get("mosaic_params", {})

        mode = ep.get("mode", "?")
        mot1 = sp.get("mot1", "?")
        mot2 = sp.get("mot2", "?")
        label = sp.get("label", "?")
        xlen = mp.get("xlen", "?")
        ylen = mp.get("ylen", "?")
        overlap = mp.get("overlap_per", 0)
        step = mp.get("step_size", "?")
        dwell = mp.get("dwell", "?")
        optics = "MLL" if mp.get("mll", False) else "ZP"
        remote = mp.get("remote_seg", True)
        fine = mp.get("followup_fine_scan", False)

        self._summary_label.setText(
            f"Label: {label}   Mode: {mode}   Motors: {mot1} / {mot2}   Optics: {optics}\n"
            f"Area: {xlen} × {ylen} µm   Overlap: {overlap}%   "
            f"Step: {step} nm   Dwell: {dwell} s\n"
            f"Remote seg: {remote}   Follow-up fine scan: {fine}"
        )

        # Tile estimate
        mot1_s = float(sp.get("mot1_s", 0))
        mot1_e = float(sp.get("mot1_e", 0))
        try:
            x_tiles, y_tiles, est_min = _calc_tile_info(
                mot1_s, mot1_e,
                float(xlen), float(ylen),
                float(overlap), float(step), float(dwell),
            )
            if x_tiles == 0 or y_tiles == 0:
                self._tile_label.setText("⚠️  Cannot calculate tiles — check scan range and area values in JSON.")
            else:
                unit = "min" if est_min < 60 else "hr"
                display_time = est_min if est_min < 60 else est_min / 60
                self._tile_label.setText(
                    f"Tiles: {x_tiles} × {y_tiles} = {x_tiles * y_tiles} total   "
                    f"Est. time: {display_time:.1f} {unit}"
                )
        except (TypeError, ValueError):
            self._tile_label.setText("⚠️  Could not compute tile estimate — check mosaic_params in JSON.")

    # ------------------------------------------------------------------
    # Collect params from JSON (no widget overrides)
    # ------------------------------------------------------------------

    def _collect_params(self):
        mp = self._json_params.get("mosaic_params", {})
        ref = mp.get("ref_scan_id")
        return {
            "xlen": mp.get("xlen", 100),
            "ylen": mp.get("ylen", 100),
            "overlap_per": mp.get("overlap_per", 0),
            "step_size": mp.get("step_size", 250),
            "dwell": mp.get("dwell", 0.01),
            "mll": mp.get("mll", False),
            "remote_seg": mp.get("remote_seg", True),
            "followup_fine_scan": mp.get("followup_fine_scan", False),
            "ref_scan_id": ref if ref else None,
            "dets": None,
            "tiled_client": None,
        }

    def _validate(self):
        if not self._json_path:
            QMessageBox.warning(self, "Missing Input", "Please load a JSON config file first.")
            return False
        return True

    # ------------------------------------------------------------------
    # Buttons
    # ------------------------------------------------------------------

    def _on_preview_json_clicked(self):
        if not self._validate():
            return
        payload = {
            "beamline_params": self._json_path,
            "initial_scan_path": self._json_path,
            **self._collect_params(),
        }
        self._preview_text.setPlainText(json.dumps(payload, indent=2, default=str))

    def _on_preview_plans_clicked(self):
        if not self._validate():
            return
        sp = self._json_params.get("scan_params", {})
        ep = self._json_params.get("execution_params", {})
        mp = self._json_params.get("mosaic_params", {})
        mot1_s = float(sp.get("mot1_s", 0))
        mot1_e = float(sp.get("mot1_e", 0))
        try:
            x_tiles, y_tiles, est_min = _calc_tile_info(
                mot1_s, mot1_e,
                float(mp.get("xlen", 100)),
                float(mp.get("ylen", 100)),
                float(mp.get("overlap_per", 0)),
                float(mp.get("step_size", 250)),
                float(mp.get("dwell", 0.01)),
            )
        except (TypeError, ValueError):
            x_tiles = y_tiles = 0
            est_min = 0.0

        scan_range = abs(mot1_e - mot1_s)
        grid_step = scan_range * (1 - float(mp.get("overlap_per", 0)) * 0.01)
        mode = ep.get("mode", "?")
        optics = "MLL (dsx/dsy)" if mp.get("mll", False) else "ZP (smarx/smary)"
        unit = "min" if est_min < 60 else "hr"
        display_time = est_min if est_min < 60 else est_min / 60

        lines = [
            f"Mode: {mode.upper()}",
            f"Optics: {optics}",
            f"Total area: {mp.get('xlen')} × {mp.get('ylen')} µm",
            f"Grid step: {grid_step:.2f} µm  ({mp.get('overlap_per', 0)}% overlap)",
            f"Tiles: {x_tiles} × {y_tiles} = {x_tiles * y_tiles} total",
            f"Per-tile scan: {sp.get('mot1','?')} / {sp.get('mot2','?')}  "
            f"[{mot1_s:.2f} → {mot1_e:.2f}]",
            f"Step size: {mp.get('step_size')} nm   Dwell: {mp.get('dwell')} s",
            f"Est. total time: {display_time:.1f} {unit}",
            f"Remote seg: {mp.get('remote_seg', True)}   "
            f"Follow-up fine scan: {mp.get('followup_fine_scan', False)}",
        ]
        if mp.get("ref_scan_id"):
            lines.append(f"ref_scan_id: {mp.get('ref_scan_id')}")
        self._preview_text.setPlainText("\n".join(lines))

    def _log(self, message):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_message.emit(f"[{ts}] {message}")

    def _on_send_clicked(self):
        if not self._validate():
            return
        if self._scan_thread and self._scan_thread.isRunning():
            QMessageBox.information(self, "Scan Running", "A mosaic scan is already in progress.")
            return

        mp = self._json_params.get("mosaic_params", {})
        sp = self._json_params.get("scan_params", {})
        ep = self._json_params.get("execution_params", {})
        try:
            x_tiles, y_tiles, est_min = _calc_tile_info(
                float(sp.get("mot1_s", 0)), float(sp.get("mot1_e", 0)),
                float(mp.get("xlen", 100)), float(mp.get("ylen", 100)),
                float(mp.get("overlap_per", 0)), float(mp.get("step_size", 250)),
                float(mp.get("dwell", 0.01)),
            )
        except (TypeError, ValueError):
            x_tiles = y_tiles = 0
            est_min = 0.0

        mode = ep.get("mode", "?")
        confirm = QMessageBox.question(
            self,
            "Confirm Mosaic Scan",
            f"Send mosaic scan to queue?\n\n"
            f"Mode: {mode.upper()}\n"
            f"Tiles: {x_tiles} × {y_tiles} = {x_tiles * y_tiles} total\n"
            f"Est. time: {est_min:.1f} min",
            QMessageBox.Yes | QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return

        tiled_uri = self._json_params.get("export_params", {}).get("tiled_uri") or None
        self._scan_thread = MosaicScanThread(
            self._json_path, self._collect_params(), tiled_uri=tiled_uri, parent=self
        )
        self._scan_thread.finished.connect(self._on_scan_finished)
        self._scan_thread.error.connect(self._on_scan_error)
        self._scan_thread.start()

        self._send_btn.setEnabled(False)
        self._send_btn.setText("Scanning…")
        self._preview_text.setPlainText(
            f"Mosaic scan started — {x_tiles * y_tiles} tile(s) queued.\n"
            "Check the terminal for tile-by-tile progress.\n"
            "The Send button will re-enable when the scan completes or fails."
        )
        mp = self._json_params.get("mosaic_params", {})
        self._log(
            f"Mosaic scan started — {x_tiles}×{y_tiles} tiles "
            f"({mp.get('xlen')}×{mp.get('ylen')} µm, mode={mode})"
        )

    def _on_scan_finished(self, message):
        self._send_btn.setEnabled(True)
        self._send_btn.setText("Send Mosaic Scan")
        self._preview_text.setPlainText(f"✓ {message}")
        self._log(f"✓ {message}")
        QMessageBox.information(self, "Scan Complete", message)

    def _on_scan_error(self, message):
        self._send_btn.setEnabled(True)
        self._send_btn.setText("Send Mosaic Scan")
        self._preview_text.setPlainText(f"✗ Error:\n{message}")
        self._log(f"✗ Error: {message.splitlines()[0]}")
        QMessageBox.critical(self, "Scan Failed", message)
