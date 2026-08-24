"""Coarse Scan Sender widget — load a JSON config and queue the initial coarse scan.

Pure Qt + stdlib (json, pathlib) at import time.
bluesky_queueserver_api is lazy-imported only when the user clicks Preview or Send.
"""

import json
from pathlib import Path

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFileDialog, QGroupBox, QTextEdit, QMessageBox,
    QScrollArea, QSizePolicy,
)
from qtpy.QtCore import Qt
from qtpy.QtGui import QFont


class CoarseScanWidget(QWidget):
    """Self-contained widget: load a JSON config, preview and send the initial coarse scan.

    Importable without any heavy dependencies (bluesky, numpy, etc.).
    Heavy imports happen lazily inside _on_preview_clicked / _on_send_clicked.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._json_path = None
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
        # Outer layout: scroll area fills the widget so buttons are always visible.
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

        header = QLabel("<b>Coarse / Mosaic Scan Sender</b>")
        header.setStyleSheet("font-size: 14px; padding: 5px;")
        layout.addWidget(header)

        layout.addWidget(self._build_json_section())
        layout.addWidget(self._build_preview_section())
        layout.addLayout(self._build_button_row())
        layout.addStretch()

        scroll.setWidget(container)
        outer.addWidget(scroll)

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

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

        self._json_summary = QLabel("")
        self._json_summary.setWordWrap(True)
        self._json_summary.setStyleSheet("color: #444; font-size: 11px; padding: 2px 4px;")
        outer.addWidget(self._json_summary)

        return group

    def _build_preview_section(self):
        group = QGroupBox("Plan Preview")
        outer = QVBoxLayout(group)
        outer.setContentsMargins(10, 22, 10, 10)

        self._preview_text = QTextEdit()
        self._preview_text.setReadOnly(True)
        mono = QFont("Monospace")
        mono.setStyleHint(QFont.TypeWriter)
        self._preview_text.setFont(mono)
        self._preview_text.setStyleSheet("background: #f8f8f8; font-size: 11px;")
        self._preview_text.setPlaceholderText(
            "Load a JSON config to preview its contents here."
        )
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
        self._preview_btn.clicked.connect(self._on_preview_clicked)
        row.addWidget(self._preview_btn)

        self._send_btn = QPushButton("Send Coarse Scan")
        self._send_btn.setStyleSheet(
            "padding: 8px 14px; font-weight: bold; background: #2a6ebb; color: white;"
        )
        self._send_btn.clicked.connect(self._on_send_clicked)
        row.addWidget(self._send_btn)

        row.addStretch()
        return row

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------

    def _browse_json(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select JSON Config", "", "JSON files (*.json)"
        )
        if not path:
            return
        self._load_json(path)

    def _load_json(self, path):
        try:
            with open(path) as f:
                params = json.load(f)
        except (OSError, json.JSONDecodeError) as err:
            QMessageBox.critical(self, "Error Loading JSON", str(err))
            return

        self._json_path = path
        self._json_path_edit.setText(path)

        sp = params.get("scan_params", {})
        ep = params.get("execution_params", {})
        mode = ep.get("mode", "?")
        mot1 = sp.get("mot1", "?")
        mot2 = sp.get("mot2", "?")
        m1s = sp.get("mot1_s", "?")
        m1e = sp.get("mot1_e", "?")
        m2s = sp.get("mot2_s", "?")
        m2e = sp.get("mot2_e", "?")
        step = sp.get("step_size", sp.get("step_size_coarse", "?"))
        dwell = sp.get("exp_t", sp.get("exp_t_coarse", "?"))
        label = sp.get("label", "?")
        proceed = ep.get("proceed_with_fine_scan", False)
        self._json_summary.setText(
            f"Label: {label}   Mode: {mode}   Proceed to fine scan: {proceed}\n"
            f"{mot1}: [{m1s}, {m1e}]   {mot2}: [{m2s}, {m2e}]   "
            f"Step: {step} µm   Dwell: {dwell} s"
        )

        self._preview_text.setPlainText(json.dumps(params, indent=2))

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _validate_inputs(self):
        if not self._json_path:
            QMessageBox.warning(self, "Missing Input", "Please load a JSON config file first.")
            return False
        return True

    def _build_requests(self):
        from automap_hxn.queue import build_coarse_scan_requests
        return build_coarse_scan_requests(self._json_path)

    def _on_preview_json_clicked(self):
        if not self._validate_inputs():
            return
        try:
            mode, requests = self._build_requests()
        except ImportError as err:
            QMessageBox.critical(self, "Dependency Missing",
                f"Could not import queue module: {err}\n\n"
                "Install bluesky-queueserver-api into this environment.")
            return
        except Exception as err:
            QMessageBox.critical(self, "Preview Failed", str(err))
            return
        self._preview_text.setPlainText(json.dumps(requests, indent=2))

    def _on_preview_clicked(self):
        if not self._validate_inputs():
            return

        try:
            mode, requests = self._build_requests()
        except ImportError as err:
            QMessageBox.critical(
                self,
                "Dependency Missing",
                f"Could not import queue module: {err}\n\n"
                "Install bluesky-queueserver-api into this environment.",
            )
            return
        except Exception as err:
            QMessageBox.critical(self, "Preview Failed", str(err))
            return

        lines = [f"Mode: {mode.upper()}  —  {len(requests)} plan(s)\n"]
        for i, req in enumerate(requests, 1):
            lines.append(f"  [{i}] {req['label']}")
            lines.append(f"      plan: {req['plan_name']}")
            if "center" in req:
                c = req["center"]
                lines.append(
                    "      center: " + "  ".join(f"{k}={v:.3f}" for k, v in c.items())
                )
            if "points" in req:
                p = req["points"]
                lines.append(f"      points: {p.get('x', '?')} × {p.get('y', '?')}")
            lines.append("")
        self._preview_text.setPlainText("\n".join(lines))

    def _on_send_clicked(self):
        if not self._validate_inputs():
            return

        try:
            mode, requests = self._build_requests()
        except ImportError as err:
            QMessageBox.critical(
                self,
                "Dependency Missing",
                f"Could not import queue module: {err}\n\n"
                "Install bluesky-queueserver-api into this environment.",
            )
            return
        except Exception as err:
            QMessageBox.critical(self, "Build Failed", str(err))
            return

        if mode != "real":
            self._preview_text.setPlainText(
                f"[{mode.upper()}] Dry run — no plans were submitted to the queue.\n\n"
                + "\n".join(f"  {r['label']}" for r in requests)
            )
            QMessageBox.information(
                self,
                "Dry Run",
                f"Mode is '{mode}' — plans were not submitted.\n"
                "Set execution_params.mode to 'real' in your JSON to submit.",
            )
            return

        scan_label = requests[-1].get("label", "coarse scan")
        confirm = QMessageBox.question(
            self,
            "Confirm Send",
            f"Submit {len(requests)} plan(s) for '{scan_label}' to the queue?\nMode: {mode.upper()}",
            QMessageBox.Yes | QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return

        try:
            from automap_hxn.queue import submit_queue_requests
            result = submit_queue_requests(requests)
        except Exception as err:
            QMessageBox.critical(self, "Submission Failed", str(err))
            return

        submitted = result.get("submitted", [])
        QMessageBox.information(
            self,
            "Sent",
            f"Successfully submitted {len(submitted)} plan(s) to the queue.",
        )
        self._preview_text.setPlainText(
            f"Submitted {len(submitted)} plan(s).\n\n"
            + "\n".join(
                f"  {s['label']}  uid={s.get('item_uid', '?')}" for s in submitted
            )
        )
