import base64
import json
import os
import pathlib
import subprocess
import shutil
import urllib.request

import cv2
import numpy as np

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QSizePolicy, QLineEdit, QFileDialog, QTabWidget,
    QScrollArea, QMessageBox,
)
from qtpy.QtCore import Qt, QThread, Signal
from qtpy.QtGui import (
    QTextCursor, QPixmap, QImage, QPainter, QPen, QColor, QFont,
)


MODEL = "qwen2.5vl:3b"
OLLAMA_URL = "http://localhost:11434/api/chat"

LOCATE_SYSTEM_PROMPT = (
    "You are a scientific image analysis assistant for XRF (X-ray fluorescence) scan data. "
    "Identify all distinct features, particles, clusters, or regions of interest visible in the image. "
    "Respond ONLY with valid JSON using this exact schema — no extra text outside the JSON object:\n"
    '{"features": [{"id": 1, "label": "particle cluster", "x1": 0.10, "y1": 0.20, '
    '"x2": 0.45, "y2": 0.60, "description": "bright region with high signal"}], "total_count": 1}\n'
    "IMPORTANT: All coordinates (x1, y1, x2, y2) must be FRACTIONAL values between 0.0 and 1.0, "
    "where (0,0) is the top-left corner and (1,1) is the bottom-right corner of the image. "
    "x1 < x2, y1 < y2. Include every distinct feature you can identify."
)

BOX_COLORS = [
    QColor(255, 80, 80),
    QColor(80, 160, 255),
    QColor(80, 220, 80),
    QColor(255, 200, 50),
    QColor(180, 80, 255),
    QColor(50, 220, 220),
    QColor(255, 140, 50),
]


def _load_image_pixmap(path):
    """Load any image (including 16/32-bit TIFF) and return (QPixmap, width, height)."""
    arr = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    if arr is None:
        raise ValueError(f"Could not read image: {path}")
    arr = arr.astype(np.float32)
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    arr = (arr * 255).astype(np.uint8)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    else:
        arr = np.stack([arr, arr, arr], axis=-1)
    h, w = arr.shape[:2]
    qimg = QImage(arr.data, w, h, w * 3, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg), w, h


def _draw_feature_boxes(base_pixmap, features):
    """Draw colored bounding boxes on a copy of base_pixmap. Coords are fractional 0–1."""
    result = base_pixmap.copy()
    painter = QPainter(result)
    painter.setRenderHint(QPainter.Antialiasing)
    pw, ph = result.width(), result.height()

    font = QFont()
    font.setPointSize(9)
    font.setBold(True)
    painter.setFont(font)

    for i, feat in enumerate(features):
        color = BOX_COLORS[i % len(BOX_COLORS)]
        x1 = int(max(0.0, min(1.0, float(feat.get("x1", 0)))) * pw)
        y1 = int(max(0.0, min(1.0, float(feat.get("y1", 0)))) * ph)
        x2 = int(max(0.0, min(1.0, float(feat.get("x2", 1)))) * pw)
        y2 = int(max(0.0, min(1.0, float(feat.get("y2", 1)))) * ph)

        painter.setPen(QPen(color, 2))
        painter.drawRect(x1, y1, x2 - x1, y2 - y1)

        label = f"{feat.get('id', i + 1)}: {feat.get('label', '')}"
        fx1 = max(0.0, min(1.0, float(feat.get("x1", 0))))
        fy1 = max(0.0, min(1.0, float(feat.get("y1", 0))))
        fx2 = max(0.0, min(1.0, float(feat.get("x2", 1))))
        fy2 = max(0.0, min(1.0, float(feat.get("y2", 1))))
        coords = f"({fx1:.2f},{fy1:.2f}) → ({fx2:.2f},{fy2:.2f})"
        lx, ly = x1 + 3, max(y1 + 12, 12)
        for text, dy in ((label, 0), (coords, 13)):
            painter.setPen(QPen(Qt.black, 1))
            painter.drawText(lx + 1, ly + dy + 1, text)
            painter.setPen(QPen(color, 1))
            painter.drawText(lx, ly + dy, text)

    painter.end()
    return result


# ── background threads ────────────────────────────────────────────────────────

class _PullThread(QThread):
    log = Signal(str)
    finished = Signal(bool)

    def run(self):
        try:
            proc = subprocess.Popen(
                ["ollama", "pull", MODEL],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            for line in proc.stdout:
                self.log.emit(line.rstrip())
            proc.wait()
            self.finished.emit(proc.returncode == 0)
        except Exception as e:
            self.log.emit(f"Error: {e}")
            self.finished.emit(False)


class _ChatThread(QThread):
    token = Signal(str)
    finished = Signal()
    error = Signal(str)

    def __init__(self, messages):
        super().__init__()
        self.messages = messages

    def run(self):
        payload = json.dumps({
            "model": MODEL,
            "messages": self.messages,
            "stream": True,
        }, ensure_ascii=False).encode("utf-8")
        try:
            req = urllib.request.Request(
                OLLAMA_URL,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=600) as resp:
                for line in resp:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        content = chunk.get("message", {}).get("content", "")
                        if content:
                            self.token.emit(content)
                    except json.JSONDecodeError:
                        pass
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


_VLM_MAX_DIM = 1024


def _encode_image_for_vlm(path, max_dim=_VLM_MAX_DIM):
    """Load image, downsample to max_dim on the longest side, return base64-PNG string."""
    arr = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    if arr is None:
        raise ValueError(f"Could not read image: {path}")
    arr = arr.astype(np.float32)
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    arr = (arr * 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    h, w = arr.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        arr = cv2.resize(arr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode()


class _LocateThread(QThread):
    result = Signal(dict)
    error = Signal(str)

    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path

    def run(self):
        try:
            img_b64 = _encode_image_for_vlm(self.image_path)

            payload = json.dumps({
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": LOCATE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": "Find all distinct features and regions of interest in this image. Return JSON only.",
                        "images": [img_b64],
                    },
                ],
                "stream": False,
                "format": "json",
            }, ensure_ascii=False).encode("utf-8")

            req = urllib.request.Request(
                OLLAMA_URL,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=600) as resp:
                data = json.loads(resp.read())

            content = data.get("message", {}).get("content", "{}")
            content = content.strip()
            if content.startswith("```"):
                lines = content.splitlines()
                content = "\n".join(lines[1:-1])

            parsed = json.loads(content)
            self.result.emit(parsed)
        except Exception as e:
            self.error.emit(str(e))


# ── tab widgets ───────────────────────────────────────────────────────────────

class _ChatTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._pull_thread = None
        self._chat_thread = None
        self._history = []
        self._attached_images = []
        self._assistant_reply = ""
        self._init_ui()
        self._check_status()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignTop)

        header_row = QHBoxLayout()
        title = QLabel("<b>LLM JSON Maker</b>")
        title.setStyleSheet("font-size: 14px; padding: 5px;")
        header_row.addWidget(title, 1)
        reset_btn = QPushButton("Reset Context")
        reset_btn.setFixedWidth(110)
        reset_btn.setToolTip("Clear conversation history and start fresh")
        reset_btn.clicked.connect(self._clear_chat)
        header_row.addWidget(reset_btn)
        layout.addLayout(header_row)

        description = QLabel(
            "Describe your scan in plain English and let an LLM generate the JSON config for you."
        )
        description.setWordWrap(True)
        description.setStyleSheet("color: #666; padding: 5px;")
        layout.addWidget(description)

        status_row = QHBoxLayout()
        self.status_label = QLabel("Checking Ollama...")
        self.status_label.setWordWrap(True)
        status_row.addWidget(self.status_label, 1)
        self.download_btn = QPushButton(f"Download {MODEL}")
        self.download_btn.setVisible(False)
        self.download_btn.clicked.connect(self._start_pull)
        status_row.addWidget(self.download_btn)
        layout.addLayout(status_row)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setVisible(False)
        self.log_box.setFixedHeight(120)
        self.log_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.log_box.setStyleSheet("font-family: monospace; font-size: 11px;")
        layout.addWidget(self.log_box)

        self.chat_area = QTextEdit()
        self.chat_area.setReadOnly(True)
        self.chat_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.chat_area.setVisible(False)
        layout.addWidget(self.chat_area, 1)

        self.file_btn = QPushButton("Attach Image")
        self.file_btn.setMinimumHeight(36)
        self.file_btn.clicked.connect(self._pick_image)
        self.file_btn.setVisible(False)
        layout.addWidget(self.file_btn)

        input_row = QHBoxLayout()
        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("Describe your scan config...")
        self.input_box.returnPressed.connect(self._send)
        input_row.addWidget(self.input_box, 1)

        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self._send)
        input_row.addWidget(self.send_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._clear_chat)
        input_row.addWidget(self.clear_btn)

        self.input_widget = QWidget()
        self.input_widget.setLayout(input_row)
        self.input_widget.setVisible(False)
        layout.addWidget(self.input_widget)

    def _check_status(self):
        if not shutil.which("ollama"):
            self.status_label.setText(
                "Ollama is not installed. Install it from https://ollama.com and restart."
            )
            self.status_label.setStyleSheet("color: #c0392b; padding: 5px;")
            return
        try:
            result = subprocess.run(
                ["ollama", "list"], capture_output=True, text=True, timeout=5
            )
            if MODEL in result.stdout:
                self._set_ready()
            else:
                self.status_label.setText(f"Model {MODEL} is not downloaded yet.")
                self.status_label.setStyleSheet("color: #e67e22; padding: 5px;")
                self.download_btn.setVisible(True)
        except Exception as e:
            self.status_label.setText(f"Could not reach Ollama: {e}")
            self.status_label.setStyleSheet("color: #c0392b; padding: 5px;")

    def _set_ready(self):
        self.status_label.setText(f"Model {MODEL} is ready.")
        self.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
        self.download_btn.setVisible(False)
        self.log_box.setVisible(False)
        self.chat_area.setVisible(True)
        self.file_btn.setVisible(True)
        self.input_widget.setVisible(True)

    def _pick_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "",
            "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp);;All Files (*)"
        )
        if path:
            self._attached_images.append(path)
            self._update_attach_label()
            self.input_box.setFocus()

    def _update_attach_label(self):
        if self._attached_images:
            names = ", ".join(os.path.basename(p) for p in self._attached_images)
            self.file_btn.setText(f"Attach Image  [{names}]")
        else:
            self.file_btn.setText("Attach Image")

    def _start_pull(self):
        self.download_btn.setEnabled(False)
        self.log_box.clear()
        self.log_box.setVisible(True)
        self.status_label.setText(f"Downloading {MODEL}... (this may take a few minutes)")
        self.status_label.setStyleSheet("color: #2980b9; padding: 5px;")
        self._pull_thread = _PullThread()
        self._pull_thread.log.connect(self._append_log)
        self._pull_thread.finished.connect(self._on_pull_finished)
        self._pull_thread.start()

    def _append_log(self, line):
        self.log_box.append(line)
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())

    def _on_pull_finished(self, success):
        self.download_btn.setEnabled(True)
        if success:
            self._set_ready()
        else:
            self.status_label.setText("Download failed. Check the log above.")
            self.status_label.setStyleSheet("color: #c0392b; padding: 5px;")

    def _send(self):
        raw = self.input_box.text().strip()
        busy = self._chat_thread and self._chat_thread.isRunning()
        if not raw or busy:
            return
        self.input_box.clear()
        image_paths = list(self._attached_images)
        self._attached_images.clear()
        self._update_attach_label()

        label = raw + (f"  [{len(image_paths)} image(s) attached]" if image_paths else "")
        self._append_chat("You", label)
        self.send_btn.setEnabled(False)
        self.input_box.setEnabled(False)
        self._assistant_reply = ""

        msg = {"role": "user", "content": raw or "Describe this image."}
        if image_paths:
            msg["images"] = [
                base64.b64encode(open(p, "rb").read()).decode()
                for p in image_paths
            ]
        self._history.append(msg)

        self.chat_area.append(f"<b>{MODEL}:</b> ")
        trimmed = []
        for i, m in enumerate(self._history):
            if i < len(self._history) - 1 and "images" in m:
                trimmed.append({k: v for k, v in m.items() if k != "images"})
            else:
                trimmed.append(m)
        self._chat_thread = _ChatThread(trimmed)
        self._chat_thread.token.connect(self._on_token)
        self._chat_thread.finished.connect(self._on_chat_done)
        self._chat_thread.error.connect(self._on_chat_error)
        self._chat_thread.start()

    def _on_token(self, token):
        self._assistant_reply += token
        cursor = self.chat_area.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(token)
        self.chat_area.setTextCursor(cursor)
        self.chat_area.verticalScrollBar().setValue(self.chat_area.verticalScrollBar().maximum())

    def _on_chat_done(self):
        self._history.append({"role": "assistant", "content": self._assistant_reply})
        self.chat_area.append("")
        self.send_btn.setEnabled(True)
        self.input_box.setEnabled(True)
        self.input_box.setFocus()

    def _on_chat_error(self, msg):
        self._append_chat("Error", msg)
        self.send_btn.setEnabled(True)
        self.input_box.setEnabled(True)

    def _append_chat(self, sender, text):
        self.chat_area.append(f"<b>{sender}:</b> {text}")

    def _clear_chat(self):
        self._history.clear()
        self._assistant_reply = ""
        self._attached_images.clear()
        self._update_attach_label()
        self.chat_area.clear()


class _LocateTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._image_path = None
        self._locate_thread = None
        self._orig_pixmap = None
        self._model_ok = True
        self._init_ui()
        self._check_model()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignTop)

        self.model_status = QLabel("")
        self.model_status.setWordWrap(True)
        layout.addWidget(self.model_status)

        pick_row = QHBoxLayout()
        self.pick_btn = QPushButton("Pick Image...")
        self.pick_btn.clicked.connect(self._pick_image)
        pick_row.addWidget(self.pick_btn)
        self.image_name_label = QLabel("No image selected")
        self.image_name_label.setStyleSheet("color: #888;")
        pick_row.addWidget(self.image_name_label, 1)
        layout.addLayout(pick_row)

        self.locate_btn = QPushButton("Get Location of Features")
        self.locate_btn.setMinimumHeight(40)
        self.locate_btn.setEnabled(False)
        self.locate_btn.clicked.connect(self._run_locate)
        layout.addWidget(self.locate_btn)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.img_display = QLabel()
        self.img_display.setAlignment(Qt.AlignCenter)
        self.img_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.scroll_area.setWidget(self.img_display)
        self.scroll_area.setVisible(False)
        layout.addWidget(self.scroll_area, 1)

        self.info_panel = QTextEdit()
        self.info_panel.setReadOnly(True)
        self.info_panel.setFixedHeight(130)
        self.info_panel.setStyleSheet("font-family: monospace; font-size: 11px;")
        self.info_panel.setVisible(False)
        layout.addWidget(self.info_panel)

    def _check_model(self):
        if not shutil.which("ollama"):
            self.model_status.setText("Ollama not installed — see the Chat tab.")
            self.model_status.setStyleSheet("color: #c0392b; padding: 4px;")
            self._model_ok = False
            return
        try:
            result = subprocess.run(
                ["ollama", "list"], capture_output=True, text=True, timeout=5
            )
            if MODEL not in result.stdout:
                self.model_status.setText(
                    f"{MODEL} not downloaded — use the Chat tab to download it first."
                )
                self.model_status.setStyleSheet("color: #e67e22; padding: 4px;")
                self._model_ok = False
        except Exception:
            self.model_status.setText("Could not reach Ollama.")
            self.model_status.setStyleSheet("color: #c0392b; padding: 4px;")
            self._model_ok = False

    def _pick_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "",
            "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp);;All Files (*)"
        )
        if not path:
            return
        self._image_path = path
        self.image_name_label.setText(os.path.basename(path))
        self.image_name_label.setStyleSheet("")
        try:
            self._orig_pixmap, _, _ = _load_image_pixmap(path)
            self._show_pixmap(self._orig_pixmap)
            self.scroll_area.setVisible(True)
        except Exception as e:
            self.status_label.setText(f"Could not load image: {e}")
            self.status_label.setStyleSheet("color: #c0392b; padding: 4px;")
            return
        if self._model_ok:
            self.locate_btn.setEnabled(True)

    def _show_pixmap(self, pixmap):
        max_w = max(400, self.scroll_area.width() - 20)
        max_h = max(300, self.scroll_area.height() - 20)
        scaled = pixmap.scaled(max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.img_display.setPixmap(scaled)

    def _run_locate(self):
        if not self._image_path or (self._locate_thread and self._locate_thread.isRunning()):
            return
        self.locate_btn.setEnabled(False)
        self.status_label.setText("Analyzing image... this may take 30–60 seconds.")
        self.status_label.setStyleSheet("color: #2980b9; padding: 4px;")

        self._locate_thread = _LocateThread(self._image_path)
        self._locate_thread.result.connect(self._on_result)
        self._locate_thread.error.connect(self._on_error)
        self._locate_thread.start()

    def _on_result(self, data):
        features = data.get("features", [])
        n = len(features)

        stem = pathlib.Path(self._image_path).stem
        json_path = pathlib.Path(self._image_path).with_name(stem + "_features.json")
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

        if self._orig_pixmap and features:
            annotated = _draw_feature_boxes(self._orig_pixmap, features)
            self._show_pixmap(annotated)

        self._populate_info(features, json_path)
        self.locate_btn.setEnabled(True)
        self.status_label.setText(f"Done. Saved: {json_path.name}")
        self.status_label.setStyleSheet("color: #27ae60; padding: 4px;")

        QMessageBox.information(
            self,
            "Features Found",
            f"{n} location(s) found.\nSaved to:\n{json_path}",
        )

    def _populate_info(self, features, json_path):
        color_names = ["red", "blue", "green", "yellow", "purple", "cyan", "orange"]
        lines = [f"Found {len(features)} feature(s)  —  {json_path.name}", ""]
        for i, feat in enumerate(features):
            color = color_names[i % len(color_names)]
            fid = feat.get("id", i + 1)
            label = feat.get("label", "")
            desc = feat.get("description", "")
            x1, y1 = feat.get("x1", 0), feat.get("y1", 0)
            x2, y2 = feat.get("x2", 0), feat.get("y2", 0)
            lines.append(f"[{color}] #{fid}  {label}")
            lines.append(f"       bbox: ({x1:.3f}, {y1:.3f}) → ({x2:.3f}, {y2:.3f})")
            if desc:
                lines.append(f"       {desc}")
            lines.append("")
        self.info_panel.setPlainText("\n".join(lines).strip())
        self.info_panel.setVisible(True)

    def _on_error(self, msg):
        self.locate_btn.setEnabled(True)
        self.status_label.setText(f"Error: {msg}")
        self.status_label.setStyleSheet("color: #c0392b; padding: 4px;")


# ── public widget ─────────────────────────────────────────────────────────────

class LLMJsonMakerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        tabs = QTabWidget()
        tabs.addTab(_ChatTab(), "Chat")
        tabs.addTab(_LocateTab(), "Feature Locator")
        layout.addWidget(tabs)
