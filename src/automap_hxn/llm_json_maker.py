import base64
import json
import os
import subprocess
import shutil
import urllib.request

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QSizePolicy, QLineEdit, QFileDialog
)
from qtpy.QtCore import Qt, QThread, Signal
from qtpy.QtGui import QTextCursor


MODEL = "qwen2.5vl:3b"
OLLAMA_URL = "http://localhost:11434/api/chat"


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


class LLMJsonMakerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._pull_thread = None
        self._chat_thread = None
        self._history = []
        self._attached_images = []
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

        # Status row
        status_row = QHBoxLayout()
        self.status_label = QLabel("Checking Ollama...")
        self.status_label.setWordWrap(True)
        status_row.addWidget(self.status_label, 1)
        self.download_btn = QPushButton(f"Download {MODEL}")
        self.download_btn.setVisible(False)
        self.download_btn.clicked.connect(self._start_pull)
        status_row.addWidget(self.download_btn)
        layout.addLayout(status_row)

        # Download log (hidden unless pulling)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setVisible(False)
        self.log_box.setFixedHeight(120)
        self.log_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.log_box.setStyleSheet("font-family: monospace; font-size: 11px;")
        layout.addWidget(self.log_box)

        # Chat area (hidden until model is ready)
        self.chat_area = QTextEdit()
        self.chat_area.setReadOnly(True)
        self.chat_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.chat_area.setVisible(False)
        layout.addWidget(self.chat_area, 1)

        # File picker button
        self.file_btn = QPushButton("Attach Image")
        self.file_btn.setMinimumHeight(36)
        self.file_btn.clicked.connect(self._pick_image)
        layout.addWidget(self.file_btn)

        # Input row
        self.input_row = QHBoxLayout()
        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("Describe your scan config...")
        self.input_box.returnPressed.connect(self._send)
        self.input_row.addWidget(self.input_box, 1)

        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self._send)
        self.input_row.addWidget(self.send_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._clear_chat)
        self.input_row.addWidget(self.clear_btn)

        self.input_widget = QWidget()
        self.input_widget.setLayout(self.input_row)
        self.input_widget.setVisible(False)
        layout.addWidget(self.input_widget)

        self.file_btn.setVisible(False)

    # ── model management ──────────────────────────────────────────────────────

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

    # ── chat ──────────────────────────────────────────────────────────────────

    def _send(self):
        raw = self.input_box.text().strip()
        busy = self._chat_thread and self._chat_thread.isRunning()
        if not raw or busy:
            return
        self.input_box.clear()
        image_paths = list(self._attached_images)
        self._attached_images.clear()
        self._update_attach_label()

        text = raw
        label = text + (f"  [{len(image_paths)} image(s) attached]" if image_paths else "")
        self._append_chat("You", label)
        self.send_btn.setEnabled(False)
        self.input_box.setEnabled(False)
        self._assistant_reply = ""

        msg = {"role": "user", "content": text or "Describe this image."}
        if image_paths:
            msg["images"] = [
                base64.b64encode(open(p, "rb").read()).decode()
                for p in image_paths
            ]
        self._history.append(msg)

        self.chat_area.append(f"<b>{MODEL}:</b> ")
        # Strip images from earlier turns to keep payload small
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
        self.chat_area.append("")  # blank line between turns
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
