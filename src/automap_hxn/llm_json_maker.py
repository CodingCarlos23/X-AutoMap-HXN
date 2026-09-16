import base64
import json
import os
import pathlib
import re
import subprocess
import shutil
import urllib.request

import cv2
import numpy as np

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QSizePolicy, QLineEdit, QFileDialog, QTabWidget,
    QScrollArea, QMessageBox, QComboBox, QCheckBox, QSpinBox,
)
from qtpy.QtCore import Qt, QThread, Signal
from qtpy.QtGui import (
    QTextCursor, QPixmap, QImage, QPainter, QPen, QColor, QFont,
)


MODEL = "qwen2.5vl:7b"
OLLAMA_URL = "http://localhost:11434/api/chat"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_API_VERSION = "2023-06-01"
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

KNOWN_MODELS = [
    # ── Local (Ollama) ──────────────────────────────
    "qwen2.5vl:7b",
    "qwen2.5vl:3b",
    "qwen2.5vl:72b",
    "minicpm-v:8b",
    "llava-llama3:8b",
    # ── Anthropic ───────────────────────────────────
    "claude-sonnet-5",
    "claude-opus-5",
    # ── OpenAI ──────────────────────────────────────
    "gpt-4o",
    "gpt-4.1",
]

_active_model = MODEL  # updated by the model picker in LLMJsonMakerWidget

# ── provider routing ──────────────────────────────────────────────────────────

_ENV_VAR = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}
_api_keys: dict = {}  # provider -> key, populated from env / GPG / user input

_GPG_ENV_CANDIDATES = [
    "~/.private.env.gpg",
    "~/.env.gpg",
    "~/.secrets.gpg",
]


def _get_provider(model: str) -> str:
    if model.startswith("claude-"):
        return "anthropic"
    if model.startswith(("gpt-", "o3", "o4")):
        return "openai"
    return "ollama"


def _try_load_key(provider: str) -> "str | None":
    """Check $ENV first, then GPG-encrypted env files. Caches result in _api_keys."""
    env_var = _ENV_VAR.get(provider, "")
    val = os.environ.get(env_var, "").strip()
    if val:
        _api_keys[provider] = val
        return val
    for path in _GPG_ENV_CANDIDATES:
        expanded = os.path.expanduser(path)
        if not os.path.exists(expanded):
            continue
        try:
            result = subprocess.run(
                ["gpg", "--quiet", "--batch", "--yes", "--decrypt", expanded],
                capture_output=True, text=True, timeout=15,
            )
            for line in result.stdout.splitlines():
                line = line.strip()
                for prefix in (f"export {env_var}=", f"{env_var}="):
                    if line.startswith(prefix):
                        key = line[len(prefix):].strip().strip('"').strip("'")
                        if key:
                            _api_keys[provider] = key
                            return key
        except Exception:
            pass
    return None


def _get_key(provider: str) -> "str | None":
    return _api_keys.get(provider) or _try_load_key(provider)


def _history_to_anthropic(history: list) -> list:
    msgs = []
    for msg in history:
        if msg.get("role") == "system":
            continue
        images = msg.get("images", [])
        if images:
            content = []
            for img_b64 in images:
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": img_b64},
                })
            if msg.get("content"):
                content.append({"type": "text", "text": msg["content"]})
            msgs.append({"role": msg["role"], "content": content})
        else:
            msgs.append({"role": msg["role"], "content": msg.get("content", "")})
    return msgs


def _history_to_openai(history: list) -> list:
    msgs = []
    for msg in history:
        images = msg.get("images", [])
        if images:
            content = []
            for img_b64 in images:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                })
            if msg.get("content"):
                content.append({"type": "text", "text": msg["content"]})
            msgs.append({"role": msg["role"], "content": content})
        else:
            msgs.append({"role": msg["role"], "content": msg.get("content", "")})
    return msgs


def _parse_locate_json(content: str) -> dict:
    if content.startswith("```"):
        content = "\n".join(content.splitlines()[1:-1])
    return json.loads(content)

LOCATE_SYSTEM_PROMPT = (
    "You are a scientific image analysis assistant for XRF (X-ray fluorescence) scan data. "
    "Identify all distinct bright features, particles, or clusters visible in the image. "
    "Respond ONLY with valid JSON using this exact schema — no extra text outside the JSON object:\n"
    '{"features": [{"id": 1, "label": "particle cluster", "x1": 0.10, "y1": 0.20, '
    '"x2": 0.45, "y2": 0.60, "description": "bright region with high signal"}], "total_count": 1}\n'
    "IMPORTANT: All coordinates (x1, y1, x2, y2) must be FRACTIONAL values between 0.0 and 1.0, "
    "where (0,0) is the top-left corner and (1,1) is the bottom-right corner of the image. "
    "x1 < x2, y1 < y2. "
    "Draw each bounding box tightly around the BRIGHT CORE of each feature — "
    "do not include surrounding dark background. Be as precise as possible with the box edges. "
    "Include every distinct feature you can identify."
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


def _load_image_pixmap(path, dilate_ksize: int = 0):
    """Load any image (including 16/32-bit TIFF) and return (QPixmap, width, height)."""
    arr = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    if arr is None:
        raise ValueError(f"Could not read image: {path}")
    arr = arr.astype(np.float32)
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    arr = (arr * 255).astype(np.uint8)
    if dilate_ksize > 0:
        kernel = np.ones((dilate_ksize, dilate_ksize), np.uint8)
        arr = cv2.dilate(arr, kernel)
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


# ── image display with mouse coord tracking ───────────────────────────────────

class _ImageDisplay(QLabel):
    """QLabel that emits fractional (0–1) mouse coordinates over the displayed pixmap."""
    mouse_moved = Signal(float, float)
    mouse_left = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)

    def mouseMoveEvent(self, event):
        pm = self.pixmap()
        if pm is None or pm.isNull():
            return
        ox = (self.width() - pm.width()) / 2
        oy = (self.height() - pm.height()) / 2
        px = event.x() - ox
        py = event.y() - oy
        if 0 <= px <= pm.width() and 0 <= py <= pm.height():
            self.mouse_moved.emit(px / pm.width(), py / pm.height())
        else:
            self.mouse_left.emit()

    def leaveEvent(self, event):
        self.mouse_left.emit()


# ── background threads ────────────────────────────────────────────────────────

class _PullThread(QThread):
    log = Signal(str)
    finished = Signal(bool)

    def run(self):
        try:
            proc = subprocess.Popen(
                ["ollama", "pull", _active_model],
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
        try:
            provider = _get_provider(_active_model)
            if provider == "anthropic":
                self._run_anthropic()
            elif provider == "openai":
                self._run_openai()
            else:
                self._run_ollama()
        except Exception as e:
            self.error.emit(str(e))

    def _run_ollama(self):
        payload = json.dumps({
            "model": _active_model,
            "messages": self.messages,
            "stream": True,
        }, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            OLLAMA_URL, data=payload,
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

    def _run_anthropic(self):
        key = _api_keys.get("anthropic", "")
        sys_msgs = [m for m in self.messages if m.get("role") == "system"]
        system = sys_msgs[0]["content"] if sys_msgs else ""
        payload = json.dumps({
            "model": _active_model,
            "max_tokens": 4096,
            "stream": True,
            "system": system,
            "messages": _history_to_anthropic(self.messages),
        }, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            ANTHROPIC_API_URL, data=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": key,
                "anthropic-version": ANTHROPIC_API_VERSION,
            },
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            for line in resp:
                line = line.strip()
                if not line or line == b"data: [DONE]":
                    continue
                if line.startswith(b"data: "):
                    line = line[6:]
                try:
                    chunk = json.loads(line)
                    if chunk.get("type") == "content_block_delta":
                        text = chunk.get("delta", {}).get("text", "")
                        if text:
                            self.token.emit(text)
                except json.JSONDecodeError:
                    pass
        self.finished.emit()

    def _run_openai(self):
        key = _api_keys.get("openai", "")
        payload = json.dumps({
            "model": _active_model,
            "stream": True,
            "messages": _history_to_openai(self.messages),
        }, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            OPENAI_API_URL, data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            for line in resp:
                line = line.strip()
                if not line or line == b"data: [DONE]":
                    continue
                if line.startswith(b"data: "):
                    line = line[6:]
                try:
                    chunk = json.loads(line)
                    content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if content:
                        self.token.emit(content)
                except json.JSONDecodeError:
                    pass
        self.finished.emit()


_VLM_MAX_DIM = 1024


def _encode_image_for_vlm(path, max_dim=_VLM_MAX_DIM, dilate_ksize: int = 0):
    """Load image, optionally dilate, downsample to max_dim, return base64-PNG string."""
    arr = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    if arr is None:
        raise ValueError(f"Could not read image: {path}")
    arr = arr.astype(np.float32)
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    arr = (arr * 255).astype(np.uint8)
    if dilate_ksize > 0:
        kernel = np.ones((dilate_ksize, dilate_ksize), np.uint8)
        arr = cv2.dilate(arr, kernel)
    # CLAHE: sharpen local contrast so blob edges are crisper for the VLM
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    if arr.ndim == 2:
        arr = clahe.apply(arr)
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    else:
        lab = cv2.cvtColor(arr, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        arr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    h, w = arr.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        arr = cv2.resize(arr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode()


_LOCATE_USER_PROMPT = (
    "Find all distinct features and regions of interest in this image. Return JSON only."
)


class _LocateThread(QThread):
    result = Signal(dict)
    error = Signal(str)

    def __init__(self, image_path, dilate_ksize: int = 0):
        super().__init__()
        self.image_path = image_path
        self.dilate_ksize = dilate_ksize

    def run(self):
        try:
            img_b64 = _encode_image_for_vlm(self.image_path, dilate_ksize=self.dilate_ksize)
            provider = _get_provider(_active_model)
            if provider == "anthropic":
                parsed = self._query_anthropic(img_b64)
            elif provider == "openai":
                parsed = self._query_openai(img_b64)
            else:
                parsed = self._query_ollama(img_b64)
            self.result.emit(parsed)
        except Exception as e:
            self.error.emit(str(e))

    def _query_ollama(self, img_b64: str) -> dict:
        payload = json.dumps({
            "model": _active_model,
            "messages": [
                {"role": "system", "content": LOCATE_SYSTEM_PROMPT},
                {"role": "user", "content": _LOCATE_USER_PROMPT, "images": [img_b64]},
            ],
            "stream": False,
            "format": "json",
        }, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            OLLAMA_URL, data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read())
        return _parse_locate_json(data.get("message", {}).get("content", "{}"))

    def _query_anthropic(self, img_b64: str) -> dict:
        key = _api_keys.get("anthropic", "")
        payload = json.dumps({
            "model": _active_model,
            "max_tokens": 4096,
            "system": LOCATE_SYSTEM_PROMPT,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_b64}},
                    {"type": "text", "text": _LOCATE_USER_PROMPT},
                ],
            }],
        }, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            ANTHROPIC_API_URL, data=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": key,
                "anthropic-version": ANTHROPIC_API_VERSION,
            },
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read())
        return _parse_locate_json(data.get("content", [{}])[0].get("text", "{}"))

    def _query_openai(self, img_b64: str) -> dict:
        key = _api_keys.get("openai", "")
        payload = json.dumps({
            "model": _active_model,
            "messages": [
                {"role": "system", "content": LOCATE_SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    {"type": "text", "text": _LOCATE_USER_PROMPT},
                ]},
            ],
        }, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            OPENAI_API_URL, data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read())
        return _parse_locate_json(
            data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
        )



def _box_iou(a: dict, b: dict) -> float:
    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class _LocateTileThread(QThread):
    """Splits the image into 4 quadrants, queries each separately, maps coords back."""
    result = Signal(dict)
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, image_path, dilate_ksize: int = 0):
        super().__init__()
        self.image_path = image_path
        self.dilate_ksize = dilate_ksize
        self._provider = _get_provider(_active_model)
        self._model = _active_model
        self._key = _api_keys.get(self._provider, "")

    def run(self):
        try:
            arr = cv2.imread(str(self.image_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
            if arr is None:
                raise ValueError(f"Could not read: {self.image_path}")
            arr = arr.astype(np.float32)
            mn, mx = arr.min(), arr.max()
            if mx > mn:
                arr = (arr - mn) / (mx - mn)
            arr = (arr * 255).astype(np.uint8)
            if self.dilate_ksize > 0:
                kernel = np.ones((self.dilate_ksize, self.dilate_ksize), np.uint8)
                arr = cv2.dilate(arr, kernel)

            h, w = arr.shape[:2]
            h2, w2 = h // 2, w // 2

            quads = [
                ("TL", arr[0:h2,  0:w2],  0.0,    0.0,    w2/w,       h2/h),
                ("TR", arr[0:h2,  w2:w],  w2/w,   0.0,    (w-w2)/w,   h2/h),
                ("BL", arr[h2:h,  0:w2],  0.0,    h2/h,   w2/w,       (h-h2)/h),
                ("BR", arr[h2:h,  w2:w],  w2/w,   h2/h,   (w-w2)/w,   (h-h2)/h),
            ]

            all_features = []
            fid = 1
            for qi, (name, crop, xoff, yoff, xscale, yscale) in enumerate(quads):
                self.progress.emit(f"Quadrant {qi + 1}/4 ({name})...")
                try:
                    img_b64 = self._encode_crop(crop)
                    features = self._query(img_b64)
                except Exception as e:
                    features = []
                    self.progress.emit(f"Quadrant {name} failed: {e}")

                for feat in features:
                    feat["id"] = fid
                    feat["x1"] = round(xoff + float(feat.get("x1", 0)) * xscale, 4)
                    feat["y1"] = round(yoff + float(feat.get("y1", 0)) * yscale, 4)
                    feat["x2"] = round(xoff + float(feat.get("x2", 1)) * xscale, 4)
                    feat["y2"] = round(yoff + float(feat.get("y2", 1)) * yscale, 4)
                    feat["description"] = f"[{name}] " + feat.get("description", "")
                    all_features.append(feat)
                    fid += 1

            all_features = self._dedup(all_features)
            for i, f in enumerate(all_features):
                f["id"] = i + 1

            self.result.emit({"features": all_features, "total_count": len(all_features)})

        except Exception as e:
            self.error.emit(str(e))

    @staticmethod
    def _encode_crop(crop: np.ndarray) -> str:
        if crop.ndim == 2:
            bgr = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        else:
            bgr = crop.copy()
        ok, buf = cv2.imencode(".png", bgr)
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        return base64.b64encode(buf.tobytes()).decode()

    def _query(self, img_b64: str) -> list:
        if self._provider == "anthropic":
            return self._query_anthropic(img_b64)
        if self._provider == "openai":
            return self._query_openai(img_b64)
        return self._query_ollama(img_b64)

    def _query_ollama(self, img_b64: str) -> list:
        payload = json.dumps({
            "model": self._model,
            "messages": [
                {"role": "system", "content": LOCATE_SYSTEM_PROMPT},
                {"role": "user", "content": _LOCATE_USER_PROMPT, "images": [img_b64]},
            ],
            "stream": False,
            "format": "json",
        }, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            OLLAMA_URL, data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read())
        return _parse_locate_json(data.get("message", {}).get("content", "{}")).get("features", [])

    def _query_anthropic(self, img_b64: str) -> list:
        payload = json.dumps({
            "model": self._model,
            "max_tokens": 4096,
            "system": LOCATE_SYSTEM_PROMPT,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_b64}},
                    {"type": "text", "text": _LOCATE_USER_PROMPT},
                ],
            }],
        }, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            ANTHROPIC_API_URL, data=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self._key,
                "anthropic-version": ANTHROPIC_API_VERSION,
            },
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read())
        return _parse_locate_json(data.get("content", [{}])[0].get("text", "{}")).get("features", [])

    def _query_openai(self, img_b64: str) -> list:
        payload = json.dumps({
            "model": self._model,
            "messages": [
                {"role": "system", "content": LOCATE_SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    {"type": "text", "text": _LOCATE_USER_PROMPT},
                ]},
            ],
        }, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            OPENAI_API_URL, data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._key}",
            },
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read())
        return _parse_locate_json(
            data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
        ).get("features", [])

    @staticmethod
    def _dedup(features: list, iou_thresh: float = 0.3) -> list:
        keep = [True] * len(features)
        for i in range(len(features)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(features)):
                if keep[j] and _box_iou(features[i], features[j]) > iou_thresh:
                    keep[j] = False
        return [f for f, k in zip(features, keep) if k]


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
        self.download_btn = QPushButton(f"Download {_active_model}")
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

        # cloud API key entry (shown only when provider is cloud and key not found)
        self._key_widget = QWidget()
        key_row = QHBoxLayout(self._key_widget)
        key_row.setContentsMargins(0, 4, 0, 0)
        self._key_label = QLabel("API Key:")
        key_row.addWidget(self._key_label)
        self._key_input = QLineEdit()
        self._key_input.setPlaceholderText("Paste your API key here...")
        self._key_input.setEchoMode(QLineEdit.Password)
        key_row.addWidget(self._key_input, 1)
        self._key_gpg_btn = QPushButton("Load from GPG")
        self._key_gpg_btn.setToolTip("Try to decrypt ~/.private.env.gpg and load key automatically")
        self._key_gpg_btn.clicked.connect(self._try_gpg)
        key_row.addWidget(self._key_gpg_btn)
        self._key_save_btn = QPushButton("Use Key")
        self._key_save_btn.clicked.connect(self._save_key)
        key_row.addWidget(self._key_save_btn)
        self._key_widget.setVisible(False)
        layout.addWidget(self._key_widget)

    def refresh_model(self):
        self.download_btn.setText(f"Download {_active_model}")
        self.download_btn.setVisible(False)
        self._key_widget.setVisible(False)
        self.chat_area.setVisible(False)
        self.file_btn.setVisible(False)
        self.input_widget.setVisible(False)
        self._history.clear()
        self._check_status()

    def _check_status(self):
        provider = _get_provider(_active_model)
        if provider == "ollama":
            self._check_ollama_status()
        else:
            self._check_cloud_status(provider)

    def _check_ollama_status(self):
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
            if _active_model in result.stdout:
                self._set_ready()
            else:
                self.status_label.setText(f"Model {_active_model} is not downloaded yet.")
                self.status_label.setStyleSheet("color: #e67e22; padding: 5px;")
                self.download_btn.setVisible(True)
        except Exception as e:
            self.status_label.setText(f"Could not reach Ollama: {e}")
            self.status_label.setStyleSheet("color: #c0392b; padding: 5px;")

    def _check_cloud_status(self, provider: str):
        key = _get_key(provider)
        if key:
            self._set_ready()
        else:
            provider_name = "Anthropic" if provider == "anthropic" else "OpenAI"
            env_var = _ENV_VAR[provider]
            self.status_label.setText(
                f"No {env_var} found. Click 'Load from GPG' or paste your {provider_name} key below."
            )
            self.status_label.setStyleSheet("color: #e67e22; padding: 5px;")
            self._key_label.setText(f"{provider_name} API Key:")
            self._key_widget.setVisible(True)

    def _try_gpg(self):
        provider = _get_provider(_active_model)
        key = _try_load_key(provider)
        if key:
            self._set_ready()
        else:
            self.status_label.setText(
                "GPG decrypt did not find the key. Enter it manually below."
            )
            self.status_label.setStyleSheet("color: #e67e22; padding: 5px;")

    def _save_key(self):
        provider = _get_provider(_active_model)
        key = self._key_input.text().strip()
        if not key:
            return
        _api_keys[provider] = key
        self._key_input.clear()
        self._set_ready()

    def _set_ready(self):
        self.status_label.setText(f"Model {_active_model} is ready.")
        self.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
        self.download_btn.setVisible(False)
        self._key_widget.setVisible(False)
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
        self.status_label.setText(f"Downloading {_active_model}... (this may take a few minutes)")
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

        self.chat_area.append(f"<b>{_active_model}:</b> ")
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
        self._dilated_pixmap = None
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

        dilate_row = QHBoxLayout()
        self._dilate_check = QCheckBox("Dilate image")
        self._dilate_check.setToolTip("Expand bright regions before display and analysis")
        self._dilate_check.toggled.connect(self._on_dilate_changed)
        dilate_row.addWidget(self._dilate_check)
        self._kernel_spin = QSpinBox()
        self._kernel_spin.setRange(3, 21)
        self._kernel_spin.setSingleStep(2)
        self._kernel_spin.setValue(5)
        self._kernel_spin.setSuffix(" px kernel")
        self._kernel_spin.setEnabled(False)
        self._kernel_spin.valueChanged.connect(self._on_dilate_changed)
        dilate_row.addWidget(self._kernel_spin)
        dilate_row.addStretch()
        layout.addLayout(dilate_row)

        tile_row = QHBoxLayout()
        self._tile_check = QCheckBox("Tile (4-quadrant)")
        self._tile_check.setToolTip(
            "Split image into 4 quadrants, query each separately, then merge coords back.\n"
            "Each feature takes up 2× more of the VLM's visual field → better precision."
        )
        tile_row.addWidget(self._tile_check)
        tile_row.addStretch()
        layout.addLayout(tile_row)

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
        self.scroll_area.viewport().setMouseTracking(True)
        self.img_display = _ImageDisplay()
        self.img_display.setAlignment(Qt.AlignCenter)
        self.img_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.img_display.mouse_moved.connect(self._on_mouse_moved)
        self.img_display.mouse_left.connect(self._on_mouse_left)
        self.scroll_area.setWidget(self.img_display)
        self.scroll_area.setVisible(False)

        self.coord_label = QLabel("X: —\nY: —")
        self.coord_label.setFixedWidth(75)
        self.coord_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.coord_label.setStyleSheet(
            "font-family: monospace; font-size: 11px; padding: 6px; color: #888;"
        )

        self.info_panel = QTextEdit()
        self.info_panel.setReadOnly(True)
        self.info_panel.setMinimumWidth(200)
        self.info_panel.setStyleSheet("font-family: monospace; font-size: 11px;")
        self.info_panel.setPlaceholderText("Feature details will appear here after analysis.")
        self.info_panel.setVisible(False)

        image_col = QVBoxLayout()
        image_col.setSpacing(0)
        image_row = QHBoxLayout()
        image_row.setSpacing(0)
        image_row.addWidget(self.scroll_area, 1)
        image_row.addWidget(self.coord_label)
        image_col.addLayout(image_row)

        content_row = QHBoxLayout()
        content_row.setSpacing(6)
        content_row.addLayout(image_col, 3)
        content_row.addWidget(self.info_panel, 1)
        layout.addLayout(content_row, 1)

    def _check_model(self):
        provider = _get_provider(_active_model)
        if provider == "ollama":
            self._check_ollama_model()
        else:
            self._check_cloud_model(provider)

    def _check_ollama_model(self):
        if not shutil.which("ollama"):
            self.model_status.setText("Ollama not installed — see the Chat tab.")
            self.model_status.setStyleSheet("color: #c0392b; padding: 4px;")
            self._model_ok = False
            return
        try:
            result = subprocess.run(
                ["ollama", "list"], capture_output=True, text=True, timeout=5
            )
            if _active_model not in result.stdout:
                self.model_status.setText(
                    f"{_active_model} not downloaded — use the Chat tab to download it first."
                )
                self.model_status.setStyleSheet("color: #e67e22; padding: 4px;")
                self._model_ok = False
        except Exception:
            self.model_status.setText("Could not reach Ollama.")
            self.model_status.setStyleSheet("color: #c0392b; padding: 4px;")
            self._model_ok = False

    def _check_cloud_model(self, provider: str):
        key = _get_key(provider)
        if key:
            self.model_status.setText("")
            self.model_status.setStyleSheet("")
            self._model_ok = True
        else:
            provider_name = "Anthropic" if provider == "anthropic" else "OpenAI"
            self.model_status.setText(
                f"No API key for {provider_name} — configure it in the Chat tab first."
            )
            self.model_status.setStyleSheet("color: #e67e22; padding: 4px;")
            self._model_ok = False

    def refresh_model(self):
        self._model_ok = True
        self.model_status.setText("")
        self.model_status.setStyleSheet("")
        self._check_model()
        if self._image_path:
            self.locate_btn.setEnabled(self._model_ok)

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
            self._dilated_pixmap = None
            self._update_display()
            self.scroll_area.setVisible(True)
            self.info_panel.setVisible(True)
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
        ksize = self._get_dilate_ksize()
        use_tile = self._tile_check.isChecked()

        if use_tile:
            self.status_label.setText("Tile mode — querying 4 quadrants... (~2–4 min)")
            self._locate_thread = _LocateTileThread(self._image_path, dilate_ksize=ksize)
            self._locate_thread.progress.connect(self._on_tile_progress)
        else:
            self.status_label.setText("Analyzing... this may take 30–60 seconds.")
            self._locate_thread = _LocateThread(self._image_path, dilate_ksize=ksize)
        self.status_label.setStyleSheet("color: #2980b9; padding: 4px;")
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

        base = self._dilated_pixmap if self._dilated_pixmap else self._orig_pixmap
        if base and features:
            annotated = _draw_feature_boxes(base, features)
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

    def _get_dilate_ksize(self) -> int:
        return self._kernel_spin.value() if self._dilate_check.isChecked() else 0

    def _on_dilate_changed(self) -> None:
        self._kernel_spin.setEnabled(self._dilate_check.isChecked())
        self._dilated_pixmap = None
        if self._image_path and self._orig_pixmap:
            self._update_display()

    def _update_display(self) -> None:
        ksize = self._get_dilate_ksize()
        if ksize > 0:
            if self._dilated_pixmap is None:
                self._dilated_pixmap, _, _ = _load_image_pixmap(self._image_path, dilate_ksize=ksize)
            self._show_pixmap(self._dilated_pixmap)
        else:
            self._show_pixmap(self._orig_pixmap)

    def _on_mouse_moved(self, fx: float, fy: float) -> None:
        self.coord_label.setText(f"X: {fx:.3f}\nY: {fy:.3f}")
        self.coord_label.setStyleSheet(
            "font-family: monospace; font-size: 11px; padding: 6px;"
        )

    def _on_mouse_left(self) -> None:
        self.coord_label.setText("X: —\nY: —")
        self.coord_label.setStyleSheet(
            "font-family: monospace; font-size: 11px; padding: 6px; color: #888;"
        )

    def _on_tile_progress(self, msg: str) -> None:
        self.status_label.setText(msg)
        self.status_label.setStyleSheet("color: #2980b9; padding: 4px;")


    def _on_error(self, msg):
        self.locate_btn.setEnabled(True)
        self.status_label.setText(f"Error: {msg}")
        self.status_label.setStyleSheet("color: #c0392b; padding: 4px;")


# ── public widget ─────────────────────────────────────────────────────────────

class LLMJsonMakerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 0)
        layout.setSpacing(4)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self._model_combo = QComboBox()
        self._model_combo.setEditable(True)
        for m in KNOWN_MODELS:
            self._model_combo.addItem(m)
        self._model_combo.setCurrentText(MODEL)
        self._model_combo.setToolTip(
            "Select or type a model name. Changes take effect on the next send or locate."
        )
        self._model_combo.currentTextChanged.connect(self._on_model_changed)
        model_row.addWidget(self._model_combo, 1)
        layout.addLayout(model_row)

        tabs = QTabWidget()
        self._chat_tab = _ChatTab()
        self._locate_tab = _LocateTab()
        tabs.addTab(self._chat_tab, "Chat")
        tabs.addTab(self._locate_tab, "Feature Locator")
        layout.addWidget(tabs)

    def _on_model_changed(self, text: str) -> None:
        global _active_model
        _active_model = text.strip()
        self._chat_tab.refresh_model()
        self._locate_tab.refresh_model()
