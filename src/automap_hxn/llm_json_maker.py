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
    QGroupBox, QRadioButton, QButtonGroup, QSplitter,
)
from qtpy.QtCore import Qt, QThread, Signal, QTimer
from qtpy.QtGui import (
    QTextCursor, QPixmap, QImage, QPainter, QPen, QColor, QFont, QCursor,
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
    # ── Anthropic (via NSLS-II APIM or direct) ──────
    "claude-sonnet-4-6",
    "claude-opus-4-5",
    "claude-fable-5",
    # ── OpenAI (via NSLS-II APIM) ───────────────────
    "gpt-4.1",
    "gpt-5.5",
    "gpt-5.6-terra",
    "gpt-5.6-sol",
]

_active_model = MODEL  # updated by the model picker in LLMJsonMakerWidget

# ── provider routing ──────────────────────────────────────────────────────────

_ENV_VAR = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}
_api_keys: dict = {}   # provider -> key, populated from env / GPG / user input
_api_urls: dict = {}   # provider -> custom base URL (e.g. AIFAPIM_HOST override)

_GPG_ENV_CANDIDATES = [
    "~/.private_env.gpg",
]


def _get_provider(model: str) -> str:
    if model.startswith("claude-"):
        return "anthropic"
    if model.startswith(("gpt-", "o3", "o4")):
        return "openai"
    return "ollama"


def _parse_env_lines(text: str) -> "dict[str, str]":
    """Parse 'export KEY=value' or 'KEY=value' lines into a dict."""
    result = {}
    for line in text.splitlines():
        line = line.strip()
        for prefix in ("export ", ""):
            if line.startswith(prefix):
                rest = line[len(prefix):]
                if "=" in rest:
                    k, _, v = rest.partition("=")
                    result[k.strip()] = v.strip().strip('"').strip("'")
                    break
    return result


def _try_load_key(provider: str) -> "tuple[str | None, str]":
    """Check $ENV first, then GPG-encrypted env files. Caches result in _api_keys.
    Returns (key_or_None, diagnostic_message)."""
    env_var = _ENV_VAR.get(provider, "")
    val = os.environ.get(env_var, "").strip()
    if val:
        _api_keys[provider] = val
        return val, ""
    for path in _GPG_ENV_CANDIDATES:
        expanded = os.path.expanduser(path)
        if not os.path.exists(expanded):
            continue
        try:
            result = subprocess.run(
                ["gpg", "--quiet", "--batch", "--yes", "--decrypt", expanded],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode != 0:
                err = result.stderr.strip() or "gpg exited non-zero (no error output)"
                return None, f"GPG error ({expanded}): {err}"
            env = _parse_env_lines(result.stdout)
            # Standard key name
            if env_var in env and env[env_var]:
                _api_keys[provider] = env[env_var]
                return env[env_var], ""
            # Fallback: AIFAPIM_API_KEY + AIFAPIM_HOST for BNL proxy (OpenAI and Anthropic)
            apim_key = env.get("AIFAPIM_API_KEY", "").strip()
            apim_host = env.get("AIFAPIM_HOST", "").strip()
            if apim_key and apim_host:
                host = apim_host.rstrip("/")
                if not host.startswith("http"):
                    host = f"https://{host}"
                _api_keys[provider] = apim_key
                if provider == "openai":
                    _api_urls[provider] = f"{host}/openai/v1/responses"
                    codex_model = env.get("AIFAPIM_CODEX_MODEL", "").strip()
                    if codex_model and codex_model not in KNOWN_MODELS:
                        KNOWN_MODELS.append(codex_model)
                    if codex_model:
                        _api_keys["_apim_model"] = codex_model
                elif provider == "anthropic":
                    _api_urls[provider] = f"{host}/anthropic/v1/messages"
                    claude_model = env.get("AIFAPIM_CLAUDE_MODEL", "").strip()
                    if claude_model and claude_model not in KNOWN_MODELS:
                        KNOWN_MODELS.append(claude_model)
                    if claude_model:
                        _api_keys["_apim_claude_model"] = claude_model
                return apim_key, ""
            return None, (
                f"GPG decrypted {expanded} but '{env_var}' was not found in the output.\n"
                f"Keys present: {list(env.keys())}"
            )
        except Exception as exc:
            return None, f"GPG exception: {exc}"
    return None, f"No GPG env file found at: {', '.join(_GPG_ENV_CANDIDATES)}"


def _get_key(provider: str) -> "str | None":
    if provider in _api_keys:
        return _api_keys[provider]
    key, _ = _try_load_key(provider)
    return key


def _get_openai_url() -> str:
    return _api_urls.get("openai", OPENAI_API_URL)


def _get_openai_model() -> str:
    return _api_keys.get("_apim_model") or _active_model


def _get_anthropic_url() -> str:
    return _api_urls.get("anthropic", ANTHROPIC_API_URL)


def _get_anthropic_model() -> str:
    return _api_keys.get("_apim_claude_model") or _active_model


def _get_openai_auth_headers(key: str) -> dict:
    # AIFAPIM gateway uses x-api-key; standard OpenAI uses Authorization: Bearer
    if "openai" in _api_urls:
        return {"x-api-key": key}
    return {"Authorization": f"Bearer {key}"}


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


def _history_to_responses_input(history: list) -> list:
    """Convert chat history to OpenAI Responses API input format (input_text/input_image)."""
    msgs = []
    for msg in history:
        role = msg.get("role")
        if role == "system":
            continue
        images = msg.get("images", [])
        if images:
            content = []
            for img_b64 in images:
                content.append({
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{img_b64}",
                })
            if msg.get("content"):
                content.append({"type": "input_text", "text": msg["content"]})
            msgs.append({"role": role, "content": content})
        else:
            msgs.append({"role": role, "content": msg.get("content", "")})
    return msgs


def _parse_responses_text(data: dict) -> str:
    """Extract text from an OpenAI Responses API (non-streaming) response."""
    if "output_text" in data:
        return data["output_text"]
    for item in data.get("output", []):
        for part in item.get("content", []):
            if part.get("type") == "output_text":
                return part.get("text", "")
    return ""


def _parse_locate_json(content: str) -> dict:
    # Strip markdown fences
    if "```" in content:
        lines = content.splitlines()
        inner = []
        in_block = False
        for line in lines:
            if line.strip().startswith("```"):
                in_block = not in_block
                continue
            if in_block or not content.startswith("```"):
                inner.append(line)
        content = "\n".join(inner).strip() if inner else content

    # Extract outermost { ... }
    start = content.find("{")
    end = content.rfind("}")
    if start >= 0 and end > start:
        content = content[start:end + 1]

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Truncated response — try to salvage complete feature objects
        features = []
        depth = 0
        buf = ""
        in_features = False
        fid = 1
        for ch in content:
            buf += ch
            if ch == "{":
                depth += 1
                if '"features"' in buf and not in_features:
                    in_features = True
            elif ch == "}":
                depth -= 1
                if in_features and depth == 1:
                    obj_start = buf.rfind('"id"', 0, len(buf) - 1)
                    if obj_start < 0:
                        obj_start = buf.rfind('"x1"', 0, len(buf) - 1)
                    if obj_start >= 0:
                        fragment = buf[obj_start - 1:]
                        try:
                            feat = json.loads("{" + fragment.lstrip("{"))
                            if any(k in feat for k in ("x1", "id", "label")):
                                feat.setdefault("id", fid)
                                features.append(feat)
                                fid += 1
                        except Exception:
                            pass
        return {"features": features, "total_count": len(features)}

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

LOCATE_SPARSE_XRF_PROMPT = (
    "You are analyzing a sparse XRF (X-ray fluorescence) scan image. "
    "XRF images are characteristically sparse: most of the image is near-zero dark background, "
    "with a small number of isolated bright clusters or particles scattered across the field. "
    "Pixel intensity directly encodes elemental signal strength — brighter means stronger signal. "
    "CRITICAL: Many particles will be very small — just a few pixels across. "
    "Do not overlook tiny bright specks; treat every localized bright point as a candidate particle. "
    "Your task is to identify every bright isolated particle, cluster, or region of elevated signal "
    "against the dark background. Ignore only pure noise (single-pixel random specks at noise floor). "
    "Focus on localized bright spots that stand clearly above the background level, regardless of size. "
    "Each feature may be a single tiny particle, an aggregate, or a compact cluster. "
    "There is no limit on how many features you can return — report ALL of them. "
    "Respond ONLY with valid JSON using this exact schema — no extra text outside the JSON object:\n"
    '{"features": [{"id": 1, "label": "XRF particle", "x1": 0.10, "y1": 0.20, '
    '"x2": 0.45, "y2": 0.60, "description": "bright isolated particle cluster"}], "total_count": 1}\n'
    "IMPORTANT: All coordinates (x1, y1, x2, y2) must be FRACTIONAL values between 0.0 and 1.0, "
    "where (0,0) is top-left and (1,1) is bottom-right. x1 < x2, y1 < y2. "
    "Draw boxes tightly around the bright core of each feature — exclude surrounding dark background. "
    "For very small particles, the box may only span 0.01–0.05 in each dimension — that is fine."
)

LOCATE_XRF_ROI_PROMPT = (
    "You are analyzing a composite XRF (X-ray fluorescence) scan image where three element channels "
    "are encoded as RGB: Red = element 1, Green = element 2, Blue = element 3. "
    "Identify all regions where TWO OR MORE elements show elevated co-localized signal — "
    "these appear as mixed colors (yellow, cyan, magenta, or white) rather than pure red, green, or blue. "
    "These co-localized regions are the areas of interest (AOI) for follow-up high-resolution scanning. "
    "Respond ONLY with valid JSON using this exact schema — no extra text outside the JSON object:\n"
    '{"features": [{"id": 1, "label": "multi-element AOI", "x1": 0.10, "y1": 0.20, '
    '"x2": 0.45, "y2": 0.60, "description": "co-localized signal from elements 1+2"}], "total_count": 1}\n'
    "IMPORTANT: All coordinates (x1, y1, x2, y2) must be FRACTIONAL values between 0.0 and 1.0, "
    "where (0,0) is top-left and (1,1) is bottom-right. x1 < x2, y1 < y2. "
    "Draw boxes tightly around regions of multi-element overlap. Ignore single-element-only regions."
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


def _apply_red_circles_transform(arr: np.ndarray, circle_color=(255, 0, 0)) -> np.ndarray:
    """Normalize → Gaussian blur → Otsu threshold → contours → draw filled circles on white.

    Helps VLMs count/box particles: converts XRF blobs into unambiguous colored circles.
    Input must be uint8 grayscale (2-D). Returns uint8 RGB (3-D).
    circle_color is an (R, G, B) tuple.
    """
    if arr.ndim != 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(arr, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.full((*arr.shape, 3), 255, dtype=np.uint8)  # white background
    for cnt in contours:
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        r = max(r, 3.0) * 1.3
        cv2.circle(out, (int(cx), int(cy)), int(r), circle_color, -1)
    return out


def _load_image_pixmap(path, dilate_ksize: int = 0, normalize: bool = True, grid_overlay: bool = False,
                       upscale: bool = False, colormap: bool = False, red_circles: bool = False,
                       circle_color=(255, 0, 0)):
    """Load any image (including 16/32-bit TIFF) and return (QPixmap, width, height)."""
    arr = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    if arr is None:
        raise ValueError(f"Could not read image: {path}")
    arr = arr.astype(np.float32)
    if normalize:
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
        arr = (arr * 255).astype(np.uint8)
    else:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if upscale:
        h, w = arr.shape[:2]
        if min(h, w) < 1092:
            scale = 1092 / min(h, w)
            arr = cv2.resize(arr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    if dilate_ksize > 0:
        kernel = np.ones((dilate_ksize, dilate_ksize), np.uint8)
        arr = cv2.dilate(arr, kernel)
    if red_circles:
        if arr.ndim != 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        arr = _apply_red_circles_transform(arr, circle_color=circle_color)  # returns RGB
    elif colormap and arr.ndim == 2:
        arr = cv2.applyColorMap(arr, cv2.COLORMAP_INFERNO)
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    elif arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    else:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    if grid_overlay:
        arr = _draw_coordinate_grid(arr)
    h, w = arr.shape[:2]
    qimg = QImage(arr.data, w, h, w * 3, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg), w, h


def _draw_feature_boxes(base_pixmap, features):
    """Draw colored bounding boxes on a copy of base_pixmap. Coords are fractional 0–1."""
    result = base_pixmap.copy()
    painter = QPainter(result)
    painter.setRenderHint(QPainter.Antialiasing)
    pw, ph = result.width(), result.height()

    for i, feat in enumerate(features):
        color = BOX_COLORS[i % len(BOX_COLORS)]
        x1 = int(max(0.0, min(1.0, float(feat.get("x1", 0)))) * pw)
        y1 = int(max(0.0, min(1.0, float(feat.get("y1", 0)))) * ph)
        x2 = int(max(0.0, min(1.0, float(feat.get("x2", 1)))) * pw)
        y2 = int(max(0.0, min(1.0, float(feat.get("y2", 1)))) * ph)
        painter.setPen(QPen(color, 2))
        painter.drawRect(x1, y1, x2 - x1, y2 - y1)

    painter.end()
    return result


def _draw_colocate_boxes(base_pixmap: QPixmap, zones: list) -> QPixmap:
    """Draw black outline boxes for co-location zones on the composite."""
    result = base_pixmap.copy()
    painter = QPainter(result)
    painter.setRenderHint(QPainter.Antialiasing)
    pw, ph = result.width(), result.height()
    pen = QPen(QColor(0, 0, 0), 3)
    pen.setStyle(Qt.SolidLine)
    painter.setPen(pen)
    for zone in zones:
        x1 = int(max(0.0, float(zone.get("x1", 0))) * pw)
        y1 = int(max(0.0, float(zone.get("y1", 0))) * ph)
        x2 = int(min(1.0, float(zone.get("x2", 1))) * pw)
        y2 = int(min(1.0, float(zone.get("y2", 1))) * ph)
        painter.drawRect(x1, y1, x2 - x1, y2 - y1)
        # small zone ID label
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        painter.setFont(QFont("monospace", 8))
        painter.drawText(x1 + 2, y1 + 10, f"Z{zone.get('id', '?')}")
        painter.setPen(pen)
    painter.end()
    return result


def _draw_feature_boxes_multi_color(base_pixmap, feature_groups):
    """Draw boxes from multiple groups, each with its own QColor.

    feature_groups: list of (features_list, QColor) tuples.
    """
    result = base_pixmap.copy()
    painter = QPainter(result)
    painter.setRenderHint(QPainter.Antialiasing)
    pw, ph = result.width(), result.height()
    for features, color in feature_groups:
        painter.setPen(QPen(color, 2))
        for feat in features:
            x1 = int(max(0.0, min(1.0, float(feat.get("x1", 0)))) * pw)
            y1 = int(max(0.0, min(1.0, float(feat.get("y1", 0)))) * ph)
            x2 = int(max(0.0, min(1.0, float(feat.get("x2", 1)))) * pw)
            y2 = int(max(0.0, min(1.0, float(feat.get("y2", 1)))) * ph)
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)
    painter.end()
    return result


# ── image display with mouse coord tracking ───────────────────────────────────

class _ImageDisplay(QLabel):
    """QLabel that emits fractional (0–1) mouse coordinates and supports drag-to-pan."""
    mouse_moved = Signal(float, float)
    mouse_left = Signal()
    zoom_scrolled = Signal(int)  # +1 zoom in, -1 zoom out

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self._drag_start = None
        self._scroll_start = None

    def _frac_pos(self, event):
        pm = self.pixmap()
        if pm is None or pm.isNull():
            return None, None
        ox = (self.width() - pm.width()) / 2
        oy = (self.height() - pm.height()) / 2
        px = event.x() - ox
        py = event.y() - oy
        if 0 <= px <= pm.width() and 0 <= py <= pm.height():
            return px / pm.width(), py / pm.height()
        return None, None

    def _scroll_area(self):
        p = self.parent()
        while p:
            if isinstance(p, QScrollArea):
                return p
            p = p.parent()
        return None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_start = event.globalPos()
            sa = self._scroll_area()
            if sa:
                self._scroll_start = (
                    sa.horizontalScrollBar().value(),
                    sa.verticalScrollBar().value(),
                )
            self.setCursor(Qt.ClosedHandCursor)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_start = None
            self._scroll_start = None
            self.setCursor(Qt.ArrowCursor)

    def mouseMoveEvent(self, event):
        if self._drag_start is not None and self._scroll_start is not None:
            delta = event.globalPos() - self._drag_start
            sa = self._scroll_area()
            if sa:
                sa.horizontalScrollBar().setValue(self._scroll_start[0] - delta.x())
                sa.verticalScrollBar().setValue(self._scroll_start[1] - delta.y())
            return
        fx, fy = self._frac_pos(event)
        if fx is not None:
            self.mouse_moved.emit(fx, fy)
        else:
            self.mouse_left.emit()

    def leaveEvent(self, event):
        self.mouse_left.emit()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta != 0:
            self.zoom_scrolled.emit(1 if delta > 0 else -1)


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
            msg = str(e)
            if provider == "openai":
                msg += f"\n[URL: {_get_openai_url()}  model: {_get_openai_model()}]"
            self.error.emit(msg)

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
        key = _get_key("anthropic") or ""
        sys_msgs = [m for m in self.messages if m.get("role") == "system"]
        system = sys_msgs[0]["content"] if sys_msgs else ""
        payload = json.dumps({
            "model": _get_anthropic_model(),
            "max_tokens": 4096,
            "stream": True,
            "system": system,
            "messages": _history_to_anthropic(self.messages),
        }, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            _get_anthropic_url(), data=payload,
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
        key = _get_key("openai") or ""
        headers = {"Content-Type": "application/json", **_get_openai_auth_headers(key)}
        if "openai" in _api_urls:
            # NSLS-II APIM — Responses API
            sys_msgs = [m for m in self.messages if m.get("role") == "system"]
            payload_dict = {
                "model": _get_openai_model(),
                "input": _history_to_responses_input(self.messages),
            }
            if sys_msgs:
                payload_dict["instructions"] = sys_msgs[0]["content"]
            payload = json.dumps(payload_dict, ensure_ascii=False).encode("utf-8")
            req = urllib.request.Request(_get_openai_url(), data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=600) as resp:
                data = json.loads(resp.read())
            text = _parse_responses_text(data)
            if text:
                self.token.emit(text)
        else:
            # Standard OpenAI — Chat Completions with streaming
            payload = json.dumps({
                "model": _active_model,
                "stream": True,
                "messages": _history_to_openai(self.messages),
            }, ensure_ascii=False).encode("utf-8")
            req = urllib.request.Request(_get_openai_url(), data=payload, headers=headers)
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


def _draw_coordinate_grid(arr: np.ndarray, steps: int = 10) -> np.ndarray:
    """Draw a coordinate grid with fractional labels onto a BGR uint8 image."""
    arr = arr.copy()
    h, w = arr.shape[:2]
    line_color = (160, 160, 160)
    label_bg   = (0, 0, 0)
    label_fg   = (255, 255, 60)
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.35, min(0.55, w / 900))
    thickness  = 1

    for i in range(steps + 1):
        frac = i / steps
        x = int(frac * (w - 1))
        y = int(frac * (h - 1))
        cv2.line(arr, (x, 0), (x, h - 1), line_color, 1)
        cv2.line(arr, (0, y), (w - 1, y), line_color, 1)

        # Only label every 2nd line (0.0, 0.2, 0.4 ...) to avoid crowding
        if i % 2 != 0:
            continue
        label = f"{frac:.1f}"
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)

        # x-axis label: centred above the line, 16px from top
        lx = max(0, min(x - tw // 2, w - tw - 1))
        ly = th + 4
        cv2.rectangle(arr, (lx - 1, 1), (lx + tw + 1, ly + 2), label_bg, -1)
        cv2.putText(arr, label, (lx, ly), font, font_scale, label_fg, thickness, cv2.LINE_AA)

        # y-axis label: 4px to right of left edge, vertically centred on line
        rx = 4
        ry = max(th, min(y + th // 2, h - 2))
        cv2.rectangle(arr, (rx - 1, ry - th - 1), (rx + tw + 1, ry + 2), label_bg, -1)
        cv2.putText(arr, label, (rx, ry), font, font_scale, label_fg, thickness, cv2.LINE_AA)

    return arr


def _encode_image_for_vlm(path, max_dim=_VLM_MAX_DIM, dilate_ksize: int = 0, normalize: bool = True,
                          grid_overlay: bool = False, upscale: bool = False, colormap: bool = False,
                          red_circles: bool = False, circle_color=(255, 0, 0)):
    """Load image, optionally normalize and dilate, downsample to max_dim, return base64-PNG string."""
    arr = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    if arr is None:
        raise ValueError(f"Could not read image: {path}")
    arr = arr.astype(np.float32)
    if normalize:
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
        arr = (arr * 255).astype(np.uint8)
    else:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if upscale:
        h, w = arr.shape[:2]
        if min(h, w) < 1092:
            scale = 1092 / min(h, w)
            arr = cv2.resize(arr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    if dilate_ksize > 0:
        kernel = np.ones((dilate_ksize, dilate_ksize), np.uint8)
        arr = cv2.dilate(arr, kernel)
    if red_circles:
        if arr.ndim != 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        rgb = _apply_red_circles_transform(arr, circle_color=circle_color)
        arr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    else:
        # CLAHE: sharpen local contrast so blob edges are crisper for the VLM
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        if arr.ndim == 2:
            arr = clahe.apply(arr)
            if colormap:
                arr = cv2.applyColorMap(arr, cv2.COLORMAP_INFERNO)
            else:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        else:
            lab = cv2.cvtColor(arr, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            arr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    if grid_overlay:
        arr = _draw_coordinate_grid(arr)
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

    def __init__(self, image_path, dilate_ksize: int = 0, system_prompt_override: str = None, normalize: bool = True,
                 grid_overlay: bool = False, upscale: bool = False, colormap: bool = False, red_circles: bool = False):
        super().__init__()
        self.image_path = image_path
        self.dilate_ksize = dilate_ksize
        self.normalize = normalize
        self.grid_overlay = grid_overlay
        self.upscale = upscale
        self.colormap = colormap
        self.red_circles = red_circles
        self._system_prompt = system_prompt_override or LOCATE_SYSTEM_PROMPT

    def run(self):
        try:
            img_b64 = _encode_image_for_vlm(self.image_path, dilate_ksize=self.dilate_ksize, normalize=self.normalize,
                                             grid_overlay=self.grid_overlay, upscale=self.upscale, colormap=self.colormap,
                                             red_circles=self.red_circles)
            filters = []
            if self.normalize:    filters.append("normalize")
            if self.grid_overlay: filters.append("grid")
            if self.dilate_ksize > 0: filters.append(f"dilate-{self.dilate_ksize}px")
            if self.upscale:      filters.append("upscale")
            if self.colormap:     filters.append("inferno")
            if self.red_circles:  filters.append("red-circles")
            filter_str = ", ".join(filters) if filters else "none"
            print(f"[Feature Locator] Sending request for image '{pathlib.Path(self.image_path).name}' "
                  f"with filters [{filter_str}] using model '{_active_model}'")
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
                {"role": "system", "content": self._system_prompt},
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
        key = _get_key("anthropic") or ""
        payload = json.dumps({
            "model": _get_anthropic_model(),
            "max_tokens": 4096,
            "system": self._system_prompt,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_b64}},
                    {"type": "text", "text": _LOCATE_USER_PROMPT},
                ],
            }],
        }, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            _get_anthropic_url(), data=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": key,
                "anthropic-version": ANTHROPIC_API_VERSION,
            },
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read())
        return _parse_locate_json((data.get("content") or [{}])[0].get("text", "{}"))

    def _query_openai(self, img_b64: str) -> dict:
        key = _get_key("openai") or ""
        headers = {"Content-Type": "application/json", **_get_openai_auth_headers(key)}
        if "openai" in _api_urls:
            payload = json.dumps({
                "model": _get_openai_model(),
                "instructions": self._system_prompt,
                "input": [{"role": "user", "content": [
                    {"type": "input_image", "image_url": f"data:image/png;base64,{img_b64}"},
                    {"type": "input_text", "text": _LOCATE_USER_PROMPT},
                ]}],
            }, ensure_ascii=False).encode("utf-8")
            req = urllib.request.Request(_get_openai_url(), data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=600) as resp:
                data = json.loads(resp.read())
            return _parse_locate_json(_parse_responses_text(data))
        else:
            payload = json.dumps({
                "model": _active_model,
                "messages": [
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                        {"type": "text", "text": _LOCATE_USER_PROMPT},
                    ]},
                ],
            }, ensure_ascii=False).encode("utf-8")
            req = urllib.request.Request(_get_openai_url(), data=payload, headers=headers)
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

    def __init__(self, image_path, dilate_ksize: int = 0, system_prompt_override: str = None, normalize: bool = True,
                 grid_overlay: bool = False, upscale: bool = False, colormap: bool = False, red_circles: bool = False):
        super().__init__()
        self.image_path = image_path
        self.dilate_ksize = dilate_ksize
        self.normalize = normalize
        self.grid_overlay = grid_overlay
        self.upscale = upscale
        self.colormap = colormap
        self.red_circles = red_circles
        self._provider = _get_provider(_active_model)
        self._model = _active_model
        self._key = _get_key(self._provider) or ""
        self._system_prompt = system_prompt_override or LOCATE_SYSTEM_PROMPT

    def run(self):
        try:
            arr = cv2.imread(str(self.image_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
            if arr is None:
                raise ValueError(f"Could not read: {self.image_path}")
            arr = arr.astype(np.float32)
            if self.normalize:
                mn, mx = arr.min(), arr.max()
                if mx > mn:
                    arr = (arr - mn) / (mx - mn)
                arr = (arr * 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            if self.upscale:
                h, w = arr.shape[:2]
                if min(h, w) < 1092:
                    scale = 1092 / min(h, w)
                    arr = cv2.resize(arr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
            if self.dilate_ksize > 0:
                kernel = np.ones((self.dilate_ksize, self.dilate_ksize), np.uint8)
                arr = cv2.dilate(arr, kernel)

            h, w = arr.shape[:2]
            h2, w2 = h // 2, w // 2

            filters = []
            if self.normalize:    filters.append("normalize")
            if self.grid_overlay: filters.append("grid")
            if self.dilate_ksize > 0: filters.append(f"dilate-{self.dilate_ksize}px")
            if self.upscale:      filters.append("upscale")
            if self.colormap:     filters.append("inferno")
            if self.red_circles:  filters.append("red-circles")
            filter_str = ", ".join(filters) if filters else "none"
            print(f"[Feature Locator] Sending tile request for image '{pathlib.Path(self.image_path).name}' "
                  f"with filters [{filter_str}] using model '{_active_model}' (4 quadrants)")

            def _prep_quad(q):
                return _draw_coordinate_grid(q) if self.grid_overlay else q

            quads = [
                ("TL", _prep_quad(arr[0:h2,  0:w2].copy()),  0.0,    0.0,    w2/w,       h2/h),
                ("TR", _prep_quad(arr[0:h2,  w2:w].copy()),  w2/w,   0.0,    (w-w2)/w,   h2/h),
                ("BL", _prep_quad(arr[h2:h,  0:w2].copy()),  0.0,    h2/h,   w2/w,       (h-h2)/h),
                ("BR", _prep_quad(arr[h2:h,  w2:w].copy()),  w2/w,   h2/h,   (w-w2)/w,   (h-h2)/h),
            ]

            all_features = []
            fid = 1
            for qi, (name, crop, xoff, yoff, xscale, yscale) in enumerate(quads):
                self.progress.emit(f"Quadrant {qi + 1}/4 ({name})...")
                try:
                    img_b64 = self._encode_crop(crop, colormap=self.colormap, red_circles=self.red_circles)
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
    def _encode_crop(crop: np.ndarray, colormap: bool = False, red_circles: bool = False) -> str:
        if red_circles:
            gray = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            rgb = _apply_red_circles_transform(gray)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        elif crop.ndim == 2:
            if colormap:
                bgr = cv2.applyColorMap(crop, cv2.COLORMAP_INFERNO)
            else:
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
                {"role": "system", "content": self._system_prompt},
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
            "model": _get_anthropic_model(),
            "max_tokens": 4096,
            "system": self._system_prompt,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_b64}},
                    {"type": "text", "text": _LOCATE_USER_PROMPT},
                ],
            }],
        }, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            _get_anthropic_url(), data=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self._key,
                "anthropic-version": ANTHROPIC_API_VERSION,
            },
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read())
        return _parse_locate_json((data.get("content") or [{}])[0].get("text", "{}")).get("features", [])

    def _query_openai(self, img_b64: str) -> list:
        headers = {"Content-Type": "application/json", **_get_openai_auth_headers(self._key)}
        if "openai" in _api_urls:
            payload = json.dumps({
                "model": _get_openai_model(),
                "instructions": self._system_prompt,
                "input": [{"role": "user", "content": [
                    {"type": "input_image", "image_url": f"data:image/png;base64,{img_b64}"},
                    {"type": "input_text", "text": _LOCATE_USER_PROMPT},
                ]}],
            }, ensure_ascii=False).encode("utf-8")
            req = urllib.request.Request(_get_openai_url(), data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=600) as resp:
                data = json.loads(resp.read())
            return _parse_locate_json(_parse_responses_text(data)).get("features", [])
        else:
            payload = json.dumps({
                "model": self._model,
                "messages": [
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                        {"type": "text", "text": _LOCATE_USER_PROMPT},
                    ]},
                ],
            }, ensure_ascii=False).encode("utf-8")
            req = urllib.request.Request(_get_openai_url(), data=payload, headers=headers)
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
        self._set_ready()

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

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

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

    def refresh_model(self):
        self._history.clear()
        self._set_ready()

    def _set_ready(self):
        self.status_label.setText(f"Model: {_active_model}")
        self.status_label.setStyleSheet("color: #27ae60; padding: 5px;")
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


_XRF_ELEM_COLORS = [
    (QColor(220, 60, 60),  (255, 0,   0)),    # Red element
    (QColor(60,  200, 60), (0,   255, 0)),    # Green element
    (QColor(60,  120, 220),(0,   0,   255)),  # Blue element
]

_MULTI_ELEM_COLORS = [
    (QColor(220, 60,  60),  (255, 0,   0)),    # Red
    (QColor(60,  200, 60),  (0,   255, 0)),    # Green
    (QColor(60,  120, 220), (0,   0,   255)),  # Blue
    (QColor(230, 200, 50),  (255, 220, 0)),    # Yellow
    (QColor(160, 60,  220), (180, 0,   255)),  # Purple
    (QColor(230, 130, 40),  (255, 140, 0)),    # Orange
    (QColor(50,  210, 210), (0,   220, 220)),  # Cyan
    (QColor(240, 100, 180), (255, 100, 180)),  # Pink
    (QColor(100, 220, 80),  (100, 255, 60)),   # Lime
    (QColor(50,  170, 180), (0,   180, 200)),  # Teal
]


class _LocateXRFTripleThread(QThread):
    """Run 3 independent VLM calls (one per element) and combine into a single result dict."""
    result = Signal(dict)
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, paths, normalize=True, upscale=True, circles=False,
                 system_prompt_override=None):
        super().__init__()
        self.paths = paths  # [r_path, g_path, b_path]
        self.normalize = normalize
        self.upscale = upscale
        self.circles = circles
        self._system_prompt = system_prompt_override or LOCATE_SPARSE_XRF_PROMPT
        self._provider = _get_provider(_active_model)
        self._model = _active_model
        self._key = _get_key(self._provider) or ""

    def run(self):
        try:
            keys = ["r_features", "g_features", "b_features"]
            labels = ["Red (Element 1)", "Green (Element 2)", "Blue (Element 3)"]
            circle_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
            combined = {}
            total = 0
            for i, (path, key, label, cc) in enumerate(
                    zip(self.paths, keys, labels, circle_colors)):
                self.progress.emit(f"Analyzing {label}... ({i+1}/3)")
                print(f"[Feature Locator XRF] Sending request for '{pathlib.Path(path).name}' "
                      f"({'circles' if self.circles else 'grayscale'}) using '{_active_model}'")
                img_b64 = _encode_image_for_vlm(
                    path, normalize=self.normalize, upscale=self.upscale,
                    red_circles=self.circles, circle_color=cc)
                features = self._query(img_b64)
                combined[key] = features
                total += len(features)
                self.progress.emit(f"{label} — {len(features)} feature(s) found.")
            combined["total_count"] = total
            self.result.emit(combined)
        except Exception as e:
            self.error.emit(str(e))

    def _query(self, img_b64: str) -> list:
        if self._provider == "anthropic":
            return self._query_anthropic(img_b64)
        if self._provider == "openai":
            return self._query_openai(img_b64)
        return self._query_ollama(img_b64)

    def _query_anthropic(self, img_b64: str) -> list:
        payload = json.dumps({
            "model": _get_anthropic_model(),
            "max_tokens": 4096,
            "system": self._system_prompt,
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64",
                                              "media_type": "image/png", "data": img_b64}},
                {"type": "text", "text": _LOCATE_USER_PROMPT},
            ]}],
        }, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            _get_anthropic_url(), data=payload,
            headers={"Content-Type": "application/json",
                     "x-api-key": self._key,
                     "anthropic-version": ANTHROPIC_API_VERSION})
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read())
        return _parse_locate_json((data.get("content") or [{}])[0].get("text", "{}")).get("features", [])

    def _query_openai(self, img_b64: str) -> list:
        headers = {"Content-Type": "application/json", **_get_openai_auth_headers(self._key)}
        if "openai" in _api_urls:
            payload = json.dumps({
                "model": _get_openai_model(),
                "instructions": self._system_prompt,
                "input": [{"role": "user", "content": [
                    {"type": "input_image", "image_url": f"data:image/png;base64,{img_b64}"},
                    {"type": "input_text", "text": _LOCATE_USER_PROMPT},
                ]}],
            }, ensure_ascii=False).encode("utf-8")
            req = urllib.request.Request(_get_openai_url(), data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=600) as resp:
                data = json.loads(resp.read())
            return _parse_locate_json(_parse_responses_text(data)).get("features", [])
        else:
            payload = json.dumps({
                "model": self._model,
                "messages": [
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                        {"type": "text", "text": _LOCATE_USER_PROMPT},
                    ]},
                ],
            }, ensure_ascii=False).encode("utf-8")
            req = urllib.request.Request(_get_openai_url(), data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=600) as resp:
                data = json.loads(resp.read())
            return _parse_locate_json(
                data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            ).get("features", [])

    def _query_ollama(self, img_b64: str) -> list:
        payload = json.dumps({
            "model": self._model,
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": _LOCATE_USER_PROMPT, "images": [img_b64]},
            ],
            "stream": False,
            "format": "json",
        }, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            OLLAMA_URL, data=payload,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as resp:
            raw = json.loads(resp.read())
        return _parse_locate_json(raw.get("message", {}).get("content", "{}")).get("features", [])


class _LocateXRFMultiThread(QThread):
    """Run N independent VLM calls (one per element) and combine into a single result dict."""
    result = Signal(dict)
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, paths, normalize=True, upscale=True, circles=False,
                 system_prompt_override=None):
        super().__init__()
        self.paths = paths
        self.normalize = normalize
        self.upscale = upscale
        self.circles = circles
        self._system_prompt = system_prompt_override or LOCATE_SPARSE_XRF_PROMPT
        self._provider = _get_provider(_active_model)
        self._model = _active_model
        self._key = _get_key(self._provider) or ""

    def run(self):
        try:
            N = len(self.paths)
            elem_features = []
            total = 0
            for i, path in enumerate(self.paths):
                color_entry = _MULTI_ELEM_COLORS[i % len(_MULTI_ELEM_COLORS)]
                cc = color_entry[1]
                stem = pathlib.Path(path).stem
                elem_name = stem.split("_")[-1]
                self.progress.emit(f"Analyzing {elem_name} ({i+1}/{N})...")
                print(f"[Feature Locator Multi] Sending '{pathlib.Path(path).name}' using '{_active_model}'")
                img_b64 = _encode_image_for_vlm(
                    path, normalize=self.normalize, upscale=self.upscale,
                    red_circles=self.circles, circle_color=cc)
                features = self._query(img_b64)
                elem_features.append(features)
                total += len(features)
                self.progress.emit(f"{elem_name} — {len(features)} feature(s) found.")
            self.result.emit({"elem_features": elem_features, "total_count": total})
        except Exception as e:
            self.error.emit(str(e))

    def _query(self, img_b64: str) -> list:
        if self._provider == "anthropic":
            return self._query_anthropic(img_b64)
        if self._provider == "openai":
            return self._query_openai(img_b64)
        return self._query_ollama(img_b64)

    def _query_anthropic(self, img_b64: str) -> list:
        key = _get_key("anthropic") or ""
        payload = json.dumps({
            "model": _get_anthropic_model(),
            "max_tokens": 4096,
            "system": self._system_prompt,
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64",
                                              "media_type": "image/png", "data": img_b64}},
                {"type": "text", "text": _LOCATE_USER_PROMPT},
            ]}],
        }, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            _get_anthropic_url(), data=payload,
            headers={"Content-Type": "application/json",
                     "x-api-key": key,
                     "anthropic-version": ANTHROPIC_API_VERSION})
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read())
        return _parse_locate_json((data.get("content") or [{}])[0].get("text", "{}")).get("features", [])

    def _query_openai(self, img_b64: str) -> list:
        key = _get_key("openai") or ""
        headers = {"Content-Type": "application/json", **_get_openai_auth_headers(key)}
        if "openai" in _api_urls:
            payload = json.dumps({
                "model": _get_openai_model(),
                "instructions": self._system_prompt,
                "input": [{"role": "user", "content": [
                    {"type": "input_image", "image_url": f"data:image/png;base64,{img_b64}"},
                    {"type": "input_text", "text": _LOCATE_USER_PROMPT},
                ]}],
            }, ensure_ascii=False).encode("utf-8")
            req = urllib.request.Request(_get_openai_url(), data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=600) as resp:
                data = json.loads(resp.read())
            return _parse_locate_json(_parse_responses_text(data)).get("features", [])
        else:
            payload = json.dumps({
                "model": self._model,
                "messages": [
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                        {"type": "text", "text": _LOCATE_USER_PROMPT},
                    ]},
                ],
            }, ensure_ascii=False).encode("utf-8")
            req = urllib.request.Request(_get_openai_url(), data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=600) as resp:
                data = json.loads(resp.read())
            return _parse_locate_json(
                data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            ).get("features", [])

    def _query_ollama(self, img_b64: str) -> list:
        payload = json.dumps({
            "model": self._model,
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": _LOCATE_USER_PROMPT, "images": [img_b64]},
            ],
            "stream": False,
            "format": "json",
        }, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            OLLAMA_URL, data=payload,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as resp:
            raw = json.loads(resp.read())
        return _parse_locate_json(raw.get("message", {}).get("content", "{}")).get("features", [])


def _box_intersect(a: dict, b: dict) -> dict | None:
    """Return intersection bbox dict, or None if no overlap."""
    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])
    if ix2 > ix1 and iy2 > iy1:
        return {"x1": ix1, "y1": iy1, "x2": ix2, "y2": iy2}
    return None


class _LocateXRFColocateThread(QThread):
    """Compute element co-location zones and send to LLM for material suggestions."""
    result = Signal(dict)
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, triple_data: dict, elem_labels: list[str],
                 system_prompt_override: str | None = None):
        super().__init__()
        self._data = triple_data          # the full triple features dict
        self._elem_labels = elem_labels   # ["Fe", "Cr", "Ni"] (user-entered names)
        self._provider = _get_provider(_active_model)
        self._model = _active_model
        self._key = _get_key(self._provider) or ""

    def run(self):
        try:
            r_feats = self._data.get("r_features", [])
            g_feats = self._data.get("g_features", [])
            b_feats = self._data.get("b_features", [])
            # elem_labels already contain the color e.g. "Ca (Red)" — use as-is
            names = [lbl if lbl.strip() else c
                     for lbl, c in zip(self._elem_labels, ["Red", "Green", "Blue"])]

            self.progress.emit("Computing co-location zones...")
            zones = self._compute_zones(r_feats, g_feats, b_feats, names)
            if not zones:
                self.result.emit({"overlap_zones": [], "total_zones": 0,
                                  "elements": dict(zip(["r", "g", "b"], names))})
                return

            self.progress.emit(f"{len(zones)} overlap zone(s) found — querying {_active_model}...")
            print(f"[XRF Co-locate] {len(zones)} zone(s). Sending to '{_active_model}' for material analysis...")
            suggestions = self._query_llm(zones, names)

            # Merge LLM suggestions into zones
            for zone, sug in zip(zones, suggestions):
                zone["llm_suggestion"] = sug.get("suggestion", "")
                zone["reasoning"] = sug.get("reasoning", "")

            result = {
                "mode": "triple",
                "method": "geometric",
                "elements": {"r": names[0], "g": names[1], "b": names[2]},
                "model": _active_model,
                "overlap_zones": zones,
                "total_zones": len(zones),
            }
            self.result.emit(result)
        except Exception as e:
            self.error.emit(str(e))

    @staticmethod
    def _compute_zones(r_feats, g_feats, b_feats, names) -> list:
        zones = []
        zone_id = 1

        def _add(elems, elem_names, box):
            nonlocal zone_id
            zones.append({
                "id": zone_id,
                "element_codes": elems,
                "elements": elem_names,
                "x1": round(box["x1"], 4), "y1": round(box["y1"], 4),
                "x2": round(box["x2"], 4), "y2": round(box["y2"], 4),
                "llm_suggestion": "", "reasoning": "",
            })
            zone_id += 1

        # R+G pairwise
        for rf in r_feats:
            for gf in g_feats:
                box = _box_intersect(rf, gf)
                if box:
                    # Check if B also overlaps the intersection
                    triple = False
                    for bf in b_feats:
                        tri_box = _box_intersect(box, bf)
                        if tri_box:
                            _add(["R", "G", "B"], [names[0], names[1], names[2]], tri_box)
                            triple = True
                            break
                    if not triple:
                        _add(["R", "G"], [names[0], names[1]], box)

        # R+B pairwise (no G)
        for rf in r_feats:
            for bf in b_feats:
                box = _box_intersect(rf, bf)
                if box:
                    has_g = any(_box_intersect(box, gf) for gf in g_feats)
                    if not has_g:
                        _add(["R", "B"], [names[0], names[2]], box)

        # G+B pairwise (no R)
        for gf in g_feats:
            for bf in b_feats:
                box = _box_intersect(gf, bf)
                if box:
                    has_r = any(_box_intersect(box, rf) for rf in r_feats)
                    if not has_r:
                        _add(["G", "B"], [names[1], names[2]], box)

        return zones

    def _query_llm(self, zones: list, names: list) -> list:
        zone_lines = []
        for z in zones:
            elem_str = " + ".join(z["elements"])
            zone_lines.append(
                f"  Zone {z['id']}: {elem_str} co-located at "
                f"({z['x1']:.3f},{z['y1']:.3f})→({z['x2']:.3f},{z['y2']:.3f})"
            )
        prompt = (
            "You are an XRF (X-ray fluorescence) materials scientist. "
            f"The scan has 3 elements: Element 1 = {names[0]}, "
            f"Element 2 = {names[1]}, Element 3 = {names[2]}.\n\n"
            "The following co-location zones were detected where multiple element signals overlap:\n"
            + "\n".join(zone_lines) + "\n\n"
            "For each zone, identify the most likely material, compound, alloy, or phase that "
            "explains why these elements co-exist in the same region. Be concise but specific.\n\n"
            "Respond ONLY with valid JSON:\n"
            '{"zones": [{"id": 1, "suggestion": "stainless steel", '
            '"reasoning": "Fe+Cr+Ni is a classic stainless steel signature"}]}'
        )
        if self._provider == "anthropic":
            return self._query_anthropic(prompt)
        if self._provider == "openai":
            return self._query_openai(prompt)
        return self._query_ollama(prompt)

    @staticmethod
    def _safe_parse_zones(text: str) -> list:
        start = text.find("{")
        if start < 0:
            return []
        fragment = text[start:]
        try:
            return json.loads(fragment).get("zones", [])
        except json.JSONDecodeError:
            # Truncated response — try to salvage complete zone objects
            zones = []
            depth = 0
            buf = ""
            in_zones = False
            for ch in fragment:
                buf += ch
                if ch == "{":
                    depth += 1
                    if '"zones"' in buf and not in_zones:
                        in_zones = True
                elif ch == "}":
                    depth -= 1
                    if in_zones and depth == 1:
                        zone_text = buf[buf.rfind('"id"', 0, len(buf) - 1) - 1:]
                        try:
                            z = json.loads("{" + zone_text.lstrip("{"))
                            if "id" in z:
                                zones.append(z)
                        except Exception:
                            pass
            return zones

    def _query_anthropic(self, prompt: str) -> list:
        payload = json.dumps({
            "model": _get_anthropic_model(),
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        }, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            _get_anthropic_url(), data=payload,
            headers={"Content-Type": "application/json",
                     "x-api-key": self._key,
                     "anthropic-version": ANTHROPIC_API_VERSION})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        text = (data.get("content") or [{}])[0].get("text", "{}")
        return self._safe_parse_zones(text)

    def _query_openai(self, prompt: str) -> list:
        headers = {"Content-Type": "application/json", **_get_openai_auth_headers(self._key)}
        payload = json.dumps({
            "model": _get_openai_model() if "openai" in _api_urls else self._model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        }, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(_get_openai_url(), data=payload, headers=headers)
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
        return self._safe_parse_zones(text)

    def _query_ollama(self, prompt: str) -> list:
        payload = json.dumps({
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False, "format": "json",
        }, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(OLLAMA_URL, data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = json.loads(resp.read())
        text = raw.get("message", {}).get("content", "{}")
        return self._safe_parse_zones(text)


class _LocateXRFMultiColocateThread(QThread):
    """Generalized N-element co-location analysis."""
    result = Signal(dict)
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, elem_features: list, elem_names: list):
        super().__init__()
        self._elem_features = elem_features
        self._elem_names = elem_names
        self._provider = _get_provider(_active_model)
        self._model = _active_model
        self._key = _get_key(self._provider) or ""

    def run(self):
        try:
            self.progress.emit("Computing multi-element co-location zones...")
            zones = self._compute_zones(self._elem_features, self._elem_names)
            if not zones:
                self.result.emit({"overlap_zones": [], "total_zones": 0,
                                  "elements": self._elem_names})
                return
            self.progress.emit(f"{len(zones)} zone(s) found — querying {_active_model}...")
            print(f"[Multi Co-locate] {len(zones)} zone(s). Sending to '{_active_model}' for material analysis...")
            suggestions = self._query_llm(zones, self._elem_names)
            for zone, sug in zip(zones, suggestions):
                zone["llm_suggestion"] = sug.get("suggestion", "")
                zone["reasoning"] = sug.get("reasoning", "")
            self.result.emit({
                "mode": "multi",
                "method": "geometric",
                "elements": self._elem_names,
                "model": _active_model,
                "overlap_zones": zones,
                "total_zones": len(zones),
            })
        except Exception as e:
            self.error.emit(str(e))

    @staticmethod
    def _compute_zones(elem_features_list, names, max_overlap=6):
        """Pairwise-seed approach: O(N^2 * F^2) instead of O(F^N).

        For every pair of features from different elements that overlap, seed a
        zone and greedily try to extend it with features from remaining elements.
        Each unique (element-set, intersection-box) is recorded once.
        """
        N = len(elem_features_list)
        zones = []
        zone_id = 1
        seen = set()  # frozenset(elem_indices) → list of boxes, for dedup

        for i in range(N):
            for j in range(i + 1, N):
                for fi in elem_features_list[i]:
                    for fj in elem_features_list[j]:
                        box = _box_intersect(fi, fj)
                        if box is None:
                            continue
                        # Greedily extend to more elements
                        active_indices = [i, j]
                        active_box = box
                        for k in range(N):
                            if k in active_indices:
                                continue
                            if len(active_indices) >= max_overlap:
                                break
                            for fk in elem_features_list[k]:
                                ext = _box_intersect(active_box, fk)
                                if ext is not None:
                                    active_indices.append(k)
                                    active_box = ext
                                    break
                        key = (frozenset(active_indices),
                               round(active_box["x1"], 3), round(active_box["y1"], 3),
                               round(active_box["x2"], 3), round(active_box["y2"], 3))
                        if key in seen:
                            continue
                        seen.add(key)
                        sorted_idx = sorted(active_indices)
                        zones.append({
                            "id": zone_id,
                            "elements": [names[idx] for idx in sorted_idx],
                            "x1": round(active_box["x1"], 4),
                            "y1": round(active_box["y1"], 4),
                            "x2": round(active_box["x2"], 4),
                            "y2": round(active_box["y2"], 4),
                            "llm_suggestion": "", "reasoning": "",
                        })
                        zone_id += 1
        return zones

    def _query_llm(self, zones, names):
        zone_lines = []
        for z in zones:
            elem_str = " + ".join(z["elements"])
            zone_lines.append(
                f"  Zone {z['id']}: {elem_str} co-located at "
                f"({z['x1']:.3f},{z['y1']:.3f})→({z['x2']:.3f},{z['y2']:.3f})"
            )
        elem_list = ", ".join(f"Element {i+1} = {n}" for i, n in enumerate(names))
        prompt = (
            "You are an XRF (X-ray fluorescence) materials scientist. "
            f"The scan has {len(names)} elements: {elem_list}.\n\n"
            "The following co-location zones were detected where multiple element signals overlap:\n"
            + "\n".join(zone_lines) + "\n\n"
            "For each zone, identify the most likely material, compound, alloy, or phase that "
            "explains why these elements co-exist in the same region. Be concise but specific.\n\n"
            "Respond ONLY with valid JSON:\n"
            '{"zones": [{"id": 1, "suggestion": "stainless steel", '
            '"reasoning": "Fe+Cr+Ni is a classic stainless steel signature"}]}'
        )
        if self._provider == "anthropic":
            return self._query_anthropic(prompt)
        if self._provider == "openai":
            return self._query_openai(prompt)
        return self._query_ollama(prompt)

    def _query_anthropic(self, prompt: str) -> list:
        key = _get_key("anthropic") or ""
        payload = json.dumps({
            "model": _get_anthropic_model(),
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        }, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            _get_anthropic_url(), data=payload,
            headers={"Content-Type": "application/json",
                     "x-api-key": key,
                     "anthropic-version": ANTHROPIC_API_VERSION})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        text = (data.get("content") or [{}])[0].get("text", "{}")
        return _LocateXRFColocateThread._safe_parse_zones(text)

    def _query_openai(self, prompt: str) -> list:
        key = _get_key("openai") or ""
        headers = {"Content-Type": "application/json", **_get_openai_auth_headers(key)}
        payload = json.dumps({
            "model": _get_openai_model() if "openai" in _api_urls else self._model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        }, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(_get_openai_url(), data=payload, headers=headers)
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
        return _LocateXRFColocateThread._safe_parse_zones(text)

    def _query_ollama(self, prompt: str) -> list:
        payload = json.dumps({
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False, "format": "json",
        }, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(OLLAMA_URL, data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = json.loads(resp.read())
        text = raw.get("message", {}).get("content", "{}")
        return _LocateXRFColocateThread._safe_parse_zones(text)


class _LocateLLMColocateThread(QThread):
    """Send full VLM features JSON to LLM; let it reason spatially and return co-location zones."""
    result = Signal(dict)
    error = Signal(str)
    progress = Signal(str)

    _SYSTEM = (
        "You are an XRF (X-ray fluorescence) materials scientist. "
        "You will receive VLM-detected features for multiple element scans. "
        "Your task is to identify co-location zones — regions where 2 or more elements "
        "overlap or are immediately adjacent — and identify the most likely material or phase. "
        "Return ONLY valid JSON. No text outside the JSON object."
    )

    _SCHEMA = (
        '{"zones": [{'
        '"id": 1, '
        '"elements": ["Fe", "Cr"], '
        '"x1": 0.10, "y1": 0.20, "x2": 0.35, "y2": 0.45, '
        '"suggestion": "ferrochrome alloy", '
        '"reasoning": "Fe+Cr co-location is characteristic of ferrochrome"'
        '}]}'
    )

    def __init__(self, elem_features: list, elem_names: list):
        super().__init__()
        self._elem_features = elem_features
        self._elem_names = elem_names
        self._provider = _get_provider(_active_model)
        self._model = _active_model
        self._key = _get_key(self._provider) or ""

    def run(self):
        try:
            self.progress.emit("Sending VLM results to LLM for spatial co-location analysis...")
            prompt = self._build_prompt()
            print(f"[LLM Co-locate] Sending {sum(len(f) for f in self._elem_features)} total features "
                  f"across {len(self._elem_names)} elements to '{_active_model}'")
            if self._provider == "anthropic":
                zones = self._query_anthropic(prompt)
            elif self._provider == "openai":
                zones = self._query_openai(prompt)
            else:
                zones = self._query_ollama(prompt)
            for i, z in enumerate(zones):
                z["id"] = i + 1
                z.setdefault("llm_suggestion", z.pop("suggestion", ""))
                z.setdefault("reasoning", "")
                z.pop("element_codes", None)
            self.result.emit({
                "elements": self._elem_names,
                "model": _active_model,
                "method": "llm",
                "overlap_zones": zones,
                "total_zones": len(zones),
            })
        except Exception as e:
            self.error.emit(str(e))

    def _build_prompt(self) -> str:
        lines = [
            f"You have VLM detection results for {len(self._elem_names)} XRF element scans.",
            "Bounding boxes are fractional coordinates (0.0–1.0), (0,0)=top-left, (1,1)=bottom-right.",
            "",
        ]
        for i, (name, feats) in enumerate(zip(self._elem_names, self._elem_features)):
            lines.append(f"Element {i+1} = {name}:  ({len(feats)} feature(s))")
            for f in feats:
                fid = f.get("id", "?")
                label = f.get("label", "")
                desc = f.get("description", "")
                x1, y1 = f.get("x1", 0), f.get("y1", 0)
                x2, y2 = f.get("x2", 0), f.get("y2", 0)
                lines.append(f"  #{fid} [{label}] ({x1:.3f},{y1:.3f})→({x2:.3f},{y2:.3f})"
                             + (f'  "{desc}"' if desc else ""))
            lines.append("")
        lines += [
            "TASK: Find ALL regions where 2 or more elements have overlapping or immediately adjacent features.",
            "For each co-location zone:",
            "  1. Return the INTERSECTION bounding box — tightest box covering where the elements touch/overlap.",
            "     If features are adjacent but not overlapping, use the gap-bridging box.",
            "  2. List which element names are present.",
            "  3. Identify the most likely material, compound, alloy, or mineral phase.",
            "",
            f"Use this exact JSON schema:\n{self._SCHEMA}",
        ]
        return "\n".join(lines)

    def _query_anthropic(self, prompt: str) -> list:
        payload = json.dumps({
            "model": _get_anthropic_model(),
            "max_tokens": 8096,
            "system": self._SYSTEM,
            "messages": [{"role": "user", "content": prompt}],
        }, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            _get_anthropic_url(), data=payload,
            headers={"Content-Type": "application/json",
                     "x-api-key": self._key,
                     "anthropic-version": ANTHROPIC_API_VERSION})
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read())
        text = (data.get("content") or [{}])[0].get("text", "{}")
        return self._parse(text)

    def _query_openai(self, prompt: str) -> list:
        headers = {"Content-Type": "application/json", **_get_openai_auth_headers(self._key)}
        payload = json.dumps({
            "model": _get_openai_model() if "openai" in _api_urls else self._model,
            "max_tokens": 8096,
            "messages": [
                {"role": "system", "content": self._SYSTEM},
                {"role": "user", "content": prompt},
            ],
        }, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(_get_openai_url(), data=payload, headers=headers)
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read())
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
        return self._parse(text)

    def _query_ollama(self, prompt: str) -> list:
        payload = json.dumps({
            "model": self._model,
            "messages": [
                {"role": "system", "content": self._SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "stream": False, "format": "json",
        }, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(OLLAMA_URL, data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as resp:
            raw = json.loads(resp.read())
        text = raw.get("message", {}).get("content", "{}")
        return self._parse(text)

    @staticmethod
    def _parse(text: str) -> list:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start:end + 1]
        try:
            return json.loads(text).get("zones", [])
        except json.JSONDecodeError:
            return _LocateXRFColocateThread._safe_parse_zones(text)


class _LocateTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._image_path = None
        self._locate_thread = None
        self._orig_pixmap = None
        self._dilated_pixmap = None
        self._annotated_pixmap = None
        self._features = []
        self._zoom = 1.0
        self._xrf_zoom = 1.0
        self._xrf_base_w = 0
        self._xrf_base_h = 0
        self._xrf_r_features = []
        self._xrf_g_features = []
        self._xrf_b_features = []
        self._xrf_colocate_zones = []
        self._xrf_last_triple_json = None  # path to last _xrf_triple_features.json
        self._boxes_visible = True         # master toggle (all boxes)
        self._xrf_show_r = True
        self._xrf_show_g = True
        self._xrf_show_b = True
        self._xrf_show_zones = True
        self._model_ok = True
        self._xrf_paths = [None, None, None]
        self._xrf_elem_names = ["Element 1", "Element 2", "Element 3"]
        self._multi_dir_path = None
        self._multi_tiff_paths = []
        self._multi_selected_paths = []
        self._multi_elem_features = []
        self._multi_elem_names = []
        self._multi_colocate_zones = []
        self._multi_composite_pixmap = None
        self._multi_show_flags = []
        self._multi_show_zones = True
        self._multi_mini_labels = []
        self._multi_checkboxes = []
        self._multi_colocate_thread = None
        self._multi_zoom = 1.0
        self._multi_base_w = 0
        self._multi_base_h = 0
        self._multi_circles_composite_pixmap = None
        self._multi_view_circles = False
        self._xrf_circles_composite_pixmap = None
        self._xrf_view_circles = False
        self._init_ui()
        self._check_model()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignTop)

        # Model + prompt header
        header = QGroupBox()
        header.setStyleSheet("QGroupBox { border: 1px solid #555; border-radius: 4px; margin-top: 2px; padding: 4px; }")
        header_layout = QVBoxLayout(header)
        header_layout.setSpacing(2)
        self._header_model_lbl = QLabel(f"Model: <b>{_active_model}</b>")
        self._header_model_lbl.setStyleSheet("font-size: 12px;")
        prompt_preview = LOCATE_SYSTEM_PROMPT[:120].rstrip() + "…"
        self._header_prompt_lbl = QLabel(f"Prompt: {prompt_preview}")
        self._header_prompt_lbl.setWordWrap(True)
        self._header_prompt_lbl.setStyleSheet("font-size: 10px; color: #aaa;")
        self._header_prompt_lbl.setToolTip(LOCATE_SYSTEM_PROMPT)
        header_layout.addWidget(self._header_model_lbl)
        header_layout.addWidget(self._header_prompt_lbl)
        layout.addWidget(header)

        self.model_status = QLabel("")
        self.model_status.setWordWrap(True)
        layout.addWidget(self.model_status)

        # Mode selector
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Single Element", "Triple Element (3 TIFFs)", "Multi Element"])
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self._mode_combo)
        mode_row.addStretch()
        layout.addLayout(mode_row)

        # Prompt preset selector (Single Element mode only)
        self._preset_row = QHBoxLayout()
        self._preset_row.addWidget(QLabel("Prompt:"))
        self._preset_combo = QComboBox()
        self._preset_combo.addItems([
            "Standard — detect all features",
            "Sparse XRF aware — particle detection",
        ])
        self._PRESETS = [LOCATE_SYSTEM_PROMPT, LOCATE_SPARSE_XRF_PROMPT]
        self._preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        self._preset_row.addWidget(self._preset_combo, 1)
        view_prompt_btn = QPushButton("View")
        view_prompt_btn.setFixedWidth(50)
        view_prompt_btn.setToolTip("View full prompt text")
        view_prompt_btn.clicked.connect(self._show_prompt_popup)
        self._preset_row.addWidget(view_prompt_btn)
        preset_container = QWidget()
        preset_container.setLayout(self._preset_row)
        self._preset_widget = preset_container
        layout.addWidget(self._preset_widget)

        self._active_prompt = LOCATE_SYSTEM_PROMPT

        # Floating hover tooltip
        self._hover_tooltip = QLabel(self)
        self._hover_tooltip.setWindowFlags(Qt.ToolTip)
        self._hover_tooltip.setStyleSheet(
            "QLabel { background: #1a1a1a; color: #f0f0f0; border: 1px solid #555;"
            " border-radius: 6px; padding: 8px; font-size: 13px; }"
        )
        self._hover_tooltip.hide()

        # --- Single Element mode controls ---
        self._single_widget = QWidget()
        single_layout = QVBoxLayout(self._single_widget)
        single_layout.setContentsMargins(0, 0, 0, 0)

        pick_row = QHBoxLayout()
        self.pick_btn = QPushButton("Pick Image...")
        self.pick_btn.clicked.connect(self._pick_image)
        pick_row.addWidget(self.pick_btn)
        self.image_name_label = QLabel("No image selected")
        self.image_name_label.setStyleSheet("color: #888;")
        pick_row.addWidget(self.image_name_label, 1)
        single_layout.addLayout(pick_row)

        norm_row = QHBoxLayout()
        self._normalize_check = QCheckBox("Normalize image (0–255)")
        self._normalize_check.setChecked(True)
        self._normalize_check.setToolTip(
            "Stretch pixel values to fill the full 0–255 range before sending to the VLM.\n"
            "Recommended for XRF images which often have a narrow raw intensity range."
        )
        self._normalize_check.toggled.connect(self._on_image_options_changed)
        norm_row.addWidget(self._normalize_check)
        norm_row.addStretch()
        single_layout.addLayout(norm_row)

        grid_row = QHBoxLayout()
        self._grid_check = QCheckBox("Grid overlay (coordinate reference)")
        self._grid_check.setChecked(False)
        self._grid_check.setToolTip(
            "Draw a 10×10 coordinate grid with labels on the image before sending to the VLM.\n"
            "Gives the model visual reference points to improve bounding box accuracy."
        )
        self._grid_check.toggled.connect(self._on_image_options_changed)
        grid_row.addWidget(self._grid_check)
        grid_row.addStretch()
        single_layout.addLayout(grid_row)

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
        single_layout.addLayout(dilate_row)

        tile_row = QHBoxLayout()
        self._tile_check = QCheckBox("Tile (4-quadrant)")
        self._tile_check.setToolTip(
            "Split image into 4 quadrants, query each separately, then merge coords back.\n"
            "Each feature takes up 2× more of the VLM's visual field → better precision."
        )
        tile_row.addWidget(self._tile_check)
        tile_row.addStretch()
        single_layout.addLayout(tile_row)

        upscale_row = QHBoxLayout()
        self._upscale_check = QCheckBox("Upscale to 1092px min")
        self._upscale_check.setChecked(True)
        self._upscale_check.setToolTip(
            "If the shortest image dimension is below 1092px, upscale before sending to the VLM.\n"
            "1092px matches Claude's native vision encoder resolution; GPT-4o tiles at similar sizes.\n"
            "Your 150×150 XRF scans become 1092×1092 — particles are visible and boxable."
        )
        self._upscale_check.toggled.connect(self._on_image_options_changed)
        upscale_row.addWidget(self._upscale_check)
        upscale_row.addStretch()
        single_layout.addLayout(upscale_row)

        colormap_row = QHBoxLayout()
        self._colormap_check = QCheckBox("Inferno colormap")
        self._colormap_check.setToolTip(
            "Render grayscale XRF images with the inferno colormap (black → red → yellow → white).\n"
            "Makes bright particles visually pop for better VLM localization."
        )
        self._colormap_check.toggled.connect(self._on_image_options_changed)
        colormap_row.addWidget(self._colormap_check)
        colormap_row.addStretch()
        single_layout.addLayout(colormap_row)

        layout.addWidget(self._single_widget)

        # Colored circles checkbox — plain style but with visible indicator outline
        _cb_style = "QCheckBox { font-size: 13px; }"

        red_circles_row = QHBoxLayout()
        red_circles_row.setContentsMargins(0, 0, 0, 0)
        self._red_circles_check = QCheckBox("Colored circles on white")
        self._red_circles_check.setStyleSheet(_cb_style)
        self._red_circles_check.setToolTip(
            "Normalize → Gaussian blur → Otsu threshold → find blob contours → draw a filled circle\n"
            "around each blob on a white background before sending to the VLM.\n"
            "In Single mode: red circles.  In XRF/Multi mode: each element gets its assigned color.\n"
            "Preview updates live. Overrides inferno colormap when enabled."
        )
        self._red_circles_check.toggled.connect(self._on_image_options_changed)
        red_circles_row.addWidget(self._red_circles_check)
        red_circles_row.addStretch()
        layout.addLayout(red_circles_row)

        # LLM co-location checkbox (Triple + Multi only)
        llm_coloc_row = QHBoxLayout()
        llm_coloc_row.setContentsMargins(0, 0, 0, 0)
        self._llm_colocate_check = QCheckBox("LLM Analyze Locations for VLM Results")
        self._llm_colocate_check.setStyleSheet(_cb_style)
        self._llm_colocate_check.setToolTip(
            "Alternative co-location method: sends the full VLM features JSON directly to the LLM\n"
            "and asks it to spatially reason about which elements overlap, returning tight intersection\n"
            "bounding boxes and material IDs — no Python geometry involved.\n"
            "When unchecked: uses the standard geometric box-intersection pipeline."
        )
        llm_coloc_row.addWidget(self._llm_colocate_check)
        llm_coloc_row.addStretch()
        self._llm_colocate_widget = QWidget()
        self._llm_colocate_widget.setContentsMargins(0, 0, 0, 0)
        self._llm_colocate_widget.setLayout(llm_coloc_row)
        self._llm_colocate_widget.setVisible(False)
        layout.addWidget(self._llm_colocate_widget)

        # Both checked by default
        self._red_circles_check.setChecked(True)
        self._llm_colocate_check.setChecked(True)

        self.locate_btn = QPushButton("Get Location of Features")
        self.locate_btn.setMinimumHeight(40)
        self.locate_btn.setEnabled(False)
        self.locate_btn.clicked.connect(self._run_locate)
        layout.addWidget(self.locate_btn)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # Co-location analysis button (XRF mode only, shown after triple analysis)
        self._colocate_btn = QPushButton("Analyze Co-locations")
        self._colocate_btn.setMinimumHeight(32)
        self._colocate_btn.setToolTip("Compute element co-location zones and query LLM for material suggestions")
        self._colocate_btn.clicked.connect(self._run_colocate)
        self._colocate_btn.setVisible(False)
        layout.addWidget(self._colocate_btn)

        # ── Single-image scroll area ──────────────────────────────────────────
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setAlignment(Qt.AlignCenter)
        self.scroll_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.scroll_area.viewport().setMouseTracking(True)
        self.img_display = _ImageDisplay()
        self.img_display.setAlignment(Qt.AlignCenter)
        self.img_display.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.img_display.mouse_moved.connect(self._on_mouse_moved)
        self.img_display.mouse_left.connect(self._on_mouse_left)
        self.img_display.zoom_scrolled.connect(self._on_zoom)
        self.scroll_area.setWidget(self.img_display)

        # Overlay toggle button — floats in top-right of single image view
        self._single_overlay_btn = QPushButton("Hide Boxes", self.scroll_area)
        self._single_overlay_btn.setFixedSize(88, 26)
        self._single_overlay_btn.setStyleSheet(
            "QPushButton { background: rgba(30,30,30,200); color: #eee; border: 1px solid #555;"
            " border-radius: 4px; font-size: 11px; padding: 2px 6px; }"
            "QPushButton:hover { background: rgba(60,60,60,220); }"
        )
        self._single_overlay_btn.clicked.connect(self._toggle_single_boxes)
        self._single_overlay_btn.raise_()

        # ── XRF mode: left column of 3 mini-previews ─────────────────────────
        self._xrf_pick_btns = []
        self._xrf_name_labels = []
        self._xrf_mini_labels = []   # QLabel pixmap displays, one per element

        _elem_cfg = [
            ("Red",   "color: #c0392b; font-weight: bold;"),
            ("Green", "color: #27ae60; font-weight: bold;"),
            ("Blue",  "color: #2980b9; font-weight: bold;"),
        ]
        xrf_left = QWidget()
        xrf_left.setMinimumWidth(160)
        xrf_left.setMaximumWidth(260)
        xrf_left_layout = QVBoxLayout(xrf_left)
        xrf_left_layout.setContentsMargins(0, 0, 4, 0)
        xrf_left_layout.setSpacing(2)
        for i, (color_name, label_style) in enumerate(_elem_cfg):
            panel = QGroupBox(f"Element {i+1} — {color_name}")
            panel.setStyleSheet(
                "QGroupBox { font-weight: bold; border: 1px solid #444; "
                "border-radius: 4px; margin-top: 8px; padding-top: 4px; }"
            )
            panel_layout = QVBoxLayout(panel)
            panel_layout.setSpacing(2)
            panel_layout.setContentsMargins(4, 4, 4, 4)

            btn_row = QHBoxLayout()
            btn = QPushButton(f"Pick {color_name}...")
            btn.clicked.connect(lambda _, idx=i: self._pick_xrf_image(idx))
            btn_row.addWidget(btn)
            panel_layout.addLayout(btn_row)
            self._xrf_pick_btns.append(btn)

            lbl = QLabel("No file selected")
            lbl.setStyleSheet("color: #888; font-size: 10px;")
            lbl.setWordWrap(True)
            panel_layout.addWidget(lbl)
            self._xrf_name_labels.append(lbl)

            mini = QLabel()
            mini.setAlignment(Qt.AlignCenter)
            mini.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            mini.setMinimumHeight(60)
            mini.setStyleSheet("background: #111; border: 1px solid #333;")
            panel_layout.addWidget(mini, 1)
            self._xrf_mini_labels.append(mini)

            xrf_left_layout.addWidget(panel, 1)

        # ── XRF mode: composite (large) scroll area ───────────────────────────
        self._xrf_composite_scroll = QScrollArea()
        self._xrf_composite_scroll.setWidgetResizable(False)
        self._xrf_composite_scroll.setAlignment(Qt.AlignCenter)
        self._xrf_composite_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._xrf_composite_scroll.viewport().setMouseTracking(True)
        self._xrf_composite_display = _ImageDisplay()
        self._xrf_composite_display.setAlignment(Qt.AlignCenter)
        self._xrf_composite_display.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._xrf_composite_display.mouse_moved.connect(self._on_xrf_mouse_moved)
        self._xrf_composite_display.mouse_left.connect(self._on_mouse_left)
        self._xrf_composite_display.zoom_scrolled.connect(self._on_xrf_zoom)
        self._xrf_composite_scroll.setWidget(self._xrf_composite_display)
        self._xrf_composite_pixmap = None   # base composite (no boxes)
        self._xrf_annotated_pixmap = None   # composite + boxes

        _ovr_style = (
            "QPushButton {{ background: rgba(30,30,30,200); color: {fg}; border: 1px solid #555;"
            " border-radius: 4px; font-size: 10px; padding: 1px 5px; }}"
            "QPushButton:hover {{ background: rgba(70,70,70,220); }}"
        )

        # Master "Hide All / Show All" overlay button — floats top-right of XRF composite
        self._xrf_overlay_btn = QPushButton("Hide All", self._xrf_composite_scroll)
        self._xrf_overlay_btn.setFixedSize(72, 24)
        self._xrf_overlay_btn.setStyleSheet(_ovr_style.format(fg="#eee"))
        self._xrf_overlay_btn.clicked.connect(self._toggle_overlay_boxes)
        self._xrf_overlay_btn.raise_()

        self._xrf_switch_view_btn = QPushButton("⬤ Circles", self._xrf_composite_scroll)
        self._xrf_switch_view_btn.setFixedSize(80, 22)
        self._xrf_switch_view_btn.setStyleSheet(_ovr_style.format(fg="#fbbf24"))
        self._xrf_switch_view_btn.clicked.connect(self._toggle_xrf_view)
        self._xrf_switch_view_btn.setVisible(False)
        self._xrf_switch_view_btn.raise_()

        # Individual element toggle buttons (hidden until analysis done)
        self._xrf_elem_overlay_btns = []
        _elem_colors_fg = ["#f87171", "#4ade80", "#60a5fa", "#d1d5db"]
        _elem_labels_dflt = ["● R", "● G", "● B", "◆ Zones"]
        _elem_callbacks = [
            lambda: self._toggle_elem_box(0),
            lambda: self._toggle_elem_box(1),
            lambda: self._toggle_elem_box(2),
            lambda: self._toggle_elem_box(3),
        ]
        for fg, txt, cb in zip(_elem_colors_fg, _elem_labels_dflt, _elem_callbacks):
            btn = QPushButton(txt, self._xrf_composite_scroll)
            btn.setFixedSize(72, 22)
            btn.setStyleSheet(_ovr_style.format(fg=fg))
            btn.clicked.connect(cb)
            btn.setVisible(False)
            btn.raise_()
            self._xrf_elem_overlay_btns.append(btn)

        # Inner splitter: xrf_left | composite  (used only in XRF mode)
        self._xrf_inner_splitter = QSplitter(Qt.Horizontal)
        self._xrf_inner_splitter.setChildrenCollapsible(False)
        self._xrf_inner_splitter.addWidget(xrf_left)
        self._xrf_inner_splitter.addWidget(self._xrf_composite_scroll)
        self._xrf_inner_splitter.setSizes([200, 520])
        self._xrf_inner_splitter.splitterMoved.connect(self._on_splitter_moved)

        # ── Multi Element mode: left panel ───────────────────────────────────────
        _multi_left = QWidget()
        _multi_left.setMinimumWidth(160)
        _multi_left.setMaximumWidth(280)
        _multi_left_layout = QVBoxLayout(_multi_left)
        _multi_left_layout.setContentsMargins(0, 0, 4, 0)
        _multi_left_layout.setSpacing(4)

        _multi_dir_row = QHBoxLayout()
        self._multi_browse_btn = QPushButton("Browse Directory...")
        self._multi_browse_btn.clicked.connect(self._pick_multi_dir)
        _multi_dir_row.addWidget(self._multi_browse_btn)
        _multi_left_layout.addLayout(_multi_dir_row)
        self._multi_dir_label = QLabel("No directory selected")
        self._multi_dir_label.setStyleSheet("color: #888; font-size: 10px;")
        self._multi_dir_label.setWordWrap(True)
        _multi_left_layout.addWidget(self._multi_dir_label)

        _multi_left_layout.addWidget(QLabel("TIFFs found:"))
        self._multi_cb_scroll = QScrollArea()
        self._multi_cb_scroll.setWidgetResizable(True)
        self._multi_cb_scroll.setMinimumHeight(80)
        self._multi_cb_container = QWidget()
        self._multi_cb_layout = QVBoxLayout(self._multi_cb_container)
        self._multi_cb_layout.setContentsMargins(4, 4, 4, 4)
        self._multi_cb_layout.setSpacing(2)
        self._multi_cb_layout.addStretch()
        self._multi_cb_scroll.setWidget(self._multi_cb_container)
        _multi_left_layout.addWidget(self._multi_cb_scroll, 1)

        self._multi_confirm_btn = QPushButton("Confirm Selection")
        self._multi_confirm_btn.setEnabled(False)
        self._multi_confirm_btn.clicked.connect(self._on_multi_confirm_selection)
        _multi_left_layout.addWidget(self._multi_confirm_btn)

        # Load from saved JSON (skip VLM re-run)
        _sep = QLabel("── Load previous results ──")
        _sep.setStyleSheet("color: #666; font-size: 10px;")
        _sep.setAlignment(Qt.AlignCenter)
        _multi_left_layout.addWidget(_sep)

        self._multi_load_features_btn = QPushButton("Load Features JSON...")
        self._multi_load_features_btn.setToolTip("Load a saved _multi_features.json to restore boxes without re-running the VLM")
        self._multi_load_features_btn.clicked.connect(self._load_multi_features_json)
        _multi_left_layout.addWidget(self._multi_load_features_btn)

        self._multi_load_colocate_btn = QPushButton("Load Co-location JSON...")
        self._multi_load_colocate_btn.setToolTip("Load a saved _multi_colocate.json to restore co-location zones")
        self._multi_load_colocate_btn.clicked.connect(self._load_multi_colocate_json)
        _multi_left_layout.addWidget(self._multi_load_colocate_btn)

        _multi_left_layout.addWidget(QLabel("Previews:"))
        self._multi_preview_scroll = QScrollArea()
        self._multi_preview_scroll.setWidgetResizable(True)
        self._multi_preview_scroll.setMinimumHeight(100)
        _multi_preview_inner = QWidget()
        self._multi_preview_layout = QVBoxLayout(_multi_preview_inner)
        self._multi_preview_layout.setContentsMargins(2, 2, 2, 2)
        self._multi_preview_layout.setSpacing(4)
        self._multi_preview_layout.addStretch()
        self._multi_preview_scroll.setWidget(_multi_preview_inner)
        _multi_left_layout.addWidget(self._multi_preview_scroll, 2)

        # Separate composite scroll for multi mode
        self._multi_composite_scroll = QScrollArea()
        self._multi_composite_scroll.setWidgetResizable(False)
        self._multi_composite_scroll.setAlignment(Qt.AlignCenter)
        self._multi_composite_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._multi_composite_scroll.viewport().setMouseTracking(True)
        self._multi_composite_display = _ImageDisplay()
        self._multi_composite_display.setAlignment(Qt.AlignCenter)
        self._multi_composite_display.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._multi_composite_display.mouse_moved.connect(self._on_multi_mouse_moved)
        self._multi_composite_display.mouse_left.connect(self._on_mouse_left)
        self._multi_composite_display.zoom_scrolled.connect(self._on_multi_zoom)
        self._multi_composite_scroll.setWidget(self._multi_composite_display)

        # Floating overlay buttons on multi composite
        _ovr_style_m = (
            "QPushButton {{ background: rgba(30,30,30,200); color: {fg}; border: 1px solid #555;"
            " border-radius: 4px; font-size: 10px; padding: 1px 5px; }}"
            "QPushButton:hover {{ background: rgba(70,70,70,220); }}"
        )
        self._multi_overlay_btn = QPushButton("Hide All", self._multi_composite_scroll)
        self._multi_overlay_btn.setFixedSize(72, 24)
        self._multi_overlay_btn.setStyleSheet(_ovr_style_m.format(fg="#eee"))
        self._multi_overlay_btn.clicked.connect(self._toggle_multi_all_boxes)
        self._multi_overlay_btn.raise_()

        self._multi_zones_btn = QPushButton("◆ Zones", self._multi_composite_scroll)
        self._multi_zones_btn.setFixedSize(72, 22)
        self._multi_zones_btn.setStyleSheet(_ovr_style_m.format(fg="#d1d5db"))
        self._multi_zones_btn.clicked.connect(self._toggle_multi_zones)
        self._multi_zones_btn.setVisible(False)
        self._multi_zones_btn.raise_()

        self._multi_boxes_btn = QPushButton("● Boxes", self._multi_composite_scroll)
        self._multi_boxes_btn.setFixedSize(72, 22)
        self._multi_boxes_btn.setStyleSheet(_ovr_style_m.format(fg="#a78bfa"))
        self._multi_boxes_btn.clicked.connect(self._toggle_multi_elem_boxes)
        self._multi_boxes_btn.setVisible(False)
        self._multi_boxes_btn.raise_()

        self._multi_switch_view_btn = QPushButton("⬤ Circles", self._multi_composite_scroll)
        self._multi_switch_view_btn.setFixedSize(80, 22)
        self._multi_switch_view_btn.setStyleSheet(_ovr_style_m.format(fg="#fbbf24"))
        self._multi_switch_view_btn.clicked.connect(self._toggle_multi_view)
        self._multi_switch_view_btn.setVisible(False)
        self._multi_switch_view_btn.raise_()

        self._multi_inner_splitter = QSplitter(Qt.Horizontal)
        self._multi_inner_splitter.setChildrenCollapsible(False)
        self._multi_inner_splitter.addWidget(_multi_left)
        self._multi_inner_splitter.addWidget(self._multi_composite_scroll)
        self._multi_inner_splitter.setSizes([220, 500])
        self._multi_inner_splitter.splitterMoved.connect(self._on_splitter_moved)

        # Left stack: index 0 = single scroll_area, index 1 = xrf inner splitter, index 2 = multi
        from qtpy.QtWidgets import QStackedWidget
        self._left_stack = QStackedWidget()
        self._left_stack.addWidget(self.scroll_area)           # index 0
        self._left_stack.addWidget(self._xrf_inner_splitter)   # index 1
        self._left_stack.addWidget(self._multi_inner_splitter) # index 2

        # ── Right info panel (always visible) ────────────────────────────────
        right_panel = QWidget()
        right_panel.setMinimumWidth(160)
        right_panel.setMaximumWidth(300)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(6, 0, 0, 0)
        right_layout.setSpacing(4)

        self.coord_label = QLabel("X: —\nY: —")
        self.coord_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.coord_label.setStyleSheet(
            "font-family: monospace; font-size: 11px; padding: 6px; color: #888;"
        )
        right_layout.addWidget(self.coord_label)

        self._send_status_label = QLabel("")
        self._send_status_label.setWordWrap(True)
        self._send_status_label.setStyleSheet(
            "font-size: 10px; color: #aaa; font-family: monospace; padding: 2px 4px;"
        )
        self._send_status_label.setVisible(False)
        right_layout.addWidget(self._send_status_label)

        self.info_panel = QTextEdit()
        self.info_panel.setReadOnly(True)
        self.info_panel.setStyleSheet("font-family: monospace; font-size: 11px;")
        self.info_panel.setPlaceholderText("Feature details will appear here after analysis.")
        self.info_panel.setVisible(False)
        right_layout.addWidget(self.info_panel, 1)

        # Outer splitter: left_stack | right info panel
        self._splitter = QSplitter(Qt.Horizontal)
        self._splitter.setChildrenCollapsible(False)
        self._splitter.addWidget(self._left_stack)
        self._splitter.addWidget(right_panel)
        self._splitter.setSizes([720, 220])
        self._splitter.splitterMoved.connect(self._on_splitter_moved)
        layout.addWidget(self._splitter, 1)

    def _check_model(self):
        provider = _get_provider(_active_model)
        if provider == "ollama":
            self._check_ollama_model()
        else:
            self._check_cloud_model(provider)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_image_views()
        self._reposition_overlay_btns()

    def _on_splitter_moved(self, pos: int, index: int):
        self._refresh_image_views()
        self._reposition_overlay_btns()

    def _reposition_overlay_btns(self):
        margin = 8
        gap = 2
        # Single image overlay
        sa = self.scroll_area
        self._single_overlay_btn.move(sa.width() - self._single_overlay_btn.width() - margin, margin)
        self._single_overlay_btn.raise_()
        # XRF composite master button
        sa = self._xrf_composite_scroll
        master = self._xrf_overlay_btn
        master.move(sa.width() - master.width() - margin, margin)
        master.raise_()
        y = margin + master.height() + gap
        for btn in self._xrf_elem_overlay_btns:
            if btn.isVisible():
                btn.move(sa.width() - btn.width() - margin, y)
                y += btn.height() + gap
                btn.raise_()
        if self._xrf_switch_view_btn.isVisible():
            self._xrf_switch_view_btn.move(sa.width() - self._xrf_switch_view_btn.width() - margin, y)
            self._xrf_switch_view_btn.raise_()
        # Multi composite buttons
        sa = self._multi_composite_scroll
        self._multi_overlay_btn.move(sa.width() - self._multi_overlay_btn.width() - margin, margin)
        self._multi_overlay_btn.raise_()
        y = margin + self._multi_overlay_btn.height() + gap
        for btn in [self._multi_boxes_btn, self._multi_zones_btn, self._multi_switch_view_btn]:
            if btn.isVisible():
                btn.move(sa.width() - btn.width() - margin, y)
                y += btn.height() + gap
                btn.raise_()

    def _toggle_overlay_boxes(self):
        self._boxes_visible = not self._boxes_visible
        self._xrf_show_r = self._boxes_visible
        self._xrf_show_g = self._boxes_visible
        self._xrf_show_b = self._boxes_visible
        self._xrf_show_zones = self._boxes_visible
        label = "Hide All" if self._boxes_visible else "Show All"
        self._xrf_overlay_btn.setText(label)
        self._single_overlay_btn.setText("Hide Boxes" if self._boxes_visible else "Show Boxes")
        self._sync_elem_btn_labels()
        self._refresh_image_views()

    def _toggle_elem_box(self, idx: int):
        flags = [self._xrf_show_r, self._xrf_show_g, self._xrf_show_b, self._xrf_show_zones]
        flags[idx] = not flags[idx]
        self._xrf_show_r, self._xrf_show_g, self._xrf_show_b, self._xrf_show_zones = flags
        self._sync_elem_btn_labels()
        self._refresh_image_views()

    def _sync_elem_btn_labels(self):
        colors = ["Red", "Green", "Blue", "Zones"]
        short = ["R", "G", "B", "Zones"]
        flags = [self._xrf_show_r, self._xrf_show_g, self._xrf_show_b, self._xrf_show_zones]
        # Use actual element names if available
        names = [self._xrf_elem_names[i] or short[i] for i in range(3)] + ["Zones"]
        for i, (btn, flag, name) in enumerate(zip(self._xrf_elem_overlay_btns, flags, names)):
            prefix = "●" if i < 3 else "◆"
            btn.setText(f"{prefix} {name}" if flag else f"○ {name}")

    def _toggle_multi_all_boxes(self):
        all_on = any(self._multi_show_flags) or self._multi_show_zones
        new_state = not all_on
        self._multi_show_flags = [new_state] * len(self._multi_show_flags)
        self._multi_show_zones = new_state
        self._multi_overlay_btn.setText("Hide All" if new_state else "Show All")
        self._multi_boxes_btn.setText("● Boxes" if new_state else "○ Boxes")
        self._multi_zones_btn.setText("◆ Zones" if new_state else "◇ Zones")
        self._refresh_image_views()

    def _toggle_multi_elem_boxes(self):
        any_on = any(self._multi_show_flags)
        self._multi_show_flags = [not any_on] * len(self._multi_show_flags)
        self._multi_boxes_btn.setText("● Boxes" if not any_on else "○ Boxes")
        self._refresh_image_views()

    def _toggle_multi_zones(self):
        self._multi_show_zones = not self._multi_show_zones
        self._multi_zones_btn.setText("◆ Zones" if self._multi_show_zones else "◇ Zones")
        self._refresh_image_views()

    def _toggle_xrf_view(self):
        self._xrf_view_circles = not self._xrf_view_circles
        self._xrf_switch_view_btn.setText("⬤ Circles" if self._xrf_view_circles else "⬤ Normal")
        pm = self._xrf_current_composite()
        if pm:
            self._show_xrf_composite(pm)

    def _toggle_multi_view(self):
        self._multi_view_circles = not self._multi_view_circles
        self._multi_switch_view_btn.setText("⬤ Circles" if self._multi_view_circles else "⬤ Normal")
        pm = self._multi_current_composite()
        if pm:
            self._show_multi_composite(pm)

    def _show_elem_overlay_btns(self, show_zones: bool = False):
        """Show/update the individual element overlay buttons after analysis."""
        for i, btn in enumerate(self._xrf_elem_overlay_btns):
            if i == 3:
                btn.setVisible(show_zones)
            else:
                btn.setVisible(True)
        self._sync_elem_btn_labels()
        self._reposition_overlay_btns()

    def _refresh_image_views(self):
        mode = self._mode_combo.currentIndex()
        if mode == 1:
            pm = self._xrf_current_composite()
            if pm:
                self._show_xrf_composite(pm)
        elif mode == 2:
            pm = self._multi_current_composite()
            if pm:
                self._show_multi_composite(pm)
        else:
            pm = self._current_pixmap()
            if pm:
                self._show_pixmap(pm)

    def _xrf_current_composite(self):
        """Dynamically build the composite with only the currently-visible box layers."""
        base = (self._xrf_circles_composite_pixmap if self._xrf_view_circles
                else self._xrf_composite_pixmap)
        if not base:
            return None
        groups = []
        if self._xrf_show_r and self._xrf_r_features:
            groups.append((self._xrf_r_features, QColor(220, 60,  60)))
        if self._xrf_show_g and self._xrf_g_features:
            groups.append((self._xrf_g_features, QColor(60,  200, 60)))
        if self._xrf_show_b and self._xrf_b_features:
            groups.append((self._xrf_b_features, QColor(60,  120, 220)))
        result = _draw_feature_boxes_multi_color(base, groups) if groups else base
        if self._xrf_show_zones and self._xrf_colocate_zones:
            result = _draw_colocate_boxes(result, self._xrf_colocate_zones)
        return result

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
                    f"{_active_model} not downloaded — use the Model tab to download it first."
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
                f"No API key for {provider_name} — configure it in the Model tab first."
            )
            self.model_status.setStyleSheet("color: #e67e22; padding: 4px;")
            self._model_ok = False

    def refresh_model(self):
        self._model_ok = True
        self.model_status.setText("")
        self.model_status.setStyleSheet("")
        self._header_model_lbl.setText(f"Model: <b>{_active_model}</b>")
        self._check_model()
        if self._image_path:
            self.locate_btn.setEnabled(self._model_ok)

    def _on_preset_changed(self, index: int):
        self._active_prompt = self._PRESETS[index]
        self._header_prompt_lbl.setText(f"Prompt: {self._active_prompt[:120].rstrip()}…")
        self._header_prompt_lbl.setToolTip(self._active_prompt)

    def _show_prompt_popup(self):
        from qtpy.QtWidgets import QDialog, QTextEdit, QVBoxLayout, QDialogButtonBox
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Prompt — {self._preset_combo.currentText()}")
        dlg.resize(640, 400)
        layout = QVBoxLayout(dlg)
        text = QTextEdit()
        text.setReadOnly(True)
        text.setPlainText(self._active_prompt)
        text.setStyleSheet("font-size: 12px; font-family: monospace;")
        layout.addWidget(text)
        btns = QDialogButtonBox(QDialogButtonBox.Close)
        btns.rejected.connect(dlg.close)
        layout.addWidget(btns)
        dlg.exec_()

    def _on_mode_changed(self, index: int):
        self._single_widget.setVisible(index == 0)
        self._preset_widget.setVisible(index == 0)
        self._llm_colocate_widget.setVisible(index in (1, 2))
        if index == 0:
            self._left_stack.setCurrentIndex(0)
            prompt = self._active_prompt
            self.locate_btn.setText("Get Location of Features")
        elif index == 1:
            self._left_stack.setCurrentIndex(1)
            prompt = LOCATE_SPARSE_XRF_PROMPT
            self.locate_btn.setText("Analyze 3 Elements")
        else:
            self._left_stack.setCurrentIndex(2)
            prompt = LOCATE_SPARSE_XRF_PROMPT
            self.locate_btn.setText("Analyze All Elements")
        self._header_prompt_lbl.setText(f"Prompt: {prompt[:120].rstrip()}…")
        self._header_prompt_lbl.setToolTip(prompt)
        self._reset_image_state()
        self.locate_btn.setEnabled(False)

    def _reset_image_state(self):
        self._orig_pixmap = None
        self._dilated_pixmap = None
        self._annotated_pixmap = None
        self._features = []
        self._zoom = 1.0
        self._xrf_composite_pixmap = None
        self._xrf_annotated_pixmap = None
        self._xrf_colocate_pixmap = None
        self._xrf_colocate_zones = []
        self._xrf_zoom = 1.0
        self._xrf_last_triple_json = None
        self._xrf_elem_names = ["", "", ""]
        self._xrf_r_features = []
        self._xrf_g_features = []
        self._xrf_b_features = []
        self._colocate_btn.setVisible(False)
        self._boxes_visible = True
        self._xrf_show_r = True
        self._xrf_show_g = True
        self._xrf_show_b = True
        self._xrf_show_zones = True
        for btn in self._xrf_elem_overlay_btns:
            btn.setVisible(False)
        self._multi_elem_features = []
        self._multi_colocate_zones = []
        self._multi_composite_pixmap = None
        self._multi_show_flags = []
        self._multi_show_zones = True
        self._multi_zoom = 1.0
        self._multi_boxes_btn.setVisible(False)
        self._multi_zones_btn.setVisible(False)
        self._multi_switch_view_btn.setVisible(False)
        self._multi_view_circles = False
        self._xrf_circles_composite_pixmap = None
        self._xrf_view_circles = False
        self._xrf_switch_view_btn.setVisible(False)
        self.info_panel.setVisible(False)

    def _pick_xrf_image(self, idx: int):
        path, _ = QFileDialog.getOpenFileName(
            self, f"Select Element {idx+1} TIFF", "",
            "TIFF Files (*.tif *.tiff);;All Files (*)"
        )
        if not path:
            return
        self._xrf_paths[idx] = path
        # Auto-extract element name from filename: scan_12345_Fe.tiff → "Fe"
        stem = pathlib.Path(path).stem
        elem_name = stem.split("_")[-1]
        self._xrf_elem_names[idx] = elem_name
        self._xrf_name_labels[idx].setText(os.path.basename(path))
        self._xrf_name_labels[idx].setStyleSheet("font-size: 10px;")
        self._update_xrf_mini_preview(idx)
        if all(self._xrf_paths):
            self._build_xrf_composite_display()
            if self._model_ok:
                self.locate_btn.setEnabled(True)

    def _update_xrf_mini_preview(self, idx: int):
        path = self._xrf_paths[idx]
        if not path:
            return
        normalize = self._normalize_check.isChecked()
        upscale = self._upscale_check.isChecked()
        circles = self._red_circles_check.isChecked()
        cc = _XRF_ELEM_COLORS[idx][1]
        try:
            pm, _, _ = _load_image_pixmap(path, normalize=normalize, upscale=upscale,
                                           red_circles=circles, circle_color=cc)
            lbl = self._xrf_mini_labels[idx]
            scaled = pm.scaled(lbl.width() or 200, lbl.height() or 120,
                               Qt.KeepAspectRatio, Qt.SmoothTransformation)
            lbl.setPixmap(scaled)
        except Exception:
            pass

    def _build_xrf_composite_display(self):
        """Build normal + circles composites for the 3 XRF elements; show whichever is active."""
        try:
            normalize = self._normalize_check.isChecked()
            upscale = self._upscale_check.isChecked()
            circle_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

            def _load_gray(p):
                arr = cv2.imread(str(p), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
                if arr is None:
                    raise ValueError(f"Could not read {p}")
                arr = arr.astype(np.float32)
                if normalize:
                    mn, mx = arr.min(), arr.max()
                    if mx > mn:
                        arr = (arr - mn) / (mx - mn)
                if upscale:
                    h_, w_ = arr.shape
                    if min(h_, w_) < 1092:
                        scale = 1092 / min(h_, w_)
                        arr = cv2.resize(arr, (int(w_ * scale), int(h_ * scale)),
                                         interpolation=cv2.INTER_CUBIC)
                return (arr * 255).astype(np.uint8)

            # Always build normal false-color composite
            channels = [_load_gray(p) for p in self._xrf_paths]
            h = max(c.shape[0] for c in channels)
            w = max(c.shape[1] for c in channels)
            padded = []
            for c in channels:
                pad = np.zeros((h, w), dtype=np.uint8)
                pad[:c.shape[0], :c.shape[1]] = c
                padded.append(pad)
            normal_comp = np.stack(padded, axis=-1)
            qimg = QImage(normal_comp.data, w, h, w * 3, QImage.Format_RGB888).copy()
            self._xrf_composite_pixmap = QPixmap.fromImage(qimg)

            # Also build circles composite
            circle_arrays = [
                _apply_red_circles_transform(channels[i], circle_color=circle_colors[i])
                for i in range(3)
            ]
            hc = max(a.shape[0] for a in circle_arrays)
            wc = max(a.shape[1] for a in circle_arrays)
            circles_comp = np.full((hc, wc, 3), 255, dtype=np.uint8)
            for ca in circle_arrays:
                mask = np.any(ca < 255, axis=-1)
                ph, pw = ca.shape[:2]
                circles_comp[:ph, :pw][mask] = ca[mask]
            qimg2 = QImage(circles_comp.data, wc, hc, wc * 3, QImage.Format_RGB888).copy()
            self._xrf_circles_composite_pixmap = QPixmap.fromImage(qimg2)

            self._xrf_annotated_pixmap = None
            # Show whichever view is active
            pm = self._xrf_current_composite()
            if pm:
                self._show_xrf_composite(pm)
            # Show switch view button since circles composite is available
            self._xrf_switch_view_btn.setVisible(True)
            self._xrf_switch_view_btn.setText(
                "⬤ Circles" if self._xrf_view_circles else "⬤ Normal"
            )
            self._reposition_overlay_btns()
            self.info_panel.setVisible(True)
        except Exception as e:
            self.status_label.setText(f"Could not build composite: {e}")
            self.status_label.setStyleSheet("color: #c0392b; padding: 4px;")

    def _show_xrf_composite(self, pixmap):
        sa = self._xrf_composite_scroll
        vp_w = max(200, sa.viewport().width())
        vp_h = max(200, sa.viewport().height())
        fit = pixmap.scaled(vp_w, vp_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._xrf_base_w, self._xrf_base_h = fit.width(), fit.height()
        w = max(1, int(self._xrf_base_w * self._xrf_zoom))
        h = max(1, int(self._xrf_base_h * self._xrf_zoom))
        scaled = pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._xrf_composite_display.setPixmap(scaled)
        self._xrf_composite_display.resize(scaled.width(), scaled.height())

    def _on_xrf_zoom(self, direction: int):
        sa = self._xrf_composite_scroll
        viewport = sa.viewport()
        cursor_vp = viewport.mapFromGlobal(self._xrf_composite_display.cursor().pos())
        old_h = sa.horizontalScrollBar().value()
        old_v = sa.verticalScrollBar().value()

        old_zoom = self._xrf_zoom
        self._xrf_zoom = max(0.25, min(8.0, self._xrf_zoom * (1.15 if direction > 0 else 1 / 1.15)))

        pm = self._xrf_current_composite()
        if pm:
            self._show_xrf_composite(pm)

        ratio = self._xrf_zoom / old_zoom
        sa.horizontalScrollBar().setValue(max(0, int((old_h + cursor_vp.x()) * ratio - cursor_vp.x())))
        sa.verticalScrollBar().setValue(max(0, int((old_v + cursor_vp.y()) * ratio - cursor_vp.y())))

    def _on_xrf_mouse_moved(self, fx: float, fy: float) -> None:
        self.coord_label.setText(f"X: {fx:.3f}\nY: {fy:.3f}")
        self.coord_label.setStyleSheet("font-family: monospace; font-size: 11px; padding: 6px;")

        lines = []

        # Zone tooltip — only when zones are visible
        if self._xrf_show_zones and self._xrf_colocate_zones:
            for zone in self._xrf_colocate_zones:
                if zone.get("x1", 0) <= fx <= zone.get("x2", 0) and \
                        zone.get("y1", 0) <= fy <= zone.get("y2", 0):
                    elem_str = " + ".join(zone.get("elements", []))
                    x1, y1 = zone.get("x1", 0), zone.get("y1", 0)
                    x2, y2 = zone.get("x2", 0), zone.get("y2", 0)
                    lines = [
                        f"<b>Zone {zone.get('id', '?')}  —  {elem_str}</b>",
                        f"bbox: ({x1:.3f},{y1:.3f})→({x2:.3f},{y2:.3f})",
                    ]
                    if zone.get("llm_suggestion"):
                        lines.append(f"<b>→ {zone['llm_suggestion']}</b>")
                    if zone.get("reasoning"):
                        lines.append(f"<i>{zone['reasoning']}</i>")
                    break

        # Element feature tooltips — only for visible elements, only if no zone hit
        if not lines:
            labels = self._get_elem_labels()
            elem_groups = [
                (labels[0], "#e74c3c", self._xrf_r_features, self._xrf_show_r),
                (labels[1], "#27ae60", self._xrf_g_features, self._xrf_show_g),
                (labels[2], "#2980b9", self._xrf_b_features, self._xrf_show_b),
            ]
            for elem_label, color, feats, visible in elem_groups:
                if not visible:
                    continue
                for feat in feats:
                    if feat.get("x1", 0) <= fx <= feat.get("x2", 0) and \
                            feat.get("y1", 0) <= fy <= feat.get("y2", 0):
                        fid = feat.get("id", "?")
                        label = feat.get("label", "")
                        desc = feat.get("description", "")
                        x1, y1 = feat.get("x1", 0), feat.get("y1", 0)
                        x2, y2 = feat.get("x2", 0), feat.get("y2", 0)
                        lines = [
                            f'<b style="color:{color};">{elem_label}</b>',
                            f"<b>Feature #{fid}</b>  {label}",
                        ]
                        if desc:
                            lines.append(f"<i>{desc}</i>")
                        lines.append(f"bbox: ({x1:.3f},{y1:.3f})→({x2:.3f},{y2:.3f})")
                        lines.append(f"size: {(x2-x1)*100:.1f}% × {(y2-y1)*100:.1f}%")
                        break
                if lines:
                    break

        if lines:
            self._hover_tooltip.setText("<br>".join(lines))
            self._hover_tooltip.adjustSize()
            gpos = QCursor.pos()
            self._hover_tooltip.move(gpos.x() + 16, gpos.y() - self._hover_tooltip.height() - 4)
            self._hover_tooltip.show()
        else:
            self._hover_tooltip.hide()

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
            self._annotated_pixmap = None
            self._features = []
            self._zoom = 1.0
            self._update_display()
            self.scroll_area.setVisible(True)
            self.info_panel.setVisible(True)
        except Exception as e:
            self.status_label.setText(f"Could not load image: {e}")
            self.status_label.setStyleSheet("color: #c0392b; padding: 4px;")
            return
        if self._model_ok:
            self.locate_btn.setEnabled(True)

    def _current_pixmap(self):
        if self._boxes_visible and self._annotated_pixmap:
            return self._annotated_pixmap
        return self._dilated_pixmap if self._dilated_pixmap else self._orig_pixmap

    def _show_pixmap(self, pixmap):
        vp_w = max(400, self.scroll_area.viewport().width())
        vp_h = max(300, self.scroll_area.viewport().height())
        fit = pixmap.scaled(vp_w, vp_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._base_w, self._base_h = fit.width(), fit.height()
        w = max(1, int(self._base_w * self._zoom))
        h = max(1, int(self._base_h * self._zoom))
        scaled = pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.img_display.setPixmap(scaled)
        self.img_display.resize(scaled.width(), scaled.height())

    def _on_zoom(self, direction: int):
        viewport = self.scroll_area.viewport()
        cursor_vp = viewport.mapFromGlobal(self.img_display.cursor().pos())
        old_h = self.scroll_area.horizontalScrollBar().value()
        old_v = self.scroll_area.verticalScrollBar().value()

        old_zoom = self._zoom
        self._zoom = max(0.25, min(8.0, self._zoom * (1.15 if direction > 0 else 1 / 1.15)))

        pm = self._current_pixmap()
        if pm:
            self._show_pixmap(pm)

        # Scroll so the point under the cursor stays fixed
        ratio = self._zoom / old_zoom
        new_h = int((old_h + cursor_vp.x()) * ratio - cursor_vp.x())
        new_v = int((old_v + cursor_vp.y()) * ratio - cursor_vp.y())
        self.scroll_area.horizontalScrollBar().setValue(max(0, new_h))
        self.scroll_area.verticalScrollBar().setValue(max(0, new_v))

    def _toggle_single_boxes(self):
        """Toggle boxes for the single image view (via the overlay button)."""
        self._boxes_visible = not self._boxes_visible
        self._single_overlay_btn.setText("Hide Boxes" if self._boxes_visible else "Show Boxes")
        pm = self._current_pixmap()
        if pm:
            self._show_pixmap(pm)

    def _run_locate(self):
        if self._locate_thread and self._locate_thread.isRunning():
            return
        mode = self._mode_combo.currentIndex()
        if mode == 2:
            if not self._multi_selected_paths:
                return
            self.locate_btn.setEnabled(False)
            normalize = self._normalize_check.isChecked()
            upscale = self._upscale_check.isChecked()
            circles = self._red_circles_check.isChecked()
            self.status_label.setText(f"Analyzing {len(self._multi_selected_paths)} elements... (may take several minutes)")
            self.status_label.setStyleSheet("color: #2980b9; padding: 4px;")
            self._locate_thread = _LocateXRFMultiThread(
                self._multi_selected_paths, normalize=normalize, upscale=upscale, circles=circles)
            self._locate_thread.progress.connect(self._on_tile_progress)
            self._locate_thread.result.connect(self._on_multi_result)
            self._locate_thread.error.connect(self._on_error)
            self._locate_thread.start()
            return
        is_xrf = mode == 1
        if is_xrf:
            if not all(self._xrf_paths):
                return
            self.locate_btn.setEnabled(False)
            normalize = self._normalize_check.isChecked()
            upscale = self._upscale_check.isChecked()
            circles = self._red_circles_check.isChecked()
            self.status_label.setText("Analyzing 3 elements individually... (~1–3 min)")
            self.status_label.setStyleSheet("color: #2980b9; padding: 4px;")
            self._locate_thread = _LocateXRFTripleThread(
                self._xrf_paths, normalize=normalize, upscale=upscale, circles=circles)
            self._locate_thread.progress.connect(self._on_tile_progress)
        else:
            if not self._image_path:
                return
            self.locate_btn.setEnabled(False)
            ksize = self._get_dilate_ksize()
            normalize = self._normalize_check.isChecked()
            grid = self._grid_check.isChecked()
            upscale = self._upscale_check.isChecked()
            colormap = self._colormap_check.isChecked()
            red_circles = self._red_circles_check.isChecked()
            use_tile = self._tile_check.isChecked()
            if use_tile:
                self.status_label.setText("Tile mode — querying 4 quadrants... (~2–4 min)")
                self._locate_thread = _LocateTileThread(self._image_path, dilate_ksize=ksize,
                                                        system_prompt_override=self._active_prompt,
                                                        normalize=normalize, grid_overlay=grid,
                                                        upscale=upscale, colormap=colormap,
                                                        red_circles=red_circles)
                self._locate_thread.progress.connect(self._on_tile_progress)
            else:
                self.status_label.setText("Analyzing... this may take 30–60 seconds.")
                self._locate_thread = _LocateThread(self._image_path, dilate_ksize=ksize,
                                                    system_prompt_override=self._active_prompt,
                                                    normalize=normalize, grid_overlay=grid,
                                                    upscale=upscale, colormap=colormap,
                                                    red_circles=red_circles)
            self.status_label.setStyleSheet("color: #2980b9; padding: 4px;")
        self._locate_thread.result.connect(self._on_result)
        self._locate_thread.error.connect(self._on_error)
        self._locate_thread.start()

    def _on_result(self, data):
        is_xrf = self._mode_combo.currentIndex() == 1

        if is_xrf:
            # Triple XRF result: r_features / g_features / b_features
            r_feats = data.get("r_features", [])
            g_feats = data.get("g_features", [])
            b_feats = data.get("b_features", [])
            n = data.get("total_count", len(r_feats) + len(g_feats) + len(b_feats))
            labels = self._get_elem_labels()
            data["elements"] = {"r": labels[0], "g": labels[1], "b": labels[2]}
            ref = pathlib.Path(self._xrf_paths[0])
            json_path = ref.with_name(ref.stem + "_xrf_triple_features.json")
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2)
            self._xrf_last_triple_json = json_path
            print(f"[Feature Locator XRF] All 3 elements done — {n} total feature(s). Saved at: {json_path}")

            # Store for tooltip hover lookup
            self._xrf_r_features = r_feats
            self._xrf_g_features = g_feats
            self._xrf_b_features = b_feats

            if self._xrf_composite_pixmap:
                groups = [
                    (r_feats, QColor(220, 60,  60)),
                    (g_feats, QColor(60,  200, 60)),
                    (b_feats, QColor(60,  120, 220)),
                ]
                self._show_xrf_composite(self._xrf_current_composite())

            self._show_elem_overlay_btns(show_zones=False)
            self._populate_info_xrf(r_feats, g_feats, b_feats, json_path)
            self.locate_btn.setEnabled(True)
            self._colocate_btn.setVisible(True)
            self._colocate_btn.setEnabled(True)
            self.status_label.setText(f"Done — {n} feature(s) across 3 elements. Saved: {json_path.name}")
            self.status_label.setStyleSheet("color: #27ae60; padding: 4px;")
            QMessageBox.information(self, "XRF Analysis Done",
                f"R: {len(r_feats)}  G: {len(g_feats)}  B: {len(b_feats)} feature(s)\n"
                f"Saved to: {json_path}")
        else:
            features = data.get("features", [])
            n = len(features)
            self._features = features
            stem = pathlib.Path(self._image_path).stem
            json_path = pathlib.Path(self._image_path).with_name(stem + "_features.json")
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"[Feature Locator] Results retrieved from '{_active_model}' — {n} feature(s) found. Saved at: {json_path}")

            base = self._dilated_pixmap if self._dilated_pixmap else self._orig_pixmap
            if base and features:
                self._annotated_pixmap = _draw_feature_boxes(base, features)
                self._show_pixmap(self._annotated_pixmap)
            else:
                self._annotated_pixmap = None

            self._populate_info(features, json_path)
            self.locate_btn.setEnabled(True)
            self.status_label.setText(f"Done. Saved: {json_path.name}")
            self.status_label.setStyleSheet("color: #27ae60; padding: 4px;")
            QMessageBox.information(self, "Features Found",
                f"{n} location(s) found.\nSaved to:\n{json_path}")

    def _populate_info(self, features, json_path):
        color_names = ["red", "blue", "green", "yellow", "purple", "cyan", "orange"]
        prompt_preview = LOCATE_SYSTEM_PROMPT[:200].rstrip() + "…"
        lines = [
            f"Model: {_active_model}",
            f"Prompt: {prompt_preview}",
            "",
            f"Found {len(features)} feature(s)  —  {json_path.name}",
            "",
        ]
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

    def _populate_info_xrf(self, r_feats, g_feats, b_feats, json_path):
        elem_cfg = [
            ("Red   (Elem 1)", r_feats),
            ("Green (Elem 2)", g_feats),
            ("Blue  (Elem 3)", b_feats),
        ]
        lines = [
            f"Model: {_active_model}",
            f"Saved: {json_path.name}",
            "",
        ]
        for elem_label, feats in elem_cfg:
            lines.append(f"── {elem_label} ─── {len(feats)} feature(s)")
            for i, feat in enumerate(feats):
                fid = feat.get("id", i + 1)
                label = feat.get("label", "")
                x1, y1 = feat.get("x1", 0), feat.get("y1", 0)
                x2, y2 = feat.get("x2", 0), feat.get("y2", 0)
                desc = feat.get("description", "")
                lines.append(f"  #{fid}  {label}")
                lines.append(f"       ({x1:.3f},{y1:.3f})→({x2:.3f},{y2:.3f})")
                if desc:
                    lines.append(f"       {desc}")
            lines.append("")
        self.info_panel.setPlainText("\n".join(lines).strip())
        self.info_panel.setVisible(True)

    def _get_elem_labels(self) -> list[str]:
        """Return element label strings like 'Fe (Red)', 'Cr (Green)', 'Ni (Blue)'.

        Element name is auto-extracted from the filename stem on pick.
        """
        colors = ["Red", "Green", "Blue"]
        return [
            f"{self._xrf_elem_names[i]} ({colors[i]})" if self._xrf_elem_names[i]
            else colors[i]
            for i in range(3)
        ]

    def _run_colocate(self):
        mode = self._mode_combo.currentIndex()
        use_llm = self._llm_colocate_check.isChecked()

        if mode == 2:
            if not self._multi_elem_features:
                return
            self._colocate_btn.setEnabled(False)
            if use_llm:
                self.status_label.setText("LLM analyzing co-locations from VLM results...")
                self.status_label.setStyleSheet("color: #2980b9; padding: 4px;")
                self._multi_colocate_thread = _LocateLLMColocateThread(
                    self._multi_elem_features, self._multi_elem_names)
                self._multi_colocate_thread.result.connect(self._on_multi_llm_colocate_result)
            else:
                self.status_label.setText("Running multi-element co-location analysis...")
                self.status_label.setStyleSheet("color: #2980b9; padding: 4px;")
                self._multi_colocate_thread = _LocateXRFMultiColocateThread(
                    self._multi_elem_features, self._multi_elem_names)
                self._multi_colocate_thread.result.connect(self._on_multi_colocate_result)
            self._multi_colocate_thread.error.connect(self._on_error)
            self._multi_colocate_thread.progress.connect(self._on_tile_progress)
            self._multi_colocate_thread.start()
            return

        if not self._xrf_last_triple_json:
            return
        self._colocate_btn.setEnabled(False)
        if use_llm:
            self.status_label.setText("LLM analyzing co-locations from VLM results...")
            self.status_label.setStyleSheet("color: #2980b9; padding: 4px;")
            labels = self._get_elem_labels()
            elem_features = [self._xrf_r_features, self._xrf_g_features, self._xrf_b_features]
            self._colocate_thread = _LocateLLMColocateThread(elem_features, labels)
            self._colocate_thread.result.connect(self._on_llm_colocate_result)
        else:
            self.status_label.setText("Running co-location analysis...")
            self.status_label.setStyleSheet("color: #2980b9; padding: 4px;")
            triple_data = {
                "r_features": self._xrf_r_features,
                "g_features": self._xrf_g_features,
                "b_features": self._xrf_b_features,
            }
            labels = self._get_elem_labels()
            self._colocate_thread = _LocateXRFColocateThread(triple_data, labels)
            self._colocate_thread.result.connect(self._on_colocate_result)
        self._colocate_thread.error.connect(self._on_error)
        self._colocate_thread.progress.connect(self._on_tile_progress)
        self._colocate_thread.start()

    def _on_colocate_result(self, data: dict):
        zones = data.get("overlap_zones", [])
        n = data.get("total_zones", len(zones))
        data.setdefault("mode", "triple")
        data.setdefault("method", "geometric")
        ref = pathlib.Path(self._xrf_paths[0])
        json_path = ref.with_name(ref.stem + "_xrf_colocate.json")
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[XRF Co-locate] {n} zone(s) identified. Saved at: {json_path}")
        for z in zones:
            elem_str = " + ".join(z.get("elements", []))
            print(f"  Zone {z['id']}: {elem_str} @ ({z['x1']:.3f},{z['y1']:.3f})→({z['x2']:.3f},{z['y2']:.3f})")

        self._xrf_colocate_zones = zones
        if self._xrf_composite_pixmap and zones:
            self._xrf_colocate_pixmap = _draw_colocate_boxes(self._xrf_composite_pixmap, zones)

        self._colocate_btn.setEnabled(True)
        self._show_elem_overlay_btns(show_zones=True)
        self._refresh_image_views()
        self.status_label.setText(f"Co-location done — {n} zone(s). Saved: {json_path.name}")
        self.status_label.setStyleSheet("color: #27ae60; padding: 4px;")
        self._populate_info_colocate(zones, json_path)

    def _on_llm_colocate_result(self, data: dict):
        """Triple mode — LLM pipeline result handler."""
        zones = data.get("overlap_zones", [])
        n = data.get("total_zones", len(zones))
        data["mode"] = "triple"
        data["method"] = "llm"
        # Strip color suffix from element names (e.g. "Fe (Red)" → "Fe")
        import re
        _strip = lambda s: re.sub(r'\s*\([^)]+\)\s*$', '', s).strip()
        if isinstance(data.get("elements"), list):
            data["elements"] = [_strip(e) for e in data["elements"]]
        for z in zones:
            z["elements"] = [_strip(e) for e in z.get("elements", [])]
        ref = pathlib.Path(self._xrf_paths[0])
        json_path = ref.with_name(ref.stem + "_xrf_llm_colocate.json")
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[XRF LLM Co-locate] {n} zone(s) identified. Saved at: {json_path}")
        self._xrf_colocate_zones = zones
        if self._xrf_composite_pixmap and zones:
            self._xrf_colocate_pixmap = _draw_colocate_boxes(self._xrf_composite_pixmap, zones)
        self._colocate_btn.setEnabled(True)
        self._show_elem_overlay_btns(show_zones=True)
        self._refresh_image_views()
        self.status_label.setText(f"LLM co-location done — {n} zone(s). Saved: {json_path.name}")
        self.status_label.setStyleSheet("color: #27ae60; padding: 4px;")
        self._populate_info_colocate(zones, json_path)

    def _on_multi_llm_colocate_result(self, data: dict):
        """Multi mode — LLM pipeline result handler."""
        zones = data.get("overlap_zones", [])
        n = data.get("total_zones", len(zones))
        data["mode"] = "multi"
        data["method"] = "llm"
        ref_path = pathlib.Path(self._multi_dir_path)
        json_path = ref_path / f"{ref_path.name}_multi_llm_colocate.json"
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[Multi LLM Co-locate] {n} zone(s) identified. Saved at: {json_path}")
        self._multi_colocate_zones = zones
        pm = self._multi_current_composite()
        if pm:
            self._show_multi_composite(pm)
        self._colocate_btn.setEnabled(True)
        self._multi_zones_btn.setVisible(True)
        self._multi_zones_btn.setText("◆ Zones")
        self._reposition_overlay_btns()
        self.status_label.setText(f"LLM co-location done — {n} zone(s). Saved: {json_path.name}")
        self.status_label.setStyleSheet("color: #27ae60; padding: 4px;")
        self._populate_info_colocate(zones, json_path)

    def _populate_info_colocate(self, zones: list, json_path):
        # Append below existing triple-features content so both are scrollable
        sep = "\n\n" + "─" * 40 + "\n"
        lines = [f"CO-LOCATION ANALYSIS  —  {json_path.name}", ""]
        for z in zones:
            elem_str = " + ".join(z.get("elements", []))
            lines.append(f"Zone {z['id']}: {elem_str}")
            lines.append(f"  bbox: ({z['x1']:.3f},{z['y1']:.3f})→({z['x2']:.3f},{z['y2']:.3f})")
            if z.get("llm_suggestion"):
                lines.append(f"  → {z['llm_suggestion']}")
            if z.get("reasoning"):
                lines.append(f"  {z['reasoning']}")
            lines.append("")
        self.info_panel.append(sep + "\n".join(lines).strip())
        self.info_panel.setVisible(True)

    # ── Multi Element methods ─────────────────────────────────────────────────

    def _pick_multi_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Directory of Element TIFFs", "")
        if not d:
            return
        self._multi_dir_path = d
        self._multi_dir_label.setText(d)
        import glob as _glob
        tiffs = sorted(
            _glob.glob(os.path.join(d, "*.tif")) + _glob.glob(os.path.join(d, "*.tiff"))
        )
        self._multi_tiff_paths = tiffs
        for cb in self._multi_checkboxes:
            cb.setParent(None)
        self._multi_checkboxes.clear()
        while self._multi_cb_layout.count():
            item = self._multi_cb_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for path in tiffs:
            cb = QCheckBox(os.path.basename(path))
            cb.setChecked(True)
            cb.setToolTip(path)
            self._multi_cb_layout.addWidget(cb)
            self._multi_checkboxes.append(cb)
        self._multi_cb_layout.addStretch()
        self._multi_confirm_btn.setEnabled(bool(tiffs))
        if not tiffs:
            self._multi_dir_label.setText(f"{d}  (no TIFFs found)")

    def _on_multi_confirm_selection(self):
        self._multi_selected_paths = [
            self._multi_tiff_paths[i]
            for i, cb in enumerate(self._multi_checkboxes)
            if cb.isChecked()
        ]
        if not self._multi_selected_paths:
            self.status_label.setText("No TIFFs selected.")
            return
        N = len(self._multi_selected_paths)
        self._multi_elem_names = [
            pathlib.Path(p).stem.split("_")[-1]
            for p in self._multi_selected_paths
        ]
        while self._multi_preview_layout.count():
            item = self._multi_preview_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._multi_mini_labels.clear()
        normalize = self._normalize_check.isChecked()
        upscale = self._upscale_check.isChecked()
        circles = self._red_circles_check.isChecked()
        for i, path in enumerate(self._multi_selected_paths):
            color_entry = _MULTI_ELEM_COLORS[i % len(_MULTI_ELEM_COLORS)]
            cc = color_entry[1]
            elem_name = self._multi_elem_names[i]
            group = QGroupBox(elem_name)
            group.setStyleSheet(
                "QGroupBox { font-weight: bold; border: 1px solid #444; "
                "border-radius: 4px; margin-top: 6px; padding-top: 2px; }"
            )
            g_layout = QVBoxLayout(group)
            g_layout.setContentsMargins(2, 2, 2, 2)
            mini = QLabel()
            mini.setAlignment(Qt.AlignCenter)
            mini.setMinimumHeight(160)
            mini.setStyleSheet("background: #111; border: 1px solid #333;")
            try:
                pm, _, _ = _load_image_pixmap(path, normalize=normalize, upscale=upscale,
                                               red_circles=circles, circle_color=cc)
                scaled = pm.scaled(200, 160, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                mini.setPixmap(scaled)
            except Exception:
                pass
            g_layout.addWidget(mini)
            self._multi_preview_layout.addWidget(group)
            self._multi_mini_labels.append(mini)
        self._multi_preview_layout.addStretch()
        self._build_multi_composite_display()
        self._multi_show_flags = [True] * N
        self._multi_show_zones = True
        if self._model_ok:
            self.locate_btn.setEnabled(True)
        self.status_label.setText(f"{N} element(s) selected. Ready to analyze.")
        self.status_label.setStyleSheet("color: #27ae60; padding: 4px;")

    def _load_multi_features_json(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Multi Features JSON", "",
            "JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return
        try:
            with open(path) as f:
                data = json.load(f)
            elem_names = data.get("elements", [])
            elem_features = data.get("elem_features", [])
            if not elem_names or not elem_features:
                self.status_label.setText("Invalid features JSON — missing elements or elem_features.")
                self.status_label.setStyleSheet("color: #c0392b; padding: 4px;")
                return
            json_dir = pathlib.Path(path).parent
            # Match TIFFs by element name suffix
            tiff_paths = []
            for name in elem_names:
                matches = sorted(json_dir.glob(f"*_{name}.tif")) + sorted(json_dir.glob(f"*_{name}.tiff"))
                tiff_paths.append(str(matches[0]) if matches else None)
            self._multi_dir_path = str(json_dir)
            self._multi_dir_label.setText(str(json_dir))
            self._multi_selected_paths = [p for p in tiff_paths if p]
            self._multi_elem_names = elem_names
            self._multi_elem_features = elem_features
            N = len(elem_names)
            self._multi_show_flags = [True] * N
            self._multi_show_zones = True
            if self._multi_selected_paths:
                self._build_multi_composite_display()
            pm = self._multi_current_composite()
            if pm:
                self._show_multi_composite(pm)
            self._colocate_btn.setVisible(True)
            self._colocate_btn.setEnabled(True)
            self._multi_boxes_btn.setVisible(True)
            self._multi_boxes_btn.setText("● Boxes")
            self._reposition_overlay_btns()
            n = data.get("total_count", sum(len(f) for f in elem_features))
            self.status_label.setText(f"Loaded {N} elements, {n} features from {pathlib.Path(path).name}")
            self.status_label.setStyleSheet("color: #27ae60; padding: 4px;")
            self._populate_info_multi(elem_features, elem_names, pathlib.Path(path))
            self.locate_btn.setEnabled(True)
            self.info_panel.setVisible(True)
        except Exception as e:
            self.status_label.setText(f"Failed to load features JSON: {e}")
            self.status_label.setStyleSheet("color: #c0392b; padding: 4px;")

    def _load_multi_colocate_json(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Co-location JSON", "",
            "JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return
        try:
            with open(path) as f:
                data = json.load(f)
            zones = data.get("overlap_zones", [])
            if not zones:
                self.status_label.setText("No overlap zones found in that JSON.")
                self.status_label.setStyleSheet("color: #e67e22; padding: 4px;")
                return
            self._multi_colocate_zones = zones
            if not self._multi_dir_path:
                self._multi_dir_path = str(pathlib.Path(path).parent)
            pm = self._multi_current_composite()
            if pm:
                self._show_multi_composite(pm)
            self._multi_zones_btn.setVisible(True)
            self._multi_zones_btn.setText("◆ Zones")
            self._reposition_overlay_btns()
            n = len(zones)
            self.status_label.setText(f"Loaded {n} co-location zone(s) from {pathlib.Path(path).name}")
            self.status_label.setStyleSheet("color: #27ae60; padding: 4px;")
            self._populate_info_colocate(zones, pathlib.Path(path))
        except Exception as e:
            self.status_label.setText(f"Failed to load co-location JSON: {e}")
            self.status_label.setStyleSheet("color: #c0392b; padding: 4px;")

    def _build_multi_composite_display(self):
        if not self._multi_selected_paths:
            return
        try:
            normalize = self._normalize_check.isChecked()
            upscale = self._upscale_check.isChecked()

            def _load_gray(path):
                arr = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
                if arr is None:
                    raise ValueError(f"Could not read {path}")
                arr = arr.astype(np.float32)
                if normalize:
                    mn, mx = arr.min(), arr.max()
                    if mx > mn:
                        arr = (arr - mn) / (mx - mn)
                if upscale:
                    h_, w_ = arr.shape
                    if min(h_, w_) < 1092:
                        scale = 1092 / min(h_, w_)
                        arr = cv2.resize(arr, (int(w_ * scale), int(h_ * scale)),
                                         interpolation=cv2.INTER_CUBIC)
                return arr

            arrays = [_load_gray(p) for p in self._multi_selected_paths]
            h = max(a.shape[0] for a in arrays)
            w = max(a.shape[1] for a in arrays)

            # Normal additive composite
            composite = np.zeros((h, w, 3), dtype=np.float32)
            for i, arr in enumerate(arrays):
                color_entry = _MULTI_ELEM_COLORS[i % len(_MULTI_ELEM_COLORS)]
                r, g, b = color_entry[1]
                padded = np.zeros((h, w), dtype=np.float32)
                padded[:arr.shape[0], :arr.shape[1]] = arr
                composite[:, :, 0] = np.clip(composite[:, :, 0] + padded * (r / 255.0), 0, 1)
                composite[:, :, 1] = np.clip(composite[:, :, 1] + padded * (g / 255.0), 0, 1)
                composite[:, :, 2] = np.clip(composite[:, :, 2] + padded * (b / 255.0), 0, 1)
            composite_u8 = (composite * 255).astype(np.uint8)
            qimg = QImage(composite_u8.data, w, h, w * 3, QImage.Format_RGB888).copy()
            self._multi_composite_pixmap = QPixmap.fromImage(qimg)

            # Circles composite — each element circles on white, overlaid by color
            circle_canvas = np.full((h, w, 3), 255, dtype=np.uint8)
            for i, arr in enumerate(arrays):
                color_entry = _MULTI_ELEM_COLORS[i % len(_MULTI_ELEM_COLORS)]
                cc = color_entry[1]
                arr_u8 = (arr * 255).astype(np.uint8)
                ca = _apply_red_circles_transform(arr_u8, circle_color=cc)
                mask = np.any(ca < 255, axis=-1)
                ph, pw = ca.shape[:2]
                circle_canvas[:ph, :pw][mask] = ca[mask]
            qimg2 = QImage(circle_canvas.data, w, h, w * 3, QImage.Format_RGB888).copy()
            self._multi_circles_composite_pixmap = QPixmap.fromImage(qimg2)

            pm = self._multi_current_composite()
            if pm:
                self._show_multi_composite(pm)
            self._multi_switch_view_btn.setVisible(True)
            self._multi_switch_view_btn.setText(
                "⬤ Circles" if self._multi_view_circles else "⬤ Normal"
            )
            self._reposition_overlay_btns()
        except Exception as e:
            self.status_label.setText(f"Could not build composite: {e}")
            self.status_label.setStyleSheet("color: #c0392b; padding: 4px;")

    def _show_multi_composite(self, pixmap):
        sa = self._multi_composite_scroll
        vp_w = max(200, sa.viewport().width())
        vp_h = max(200, sa.viewport().height())
        fit = pixmap.scaled(vp_w, vp_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._multi_base_w, self._multi_base_h = fit.width(), fit.height()
        w = max(1, int(self._multi_base_w * self._multi_zoom))
        h = max(1, int(self._multi_base_h * self._multi_zoom))
        scaled = pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._multi_composite_display.setPixmap(scaled)
        self._multi_composite_display.resize(scaled.width(), scaled.height())

    def _multi_current_composite(self):
        base = (self._multi_circles_composite_pixmap if self._multi_view_circles
                else self._multi_composite_pixmap)
        if not base:
            return None
        groups = []
        for i, (feats, show) in enumerate(zip(self._multi_elem_features, self._multi_show_flags)):
            if show and feats:
                color_entry = _MULTI_ELEM_COLORS[i % len(_MULTI_ELEM_COLORS)]
                groups.append((feats, color_entry[0]))
        result = _draw_feature_boxes_multi_color(base, groups) if groups else base
        if self._multi_show_zones and self._multi_colocate_zones:
            result = _draw_colocate_boxes(result, self._multi_colocate_zones)
        return result

    def _on_multi_zoom(self, direction: int):
        sa = self._multi_composite_scroll
        viewport = sa.viewport()
        cursor_vp = viewport.mapFromGlobal(self._multi_composite_display.cursor().pos())
        old_h = sa.horizontalScrollBar().value()
        old_v = sa.verticalScrollBar().value()
        old_zoom = self._multi_zoom
        self._multi_zoom = max(0.25, min(8.0, self._multi_zoom * (1.15 if direction > 0 else 1 / 1.15)))
        pm = self._multi_current_composite()
        if pm:
            self._show_multi_composite(pm)
        ratio = self._multi_zoom / old_zoom
        sa.horizontalScrollBar().setValue(max(0, int((old_h + cursor_vp.x()) * ratio - cursor_vp.x())))
        sa.verticalScrollBar().setValue(max(0, int((old_v + cursor_vp.y()) * ratio - cursor_vp.y())))

    def _on_multi_mouse_moved(self, fx: float, fy: float) -> None:
        self.coord_label.setText(f"X: {fx:.3f}\nY: {fy:.3f}")
        self.coord_label.setStyleSheet("font-family: monospace; font-size: 11px; padding: 6px;")
        lines = []
        if self._multi_show_zones and self._multi_colocate_zones:
            for zone in self._multi_colocate_zones:
                if zone.get("x1", 0) <= fx <= zone.get("x2", 0) and \
                        zone.get("y1", 0) <= fy <= zone.get("y2", 0):
                    elem_str = " + ".join(zone.get("elements", []))
                    lines = [
                        f"<b>Zone {zone.get('id', '?')}  —  {elem_str}</b>",
                        f"bbox: ({zone['x1']:.3f},{zone['y1']:.3f})→({zone['x2']:.3f},{zone['y2']:.3f})",
                    ]
                    if zone.get("llm_suggestion"):
                        lines.append(f"<b>→ {zone['llm_suggestion']}</b>")
                    if zone.get("reasoning"):
                        lines.append(f"<i>{zone['reasoning']}</i>")
                    break
        if not lines:
            for i, (feats, show) in enumerate(zip(self._multi_elem_features, self._multi_show_flags)):
                if not show:
                    continue
                for feat in feats:
                    if feat.get("x1", 0) <= fx <= feat.get("x2", 0) and \
                            feat.get("y1", 0) <= fy <= feat.get("y2", 0):
                        color_entry = _MULTI_ELEM_COLORS[i % len(_MULTI_ELEM_COLORS)]
                        elem_name = self._multi_elem_names[i] if i < len(self._multi_elem_names) else f"Elem {i+1}"
                        fid = feat.get("id", "?")
                        label = feat.get("label", "")
                        desc = feat.get("description", "")
                        x1, y1 = feat.get("x1", 0), feat.get("y1", 0)
                        x2, y2 = feat.get("x2", 0), feat.get("y2", 0)
                        hex_color = color_entry[0].name()
                        lines = [
                            f'<b style="color:{hex_color};">{elem_name}</b>',
                            f"<b>Feature #{fid}</b>  {label}",
                        ]
                        if desc:
                            lines.append(f"<i>{desc}</i>")
                        lines.append(f"bbox: ({x1:.3f},{y1:.3f})→({x2:.3f},{y2:.3f})")
                        break
                if lines:
                    break
        if lines:
            self._hover_tooltip.setText("<br>".join(lines))
            self._hover_tooltip.adjustSize()
            gpos = QCursor.pos()
            self._hover_tooltip.move(gpos.x() + 16, gpos.y() - self._hover_tooltip.height() - 4)
            self._hover_tooltip.show()
        else:
            self._hover_tooltip.hide()

    def _on_multi_result(self, data: dict):
        self._multi_elem_features = data.get("elem_features", [])
        n = data.get("total_count", sum(len(f) for f in self._multi_elem_features))
        N = len(self._multi_selected_paths)
        ref_path = pathlib.Path(self._multi_dir_path)
        dir_name = ref_path.name
        json_path = ref_path / f"{dir_name}_multi_features.json"
        save_data = {
            "elements": self._multi_elem_names,
            "elem_features": self._multi_elem_features,
            "total_count": n,
            "model": _active_model,
        }
        with open(json_path, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"[Feature Locator Multi] All {N} elements done — {n} total feature(s). Saved at: {json_path}")
        self._multi_show_flags = [True] * N
        self._multi_show_zones = True
        pm = self._multi_current_composite()
        if pm:
            self._show_multi_composite(pm)
        self.locate_btn.setEnabled(True)
        self._colocate_btn.setVisible(True)
        self._colocate_btn.setEnabled(True)
        self._multi_boxes_btn.setVisible(True)
        self._multi_boxes_btn.setText("● Boxes")
        self._reposition_overlay_btns()
        self.status_label.setText(f"Done — {n} feature(s) across {N} elements. Saved: {json_path.name}")
        self.status_label.setStyleSheet("color: #27ae60; padding: 4px;")
        self._populate_info_multi(self._multi_elem_features, self._multi_elem_names, json_path)
        QMessageBox.information(self, "Multi Element Analysis Done",
            f"{N} elements analyzed — {n} total feature(s)\nSaved to: {json_path}")

    def _populate_info_multi(self, elem_features, elem_names, json_path):
        lines = [f"Model: {_active_model}", f"Saved: {json_path.name}", ""]
        for i, (feats, name) in enumerate(zip(elem_features, elem_names)):
            lines.append(f"── {name}  ({len(feats)} feature(s))")
            for j, feat in enumerate(feats):
                fid = feat.get("id", j + 1)
                label = feat.get("label", "")
                x1, y1 = feat.get("x1", 0), feat.get("y1", 0)
                x2, y2 = feat.get("x2", 0), feat.get("y2", 0)
                desc = feat.get("description", "")
                lines.append(f"  #{fid}  {label}")
                lines.append(f"       ({x1:.3f},{y1:.3f})→({x2:.3f},{y2:.3f})")
                if desc:
                    lines.append(f"       {desc}")
            lines.append("")
        self.info_panel.setPlainText("\n".join(lines).strip())
        self.info_panel.setVisible(True)

    def _on_multi_colocate_result(self, data: dict):
        zones = data.get("overlap_zones", [])
        n = data.get("total_zones", len(zones))
        data.setdefault("mode", "multi")
        data.setdefault("method", "geometric")
        ref_path = pathlib.Path(self._multi_dir_path)
        json_path = ref_path / f"{ref_path.name}_multi_colocate.json"
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[Multi Co-locate] {n} zone(s) identified. Saved at: {json_path}")
        self._multi_colocate_zones = zones
        pm = self._multi_current_composite()
        if pm:
            self._show_multi_composite(pm)
        self._colocate_btn.setEnabled(True)
        self._multi_zones_btn.setVisible(True)
        self._multi_zones_btn.setText("◆ Zones")
        self._reposition_overlay_btns()
        self.status_label.setText(f"Co-location done — {n} zone(s). Saved: {json_path.name}")
        self.status_label.setStyleSheet("color: #27ae60; padding: 4px;")
        self._populate_info_colocate(zones, json_path)

    def _get_dilate_ksize(self) -> int:
        return self._kernel_spin.value() if self._dilate_check.isChecked() else 0

    def _on_dilate_changed(self) -> None:
        self._kernel_spin.setEnabled(self._dilate_check.isChecked())
        self._dilated_pixmap = None
        if self._image_path and self._orig_pixmap:
            self._update_display()
        else:
            self._refresh_send_status()

    def _on_image_options_changed(self) -> None:
        mode = self._mode_combo.currentIndex()
        if mode == 1:
            for i in range(3):
                self._update_xrf_mini_preview(i)
            if all(self._xrf_paths):
                self._build_xrf_composite_display()
        elif mode == 2:
            if self._multi_selected_paths:
                self._on_multi_confirm_selection()
        else:
            self._orig_pixmap = None
            self._dilated_pixmap = None
            if self._image_path:
                self._update_display()

    def _update_display(self) -> None:
        normalize = self._normalize_check.isChecked()
        grid = self._grid_check.isChecked()
        upscale = self._upscale_check.isChecked()
        colormap = self._colormap_check.isChecked()
        red_circles = self._red_circles_check.isChecked()
        ksize = self._get_dilate_ksize()
        self._orig_pixmap, _, _ = _load_image_pixmap(
            self._image_path, dilate_ksize=0, normalize=normalize, grid_overlay=grid,
            upscale=upscale, colormap=colormap, red_circles=red_circles)
        if ksize > 0:
            self._dilated_pixmap, _, _ = _load_image_pixmap(
                self._image_path, dilate_ksize=ksize, normalize=normalize, grid_overlay=grid,
                upscale=upscale, colormap=colormap, red_circles=red_circles)
        pm = self._current_pixmap()
        if pm:
            self._show_pixmap(pm)
        self._refresh_send_status()

    def _refresh_send_status(self) -> None:
        if not self._image_path:
            self._send_status_label.setVisible(False)
            return
        parts = []
        if self._normalize_check.isChecked():
            parts.append("Normalize ✓")
        if self._grid_check.isChecked():
            parts.append("Grid overlay ✓")
        ksize = self._get_dilate_ksize()
        if ksize > 0:
            parts.append(f"Dilate {ksize}px ✓")
        if self._upscale_check.isChecked():
            parts.append("Upscale ✓")
        if self._red_circles_check.isChecked():
            parts.append("Red Circles ✓")
        elif self._colormap_check.isChecked():
            parts.append("Inferno ✓")
        text = "Sending to Claude:  " + "  |  ".join(parts) if parts else "Sending to Claude: raw image"
        self._send_status_label.setText(text)
        self._send_status_label.setVisible(True)

    def _on_mouse_moved(self, fx: float, fy: float) -> None:
        self.coord_label.setText(f"X: {fx:.3f}\nY: {fy:.3f}")
        self.coord_label.setStyleSheet("font-family: monospace; font-size: 11px; padding: 6px;")

        # Hover tooltip: only when boxes are visible
        hit = None
        if self._boxes_visible:
            for feat in self._features:
                if feat.get("x1", 0) <= fx <= feat.get("x2", 0) and feat.get("y1", 0) <= fy <= feat.get("y2", 0):
                    hit = feat
                    break

        if hit:
            fid = hit.get("id", "?")
            label = hit.get("label", "")
            desc = hit.get("description", "")
            x1, y1, x2, y2 = hit.get("x1", 0), hit.get("y1", 0), hit.get("x2", 0), hit.get("y2", 0)
            w_pct = (x2 - x1) * 100
            h_pct = (y2 - y1) * 100
            lines = [f"<b>Feature #{fid}</b>", label]
            if desc:
                lines.append(f"<i>{desc}</i>")
            lines.append(f"bbox: ({x1:.3f}, {y1:.3f}) → ({x2:.3f}, {y2:.3f})")
            lines.append(f"size: {w_pct:.1f}% × {h_pct:.1f}% of image")
            self._hover_tooltip.setText("<br>".join(lines))
            self._hover_tooltip.adjustSize()
            gpos = QCursor.pos()
            self._hover_tooltip.move(gpos.x() + 16, gpos.y() - self._hover_tooltip.height() - 4)
            self._hover_tooltip.show()
        else:
            self._hover_tooltip.hide()

    def _on_mouse_left(self) -> None:
        self.coord_label.setText("X: —\nY: —")
        self.coord_label.setStyleSheet("font-family: monospace; font-size: 11px; padding: 6px; color: #888;")
        self._hover_tooltip.hide()

    def _on_tile_progress(self, msg: str) -> None:
        self.status_label.setText(msg)
        self.status_label.setStyleSheet("color: #2980b9; padding: 4px;")


    def _on_error(self, msg):
        self.locate_btn.setEnabled(True)
        self.status_label.setText(f"Error: {msg}")
        self.status_label.setStyleSheet("color: #c0392b; padding: 4px;")


# ── model selector tab ────────────────────────────────────────────────────────

class _ModelTab(QWidget):
    model_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pull_thread: "_PullThread | None" = None
        self._radio_by_model: "dict[str, QRadioButton]" = {}
        self._init_ui()
        QTimer.singleShot(0, self._check_current)

    def _init_ui(self):
        outer = QVBoxLayout(self)
        outer.setAlignment(Qt.AlignTop)

        self._btn_grp = QButtonGroup(self)
        self._btn_grp.setExclusive(True)

        sections = [
            ("Local (Ollama)", "ollama"),
            ("Anthropic", "anthropic"),
            ("OpenAI", "openai"),
        ]
        for title, prov in sections:
            box = QGroupBox(title)
            bl = QVBoxLayout(box)
            bl.setSpacing(2)
            for model in KNOWN_MODELS:
                if _get_provider(model) != prov:
                    continue
                rb = QRadioButton(model)
                if model == _active_model:
                    rb.setChecked(True)
                rb.toggled.connect(lambda checked, m=model: checked and self._select(m))
                bl.addWidget(rb)
                self._btn_grp.addButton(rb)
                self._radio_by_model[model] = rb
            outer.addWidget(box)

        custom_row = QHBoxLayout()
        custom_row.addWidget(QLabel("Custom model:"))
        self._custom = QLineEdit()
        self._custom.setPlaceholderText("e.g. qwen2.5vl:14b or claude-opus-5")
        self._custom.returnPressed.connect(self._use_custom)
        custom_row.addWidget(self._custom, 1)
        use_btn = QPushButton("Use")
        use_btn.clicked.connect(self._use_custom)
        custom_row.addWidget(use_btn)
        outer.addLayout(custom_row)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        outer.addWidget(self._status)

        self._key_widget = QWidget()
        kr = QHBoxLayout(self._key_widget)
        kr.setContentsMargins(0, 4, 0, 0)
        self._key_label = QLabel("API Key:")
        kr.addWidget(self._key_label)
        self._key_input = QLineEdit()
        self._key_input.setEchoMode(QLineEdit.Password)
        self._key_input.setPlaceholderText("Paste API key here…")
        kr.addWidget(self._key_input, 1)
        save_btn = QPushButton("Use Key")
        save_btn.clicked.connect(self._save_key)
        kr.addWidget(save_btn)
        self._key_widget.setVisible(False)
        outer.addWidget(self._key_widget)

        self._dl_btn = QPushButton(f"Download {_active_model}")
        self._dl_btn.setVisible(False)
        self._dl_btn.clicked.connect(self._start_pull)
        outer.addWidget(self._dl_btn)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setVisible(False)
        self._log.setFixedHeight(100)
        self._log.setStyleSheet("font-family: monospace; font-size: 11px;")
        outer.addWidget(self._log)

        outer.addStretch()

    def _select(self, model: str):
        global _active_model
        _active_model = model
        self._dl_btn.setText(f"Download {model}")
        self.model_changed.emit(model)
        self._check_current()

    def _use_custom(self):
        m = self._custom.text().strip()
        if m:
            self._select(m)

    def _check_current(self):
        self._key_widget.setVisible(False)
        self._dl_btn.setVisible(False)
        self._log.setVisible(False)
        provider = _get_provider(_active_model)
        if provider == "ollama":
            self._check_ollama()
        else:
            self._check_cloud(provider)

    def _check_ollama(self):
        if not shutil.which("ollama"):
            self._set_status("Ollama not installed. Get it at https://ollama.com then restart.", "#c0392b")
            return
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
            if _active_model in result.stdout:
                self._set_status(f"{_active_model} is ready.", "#27ae60")
            else:
                self._set_status(f"{_active_model} is not downloaded yet.", "#e67e22")
                self._dl_btn.setVisible(True)
        except Exception as exc:
            self._set_status(f"Could not reach Ollama: {exc}", "#c0392b")

    def _check_cloud(self, provider: str):
        key, diag = _try_load_key(provider)
        if key:
            effective = _get_openai_model() if provider == "openai" else _active_model
            extra = f" (gateway model: {effective})" if effective != _active_model else ""
            self._set_status(f"{_active_model} is ready.{extra}", "#27ae60")
        else:
            provider_name = "Anthropic" if provider == "anthropic" else "OpenAI"
            env_var = _ENV_VAR[provider]
            msg = diag or f"No {env_var} found."
            self._set_status(msg, "#e67e22")
            self._key_label.setText(f"{provider_name} API Key:")
            self._key_widget.setVisible(True)

    def _save_key(self):
        provider = _get_provider(_active_model)
        key = self._key_input.text().strip()
        if not key:
            return
        _api_keys[provider] = key
        self._key_input.clear()
        self._key_widget.setVisible(False)
        self._set_status(f"{_active_model} is ready.", "#27ae60")

    def _set_status(self, msg: str, color: str):
        self._status.setText(msg)
        self._status.setStyleSheet(f"color: {color}; padding: 5px;")

    def _start_pull(self):
        if self._pull_thread and self._pull_thread.isRunning():
            return
        self._dl_btn.setVisible(False)
        self._log.clear()
        self._log.setVisible(True)
        self._set_status(f"Downloading {_active_model}…", "#e67e22")
        self._pull_thread = _PullThread()
        self._pull_thread.log.connect(self._log.append)
        self._pull_thread.finished.connect(self._on_pull_done)
        self._pull_thread.start()

    def _on_pull_done(self, ok: bool):
        if ok:
            self._log.setVisible(False)
            self._set_status(f"{_active_model} downloaded and ready.", "#27ae60")
            self.model_changed.emit(_active_model)
        else:
            self._set_status("Download failed. See log above.", "#c0392b")
            self._dl_btn.setVisible(True)

    def refresh_model(self):
        model = _active_model
        if model in self._radio_by_model:
            rb = self._radio_by_model[model]
            if not rb.isChecked():
                rb.setChecked(True)
        self._dl_btn.setText(f"Download {model}")
        self._check_current()


# ── Phase ID tab ──────────────────────────────────────────────────────────────

_PHASE_ID_SYSTEM_PROMPT = """\
You are an XRF (X-ray fluorescence) data analyst. You will receive a JSON of elemental blobs \
detected in an XRF scan. Each JSON entry represents one detected blob with these fields:
  - element: chemical symbol (e.g. Al, Fe, Ca, S)
  - image_center: [x, y] pixel coordinates in the scan image (x = column, y = row)
  - image_radius: blob radius in pixels
  - mean_intensity: average XRF signal within the blob (0–255 normalized)
  - max_intensity: peak XRF signal
  - cx, cy: physical scan position in microns (for beamline use — ignore for spatial grouping)
  - num_x, num_y: physical blob size in microns (for beamline use — ignore for spatial grouping)
  - fx_center: [fx, fy] pre-computed fractional position (image_center / image dimensions, 0–1)
  - fx_radius: pre-computed fractional radius (image_radius / image dimensions, 0–1)

Use image_center and image_radius for spatial proximity. Use fx_center and fx_radius for boxes.

Go through ALL blobs in the JSON. Group blobs of different elements whose image_center positions \
are within ~(image_radius_a + image_radius_b + 5) pixels of each other — these are co-located \
and likely the same particle or phase. For each group, compute elemental ratios:
  ratio(element) = mean_intensity(element) / sum(mean_intensities in group) × 100%

Record any recognizable elemental compositions or material phases, showing ratio arithmetic. \
If no known phase matches, still report the elements and their ratios.

For each cluster, derive the bounding box from fx_center and fx_radius of all contributing blobs:
  x1 = min(fx_center[0] - fx_radius)  ← top-left X
  y1 = min(fx_center[1] - fx_radius)  ← top-left Y
  x2 = max(fx_center[0] + fx_radius)  ← bottom-right X
  y2 = max(fx_center[1] + fx_radius)  ← bottom-right Y
Clamp all to [0.0, 1.0].

Respond ONLY with valid JSON:
{
  "phases": [
    {
      "id": 1,
      "material": "Stainless Steel 304",
      "x1": 0.47, "y1": 0.45, "x2": 0.63, "y2": 0.61,
      "elements": {"Fe": 52.0, "Cr": 29.0, "Ni": 19.0},
      "confidence": 0.75,
      "reasoning": "Fe at image_center=[83,79] r=8, Cr at [81,81] r=9, Ni at [84,79] r=7 — all within 5px. fx box: x1=0.47, y1=0.45, x2=0.63, y2=0.61. Ratios: Fe=233, Cr=130, Ni=85, sum=448 → Fe=52%, Cr=29%, Ni=19%. Fe-dominant with Cr+Ni — stainless steel family."
    }
  ],
  "summary": "Brief overall summary of what was found across all blobs."
}
"""

_PHASE_ID_USER_PROMPT = (
    "Analyze the XRF elemental blob data below. Group spatially co-located blobs, compute elemental "
    "ratios as instructed, identify the most likely material phase for each cluster, and return the "
    "JSON response with bounding boxes and reasoning.\n\nDATA:\n"
)


class _PhaseIDThread(QThread):
    result = Signal(dict)
    error = Signal(str)

    def __init__(self, json_data: dict, img_w: int, img_h: int,
                 system_prompt: str, user_note: str = ""):
        super().__init__()
        self._json_data = json_data
        self._img_w = img_w
        self._img_h = img_h
        self._system_prompt = system_prompt
        self._user_note = user_note

    def run(self):
        try:
            # Enrich entries with fractional coords before sending
            enriched = {}
            for k, v in self._json_data.items():
                entry = dict(v)
                cx, cy = entry.get("image_center", [0, 0])
                r = entry.get("image_radius", 1)
                entry["fx_center"] = [round(cx / self._img_w, 4), round(cy / self._img_h, 4)]
                entry["fx_radius"] = round(r / min(self._img_w, self._img_h), 4)
                enriched[k] = entry

            import json as _json
            data_str = _json.dumps(enriched, indent=2)
            note = f"\n\nAdditional focus: {self._user_note}" if self._user_note.strip() else ""
            user_msg = _PHASE_ID_USER_PROMPT + data_str + note

            provider = _get_provider(_active_model)
            if provider == "anthropic":
                parsed = self._query_anthropic(user_msg)
            elif provider == "openai":
                parsed = self._query_openai(user_msg)
            else:
                parsed = self._query_ollama(user_msg)
            self.result.emit(parsed)
        except Exception as e:
            self.error.emit(str(e))

    def _query_anthropic(self, user_msg: str) -> dict:
        import json as _json, urllib.request as _req
        key = _get_key("anthropic") or ""
        payload = _json.dumps({
            "model": _get_anthropic_model(),
            "max_tokens": 8192,
            "system": self._system_prompt,
            "messages": [{"role": "user", "content": user_msg}],
        }, ensure_ascii=False).encode("utf-8")
        request = _req.Request(
            _get_anthropic_url(), data=payload,
            headers={"Content-Type": "application/json", "x-api-key": key,
                     "anthropic-version": ANTHROPIC_API_VERSION},
        )
        with _req.urlopen(request, timeout=600) as resp:
            data = _json.loads(resp.read())
        return _parse_locate_json((data.get("content") or [{}])[0].get("text", "{}"))

    def _query_openai(self, user_msg: str) -> dict:
        import json as _json, urllib.request as _req
        key = _get_key("openai") or ""
        headers = {"Content-Type": "application/json", **_get_openai_auth_headers(key)}
        if "openai" in _api_urls:
            payload = _json.dumps({
                "model": _get_openai_model(),
                "instructions": self._system_prompt,
                "input": [{"role": "user", "content": [{"type": "input_text", "text": user_msg}]}],
            }, ensure_ascii=False).encode("utf-8")
            request = _req.Request(_get_openai_url(), data=payload, headers=headers)
            with _req.urlopen(request, timeout=600) as resp:
                data = _json.loads(resp.read())
            return _parse_locate_json(_parse_responses_text(data))
        else:
            payload = _json.dumps({
                "model": _active_model,
                "messages": [
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": user_msg},
                ],
            }, ensure_ascii=False).encode("utf-8")
            request = _req.Request(_get_openai_url(), data=payload, headers=headers)
            with _req.urlopen(request, timeout=600) as resp:
                data = _json.loads(resp.read())
            return _parse_locate_json(
                data.get("choices", [{}])[0].get("message", {}).get("content", "{}"))

    def _query_ollama(self, user_msg: str) -> dict:
        import json as _json, urllib.request as _req
        payload = _json.dumps({
            "model": _active_model,
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_msg},
            ],
            "stream": False,
            "format": "json",
        }, ensure_ascii=False).encode("utf-8")
        request = _req.Request(OLLAMA_URL, data=payload,
                               headers={"Content-Type": "application/json"})
        with _req.urlopen(request, timeout=600) as resp:
            data = _json.loads(resp.read())
        return _parse_locate_json(data.get("message", {}).get("content", "{}"))


def _draw_phase_boxes(base_pixmap, phases):
    """Draw phase boxes with material label on a copy of base_pixmap."""
    result = base_pixmap.copy()
    painter = QPainter(result)
    painter.setRenderHint(QPainter.Antialiasing)
    pw, ph = result.width(), result.height()
    font = painter.font()
    font.setPointSize(9)
    font.setBold(True)
    painter.setFont(font)

    for i, phase in enumerate(phases):
        color = BOX_COLORS[i % len(BOX_COLORS)]
        x1 = int(max(0.0, min(1.0, float(phase.get("x1", 0)))) * pw)
        y1 = int(max(0.0, min(1.0, float(phase.get("y1", 0)))) * ph)
        x2 = int(max(0.0, min(1.0, float(phase.get("x2", 1)))) * pw)
        y2 = int(max(0.0, min(1.0, float(phase.get("y2", 1)))) * ph)
        painter.setPen(QPen(color, 2))
        painter.drawRect(x1, y1, x2 - x1, y2 - y1)
        label = phase.get("material", f"Phase {phase.get('id', i+1)}")
        fm = painter.fontMetrics()
        tw = fm.horizontalAdvance(label) + 6
        th = fm.height() + 2
        painter.fillRect(x1, y1 - th, tw, th, QColor(0, 0, 0, 160))
        painter.setPen(color)
        painter.drawText(x1 + 3, y1 - 3, label)

    painter.end()
    return result


class _PhaseIDTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._json_path = None
        self._json_data = None
        self._img_w = 150
        self._img_h = 150
        self._thread = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # ── prompt box (always visible) ──
        layout.addWidget(QLabel("System Prompt:"))
        self._prompt_edit = QTextEdit()
        self._prompt_edit.setPlainText(_PHASE_ID_SYSTEM_PROMPT)
        self._prompt_edit.setMinimumHeight(140)
        self._prompt_edit.setMaximumHeight(180)
        layout.addWidget(self._prompt_edit)

        # ── JSON file picker ──
        json_row = QHBoxLayout()
        json_btn = QPushButton("Pick Blob JSON...")
        json_btn.clicked.connect(self._pick_json)
        json_row.addWidget(json_btn)
        self._json_label = QLabel("No JSON selected  (MASTER_raw_detections.json or group_all_boxes_*.json)")
        self._json_label.setStyleSheet("color: #888;")
        json_row.addWidget(self._json_label, 1)
        layout.addLayout(json_row)

        # ── optional focus note ──
        note_row = QHBoxLayout()
        note_row.addWidget(QLabel("Focus (optional):"))
        self._note_edit = QLineEdit()
        self._note_edit.setPlaceholderText(
            "e.g. 'look for iron-sulfide phases' or 'focus on stainless steel'")
        note_row.addWidget(self._note_edit, 1)
        layout.addLayout(note_row)

        # ── run button ──
        self._run_btn = QPushButton("Identify Phases")
        self._run_btn.setMinimumHeight(40)
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._run)
        layout.addWidget(self._run_btn)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        # ── results text (main output) ──
        layout.addWidget(QLabel("Results:"))
        self._results_text = QTextEdit()
        self._results_text.setReadOnly(True)
        self._results_text.setPlaceholderText("Phase analysis results will appear here…")
        self._results_text.setStyleSheet("font-family: monospace; font-size: 12px;")
        layout.addWidget(self._results_text, 1)

    def refresh_model(self):
        pass

    def _pick_json(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Blob JSON", "",
            "JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return
        import json as _json
        try:
            with open(path) as f:
                self._json_data = _json.load(f)
            self._json_path = path
            self._json_label.setText(os.path.basename(path))
            self._json_label.setStyleSheet("")
            self._run_btn.setEnabled(True)
        except Exception as e:
            self._status.setText(f"Could not load JSON: {e}")
            self._status.setStyleSheet("color: #c0392b; padding: 4px;")

    def _run(self):
        if self._thread and self._thread.isRunning():
            return
        if not self._json_data:
            return
        self._run_btn.setEnabled(False)
        self._status.setText("Analyzing elemental data… this may take 30–90 seconds.")
        self._status.setStyleSheet("color: #2980b9; padding: 4px;")
        system_prompt = self._prompt_edit.toPlainText().strip() or _PHASE_ID_SYSTEM_PROMPT
        self._thread = _PhaseIDThread(
            self._json_data, self._img_w, self._img_h,
            system_prompt, self._note_edit.text()
        )
        self._thread.result.connect(self._on_result)
        self._thread.error.connect(self._on_error)
        self._thread.start()

    def _on_result(self, data):
        self._run_btn.setEnabled(True)
        try:
            phases = data.get("phases", [])
            if not isinstance(phases, list):
                raise ValueError(f"Expected 'phases' list, got: {type(phases).__name__}")
        except Exception as e:
            self._on_error(f"Could not parse response: {e}\n\nRaw: {data}")
            return

        summary = data.get("summary", "")
        lines = []
        if summary:
            lines.append(f"SUMMARY: {summary}\n")
        for ph in phases:
            mat = ph.get("material", "Unknown")
            conf = ph.get("confidence", "?")
            conf_str = f"{float(conf)*100:.0f}%" if isinstance(conf, (int, float)) else str(conf)
            elems = ph.get("elements", {})
            elem_str = "  |  ".join(f"{el}: {pct}%" for el, pct in sorted(
                elems.items(), key=lambda x: -float(x[1])))
            reasoning = ph.get("reasoning", "")
            lines.append(f"{'─'*60}")
            lines.append(f"Phase {ph.get('id','?')}: {mat}  (confidence: {conf_str})")
            if elem_str:
                lines.append(f"Composition:  {elem_str}")
            lines.append(f"Reasoning:    {reasoning}")
            lines.append("")
        self._results_text.setPlainText("\n".join(lines))
        n = len(phases)
        self._status.setText(f"Found {n} phase{'s' if n != 1 else ''}.")
        self._status.setStyleSheet("color: #27ae60; padding: 4px;")

    def _on_error(self, msg):
        self._run_btn.setEnabled(True)
        self._status.setText(f"Error: {msg}")
        self._status.setStyleSheet("color: #c0392b; padding: 4px;")


# ── public widget ─────────────────────────────────────────────────────────────

class LLMJsonMakerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 0)
        layout.setSpacing(4)

        self._model_tab = _ModelTab()
        self._chat_tab = _ChatTab()
        self._locate_tab = _LocateTab()
        self._phase_id_tab = _PhaseIDTab()

        tabs = QTabWidget()
        tabs.addTab(self._model_tab, "Model")
        tabs.addTab(self._chat_tab, "Chat")
        tabs.addTab(self._locate_tab, "Feature Locator")
        tabs.addTab(self._phase_id_tab, "Phase ID")
        layout.addWidget(tabs)

        self._model_tab.model_changed.connect(self._on_model_changed)

    def _on_model_changed(self, text: str) -> None:
        global _active_model
        _active_model = text.strip()
        self._chat_tab.refresh_model()
        self._locate_tab.refresh_model()
        self._phase_id_tab.refresh_model()
