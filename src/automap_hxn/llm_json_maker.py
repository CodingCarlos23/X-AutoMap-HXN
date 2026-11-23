import base64
import json
import os
import pathlib
import re
import subprocess
import shutil
import urllib.request
from datetime import datetime

import cv2
import numpy as np

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QSizePolicy, QLineEdit, QFileDialog, QTabWidget,
    QScrollArea, QMessageBox, QComboBox, QCheckBox, QSpinBox,
    QGroupBox, QRadioButton, QButtonGroup,
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

# Common chemical element symbols (most relevant for XRF)
ELEMENT_SYMBOLS = {
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
    'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
    'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
    'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf'
}


def extract_element_from_filename(filepath: str) -> str:
    """
    Extract element symbol from XRF TIFF filename.

    Supports patterns like:
    - detsum_Ca_K.tiff → Ca
    - Ca_K.tiff → Ca
    - Fe.tiff → Fe
    - element_Ca.tif → Ca
    - scan_001_P_data.tif → P

    Returns:
        Element symbol (e.g., "Ca", "Fe", "P") or empty string if not found
    """
    filename = os.path.basename(filepath)
    # Remove extension
    name_no_ext = os.path.splitext(filename)[0]

    # Split by common delimiters
    parts = re.split(r'[_\-\.\s]+', name_no_ext)

    # Look for element symbols in parts (case-insensitive match)
    for part in parts:
        # Check if this part matches a known element symbol
        if part in ELEMENT_SYMBOLS:
            return part
        # Check capitalized version (e.g., "ca" → "Ca")
        if part.capitalize() in ELEMENT_SYMBOLS:
            return part.capitalize()
        # Check uppercase version (e.g., "fe" → "Fe")
        if len(part) == 2 and part.upper() in ELEMENT_SYMBOLS:
            return part.upper()
        if len(part) == 1 and part.upper() in ELEMENT_SYMBOLS:
            return part.upper()

    # Fallback: look for element pattern anywhere in the filename
    for elem in ELEMENT_SYMBOLS:
        # Match element symbol as a separate word (with word boundaries)
        pattern = r'\b' + re.escape(elem) + r'\b'
        if re.search(pattern, filename, re.IGNORECASE):
            return elem

    return ""


# ============================================================================
# H5 File Reading Functions (using xraylarch python which has h5py)
# ============================================================================

XRAYLARCH_PYTHON = "/Users/Lawrence/xraylarch/bin/python"


def _read_h5_positions(h5_path: str):
    """
    Read position data from h5 file.

    Returns:
        tuple: (x_pos_2d, y_pos_2d) - 2D arrays of physical coordinates in microns
               Both arrays have shape (height, width)
    """
    script = f"""
import h5py
import numpy as np
import json

with h5py.File('{h5_path}', 'r') as f:
    pos_data = f['/xrfmap/positions/pos'][:]  # Shape: (2, height, width)
    x_pos = pos_data[0]  # Shape: (height, width)
    y_pos = pos_data[1]  # Shape: (height, width)

    # Convert to lists for JSON serialization
    result = {{
        'x_pos': x_pos.tolist(),
        'y_pos': y_pos.tolist(),
        'shape': list(x_pos.shape)
    }}
    print(json.dumps(result))
"""

    result = subprocess.run(
        [XRAYLARCH_PYTHON, "-c", script],
        capture_output=True,
        text=True,
        check=True
    )

    data = json.loads(result.stdout)
    x_pos = np.array(data['x_pos'])
    y_pos = np.array(data['y_pos'])

    return x_pos, y_pos


def _read_h5_element_data(h5_path: str, element_name: str):
    """
    Read element intensity data from h5 file.

    Args:
        h5_path: Path to h5 file
        element_name: Element to read (e.g., "Ca_K", "P_K", "Ce_L")

    Returns:
        np.ndarray: 2D array of element intensities, shape (height, width)
    """
    script = f"""
import h5py
import numpy as np
import json

with h5py.File('{h5_path}', 'r') as f:
    # Get element names
    elem_names = [name.decode('utf-8').rstrip('\\x00') for name in f['/xrfmap/detsum/xrf_fit_name'][:]]

    # Find index of requested element
    try:
        elem_idx = elem_names.index('{element_name}')
    except ValueError:
        print(json.dumps({{'error': 'Element not found', 'available': elem_names}}))
        exit(1)

    # Read data for this element
    data = f['/xrfmap/detsum/xrf_fit'][elem_idx]  # Shape: (height, width)

    result = {{
        'data': data.tolist(),
        'shape': list(data.shape)
    }}
    print(json.dumps(result))
"""

    result = subprocess.run(
        [XRAYLARCH_PYTHON, "-c", script],
        capture_output=True,
        text=True,
        check=True
    )

    data = json.loads(result.stdout)
    if 'error' in data:
        raise ValueError(f"{data['error']}: {data.get('available', [])}")

    return np.array(data['data'])


def _get_h5_element_list(h5_path: str):
    """
    Get list of available elements in h5 file.

    Returns:
        list: Element names (e.g., ["Ca_K", "P_K", "Ce_L", ...])
    """
    script = f"""
import h5py
import json

with h5py.File('{h5_path}', 'r') as f:
    elem_names = [name.decode('utf-8').rstrip('\\x00') for name in f['/xrfmap/detsum/xrf_fit_name'][:]]
    print(json.dumps(elem_names))
"""

    result = subprocess.run(
        [XRAYLARCH_PYTHON, "-c", script],
        capture_output=True,
        text=True,
        check=True
    )

    return json.loads(result.stdout)


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
    "gpt-5.6-luna",
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
    if content.startswith("```"):
        content = "\n".join(content.splitlines()[1:-1])
    return json.loads(content)

LOCATE_SYSTEM_PROMPT = (
    "You are a scientific image analysis assistant for XRF (X-ray fluorescence) scan data. "
    "Identify all distinct bright features, particles, or clusters visible in the image. "
    "Respond ONLY with valid JSON using this exact schema — no extra text outside the JSON object:\n"
    '{"features": [{"id": 1, "label": "particle cluster", "x1": 0.10, "y1": 0.20, '
    '"x2": 0.45, "y2": 0.60, "confidence": 0.92, "description": "bright region with high signal"}], "total_count": 1}\n'
    "IMPORTANT: All coordinates (x1, y1, x2, y2) must be FRACTIONAL values between 0.0 and 1.0, "
    "where (0,0) is the top-left corner and (1,1) is the bottom-right corner of the image. "
    "x1 < x2, y1 < y2. "
    "Draw each bounding box tightly around the BRIGHT CORE of each feature — "
    "do not include surrounding dark background. Be as precise as possible with the box edges. "
    "Include every distinct feature you can identify.\n"
    "The 'confidence' field (0.0 to 1.0) indicates how certain you are that this is a distinct feature "
    "based on signal brightness, clarity, and distinctness. Use the full scale: higher values for "
    "clear/bright features, lower values for faint/ambiguous regions."
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
    '"x2": 0.45, "y2": 0.60, "confidence": 0.88, "description": "bright isolated particle cluster"}], "total_count": 1}\n'
    "IMPORTANT: All coordinates (x1, y1, x2, y2) must be FRACTIONAL values between 0.0 and 1.0, "
    "where (0,0) is top-left and (1,1) is bottom-right. x1 < x2, y1 < y2. "
    "Draw boxes tightly around the bright core of each feature — exclude surrounding dark background. "
    "For very small particles, the box may only span 0.01–0.05 in each dimension — that is fine.\n"
    "The 'confidence' field (0.0 to 1.0) indicates your certainty that this is a real particle vs noise. "
    "Use the full scale based on brightness, clarity, and signal-to-background ratio."
)

LOCATE_XRF_ROI_PROMPT = (
    "You are analyzing a composite XRF (X-ray fluorescence) scan image where three element channels "
    "are encoded as RGB: Red = element 1, Green = element 2, Blue = element 3. "
    "Identify ALL distinct regions of interest (ROIs), including:\n"
    "1. MULTI-ELEMENT overlaps (2+ elements) — mixed colors (yellow, cyan, magenta, white)\n"
    "2. SINGLE-ELEMENT regions with significant signal — pure red, green, or blue\n"
    "Respond ONLY with valid JSON using this exact schema — no extra text outside the JSON object:\n"
    '{"features": [{"id": 1, "label": "multi-element AOI", "x1": 0.10, "y1": 0.20, '
    '"x2": 0.45, "y2": 0.60, "confidence": 0.91, "description": "co-localized signal from element1+element2"}, '
    '{"id": 2, "label": "single-element AOI", "x1": 0.50, "y1": 0.30, "x2": 0.65, "y2": 0.45, '
    '"confidence": 0.73, "description": "single-element signal from element3"}], "total_count": 2}\n'
    "IMPORTANT: All coordinates (x1, y1, x2, y2) must be FRACTIONAL values between 0.0 and 1.0, "
    "where (0,0) is top-left and (1,1) is bottom-right. x1 < x2, y1 < y2. "
    "For multi-element: 'co-localized signal from X+Y'. For single: 'single-element signal from X'.\n"
    "The 'confidence' field (0.0 to 1.0) indicates your certainty that this is a distinct ROI. "
    "Use the full scale based on signal brightness, clarity, and how well-defined the region is."
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


def _load_image_pixmap(path, dilate_ksize: int = 0, normalize: bool = True, grid_overlay: bool = False,
                       upscale: bool = False, colormap: bool = False):
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
    if colormap and arr.ndim == 2:
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


def _draw_boxes_on_matplotlib_figure(fig, ax, features, draw_boxes=True):
    """Draw colored bounding boxes on matplotlib axes.

    Args:
        fig: Matplotlib figure
        ax: Main axes to draw on
        features: List of feature dicts with x1, y1, x2, y2 (in axes coordinates)
        draw_boxes: If True, draw the boxes; if False, just render the figure

    Returns:
        QPixmap with the rendered figure
    """
    from matplotlib.patches import Rectangle
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import io

    # Color palette matching BOX_COLORS
    box_colors_mpl = [
        '#FF5050',  # Red
        '#50A0FF',  # Blue
        '#50DC50',  # Green
        '#FFC832',  # Yellow
        '#B450FF',  # Purple
        '#32DCDC',  # Cyan
        '#FF8C32',  # Orange
    ]

    # Store existing patches to remove them later if needed
    existing_rects = [p for p in ax.patches if isinstance(p, Rectangle)]

    if draw_boxes and features:
        # Remove old boxes first
        for rect in existing_rects:
            rect.remove()

        # Get axes limits for clamping (works for both fractional and physical coordinates)
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        # Note: y-axis is inverted (y_min > y_max typically), so we need to handle that
        y_lower = min(y_min, y_max)
        y_upper = max(y_min, y_max)

        # Draw rectangles on the axes (coordinates match the axes data coordinates)
        for i, feat in enumerate(features):
            color = box_colors_mpl[i % len(box_colors_mpl)]
            # Get coordinates and clamp to axes limits
            x1 = max(x_min, min(x_max, float(feat.get("x1", x_min))))
            y1 = max(y_lower, min(y_upper, float(feat.get("y1", y_lower))))
            x2 = max(x_min, min(x_max, float(feat.get("x2", x_max))))
            y2 = max(y_lower, min(y_upper, float(feat.get("y2", y_upper))))

            width = x2 - x1
            height = y2 - y1

            rect = Rectangle((x1, y1), width, height, linewidth=2,
                             edgecolor=color, facecolor='none', zorder=100)
            ax.add_patch(rect)
    else:
        # Remove all boxes
        for rect in existing_rects:
            rect.remove()

    # Save to buffer - use fig.savefig() directly instead of plt.savefig()
    buf = io.BytesIO()
    try:
        # Draw the canvas to ensure all patches are rendered
        fig.canvas.draw()
        # Save the figure directly (don't use plt.savefig which might target wrong figure)
        # Preserve dark background
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                   facecolor=fig.get_facecolor(), edgecolor='none')
        buf.seek(0)

        # Load into QPixmap
        qimg = QImage()
        success = qimg.loadFromData(buf.getvalue())
        if not success or qimg.isNull():
            raise RuntimeError("Failed to load image data into QImage")

        pixmap = QPixmap.fromImage(qimg)
        if pixmap.isNull():
            raise RuntimeError("Failed to create QPixmap from QImage")

        return pixmap

    except Exception as e:
        print(f"Error saving matplotlib figure: {e}")
        # Return a blank pixmap as fallback
        return QPixmap()


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


def _encode_xrf_figure_for_vlm(path):
    """Encode XRF matplotlib figure for VLM - no processing, just load and encode."""
    arr = cv2.imread(str(path), cv2.IMREAD_ANYCOLOR)
    if arr is None:
        raise ValueError(f"Could not read image: {path}")

    # No CLAHE, no resizing, no processing - figure is already perfect
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise RuntimeError("cv2.imencode failed")

    # Save debug copy
    import pathlib
    debug_final = pathlib.Path("/tmp/_llm_input_final_processed.png")
    cv2.imwrite(str(debug_final), arr)
    print(f"DEBUG: Final XRF figure (sent to LLM) saved to: {debug_final}")

    return base64.b64encode(buf.tobytes()).decode()


def _encode_image_for_vlm(path, max_dim=_VLM_MAX_DIM, dilate_ksize: int = 0, normalize: bool = True,
                          grid_overlay: bool = False, upscale: bool = False, colormap: bool = False):
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

    # Save a debug copy of the final processed image sent to LLM (for XRF mode)
    # Save to /tmp with a fixed name so user can easily find it
    import pathlib
    if "/tmp" in str(path) or "/var/folders" in str(path):
        debug_final = pathlib.Path("/tmp/_llm_input_final_processed.png")
        try:
            cv2.imwrite(str(debug_final), arr)
            print(f"DEBUG: Final processed image (sent to LLM) saved to: {debug_final}")
        except:
            pass  # Don't fail detection if debug save fails

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
                 grid_overlay: bool = False, upscale: bool = False, colormap: bool = False, is_xrf_figure: bool = False):
        super().__init__()
        self.image_path = image_path
        self.dilate_ksize = dilate_ksize
        self.normalize = normalize
        self.grid_overlay = grid_overlay
        self.upscale = upscale
        self.colormap = colormap
        self.is_xrf_figure = is_xrf_figure
        self._system_prompt = system_prompt_override or LOCATE_SYSTEM_PROMPT

    def run(self):
        try:
            if self.is_xrf_figure:
                # XRF figure: already processed, just encode as-is
                img_b64 = _encode_xrf_figure_for_vlm(self.image_path)
            else:
                # Single image: apply normal preprocessing
                img_b64 = _encode_image_for_vlm(self.image_path, dilate_ksize=self.dilate_ksize, normalize=self.normalize,
                                                 grid_overlay=self.grid_overlay, upscale=self.upscale, colormap=self.colormap)
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
                 grid_overlay: bool = False, upscale: bool = False, colormap: bool = False):
        super().__init__()
        self.image_path = image_path
        self.dilate_ksize = dilate_ksize
        self.normalize = normalize
        self.grid_overlay = grid_overlay
        self.upscale = upscale
        self.colormap = colormap
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
                    img_b64 = self._encode_crop(crop, colormap=self.colormap)
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
    def _encode_crop(crop: np.ndarray, colormap: bool = False) -> str:
        if crop.ndim == 2:
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


class _LocateTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._image_path = None
        self._locate_thread = None
        self._orig_pixmap = None
        self._dilated_pixmap = None
        self._annotated_pixmap = None
        self._features = []
        self._displayed_features = []  # Features currently shown in the list (after filtering)
        self._filtered_features_cache = []  # Cache of filtered features
        self._zoom = 1.0
        self._model_ok = True
        self._xrf_paths = [None, None, None]
        self._xrf_elem_names = ["Element 1", "Element 2", "Element 3"]
        self._last_used_prompt = LOCATE_SYSTEM_PROMPT  # Stores the actual prompt used for detection
        self._xrf_source_type = "TIFF"  # "TIFF" or "H5"
        self._xrf_h5_path = None  # Path to h5 file
        self._xrf_h5_elements = []  # Available elements in h5 file
        self._xrf_h5_selected = [None, None, None]  # Selected element indices for RGB
        self._xrf_positions = None  # (x_pos, y_pos) arrays from h5 file
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
        self._mode_combo.addItems(["Single Image", "XRF ROI (3 Elements)"])
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self._mode_combo)
        mode_row.addStretch()
        layout.addLayout(mode_row)

        # Prompt preset selector (Single Image mode only)
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

        # --- Single Image mode controls ---
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
        self._grid_check.setChecked(True)
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

        # --- XRF ROI mode controls ---
        self._xrf_widget = QWidget()
        self._xrf_widget.setVisible(False)
        xrf_layout = QVBoxLayout(self._xrf_widget)
        xrf_layout.setContentsMargins(0, 0, 0, 0)
        xrf_layout.setSpacing(4)

        # Source type selector
        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Data source:"))
        self._source_btn_group = QButtonGroup(self)
        self._tiff_radio = QRadioButton("TIFF files (3 separate)")
        self._h5_radio = QRadioButton("H5 file (with positions)")
        self._tiff_radio.setChecked(True)
        self._source_btn_group.addButton(self._tiff_radio)
        self._source_btn_group.addButton(self._h5_radio)
        self._tiff_radio.toggled.connect(self._on_source_type_changed)
        source_row.addWidget(self._tiff_radio)
        source_row.addWidget(self._h5_radio)
        source_row.addStretch()
        xrf_layout.addLayout(source_row)

        # TIFF mode controls (3 separate file pickers)
        self._tiff_controls = QWidget()
        tiff_layout = QVBoxLayout(self._tiff_controls)
        tiff_layout.setContentsMargins(0, 0, 0, 0)
        tiff_layout.setSpacing(2)

        colors = ["Red", "Green", "Blue"]
        self._xrf_pick_btns = []
        self._xrf_name_labels = []
        self._xrf_elem_inputs = []
        for i, color in enumerate(colors):
            row = QHBoxLayout()
            btn = QPushButton(f"Pick {color} Channel...")
            btn.clicked.connect(lambda _, idx=i: self._pick_xrf_image(idx))
            lbl = QLabel("No file selected")
            lbl.setStyleSheet("color: #888;")
            elem_input = QLineEdit()
            elem_input.setPlaceholderText(f"Element {i+1}")
            elem_input.setText(self._xrf_elem_names[i])
            elem_input.setMaximumWidth(100)
            elem_input.textChanged.connect(lambda text, idx=i: self._on_elem_name_changed(idx, text))
            row.addWidget(btn)
            row.addWidget(elem_input)
            row.addWidget(lbl, 1)
            tiff_layout.addLayout(row)
            self._xrf_pick_btns.append(btn)
            self._xrf_name_labels.append(lbl)
            self._xrf_elem_inputs.append(elem_input)

        xrf_layout.addWidget(self._tiff_controls)

        # H5 mode controls (1 file picker + 3 element selectors)
        self._h5_controls = QWidget()
        self._h5_controls.setVisible(False)
        h5_layout = QVBoxLayout(self._h5_controls)
        h5_layout.setContentsMargins(0, 0, 0, 0)
        h5_layout.setSpacing(2)

        # H5 file picker
        h5_file_row = QHBoxLayout()
        self._h5_pick_btn = QPushButton("Pick H5 File...")
        self._h5_pick_btn.clicked.connect(self._pick_h5_file)
        self._h5_file_label = QLabel("No H5 file selected")
        self._h5_file_label.setStyleSheet("color: #888;")
        h5_file_row.addWidget(self._h5_pick_btn)
        h5_file_row.addWidget(self._h5_file_label, 1)
        h5_layout.addLayout(h5_file_row)

        # Element selectors for RGB channels
        self._h5_elem_combos = []
        for i, color in enumerate(colors):
            row = QHBoxLayout()
            row.addWidget(QLabel(f"{color} Channel:"))
            combo = QComboBox()
            combo.setEnabled(False)
            combo.currentIndexChanged.connect(lambda idx, ch=i: self._on_h5_elem_selected(ch, idx))
            row.addWidget(combo, 1)
            h5_layout.addLayout(row)
            self._h5_elem_combos.append(combo)

        xrf_layout.addWidget(self._h5_controls)

        layout.addWidget(self._xrf_widget)

        # Detection buttons row
        detect_row = QHBoxLayout()
        self.locate_btn = QPushButton("Get Location of Features")
        self.locate_btn.setMinimumHeight(40)
        self.locate_btn.setEnabled(False)
        self.locate_btn.clicked.connect(self._run_locate)
        detect_row.addWidget(self.locate_btn, 3)

        self.load_json_btn = QPushButton("Load JSON")
        self.load_json_btn.setMinimumHeight(40)
        self.load_json_btn.setToolTip("Load previously saved detection results from JSON file")
        self.load_json_btn.clicked.connect(self._load_json_results)
        detect_row.addWidget(self.load_json_btn, 1)
        layout.addLayout(detect_row)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # Element filter checkboxes (for XRF mode)
        self._filter_group = QGroupBox("Filter by Elements (co-localization)")
        self._filter_group.setVisible(False)  # Hidden until features are loaded
        filter_layout = QHBoxLayout(self._filter_group)
        filter_layout.setContentsMargins(8, 8, 8, 8)

        filter_label = QLabel("Show features containing:")
        filter_label.setStyleSheet("font-weight: bold;")
        filter_layout.addWidget(filter_label)

        self._elem_filter_checks = []
        for i, elem_name in enumerate(self._xrf_elem_names):
            check = QCheckBox(elem_name)
            check.setChecked(True)  # All selected by default
            check.toggled.connect(self._on_element_filter_changed)
            filter_layout.addWidget(check)
            self._elem_filter_checks.append(check)

        # Add "Exact match" checkbox
        self._exact_match_check = QCheckBox("Exact match only")
        self._exact_match_check.setToolTip(
            "When checked: show ONLY features with exactly the selected elements (e.g., Ca+P excludes Ca+P+Ce)\n"
            "When unchecked: show features containing AT LEAST the selected elements (e.g., Ca+P includes Ca+P+Ce)"
        )
        self._exact_match_check.setStyleSheet("font-weight: bold; color: #e67e22;")
        self._exact_match_check.toggled.connect(self._on_element_filter_changed)
        filter_layout.addWidget(self._exact_match_check)

        filter_layout.addStretch()

        # Add "Select All" and "Clear All" buttons
        select_all_btn = QPushButton("All")
        select_all_btn.setMaximumWidth(60)
        select_all_btn.clicked.connect(lambda: self._set_all_filters(True))
        filter_layout.addWidget(select_all_btn)

        clear_all_btn = QPushButton("None")
        clear_all_btn.setMaximumWidth(60)
        clear_all_btn.clicked.connect(lambda: self._set_all_filters(False))
        filter_layout.addWidget(clear_all_btn)

        layout.addWidget(self._filter_group)

        # Confidence threshold filter
        conf_row = QHBoxLayout()
        conf_label = QLabel("Min confidence:")
        conf_label.setToolTip("Only show features with confidence >= this threshold")
        conf_row.addWidget(conf_label)

        self._conf_spin = QSpinBox()
        self._conf_spin.setRange(0, 100)
        self._conf_spin.setValue(0)
        self._conf_spin.setSuffix("%")
        self._conf_spin.setMaximumWidth(80)
        self._conf_spin.valueChanged.connect(self._on_confidence_filter_changed)
        conf_row.addWidget(self._conf_spin)
        conf_row.addStretch()
        layout.addLayout(conf_row)

        # Display toggles
        display_toggle_row = QHBoxLayout()
        self._show_boxes_check = QCheckBox("Show Boxes")
        self._show_boxes_check.setChecked(True)
        self._show_boxes_check.toggled.connect(self._on_boxes_toggled)
        display_toggle_row.addWidget(self._show_boxes_check)

        self._show_grid_check = QCheckBox("Show Grid Lines")
        self._show_grid_check.setChecked(True)
        self._show_grid_check.setToolTip("Toggle grid display (LLM always sees grid)")
        self._show_grid_check.toggled.connect(self._on_grid_display_toggled)
        display_toggle_row.addWidget(self._show_grid_check)

        display_toggle_row.addStretch()
        layout.addLayout(display_toggle_row)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.viewport().setMouseTracking(True)
        self.img_display = _ImageDisplay()
        self.img_display.setAlignment(Qt.AlignCenter)
        self.img_display.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.img_display.mouse_moved.connect(self._on_mouse_moved)
        self.img_display.mouse_left.connect(self._on_mouse_left)
        self.img_display.zoom_scrolled.connect(self._on_zoom)
        self.scroll_area.setWidget(self.img_display)
        self.scroll_area.setVisible(False)

        self.coord_label = QLabel("X: —\nY: —")
        self.coord_label.setFixedWidth(75)
        self.coord_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.coord_label.setStyleSheet(
            "font-family: monospace; font-size: 11px; padding: 6px; color: #888;"
        )

        # Right panel with features list and selection
        right_panel = QWidget()
        right_panel_layout = QVBoxLayout(right_panel)
        right_panel_layout.setContentsMargins(0, 0, 0, 0)
        right_panel_layout.setSpacing(4)

        # Features list (all filtered features)
        features_label = QLabel("<b>Detected Features</b> (click to highlight)")
        features_label.setStyleSheet("padding: 4px;")
        right_panel_layout.addWidget(features_label)

        from qtpy.QtWidgets import QListWidget, QListWidgetItem
        self.features_list = QListWidget()
        self.features_list.setSelectionMode(QListWidget.SingleSelection)
        self.features_list.itemClicked.connect(self._on_feature_clicked)
        self.features_list.setStyleSheet("font-family: monospace; font-size: 11px;")
        right_panel_layout.addWidget(self.features_list, 3)

        # Selection control buttons
        selection_btn_row = QHBoxLayout()
        add_to_selection_btn = QPushButton("Add to Selection")
        add_to_selection_btn.setToolTip("Add selected feature to the export list")
        add_to_selection_btn.clicked.connect(self._add_to_selection)
        selection_btn_row.addWidget(add_to_selection_btn)
        right_panel_layout.addLayout(selection_btn_row)

        # Selected features panel
        selected_label = QLabel("<b>Selected for Export</b>")
        selected_label.setStyleSheet("padding: 4px; margin-top: 8px;")
        right_panel_layout.addWidget(selected_label)

        self.selected_list = QListWidget()
        self.selected_list.setStyleSheet("font-family: monospace; font-size: 11px; background: #000000; color: #ffffff;")
        right_panel_layout.addWidget(self.selected_list, 2)

        # Export controls
        export_btn_row = QHBoxLayout()
        remove_selected_btn = QPushButton("Remove")
        remove_selected_btn.setToolTip("Remove from export list")
        remove_selected_btn.clicked.connect(self._remove_from_selection)
        export_btn_row.addWidget(remove_selected_btn)

        clear_selection_btn = QPushButton("Clear All")
        clear_selection_btn.clicked.connect(self._clear_all_selection)
        export_btn_row.addWidget(clear_selection_btn)
        right_panel_layout.addLayout(export_btn_row)

        export_json_btn = QPushButton("Export to JSON")
        export_json_btn.setStyleSheet("font-weight: bold; padding: 6px; background: #27ae60; color: white;")
        export_json_btn.clicked.connect(self._export_selected_json)
        right_panel_layout.addWidget(export_json_btn)

        right_panel.setMinimumWidth(250)
        right_panel.setVisible(False)
        self.info_panel = right_panel  # Keep same name for compatibility

        image_col = QVBoxLayout()
        image_col.setSpacing(2)
        image_row = QHBoxLayout()
        image_row.setSpacing(0)
        image_row.addWidget(self.scroll_area, 1)
        image_row.addWidget(self.coord_label)
        image_col.addLayout(image_row)

        self._send_status_label = QLabel("")
        self._send_status_label.setStyleSheet(
            "font-size: 10px; color: #aaa; font-family: monospace; padding: 2px 4px;"
        )
        self._send_status_label.setVisible(False)
        image_col.addWidget(self._send_status_label)

        content_row = QHBoxLayout()
        content_row.setSpacing(6)
        content_row.addLayout(image_col, 3)
        content_row.addWidget(right_panel, 1)
        layout.addLayout(content_row, 1)

        # Store currently displayed features and filtered features
        self._displayed_features = []
        self._filtered_features_cache = []

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
        is_xrf = index == 1
        self._single_widget.setVisible(not is_xrf)
        self._xrf_widget.setVisible(is_xrf)
        self._preset_widget.setVisible(not is_xrf)
        if is_xrf:
            prompt = LOCATE_XRF_ROI_PROMPT
        else:
            prompt = self._active_prompt
        self._header_prompt_lbl.setText(f"Prompt: {prompt[:120].rstrip()}…")
        self._header_prompt_lbl.setToolTip(prompt)
        self.locate_btn.setText("Find XRF Areas of Interest" if is_xrf else "Get Location of Features")
        self._reset_image_state()
        self.locate_btn.setEnabled(False)

    def _reset_image_state(self):
        self._orig_pixmap = None
        self._dilated_pixmap = None
        self._annotated_pixmap = None
        self._features = []
        self._zoom = 1.0
        self.scroll_area.setVisible(False)
        self.info_panel.setVisible(False)

    def _on_elem_name_changed(self, idx: int, text: str):
        """Update element name and refresh composite if all images are loaded."""
        self._xrf_elem_names[idx] = text or f"Element {idx+1}"
        # Update filter checkbox label if it exists
        if hasattr(self, '_elem_filter_checks') and idx < len(self._elem_filter_checks):
            self._elem_filter_checks[idx].setText(self._xrf_elem_names[idx])
        if all(self._xrf_paths):
            self._build_xrf_composite_display()

    def _on_source_type_changed(self):
        """Switch between TIFF and H5 modes."""
        if self._tiff_radio.isChecked():
            self._xrf_source_type = "TIFF"
            self._tiff_controls.setVisible(True)
            self._h5_controls.setVisible(False)
        else:
            self._xrf_source_type = "H5"
            self._tiff_controls.setVisible(False)
            self._h5_controls.setVisible(True)

        # Reset selections
        self._xrf_paths = [None, None, None]
        self._xrf_h5_path = None
        self.locate_btn.setEnabled(False)

    def _pick_h5_file(self):
        """Pick an H5 file and load available elements."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select H5 XRF File", "",
            "H5 Files (*.h5 *.hdf5);;All Files (*)"
        )
        if not path:
            return

        try:
            # Load available elements
            elements = _get_h5_element_list(path)

            if not elements:
                QMessageBox.warning(self, "No Elements", "No elements found in H5 file")
                return

            self._xrf_h5_path = path
            self._xrf_h5_elements = elements
            self._h5_file_label.setText(os.path.basename(path))
            self._h5_file_label.setStyleSheet("")

            # Populate element combo boxes
            for combo in self._h5_elem_combos:
                combo.clear()
                combo.addItems(elements)
                combo.setEnabled(True)

            # Auto-select first 3 elements if available
            for i in range(min(3, len(elements))):
                self._h5_elem_combos[i].setCurrentIndex(i)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load H5 file: {e}")

    def _on_h5_elem_selected(self, channel: int, elem_idx: int):
        """Handle element selection for a channel."""
        if elem_idx < 0 or not self._xrf_h5_elements:
            return

        # Store selected element index
        self._xrf_h5_selected[channel] = elem_idx

        # Extract element name (e.g., "Ca_K" -> "Ca")
        elem_full_name = self._xrf_h5_elements[elem_idx]
        elem_name = elem_full_name.split('_')[0]  # Get just the element symbol
        self._xrf_elem_names[channel] = elem_name

        # Update filter checkbox label if it exists
        if hasattr(self, '_elem_filter_checks') and channel < len(self._elem_filter_checks):
            self._elem_filter_checks[channel].setText(elem_name)

        # Enable locate button if all 3 elements selected
        if all(idx is not None for idx in self._xrf_h5_selected) and self._model_ok:
            self._build_xrf_composite_display()
            self.locate_btn.setEnabled(True)

    def _set_all_filters(self, state: bool):
        """Select or deselect all element filters."""
        for check in self._elem_filter_checks:
            check.setChecked(state)

    def _update_features_list(self, filtered_features):
        """Update the features QListWidget with filtered features."""
        from qtpy.QtWidgets import QListWidgetItem

        # Store for later use
        self._displayed_features = filtered_features
        self._filtered_features_cache = filtered_features

        # Clear and repopulate
        self.features_list.clear()

        for i, feat in enumerate(filtered_features):
            fid = feat.get("id", i + 1)
            label = feat.get("label", "")
            desc = feat.get("description", "")
            confidence = feat.get("confidence", None)
            x1, y1 = feat.get("x1", 0), feat.get("y1", 0)
            x2, y2 = feat.get("x2", 0), feat.get("y2", 0)

            # Build display text
            display_lines = [f"#{fid} {label}"]
            display_lines.append(f"  ({x1:.3f}, {y1:.3f}) → ({x2:.3f}, {y2:.3f})")

            if confidence is not None:
                conf_pct = confidence * 100
                display_lines.append(f"  conf: {conf_pct:.0f}%")

            if desc:
                short_desc = desc if len(desc) < 35 else desc[:32] + "..."
                display_lines.append(f"  {short_desc}")

            display_text = "\n".join(display_lines)

            item = QListWidgetItem(display_text)
            item.setData(Qt.UserRole, i)
            item.setToolTip(f"{desc}\nClick to highlight on map")
            self.features_list.addItem(item)

    def _select_all_features(self):
        """Select all features in the features list."""
        for i in range(self.features_list.count()):
            item = self.features_list.item(i)
            item.setSelected(True)

    def _on_feature_clicked(self, item):
        """Highlight the clicked feature on the map."""
        # Get the feature data stored in the item
        feature_idx = item.data(Qt.UserRole)
        if feature_idx is None or feature_idx >= len(self._displayed_features):
            return

        feature = self._displayed_features[feature_idx]

        # Redraw with only this feature highlighted
        is_xrf = self._mode_combo.currentIndex() == 1
        if is_xrf and hasattr(self, '_xrf_fig') and hasattr(self, '_xrf_ax_main'):
            # XRF mode: draw with matplotlib
            highlighted_pixmap = _draw_boxes_on_matplotlib_figure(
                self._xrf_fig, self._xrf_ax_main, [feature], draw_boxes=True
            )
            self._show_pixmap(highlighted_pixmap)
        else:
            # Single element mode: draw on QPixmap
            base = self._dilated_pixmap if self._dilated_pixmap else self._orig_pixmap
            if base:
                highlighted_pixmap = _draw_feature_boxes(base, [feature])
                self._show_pixmap(highlighted_pixmap)

    def _add_to_selection(self):
        """Add selected features from the list to the export selection."""
        from qtpy.QtWidgets import QListWidgetItem

        selected_items = self.features_list.selectedItems()
        if not selected_items:
            return

        for item in selected_items:
            feature_idx = item.data(Qt.UserRole)
            if feature_idx is None or feature_idx >= len(self._displayed_features):
                continue

            feature = self._displayed_features[feature_idx]

            # Check if already in selection
            already_added = False
            for i in range(self.selected_list.count()):
                existing_item = self.selected_list.item(i)
                existing_feature = existing_item.data(Qt.UserRole)
                if existing_feature.get('id') == feature.get('id'):
                    already_added = True
                    break

            if not already_added:
                # Add to selected list
                display_text = item.text()
                selected_item = QListWidgetItem(display_text)
                selected_item.setData(Qt.UserRole, feature)
                self.selected_list.addItem(selected_item)

        # Update status
        count = self.selected_list.count()
        self.status_label.setText(f"{count} feature(s) selected for export")
        self.status_label.setStyleSheet("color: #2980b9; padding: 4px;")

    def _remove_from_selection(self):
        """Remove selected items from the export selection."""
        selected_items = self.selected_list.selectedItems()
        for item in selected_items:
            row = self.selected_list.row(item)
            self.selected_list.takeItem(row)

        # Update status
        count = self.selected_list.count()
        if count > 0:
            self.status_label.setText(f"{count} feature(s) selected for export")
        else:
            self.status_label.setText("Selection cleared")
        self.status_label.setStyleSheet("color: #2980b9; padding: 4px;")

    def _clear_all_selection(self):
        """Clear all items from the export selection."""
        self.selected_list.clear()
        self.status_label.setText("Selection cleared")
        self.status_label.setStyleSheet("color: #2980b9; padding: 4px;")

    def _export_selected_json(self):
        """Export selected features to a new JSON file."""
        if self.selected_list.count() == 0:
            QMessageBox.warning(
                self, "No Selection",
                "Please add features to the selection list before exporting."
            )
            return

        # Collect selected features
        selected_features = []
        for i in range(self.selected_list.count()):
            item = self.selected_list.item(i)
            feature = item.data(Qt.UserRole)
            if feature:
                selected_features.append(feature)

        # Ask for save location
        default_name = "selected_features.json"
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Export Selected Features", default_name,
            "JSON Files (*.json);;All Files (*)"
        )

        if not save_path:
            return

        try:
            # Create output data
            output_data = {
                "features": selected_features,
                "total_count": len(selected_features)
            }

            # Save to file
            with open(save_path, 'w') as f:
                json.dump(output_data, f, indent=2)

            QMessageBox.information(
                self, "Export Successful",
                f"Exported {len(selected_features)} feature(s) to:\n{save_path}"
            )

            self.status_label.setText(f"Exported {len(selected_features)} features to {pathlib.Path(save_path).name}")
            self.status_label.setStyleSheet("color: #27ae60; padding: 4px;")

        except Exception as e:
            QMessageBox.critical(
                self, "Export Failed",
                f"Failed to export features:\n{str(e)}"
            )

    def _on_element_filter_changed(self):
        """Called when any element filter checkbox is toggled."""
        # Only redraw if we have features loaded
        if not self._features:
            return

        # Get filtered features
        filtered_features = self._get_filtered_features()

        # Update the features list UI
        self._update_features_list(filtered_features)

        # Redraw with filtered features
        is_xrf = self._mode_combo.currentIndex() == 1
        if is_xrf and hasattr(self, '_xrf_fig') and hasattr(self, '_xrf_ax_main'):
            # Regenerate both versions with filtered features
            self._orig_pixmap = _draw_boxes_on_matplotlib_figure(
                self._xrf_fig, self._xrf_ax_main, filtered_features, draw_boxes=False
            )
            if filtered_features:
                self._annotated_pixmap = _draw_boxes_on_matplotlib_figure(
                    self._xrf_fig, self._xrf_ax_main, filtered_features, draw_boxes=True
                )
            else:
                self._annotated_pixmap = None

            # Show the appropriate version
            if self._show_boxes_check.isChecked() and self._annotated_pixmap:
                self._show_pixmap(self._annotated_pixmap)
            else:
                self._show_pixmap(self._orig_pixmap)

            # Update status
            total = len(self._features)
            shown = len(filtered_features)
            if shown < total:
                self.status_label.setText(f"Showing {shown} of {total} features (filtered)")
                self.status_label.setStyleSheet("color: #2980b9; padding: 4px;")

    def _on_confidence_filter_changed(self):
        """Called when confidence threshold is changed."""
        if not self._features:
            return
        # Trigger the same filtering logic as element filter
        self._on_element_filter_changed()

    def _on_grid_display_toggled(self):
        """Toggle grid lines in the UI display only (not in LLM input)."""
        if not hasattr(self, '_xrf_fig') or not hasattr(self, '_xrf_ax_main'):
            return
        # Rebuild the display with or without grid
        # Check if data is ready based on source type
        data_ready = False
        if self._xrf_source_type == "TIFF":
            data_ready = all(self._xrf_paths)
        else:  # H5 mode
            data_ready = self._xrf_h5_path is not None and all(idx is not None for idx in self._xrf_h5_selected)

        if data_ready:
            self._build_xrf_composite_display()

    def _get_filtered_features(self):
        """Filter features based on selected elements and confidence threshold.

        Returns features based on selected elements and exact match mode:
        - Exact match OFF: features containing AT LEAST the selected elements (AND logic)
        - Exact match ON: features containing EXACTLY the selected elements (no extras)
        - Also filters by minimum confidence threshold
        """
        # Get selected element names
        selected_elements = set()
        for i, check in enumerate(self._elem_filter_checks):
            if check.isChecked():
                elem_name = self._xrf_elem_names[i].strip()
                if elem_name:
                    selected_elements.add(elem_name.lower())

        # If no elements selected, return empty list
        if not selected_elements:
            return []

        # Get confidence threshold
        min_confidence = self._conf_spin.value() / 100.0  # Convert percentage to 0-1

        # Check if exact match mode is enabled
        exact_match = self._exact_match_check.isChecked()

        # Filter features
        filtered = []
        for feat in self._features:
            # Check confidence threshold first
            confidence = feat.get("confidence", 0.0)
            if confidence < min_confidence:
                continue

            desc = feat.get("description", "")

            # Extract elements mentioned in the description
            # Look for patterns like "Ca+P", "Ce+P", "Ca+Ce+P"
            mentioned_elements = self._extract_elements_from_description(desc)

            if exact_match:
                # Exact match: sets must be equal
                if mentioned_elements == selected_elements:
                    filtered.append(feat)
            else:
                # Contains mode: selected must be subset of mentioned
                if selected_elements.issubset(mentioned_elements):
                    filtered.append(feat)

        return filtered

    def _extract_elements_from_description(self, description: str):
        """Extract element symbols from a feature description.

        Examples:
        - "co-localized signal from Ca+P" → {'ca', 'p'}
        - "co-localized signal from Ca+Ce+P" → {'ca', 'ce', 'p'}
        - "single-element signal from Ce" → {'ce'}
        """
        import re
        elements = set()

        # Look for element symbols (1-2 letter capitalized words)
        # This pattern finds sequences of 1-2 letter element symbols
        pattern = r'\b([A-Z][a-z]?)\b'
        matches = re.findall(pattern, description, re.IGNORECASE)

        # Filter to only include known element names
        all_elem_names = [name.lower() for name in self._xrf_elem_names if name.strip()]

        for match in matches:
            elem_lower = match.lower()
            if elem_lower in all_elem_names:
                elements.add(elem_lower)

        return elements

    def _pick_xrf_image(self, idx: int):
        path, _ = QFileDialog.getOpenFileName(
            self, f"Select Element {idx+1} TIFF", "",
            "TIFF Files (*.tif *.tiff);;All Files (*)"
        )
        if not path:
            return
        self._xrf_paths[idx] = path
        self._xrf_name_labels[idx].setText(os.path.basename(path))
        self._xrf_name_labels[idx].setStyleSheet("")

        # Auto-extract element name from filename
        elem_name = extract_element_from_filename(path)
        if elem_name:
            self._xrf_elem_inputs[idx].setText(elem_name)
            self._xrf_elem_names[idx] = elem_name
        else:
            # Fallback to default if no element detected
            default_name = f"Element {idx+1}"
            self._xrf_elem_inputs[idx].setText(default_name)
            self._xrf_elem_names[idx] = default_name

        # Enable run button only when all 3 are selected
        if all(self._xrf_paths) and self._model_ok:
            self._build_xrf_composite_display()
            self.locate_btn.setEnabled(True)

    def _build_xrf_composite_display(self):
        """Load data (TIFF or H5), create matplotlib composite with RGB + individual element panels."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
            from matplotlib.patches import Patch

            channels = []
            normalized_channels = []

            if self._xrf_source_type == "TIFF":
                # Load from 3 separate TIFF files
                for p in self._xrf_paths:
                    arr = cv2.imread(str(p), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
                    if arr is None:
                        raise ValueError(f"Could not read {p}")
                    arr = arr.astype(np.float32)
                    mn, mx = arr.min(), arr.max()
                    if mx > mn:
                        normalized = (arr - mn) / (mx - mn)
                    else:
                        normalized = np.zeros_like(arr, dtype=np.float32)
                    normalized_channels.append(normalized)
                    channels.append((normalized * 255).astype(np.uint8))

                # No position data from TIFFs
                self._xrf_positions = None

            else:  # H5 mode
                # Load position data from h5 file
                x_pos, y_pos = _read_h5_positions(self._xrf_h5_path)
                self._xrf_positions = (x_pos, y_pos)

                # Load element data for each selected element
                for elem_idx in self._xrf_h5_selected:
                    elem_name = self._xrf_h5_elements[elem_idx]
                    arr = _read_h5_element_data(self._xrf_h5_path, elem_name)
                    arr = arr.astype(np.float32)
                    mn, mx = arr.min(), arr.max()
                    if mx > mn:
                        normalized = (arr - mn) / (mx - mn)
                    else:
                        normalized = np.zeros_like(arr, dtype=np.float32)
                    normalized_channels.append(normalized)
                    channels.append((normalized * 255).astype(np.uint8))

            # Pad to same shape
            h = max(c.shape[0] for c in channels)
            w = max(c.shape[1] for c in channels)
            padded = []
            padded_norm = []
            for c, n in zip(channels, normalized_channels):
                p = np.zeros((h, w), dtype=np.uint8)
                p[:c.shape[0], :c.shape[1]] = c
                padded.append(p)
                pn = np.zeros((h, w), dtype=np.float32)
                pn[:n.shape[0], :n.shape[1]] = n
                padded_norm.append(pn)

            # Create RGB composite
            rgb = np.stack(padded, axis=-1)

            # Store raw RGB composite for LLM analysis (without matplotlib decorations)
            self._xrf_rgb_composite = rgb.copy()

            # Create matplotlib figure with gridspec layout
            # Larger figure size for better resolution
            fig = plt.figure(figsize=(16, 8), facecolor='#1a1a1a')
            gs = gridspec.GridSpec(3, 2, width_ratios=[3, 1], hspace=0.35, wspace=0.12,
                                   left=0.08, right=0.95, top=0.92, bottom=0.08)

            # Main RGB composite (left, spans all rows)
            ax_main = fig.add_subplot(gs[:, 0])
            ax_main.set_facecolor('#1a1a1a')

            # Determine coordinate system (physical vs fractional)
            if self._xrf_positions is not None:
                # H5 mode with physical coordinates
                x_pos, y_pos = self._xrf_positions
                # Get coordinate ranges
                x_min, x_max = x_pos.min(), x_pos.max()
                y_min, y_max = y_pos.min(), y_pos.max()
                extent = [x_min, x_max, y_max, y_min]  # [left, right, bottom, top] with origin='upper'
                xlabel = 'X (μm)'
                ylabel = 'Y (μm)'
                use_physical = True
            else:
                # TIFF mode with fractional coordinates
                extent = [0, 1, 1, 0]
                xlabel = 'X'
                ylabel = 'Y'
                use_physical = False

            ax_main.imshow(rgb, aspect='auto', origin='upper', interpolation='nearest',
                          extent=extent)

            # Store figure, axes, and coordinate info for later drawing of boxes
            self._xrf_fig = fig
            self._xrf_ax_main = ax_main
            self._xrf_use_physical = use_physical
            self._xrf_extent = extent

            # Set axis labels
            ax_main.set_xlabel(xlabel, fontsize=11, fontweight='bold', color='white')
            ax_main.set_ylabel(ylabel, fontsize=11, fontweight='bold', color='white')

            # Set tick marks
            if use_physical:
                # Physical coordinates: use smart tick spacing
                x_range = x_max - x_min
                y_range = y_max - y_min
                num_ticks = 6
                x_ticks = np.linspace(x_min, x_max, num_ticks)
                y_ticks = np.linspace(y_min, y_max, num_ticks)
                ax_main.set_xticks(x_ticks)
                ax_main.set_yticks(y_ticks)
                ax_main.set_xticklabels([f'{x:.1f}' for x in x_ticks], fontsize=9, color='white')
                ax_main.set_yticklabels([f'{y:.1f}' for y in y_ticks], fontsize=9, color='white')
                ax_main.set_xlim(x_min, x_max)
                ax_main.set_ylim(y_max, y_min)  # Inverted for origin='upper'
            else:
                # Fractional coordinates: 0-1
                tick_positions = np.linspace(0, 1, 6)
                ax_main.set_xticks(tick_positions)
                ax_main.set_yticks(tick_positions)
                ax_main.set_xticklabels([f'{x:.1f}' for x in tick_positions], fontsize=9, color='white')
                ax_main.set_yticklabels([f'{y:.1f}' for y in tick_positions], fontsize=9, color='white')
                ax_main.set_xlim(0, 1)
                ax_main.set_ylim(1, 0)  # Inverted for origin='upper'

            ax_main.tick_params(axis='both', which='both', direction='out', length=4, colors='white')

            # Style spines for dark background
            for spine in ax_main.spines.values():
                spine.set_edgecolor('white')
                spine.set_linewidth(1.0)

            # Add grid lines to main RGB if checkbox is enabled
            if hasattr(self, '_show_grid_check') and self._show_grid_check.isChecked():
                if use_physical:
                    # Physical coordinates: dense grid for better precision
                    x_range = x_max - x_min
                    y_range = y_max - y_min

                    # Major grid: aim for 0.5 μm spacing for better precision
                    # Round to nearest 0.5
                    x_spacing = 0.5
                    y_spacing = 0.5

                    # Create major grid ticks
                    x_grid_ticks = np.arange(
                        np.floor(x_min * 2) / 2,  # Round down to nearest 0.5
                        np.ceil(x_max * 2) / 2 + x_spacing,
                        x_spacing
                    )
                    y_grid_ticks = np.arange(
                        np.floor(y_min * 2) / 2,
                        np.ceil(y_max * 2) / 2 + y_spacing,
                        y_spacing
                    )

                    ax_main.set_xticks(x_grid_ticks)
                    ax_main.set_yticks(y_grid_ticks)
                    ax_main.set_xticklabels([f'{x:.1f}' for x in x_grid_ticks], fontsize=7, color='white')
                    ax_main.set_yticklabels([f'{y:.1f}' for y in y_grid_ticks], fontsize=7, color='white')
                else:
                    # Fractional coordinates: 0.05 intervals (20 grid lines instead of 10)
                    all_ticks = np.linspace(0, 1, 21)
                    ax_main.set_xticks(all_ticks)
                    ax_main.set_yticks(all_ticks)
                    # Only label every other tick (0.1 intervals) to avoid crowding
                    labels = [f'{x:.2f}' if i % 2 == 0 else '' for i, x in enumerate(all_ticks)]
                    ax_main.set_xticklabels(labels, fontsize=7, color='white')
                    ax_main.set_yticklabels(labels, fontsize=7, color='white')
                # Grid lines at all tick positions
                ax_main.grid(which='both', color='white', linestyle='-', linewidth=0.4, alpha=0.5)

            # Individual element panels (right, stacked vertically)
            # Create custom colormaps that go from black to the target color (for dark background)
            from matplotlib.colors import LinearSegmentedColormap

            # Black to Red
            colors_red = [(0, 0, 0), (1, 0, 0)]
            cmap_red = LinearSegmentedColormap.from_list('black_to_red', colors_red, N=256)

            # Black to Green
            colors_green = [(0, 0, 0), (0, 1, 0)]
            cmap_green = LinearSegmentedColormap.from_list('black_to_green', colors_green, N=256)

            # Black to Blue
            colors_blue = [(0, 0, 0), (0, 0, 1)]
            cmap_blue = LinearSegmentedColormap.from_list('black_to_blue', colors_blue, N=256)

            color_cmaps = [cmap_red, cmap_green, cmap_blue]

            for i, (elem_name, img_norm) in enumerate(zip(self._xrf_elem_names, padded_norm)):
                ax_panel = fig.add_subplot(gs[i, 1])

                # Set dark background for the panel
                ax_panel.set_facecolor('#1a1a1a')

                # Use same coordinate system as main panel
                ax_panel.imshow(img_norm, cmap=color_cmaps[i], aspect='auto', origin='upper',
                               interpolation='nearest', extent=extent)

                # Use actual element name as title at the top, with extra padding to prevent overlap
                display_name = elem_name if elem_name.strip() else f"Element {i+1}"
                ax_panel.set_title(display_name, fontsize=12, fontweight='bold', pad=12,
                                  loc='center', verticalalignment='baseline', color='white')

                # Add coordinate ticks matching the main panel
                if use_physical:
                    # Physical coordinates: same as main panel
                    num_ticks_panel = 6
                    x_ticks_panel = np.linspace(x_min, x_max, num_ticks_panel)
                    y_ticks_panel = np.linspace(y_min, y_max, num_ticks_panel)
                    ax_panel.set_xticks(x_ticks_panel)
                    ax_panel.set_yticks(y_ticks_panel)
                    ax_panel.set_xticklabels([f'{x:.1f}' for x in x_ticks_panel], fontsize=6, color='white')
                    ax_panel.set_yticklabels([f'{y:.1f}' for y in y_ticks_panel], fontsize=6, color='white')
                    ax_panel.set_xlim(x_min, x_max)
                    ax_panel.set_ylim(y_max, y_min)
                else:
                    # Fractional coordinates: 0.1 intervals
                    all_ticks = np.linspace(0, 1, 11)
                    ax_panel.set_xticks(all_ticks)
                    ax_panel.set_yticks(all_ticks)
                    ax_panel.set_xticklabels([f'{x:.1f}' for x in all_ticks], fontsize=6, color='white')
                    ax_panel.set_yticklabels([f'{y:.1f}' for y in all_ticks], fontsize=6, color='white')
                    ax_panel.set_xlim(0, 1)
                    ax_panel.set_ylim(1, 0)  # Inverted for origin='upper'

                ax_panel.tick_params(axis='both', which='both', direction='out', length=3, labelsize=6,
                                    colors='white')

                # Add grid lines to individual panels if enabled
                if hasattr(self, '_show_grid_check') and self._show_grid_check.isChecked():
                    ax_panel.grid(which='both', color='white', linestyle='-', linewidth=0.3, alpha=0.3)

                # Add thin border around each panel
                for spine in ax_panel.spines.values():
                    spine.set_edgecolor('white')
                    spine.set_linewidth(0.8)

            # Save to temporary buffer
            import io

            # Save as PNG to buffer with dark background preserved
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                       facecolor='#1a1a1a', edgecolor='none')
            buf.seek(0)

            # Load into QPixmap
            qimg = QImage()
            qimg.loadFromData(buf.getvalue())
            self._orig_pixmap = QPixmap.fromImage(qimg)

            # DON'T close the figure - we need it to draw boxes later!
            # The figure will be reused when loading JSON or running detection
            # plt.close(fig)  # Commented out to keep figure alive for drawing boxes

            self._dilated_pixmap = None
            self._annotated_pixmap = None
            self._zoom = 1.0
            self._show_pixmap(self._orig_pixmap)
            self.scroll_area.setVisible(True)
            self.info_panel.setVisible(True)
        except Exception as e:
            self.status_label.setText(f"Could not build composite: {e}")
            self.status_label.setStyleSheet("color: #c0392b; padding: 4px;")

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
        if self._show_boxes_check.isChecked() and self._annotated_pixmap:
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

    def _on_boxes_toggled(self, checked: bool):
        pm = self._current_pixmap()
        if pm:
            self._show_pixmap(pm)

    def _run_locate(self):
        if self._locate_thread and self._locate_thread.isRunning():
            return
        is_xrf = self._mode_combo.currentIndex() == 1
        if is_xrf:
            # Check if data is ready based on source type
            if self._xrf_source_type == "TIFF":
                if not all(self._xrf_paths):
                    return
            else:  # H5 mode
                if not self._xrf_h5_path or not all(idx is not None for idx in self._xrf_h5_selected):
                    return
            self.locate_btn.setEnabled(False)

            # TEMPORARY: Save full matplotlib figure (RGB + individual panels) for LLM analysis
            import tempfile
            buf_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            buf_file.close()

            # Add grid to the main RGB composite axes
            if hasattr(self, '_xrf_fig') and hasattr(self, '_xrf_ax_main'):
                # Use stored coordinate system information
                if hasattr(self, '_xrf_use_physical') and self._xrf_use_physical:
                    # Physical coordinates: dense grid at 0.5 μm intervals
                    x_min, x_max = self._xrf_extent[0], self._xrf_extent[1]
                    y_min, y_max = self._xrf_extent[3], self._xrf_extent[2]

                    # Dense grid: 0.5 μm spacing for precision
                    x_spacing = 0.5
                    y_spacing = 0.5

                    x_ticks = np.arange(
                        np.floor(x_min * 2) / 2,
                        np.ceil(x_max * 2) / 2 + x_spacing,
                        x_spacing
                    )
                    y_ticks = np.arange(
                        np.floor(y_min * 2) / 2,
                        np.ceil(y_max * 2) / 2 + y_spacing,
                        y_spacing
                    )

                    self._xrf_ax_main.set_xticks(x_ticks)
                    self._xrf_ax_main.set_yticks(y_ticks)
                    self._xrf_ax_main.set_xticklabels([f'{x:.1f}' for x in x_ticks], fontsize=7, color='white')
                    self._xrf_ax_main.set_yticklabels([f'{y:.1f}' for y in y_ticks], fontsize=7, color='white')
                else:
                    # Fractional coordinates: 0.05 intervals (denser grid)
                    all_ticks = np.linspace(0, 1, 21)
                    self._xrf_ax_main.set_xticks(all_ticks)
                    self._xrf_ax_main.set_yticks(all_ticks)
                    # Label every other tick
                    labels = [f'{x:.2f}' if i % 2 == 0 else '' for i, x in enumerate(all_ticks)]
                    self._xrf_ax_main.set_xticklabels(labels, fontsize=7, color='white')
                    self._xrf_ax_main.set_yticklabels(labels, fontsize=7, color='white')

                self._xrf_ax_main.tick_params(axis='both', which='both', direction='out', length=4, colors='white')

                # Grid lines at all tick positions - slightly more prominent for LLM
                self._xrf_ax_main.grid(which='both', color='white', linestyle='-', linewidth=0.5, alpha=0.6)

                # Save the entire matplotlib figure (with all panels)
                # Higher DPI for better LLM precision
                self._xrf_fig.savefig(buf_file.name, format='png', dpi=200, bbox_inches='tight',
                                     facecolor='#1a1a1a', edgecolor='none')

                print(f"DEBUG: Full matplotlib figure (RGB + panels + grid) saved to: {buf_file.name}")

            self._xrf_tmp_path = buf_file.name
            self.status_label.setText(f"Analyzing XRF composite with element panels... this may take 30–60 seconds.")
            self.status_label.setStyleSheet("color: #2980b9; padding: 4px;")

            # IMPORTANT: Disable preprocessing since figure is already processed
            normalize = False  # Already normalized
            grid = False  # Grid already added to figure
            upscale = False  # Figure is already high-res (dpi=200)
            colormap = False  # Already in color

            # Create dynamic prompt with actual element names
            elem_red = self._xrf_elem_names[0] if self._xrf_elem_names[0].strip() else "Element 1"
            elem_green = self._xrf_elem_names[1] if self._xrf_elem_names[1].strip() else "Element 2"
            elem_blue = self._xrf_elem_names[2] if self._xrf_elem_names[2].strip() else "Element 3"

            # Determine coordinate system for prompt
            if hasattr(self, '_xrf_use_physical') and self._xrf_use_physical:
                # Physical coordinates mode (H5 file)
                x_min, x_max = self._xrf_extent[0], self._xrf_extent[1]
                y_min, y_max = self._xrf_extent[3], self._xrf_extent[2]  # Note: extent is [left, right, bottom, top]
                coord_desc = f"physical coordinates in microns (μm)"
                coord_range = f"X: {x_min:.1f} to {x_max:.1f} μm, Y: {y_min:.1f} to {y_max:.1f} μm"
                grid_desc = "Grid lines show 0.5 μm intervals (dense grid for precise positioning)"
                # Example coordinates using physical range
                ex1_x1, ex1_y1 = x_min + (x_max - x_min) * 0.1, y_min + (y_max - y_min) * 0.2
                ex1_x2, ex1_y2 = x_min + (x_max - x_min) * 0.4, y_min + (y_max - y_min) * 0.6
                ex2_x1, ex2_y1 = x_min + (x_max - x_min) * 0.5, y_min + (y_max - y_min) * 0.3
                ex2_x2, ex2_y2 = x_min + (x_max - x_min) * 0.7, y_min + (y_max - y_min) * 0.5
            else:
                # Fractional coordinates mode (TIFF files)
                coord_desc = "fractional 0.0-1.0 scale"
                coord_range = "X: 0.0 to 1.0, Y: 0.0 to 1.0"
                grid_desc = "Grid lines show 0.05 intervals (dense grid for precise positioning)"
                ex1_x1, ex1_y1, ex1_x2, ex1_y2 = 0.10, 0.20, 0.45, 0.60
                ex2_x1, ex2_y1, ex2_x2, ex2_y2 = 0.50, 0.30, 0.65, 0.45

            xrf_prompt = (
                f"You are analyzing an XRF (X-ray fluorescence) display with:\n"
                f"- LEFT (large): RGB composite image where Red={elem_red}, Green={elem_green}, Blue={elem_blue}\n"
                f"- RIGHT (3 panels stacked): Individual element intensity maps for {elem_red} (top), {elem_green} (middle), {elem_blue} (bottom)\n"
                f"- {grid_desc}\n\n"
                f"Use BOTH the RGB composite AND the individual panels to identify ALL distinct regions:\n"
                f"1. MULTI-ELEMENT overlaps (2+ elements) — mixed colors: yellow, cyan, magenta, white\n"
                f"2. SINGLE-ELEMENT regions — pure red, green, or blue (verify with individual panels)\n\n"
                f"Respond with JSON only:\n"
                f'{{"features": [{{"id": 1, "label": "multi-element AOI", "x1": {ex1_x1:.2f}, "y1": {ex1_y1:.2f}, '
                f'"x2": {ex1_x2:.2f}, "y2": {ex1_y2:.2f}, "confidence": 0.91, "description": "co-localized signal from {elem_red}+{elem_green}"}}, '
                f'{{"id": 2, "label": "single-element AOI", "x1": {ex2_x1:.2f}, "y1": {ex2_y1:.2f}, "x2": {ex2_x2:.2f}, "y2": {ex2_y2:.2f}, '
                f'"confidence": 0.73, "description": "single-element signal from {elem_blue}"}}], "total_count": 2}}\n\n'
                f"CRITICAL:\n"
                f"- Coordinates (x1,y1,x2,y2) are ONLY for the MAIN RGB IMAGE on the left, in {coord_desc}\n"
                f"- Coordinate range: {coord_range}\n"
                f"- Use the grid lines to accurately place coordinates\n"
                f"- Use the individual element panels to verify which elements are present\n"
                f"- Confidence: 0.0-1.0 based on signal strength and boundary clarity\n"
                f"- Description format: 'co-localized signal from X+Y' or 'single-element signal from X'"
            )

            # Store the prompt that's actually being used for display in results
            self._last_used_prompt = xrf_prompt

            # Update the header to show the actual prompt being used
            self._header_prompt_lbl.setText(f"Prompt: {xrf_prompt[:120].rstrip()}…")
            self._header_prompt_lbl.setToolTip(xrf_prompt)

            self._locate_thread = _LocateThread(buf_file.name, dilate_ksize=0,
                                                system_prompt_override=xrf_prompt,
                                                normalize=normalize, grid_overlay=grid,
                                                upscale=upscale, colormap=colormap,
                                                is_xrf_figure=True)
        else:
            if not self._image_path:
                return
            self.locate_btn.setEnabled(False)
            ksize = self._get_dilate_ksize()
            normalize = self._normalize_check.isChecked()
            grid = self._grid_check.isChecked()
            upscale = self._upscale_check.isChecked()
            colormap = self._colormap_check.isChecked()
            use_tile = self._tile_check.isChecked()

            # Store the prompt that's actually being used for display in results
            self._last_used_prompt = self._active_prompt

            if use_tile:
                self.status_label.setText("Tile mode — querying 4 quadrants... (~2–4 min)")
                self._locate_thread = _LocateTileThread(self._image_path, dilate_ksize=ksize,
                                                        system_prompt_override=self._active_prompt,
                                                        normalize=normalize, grid_overlay=grid,
                                                        upscale=upscale, colormap=colormap)
                self._locate_thread.progress.connect(self._on_tile_progress)
            else:
                self.status_label.setText("Analyzing... this may take 30–60 seconds.")
                self._locate_thread = _LocateThread(self._image_path, dilate_ksize=ksize,
                                                    system_prompt_override=self._active_prompt,
                                                    normalize=normalize, grid_overlay=grid,
                                                    upscale=upscale, colormap=colormap)
            self.status_label.setStyleSheet("color: #2980b9; padding: 4px;")
        self._locate_thread.result.connect(self._on_result)
        self._locate_thread.error.connect(self._on_error)
        self._locate_thread.start()

    def _load_json_results(self):
        """Load previously saved detection results from a JSON file."""
        # Open file dialog
        json_path, _ = QFileDialog.getOpenFileName(
            self, "Load Detection Results JSON", "",
            "JSON Files (*.json);;All Files (*)"
        )
        if not json_path:
            return

        try:
            # Read JSON file
            with open(json_path, 'r') as f:
                content = f.read()

            # Clean up the content - find the first '{' or '[' to handle malformed files
            content = content.strip()

            # Try to find where JSON actually starts
            json_start = -1
            for i, char in enumerate(content):
                if char in ('{', '['):
                    json_start = i
                    break

            if json_start > 0:
                # Found leading garbage, skip it
                content = content[json_start:]

            # Parse cleaned JSON
            data = json.loads(content)

            # Validate JSON structure
            if "features" not in data:
                QMessageBox.warning(
                    self, "Invalid JSON",
                    "The selected JSON file does not contain a 'features' field.\n"
                    "Please select a valid detection results file."
                )
                return

            # Extract features
            features = data.get("features", [])
            n = len(features)

            if n == 0:
                QMessageBox.information(
                    self, "No Features",
                    "The JSON file contains no features."
                )
                return

            # Store features and JSON path
            self._features = features
            json_file = pathlib.Path(json_path)

            # Determine mode from current UI state
            is_xrf = self._mode_combo.currentIndex() == 1

            # Draw boxes on the appropriate image type
            if is_xrf:
                # For XRF mode: ensure images are loaded and composite is built
                if not all(self._xrf_paths):
                    QMessageBox.warning(
                        self, "No XRF Images Loaded",
                        "Please load the 3 XRF element images first.\n"
                        "The JSON file contains coordinates but there's no image to display them on."
                    )
                    return

                # Check if composite was built
                if not hasattr(self, '_xrf_fig') or not hasattr(self, '_xrf_ax_main'):
                    # Try to rebuild composite
                    self._build_xrf_composite_display()

                # Verify figure is valid
                if not hasattr(self, '_xrf_fig') or self._xrf_fig is None:
                    QMessageBox.warning(
                        self, "Display Error",
                        "Failed to build XRF composite display.\n"
                        "Please try reloading the XRF images."
                    )
                    return

                # Draw boxes on matplotlib figure (uses fractional 0-1 coordinates)
                try:
                    # Regenerate both versions: with and without boxes
                    self._orig_pixmap = _draw_boxes_on_matplotlib_figure(
                        self._xrf_fig, self._xrf_ax_main, features, draw_boxes=False
                    )
                    if features:
                        self._annotated_pixmap = _draw_boxes_on_matplotlib_figure(
                            self._xrf_fig, self._xrf_ax_main, features, draw_boxes=True
                        )
                    else:
                        self._annotated_pixmap = None

                    # Show the appropriate version based on checkbox state
                    if self._show_boxes_check.isChecked() and self._annotated_pixmap:
                        self._show_pixmap(self._annotated_pixmap)
                    else:
                        self._show_pixmap(self._orig_pixmap)

                    # Make sure scroll area is visible
                    self.scroll_area.setVisible(True)
                    self.info_panel.setVisible(True)

                except Exception as e:
                    QMessageBox.critical(
                        self, "Drawing Error",
                        f"Failed to draw boxes on composite:\n{str(e)}"
                    )
                    return
            else:
                # For single element mode: draw on QPixmap (converts fractional to pixels)
                base = self._dilated_pixmap if self._dilated_pixmap else self._orig_pixmap
                if base and features:
                    self._annotated_pixmap = _draw_feature_boxes(base, features)
                    self._show_pixmap(self._annotated_pixmap)
                else:
                    QMessageBox.warning(
                        self, "No Image Loaded",
                        "Please load an image first before loading JSON results."
                    )
                    return

            # Update info panel
            self._populate_info(features, json_file)

            # Show element filter panel if in XRF mode
            if is_xrf:
                self._filter_group.setVisible(True)
                # Reset all filters to checked
                for check in self._elem_filter_checks:
                    check.setChecked(True)
                # Default to "contains" mode (exact match OFF)
                self._exact_match_check.setChecked(False)

            # Update status
            self.status_label.setText(f"Loaded {n} feature(s) from: {json_file.name}")
            self.status_label.setStyleSheet("color: #27ae60; padding: 4px;")

            QMessageBox.information(
                self,
                "JSON Loaded",
                f"Successfully loaded {n} feature(s) from:\n{json_file.name}"
            )

        except json.JSONDecodeError as e:
            QMessageBox.critical(
                self, "JSON Parse Error",
                f"Failed to parse JSON file:\n{str(e)}"
            )
        except Exception as e:
            QMessageBox.critical(
                self, "Error Loading JSON",
                f"Failed to load JSON file:\n{str(e)}"
            )

    def _on_result(self, data):
        features = data.get("features", [])
        n = len(features)
        self._features = features

        is_xrf = self._mode_combo.currentIndex() == 1
        if is_xrf:
            # Determine reference path based on source type
            if self._xrf_source_type == "TIFF" and self._xrf_paths[0]:
                ref = pathlib.Path(self._xrf_paths[0])
            elif self._xrf_source_type == "H5" and self._xrf_h5_path:
                ref = pathlib.Path(self._xrf_h5_path)
            else:
                # Fallback to current directory
                ref = pathlib.Path.cwd() / "xrf_features"

            # Build filename with all three element names and timestamp
            elem_names = []
            for elem in self._xrf_elem_names:
                # Sanitize element name: strip whitespace and remove special chars
                clean_name = elem.strip() if elem.strip() else "Element"
                # Remove/replace characters that are problematic in filenames
                clean_name = re.sub(r'[^\w\-]', '', clean_name)
                elem_names.append(clean_name)

            # Create timestamp string (YYYYMMDD_HHMMSS)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Combine: ElementA_ElementB_ElementC_xrf_roi_features_YYYYMMDD_HHMMSS.json
            filename = f"{'_'.join(elem_names)}_xrf_roi_features_{timestamp}.json"
            json_path = ref.with_name(filename)
        else:
            stem = pathlib.Path(self._image_path).stem
            json_path = pathlib.Path(self._image_path).with_name(stem + "_features.json")

        # Add metadata to JSON (model used, timestamp, coordinate system)
        data["metadata"] = {
            "model": _active_model,
            "timestamp": datetime.now().isoformat(),
            "mode": "XRF" if is_xrf else "Single Image",
        }

        # Add coordinate system info for XRF mode
        if is_xrf and hasattr(self, '_xrf_use_physical') and self._xrf_use_physical:
            data["metadata"]["coordinate_system"] = "physical"
            data["metadata"]["units"] = "micrometers"
            data["metadata"]["extent"] = {
                "x_min": self._xrf_extent[0],
                "x_max": self._xrf_extent[1],
                "y_min": self._xrf_extent[3],
                "y_max": self._xrf_extent[2]
            }
        elif is_xrf:
            data["metadata"]["coordinate_system"] = "fractional"
            data["metadata"]["units"] = "0-1 scale"

        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

        # Draw boxes on the appropriate image type
        if is_xrf:
            # For XRF mode: draw on matplotlib figure (uses fractional 0-1 coordinates)
            if hasattr(self, '_xrf_fig') and hasattr(self, '_xrf_ax_main'):
                # Regenerate both versions: with and without boxes
                self._orig_pixmap = _draw_boxes_on_matplotlib_figure(
                    self._xrf_fig, self._xrf_ax_main, features, draw_boxes=False
                )
                if features:
                    self._annotated_pixmap = _draw_boxes_on_matplotlib_figure(
                        self._xrf_fig, self._xrf_ax_main, features, draw_boxes=True
                    )
                else:
                    self._annotated_pixmap = None

                # Show the appropriate version based on checkbox state
                if self._show_boxes_check.isChecked() and self._annotated_pixmap:
                    self._show_pixmap(self._annotated_pixmap)
                else:
                    self._show_pixmap(self._orig_pixmap)
            else:
                self._annotated_pixmap = None
        else:
            # For single element mode: draw on QPixmap (converts fractional to pixels)
            base = self._dilated_pixmap if self._dilated_pixmap else self._orig_pixmap
            if base and features:
                self._annotated_pixmap = _draw_feature_boxes(base, features)
                self._show_pixmap(self._annotated_pixmap)
            else:
                self._annotated_pixmap = None

        self._populate_info(features, json_path)

        # Show element filter panel if in XRF mode
        if is_xrf:
            self._filter_group.setVisible(True)
            # Reset all filters to checked
            for check in self._elem_filter_checks:
                check.setChecked(True)
            # Default to "contains" mode (exact match OFF)
            self._exact_match_check.setChecked(False)

        self.locate_btn.setEnabled(True)
        self.status_label.setText(f"Done. Saved: {json_path.name}")
        self.status_label.setStyleSheet("color: #27ae60; padding: 4px;")

        QMessageBox.information(
            self,
            "Features Found",
            f"{n} location(s) found.\nSaved to:\n{json_path}",
        )

    def _populate_info(self, features, json_path):
        """Populate the features list with clickable, selectable items."""
        from qtpy.QtWidgets import QListWidgetItem

        # Get filtered features (respects element filter checkboxes)
        is_xrf = self._mode_combo.currentIndex() == 1
        if is_xrf and hasattr(self, '_elem_filter_checks'):
            filtered_features = self._get_filtered_features()
        else:
            filtered_features = features

        # Store for later use
        self._displayed_features = filtered_features
        self._filtered_features_cache = filtered_features

        # Clear existing list
        self.features_list.clear()

        # Populate with filtered features
        for i, feat in enumerate(filtered_features):
            fid = feat.get("id", i + 1)
            label = feat.get("label", "")
            desc = feat.get("description", "")
            confidence = feat.get("confidence", None)
            x1, y1 = feat.get("x1", 0), feat.get("y1", 0)
            x2, y2 = feat.get("x2", 0), feat.get("y2", 0)

            # Build display text
            display_lines = [f"#{fid} {label}"]
            display_lines.append(f"  ({x1:.3f}, {y1:.3f}) → ({x2:.3f}, {y2:.3f})")

            if confidence is not None:
                conf_pct = confidence * 100
                display_lines.append(f"  conf: {conf_pct:.0f}%")

            if desc:
                # Truncate long descriptions
                short_desc = desc if len(desc) < 35 else desc[:32] + "..."
                display_lines.append(f"  {short_desc}")

            display_text = "\n".join(display_lines)

            # Create list item
            item = QListWidgetItem(display_text)
            item.setData(Qt.UserRole, i)  # Store feature index
            item.setToolTip(f"{desc}\nClick to highlight on map")
            self.features_list.addItem(item)

        self.info_panel.setVisible(True)

        # Update status if filtering
        total = len(features)
        shown = len(filtered_features)
        if shown < total:
            self.status_label.setText(f"Showing {shown} of {total} features (filtered)")
            self.status_label.setStyleSheet("color: #2980b9; padding: 4px;")

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
        self._orig_pixmap = None
        self._dilated_pixmap = None
        if self._image_path:
            self._update_display()

    def _update_display(self) -> None:
        normalize = self._normalize_check.isChecked()
        grid = self._grid_check.isChecked()
        upscale = self._upscale_check.isChecked()
        colormap = self._colormap_check.isChecked()
        ksize = self._get_dilate_ksize()
        self._orig_pixmap, _, _ = _load_image_pixmap(
            self._image_path, dilate_ksize=0, normalize=normalize, grid_overlay=grid,
            upscale=upscale, colormap=colormap)
        if ksize > 0:
            self._dilated_pixmap, _, _ = _load_image_pixmap(
                self._image_path, dilate_ksize=ksize, normalize=normalize, grid_overlay=grid,
                upscale=upscale, colormap=colormap)
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
        if self._colormap_check.isChecked():
            parts.append("Inferno ✓")
        text = "Sending to Claude:  " + "  |  ".join(parts) if parts else "Sending to Claude: raw image"
        self._send_status_label.setText(text)
        self._send_status_label.setVisible(True)

    def _on_mouse_moved(self, fx: float, fy: float) -> None:
        self.coord_label.setText(f"X: {fx:.3f}\nY: {fy:.3f}")
        self.coord_label.setStyleSheet("font-family: monospace; font-size: 11px; padding: 6px;")

        # Hover tooltip: find if cursor is inside any feature bbox
        hit = None
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
        self._json_label = QLabel("No JSON selected  (raw_detections_master.json or all_boxes_*.json)")
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
