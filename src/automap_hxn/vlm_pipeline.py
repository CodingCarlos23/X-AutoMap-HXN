"""
vlm_pipeline.py — shared VLM/LLM logic, no Qt dependencies.

Imported by both llm_json_maker.py (GUI) and feature_locator_headless.py (CLI).
A change here propagates to both. No module-level mutable state.
"""

import base64
import json
import os
import subprocess
import urllib.request

import cv2
import numpy as np

# ── API constants ──────────────────────────────────────────────────────────────

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_API_VERSION = "2023-06-01"
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OLLAMA_URL = "http://localhost:11434/api/chat"
VLM_MAX_DIM = 1024  # matches _VLM_MAX_DIM in llm_json_maker.py
# ~80–100 tokens per box → 8k allows roughly 80–100 features before the reply is cut off
LOCATE_MAX_TOKENS = 8192

# ── Prompts ────────────────────────────────────────────────────────────────────

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

LOCATE_USER_PROMPT = (
    "Find all distinct features and regions of interest in this image. Return JSON only."
)

LLM_COLOCATE_SYSTEM = (
    "You are an XRF (X-ray fluorescence) materials scientist. "
    "You will receive VLM-detected features for multiple element scans. "
    "Your task is to identify co-location zones — regions where 2 or more elements "
    "overlap or are immediately adjacent — and identify the most likely material or phase. "
    "For each zone provide a suggestion, confidence (high/medium/low), one-sentence reasoning, "
    "and up to 2 short alternatives. Be concise — one sentence per field maximum. "
    "Be explicit when evidence is ambiguous — low confidence answers are valid. "
    "Return ONLY valid JSON. No text outside the JSON object."
)

LLM_COLOCATE_SCHEMA = (
    '{"zones": [{'
    '"id": 1, '
    '"elements": ["Fe", "Cr"], '
    '"x1": 0.10, "y1": 0.20, "x2": 0.35, "y2": 0.45, '
    '"suggestion": "ferrochrome alloy (FeCr)", '
    '"confidence": "medium", '
    '"reasoning": "Fe+Cr co-location without O rules out chromite and points to metallic alloy.", '
    '"alternatives": ['
    '{"name": "chromite (FeCr2O4)", "note": "requires O co-location to confirm"}, '
    '{"name": "Fe-Cr carbide", "note": "requires C signal to confirm"}'
    ']'
    '}]}'
)

# ── Colors (pure RGB tuples, no Qt) ───────────────────────────────────────────
# These match the QColor + cv2 tuple pairs in _MULTI_ELEM_COLORS / _XRF_ELEM_COLORS.

ELEM_COLORS_RGB = [
    (220, 60,  60),
    (60,  200, 60),
    (60,  120, 220),
    (230, 200, 50),
    (160, 60,  220),
    (230, 130, 40),
    (50,  210, 210),
    (240, 100, 180),
    (100, 220, 80),
    (50,  170, 180),
]

CIRCLE_COLORS_RGB = [
    (255, 0,   0),
    (0,   255, 0),
    (0,   0,   255),
    (255, 220, 0),
    (180, 0,   255),
    (255, 140, 0),
    (0,   220, 220),
    (255, 100, 180),
    (100, 255, 60),
    (0,   180, 200),
]

# ── Provider routing ───────────────────────────────────────────────────────────

def get_provider(model: str) -> str:
    if model.startswith("claude-"):
        return "anthropic"
    if model.startswith(("gpt-", "o3", "o4")):
        return "openai"
    return "ollama"


# ── Key / URL resolution (stateless) ──────────────────────────────────────────

GPG_ENV_CANDIDATES = ["~/.private_env.gpg"]
_ENV_VAR = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}


def parse_env_lines(text: str) -> "dict[str, str]":
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


def resolve_key(provider: str, gpg_candidates: "list[str] | None" = None) -> "tuple[str|None, str, str]":
    """Resolve API key and URL overrides for the given provider.

    Returns (key_or_None, anthropic_url, openai_url).
    URLs are defaults unless AIFAPIM_HOST is found in env/GPG.
    """
    candidates = gpg_candidates or GPG_ENV_CANDIDATES
    env_var = _ENV_VAR.get(provider, "")
    anthropic_url = ANTHROPIC_API_URL
    openai_url = OPENAI_API_URL

    # 1. Direct env var
    val = os.environ.get(env_var, "").strip()
    if val:
        return val, anthropic_url, openai_url

    # 2. GPG-encrypted env file
    for path in candidates:
        expanded = os.path.expanduser(path)
        if not os.path.exists(expanded):
            continue
        try:
            result = subprocess.run(
                ["gpg", "--quiet", "--batch", "--yes", "--decrypt", expanded],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode != 0:
                continue
            env = parse_env_lines(result.stdout)

            # Standard key
            if env_var in env and env[env_var]:
                return env[env_var], anthropic_url, openai_url

            # AIFAPIM proxy fallback (BNL NSLS-II)
            apim_key = env.get("AIFAPIM_API_KEY", "").strip()
            apim_host = env.get("AIFAPIM_HOST", "").strip()
            if apim_key and apim_host:
                host = apim_host.rstrip("/")
                if not host.startswith("http"):
                    host = f"https://{host}"
                if provider == "openai":
                    openai_url = f"{host}/openai/v1/responses"
                elif provider == "anthropic":
                    anthropic_url = f"{host}/anthropic/v1/messages"
                return apim_key, anthropic_url, openai_url
        except Exception:
            continue

    return None, anthropic_url, openai_url


# ── Image processing ───────────────────────────────────────────────────────────

def draw_coordinate_grid(arr: np.ndarray, steps: int = 10) -> np.ndarray:
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

        if i % 2 != 0:
            continue
        label = f"{frac:.1f}"
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)

        lx = max(0, min(x - tw // 2, w - tw - 1))
        ly = th + 4
        cv2.rectangle(arr, (lx - 1, 1), (lx + tw + 1, ly + 2), label_bg, -1)
        cv2.putText(arr, label, (lx, ly), font, font_scale, label_fg, thickness, cv2.LINE_AA)

        rx = 4
        ry = max(th, min(y + th // 2, h - 2))
        cv2.rectangle(arr, (rx - 1, ry - th - 1), (rx + tw + 1, ry + 2), label_bg, -1)
        cv2.putText(arr, label, (rx, ry), font, font_scale, label_fg, thickness, cv2.LINE_AA)

    return arr


def apply_circles_transform(arr: np.ndarray, circle_color=(255, 0, 0)) -> np.ndarray:
    """Gaussian blur → Otsu threshold → contours → colored circles on white background.

    Input: uint8 grayscale (2-D). Returns uint8 RGB (3-D).
    circle_color is an (R, G, B) tuple.
    """
    if arr.ndim != 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(arr, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.full((*arr.shape, 3), 255, dtype=np.uint8)
    for cnt in contours:
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        r = max(r, 3.0) * 1.3
        cv2.circle(out, (int(cx), int(cy)), int(r), circle_color, -1)
    return out


# ── Emoji marker transform (benchmark variants) ───────────────────────────────

EMOJI_MARKERS = {"flower": "\U0001F338", "car": "\U0001F697"}  # 🌸 🚗
EMOJI_FONT_PATHS = [
    "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
    "/usr/share/fonts/noto/NotoColorEmoji.ttf",
    "/System/Library/Fonts/Apple Color Emoji.ttc",
]
_EMOJI_SPRITE_CACHE: dict = {}


def _emoji_sprite(char: str) -> np.ndarray:
    """Render an emoji once to a tightly cropped RGBA uint8 array (cached)."""
    if char in _EMOJI_SPRITE_CACHE:
        return _EMOJI_SPRITE_CACHE[char]
    from PIL import Image, ImageDraw, ImageFont

    font = None
    for fp in EMOJI_FONT_PATHS:
        if os.path.exists(fp):
            # Noto Color Emoji is a bitmap font that only renders at size 109
            font = ImageFont.truetype(fp, 109)
            break
    if font is None:
        raise RuntimeError(f"No emoji font found (looked in {EMOJI_FONT_PATHS})")
    canvas = Image.new("RGBA", (160, 160), (0, 0, 0, 0))
    ImageDraw.Draw(canvas).text((10, 10), char, font=font, embedded_color=True)
    canvas = canvas.crop(canvas.getbbox())
    sprite = np.array(canvas)
    _EMOJI_SPRITE_CACHE[char] = sprite
    return sprite


def apply_emoji_transform(arr: np.ndarray, marker: str) -> np.ndarray:
    """Same blob finding + sizing as apply_circles_transform, but stamps an emoji per blob.

    Input: uint8 grayscale (2-D). Returns uint8 RGB (3-D) on a white background.
    marker is a key of EMOJI_MARKERS ("flower" or "car").
    """
    if arr.ndim != 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    sprite = _emoji_sprite(EMOJI_MARKERS[marker])
    blurred = cv2.GaussianBlur(arr, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.full((*arr.shape, 3), 255, dtype=np.float32)
    H, W = arr.shape
    for cnt in contours:
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        r = max(r, 3.0) * 1.3  # identical sizing to the circle variant
        d = max(2, int(round(2 * r)))
        sp = cv2.resize(sprite, (d, d), interpolation=cv2.INTER_AREA).astype(np.float32)
        x0, y0 = int(cx) - d // 2, int(cy) - d // 2
        # clip sprite to image bounds
        sx0, sy0 = max(0, -x0), max(0, -y0)
        dx0, dy0 = max(0, x0), max(0, y0)
        dx1, dy1 = min(W, x0 + d), min(H, y0 + d)
        if dx1 <= dx0 or dy1 <= dy0:
            continue
        patch = sp[sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)]
        alpha = patch[..., 3:4] / 255.0
        region = out[dy0:dy1, dx0:dx1]
        out[dy0:dy1, dx0:dx1] = region * (1 - alpha) + patch[..., :3] * alpha
    return out.clip(0, 255).astype(np.uint8)


def encode_image_for_vlm(path, max_dim: int = VLM_MAX_DIM, dilate_ksize: int = 0,
                          normalize: bool = True, grid_overlay: bool = False,
                          upscale: bool = False, colormap: bool = False,
                          red_circles: bool = False, circle_color=(255, 0, 0),
                          emoji_marker: "str | None" = None, clahe: bool = True) -> str:
    """Load image, optionally preprocess, downsample to max_dim, return base64-PNG string.

    emoji_marker ("flower" / "car") stamps emoji instead of circles; ignored if red_circles.
    clahe=False skips contrast enhancement (grayscale stays min-max scaled only).
    """
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
        rgb = apply_circles_transform(arr, circle_color=circle_color)
        arr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    elif emoji_marker:
        if arr.ndim != 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        rgb = apply_emoji_transform(arr, emoji_marker)
        arr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    else:
        _clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)) if clahe else None
        if arr.ndim == 2:
            if _clahe is not None:
                arr = _clahe.apply(arr)
            if colormap:
                arr = cv2.applyColorMap(arr, cv2.COLORMAP_INFERNO)
            else:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        else:
            lab = cv2.cvtColor(arr, cv2.COLOR_BGR2LAB)
            if _clahe is not None:
                lab[:, :, 0] = _clahe.apply(lab[:, :, 0])
            arr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    if grid_overlay:
        arr = draw_coordinate_grid(arr)
    h, w = arr.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        arr = cv2.resize(arr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode()


# ── JSON parsing ───────────────────────────────────────────────────────────────

def parse_locate_json_status(content: str) -> "tuple[dict, str]":
    """Like parse_locate_json, also returns 'ok' | 'recovered' (truncated JSON salvaged) | 'failed'."""
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

    start = content.find("{")
    end = content.rfind("}")
    if start >= 0 and end > start:
        content = content[start:end + 1]

    try:
        return json.loads(content), "ok"
    except json.JSONDecodeError:
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
        return ({"features": features, "total_count": len(features)},
                "recovered" if features else "failed")


def parse_locate_json(content: str) -> dict:
    """Parse VLM feature-location JSON response, with truncation recovery."""
    return parse_locate_json_status(content)[0]


def parse_zones(text: str) -> list:
    """Extract zones list from an LLM co-locate JSON response."""
    if "```" in text:
        inner, in_block = [], False
        for line in text.splitlines():
            if line.strip().startswith("```"):
                in_block = not in_block
                continue
            if in_block or not text.startswith("```"):
                inner.append(line)
        text = "\n".join(inner).strip() or text
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]
    try:
        return json.loads(text).get("zones", [])
    except Exception:
        return []


def parse_responses_text(data: dict) -> str:
    """Extract text from an OpenAI Responses API (non-streaming) response."""
    if "output_text" in data:
        return data["output_text"]
    for item in data.get("output", []):
        for part in item.get("content", []):
            if part.get("type") == "output_text":
                return part.get("text", "")
    return ""


# ── VLM feature query ──────────────────────────────────────────────────────────

_TRUNCATED_REASONS = {"max_tokens", "length", "max_output_tokens"}


def _fill_meta(meta, provider: str, data: dict, text: str):
    """Record stop reason, truncation and token usage from a raw API response (provider-specific shapes)."""
    if meta is None:
        return
    if provider == "anthropic":
        stop = data.get("stop_reason")
        u = data.get("usage") or {}
        tin, tout = u.get("input_tokens"), u.get("output_tokens")
    elif provider == "openai" and "choices" in data:
        stop = (data.get("choices") or [{}])[0].get("finish_reason")
        u = data.get("usage") or {}
        tin, tout = u.get("prompt_tokens"), u.get("completion_tokens")
    elif provider == "openai":  # Responses API
        stop = (data.get("incomplete_details") or {}).get("reason") or data.get("status")
        u = data.get("usage") or {}
        tin, tout = u.get("input_tokens"), u.get("output_tokens")
    else:  # ollama
        stop = data.get("done_reason")
        tin, tout = data.get("prompt_eval_count"), data.get("eval_count")
    meta.update(stop_reason=stop, truncated=stop in _TRUNCATED_REASONS,
                input_tokens=tin, output_tokens=tout, text=text)


def _anthropic_text(data: dict) -> str:
    """First text block — skips thinking blocks that thinking-on models put first."""
    for block in data.get("content") or []:
        if block.get("type") == "text":
            return block.get("text", "")
    return ""


def _post(url: str, payload: dict, headers: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                                 headers=headers)
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())


def query_vlm_features(img_b64: str, system_prompt: str, model: str, provider: str,
                        key: str = "",
                        anthropic_url: str = ANTHROPIC_API_URL,
                        openai_url: str = OPENAI_API_URL,
                        meta: "dict | None" = None) -> list:
    """Send a single image to the VLM and return a list of feature dicts.

    Detects the OpenAI Responses API (NSLS-II APIM) from openai_url ending with /responses.
    If `meta` is a dict it is filled with stop_reason, truncated, input/output tokens,
    parse_status ('ok' | 'recovered' | 'failed') and the raw reply text.
    """
    if provider == "anthropic":
        data = _post(anthropic_url, {
            "model": model,
            "max_tokens": LOCATE_MAX_TOKENS,
            "system": system_prompt,
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64",
                                              "media_type": "image/png", "data": img_b64}},
                {"type": "text", "text": LOCATE_USER_PROMPT},
            ]}],
        }, {"Content-Type": "application/json", "x-api-key": key,
            "anthropic-version": ANTHROPIC_API_VERSION})
        text = _anthropic_text(data) or "{}"
    elif provider == "openai":
        is_responses_api = openai_url.rstrip("/").endswith("/responses")
        auth_header = {"x-api-key": key} if is_responses_api else {"Authorization": f"Bearer {key}"}
        headers = {"Content-Type": "application/json", **auth_header}
        if is_responses_api:
            data = _post(openai_url, {
                "model": model,
                "instructions": system_prompt,
                "input": [{"role": "user", "content": [
                    {"type": "input_image", "image_url": f"data:image/png;base64,{img_b64}"},
                    {"type": "input_text", "text": LOCATE_USER_PROMPT},
                ]}],
            }, headers)
            text = parse_responses_text(data)
        else:
            data = _post(openai_url, {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                        {"type": "text", "text": LOCATE_USER_PROMPT},
                    ]},
                ],
            }, headers)
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
    else:  # Ollama
        data = _post(OLLAMA_URL, {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": LOCATE_USER_PROMPT, "images": [img_b64]},
            ],
            "stream": False,
            "format": "json",
        }, {"Content-Type": "application/json"})
        text = data.get("message", {}).get("content", "{}")

    parsed, status = parse_locate_json_status(text or "{}")
    _fill_meta(meta, provider, data, text)
    if meta is not None:
        meta["parse_status"] = status
    return parsed.get("features", [])


def query_llm_text(system: str, prompt: str, model: str, provider: str, key: str = "",
                   anthropic_url: str = ANTHROPIC_API_URL, openai_url: str = OPENAI_API_URL,
                   max_tokens: int = 16384, meta: "dict | None" = None) -> str:
    """Text-only LLM call (system + user prompt) → reply text. Fills `meta` like query_vlm_features."""
    if provider == "anthropic":
        data = _post(anthropic_url, {
            "model": model, "max_tokens": max_tokens, "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }, {"Content-Type": "application/json", "x-api-key": key,
            "anthropic-version": ANTHROPIC_API_VERSION})
        text = _anthropic_text(data) or "{}"
    elif provider == "openai":
        is_responses_api = openai_url.rstrip("/").endswith("/responses")
        auth_header = {"x-api-key": key} if is_responses_api else {"Authorization": f"Bearer {key}"}
        data = _post(openai_url, {
            "model": model, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        }, {"Content-Type": "application/json", **auth_header})
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
    else:  # Ollama
        data = _post(OLLAMA_URL, {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "stream": False, "format": "json",
        }, {"Content-Type": "application/json"})
        text = data.get("message", {}).get("content", "{}")
    _fill_meta(meta, provider, data, text)
    return text


# ── LLM co-location ────────────────────────────────────────────────────────────

def build_colocate_prompt(elem_features: list, elem_names: list,
                           sample_context: str = "") -> str:
    """Build the user prompt for LLM co-location analysis."""
    lines = []
    if sample_context:
        lines.append(f"Sample context provided by the user: {sample_context}. "
                     "Use this to inform your material suggestions and reasoning.")
        lines.append("")
    lines += [
        f"You have VLM detection results for {len(elem_names)} XRF element scans.",
        "Bounding boxes are fractional coordinates (0.0–1.0), (0,0)=top-left, (1,1)=bottom-right.",
        "",
    ]
    for i, (name, feats) in enumerate(zip(elem_names, elem_features)):
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
        "  2. List which element names are present.",
        "  3. Give your best-guess material, compound, alloy, or mineral phase as 'suggestion'.",
        "  4. Rate your confidence as 'high', 'medium', or 'low'. Use 'low' when ambiguous.",
        "  5. Write ONE concise sentence for 'reasoning' — why this suggestion fits the element combination.",
        "  6. List up to 2 alternatives as objects with 'name' and a short 'note' (what signal would confirm it).",
        "     Return [] if no plausible alternatives exist.",
        "Be concise — one sentence per field. Do not write paragraphs.",
        "",
        f"Use this exact JSON schema:\n{LLM_COLOCATE_SCHEMA}",
    ]
    return "\n".join(lines)


def query_llm_colocate(prompt: str, model: str, provider: str,
                        key: str = "",
                        anthropic_url: str = ANTHROPIC_API_URL,
                        openai_url: str = OPENAI_API_URL,
                        meta: "dict | None" = None) -> list:
    """Send co-location prompt to LLM and return list of zone dicts."""
    text = query_llm_text(LLM_COLOCATE_SYSTEM, prompt, model, provider, key,
                          anthropic_url=anthropic_url, openai_url=openai_url,
                          max_tokens=16384, meta=meta)
    zones = parse_zones(text)
    if meta is not None:
        meta["parse_status"] = "ok" if (zones or '"zones"' in text) else "failed"
    return zones
