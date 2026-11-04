"""
Feature Locator headless runner — multi-element VLM pipeline with optional LLM co-location.

Usage:
    feature-locator-headless config.json
    python -m automap_hxn.feature_locator_headless config.json

Config JSON fields:
    input_dir      : path to folder of element TIFFs (required)
    output_dir     : output folder (default: same as input_dir)
    model          : VLM/LLM model name (default: claude-sonnet-4-6)
    circles        : true/false — colored circles on white bg (default: true)
    normalize      : true/false (default: true)
    upscale        : true/false — upscale to 1092px (default: true)
    llm_colocate   : true/false — run LLM co-location analysis (default: true)
    sample_context : optional string describing the sample
    elements       : optional list of element names to include (e.g. ["Fe","Ca","Si"])

Outputs (matching GUI naming conventions):
    <name>_multi_features.json
    <name>_multi_llm_colocate.json
    <name>_multi_elements_composite.png
    <name>_multi_elements_circles.png   (if circles=true)
    <name>_multi_colocate_composite.png (if llm_colocate=true)
    <name>_multi_colocate_circles.png   (if circles=true and llm_colocate=true)
"""

import concurrent.futures
import json
import pathlib
import sys
import time

import cv2
import numpy as np

from automap_hxn.vlm_pipeline import (
    LOCATE_SPARSE_XRF_PROMPT,
    ELEM_COLORS_RGB,
    CIRCLE_COLORS_RGB,
    get_provider,
    resolve_key,
    encode_image_for_vlm,
    apply_circles_transform,
    query_vlm_features,
    build_colocate_prompt,
    query_llm_colocate,
)


# ── Composite building (headless-only, no Qt) ─────────────────────────────────

def _load_gray(path: str, normalize: bool, upscale: bool) -> np.ndarray:
    """Load any TIFF as float32 [0,1], matching the GUI's _load_gray exactly."""
    arr = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise ValueError(f"Could not read: {path}")
    arr = arr.astype(np.float32)
    if normalize:
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
    if upscale:
        h, w = arr.shape
        if min(h, w) < 1092:
            scale = 1092 / min(h, w)
            arr = cv2.resize(arr, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_CUBIC)
    return arr


def _build_composite(tiff_paths: list, normalize: bool, upscale: bool,
                     circles: bool) -> np.ndarray:
    """False-color composite matching the GUI's _build_multi_composite_display. Returns RGB uint8."""
    arrays = [_load_gray(p, normalize, upscale) for p in tiff_paths]
    h = max(a.shape[0] for a in arrays)
    w = max(a.shape[1] for a in arrays)

    if circles:
        # White canvas; paste each element's colored circles via mask — same as GUI
        canvas = np.full((h, w, 3), 255, dtype=np.uint8)
        for i, arr in enumerate(arrays):
            cc_rgb = ELEM_COLORS_RGB[i % len(ELEM_COLORS_RGB)]
            arr_u8 = (arr * 255).astype(np.uint8)
            ca = apply_circles_transform(arr_u8, circle_color=cc_rgb)  # RGB uint8
            mask = np.any(ca < 255, axis=-1)
            ph, pw = ca.shape[:2]
            canvas[:ph, :pw][mask] = ca[mask]
        return canvas
    else:
        # Additive blend into float composite — same as GUI
        composite = np.zeros((h, w, 3), dtype=np.float32)
        for i, arr in enumerate(arrays):
            r, g, b = ELEM_COLORS_RGB[i % len(ELEM_COLORS_RGB)]
            padded = np.zeros((h, w), dtype=np.float32)
            padded[:arr.shape[0], :arr.shape[1]] = arr
            composite[:, :, 0] = np.clip(composite[:, :, 0] + padded * (r / 255.0), 0, 1)
            composite[:, :, 1] = np.clip(composite[:, :, 1] + padded * (g / 255.0), 0, 1)
            composite[:, :, 2] = np.clip(composite[:, :, 2] + padded * (b / 255.0), 0, 1)
        return (composite * 255).astype(np.uint8)


def _draw_element_boxes(img_rgb: np.ndarray, elem_features: list) -> np.ndarray:
    """Draw per-element feature boxes on the composite, one color per element. Returns RGB uint8."""
    out = img_rgb.copy()
    h, w = out.shape[:2]
    for i, feats in enumerate(elem_features):
        color = ELEM_COLORS_RGB[i % len(ELEM_COLORS_RGB)]
        for feat in feats:
            x1 = int(feat.get("x1", 0) * w)
            y1 = int(feat.get("y1", 0) * h)
            x2 = int(feat.get("x2", 1) * w)
            y2 = int(feat.get("y2", 1) * h)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)  # 2px matches GUI QPen width
    return out


def _draw_zone_boxes(img_rgb: np.ndarray, zones: list) -> np.ndarray:
    """Draw confidence-colored zone boxes on a copy of img_rgb. Returns RGB uint8."""
    out = img_rgb.copy()
    h, w = out.shape[:2]
    # Exact RGB values matching GUI _draw_colocate_boxes_confidence QColor definitions
    conf_color = {"high": (60, 200, 60), "medium": (220, 200, 50), "low": (200, 60, 60)}
    for z in zones:
        x1 = int(z.get("x1", 0) * w)
        y1 = int(z.get("y1", 0) * h)
        x2 = int(z.get("x2", 1) * w)
        y2 = int(z.get("y2", 1) * h)
        conf = z.get("confidence", "medium")
        color = conf_color.get(conf, (200, 200, 200))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)  # 3px matches GUI QPen width
        label = f"Z{z.get('id', '?')}"
        cv2.putText(out, label, (x1 + 2, y1 + 10),  # inside box top-left, matches GUI drawText
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return out


def _save_rgb(img_rgb: np.ndarray, path: pathlib.Path):
    cv2.imwrite(str(path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(config: dict):
    input_dir = pathlib.Path(config["input_dir"])
    output_dir = pathlib.Path(config.get("output_dir", input_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    model = config.get("model", "claude-sonnet-4-6")
    circles = bool(config.get("circles", True))
    normalize = bool(config.get("normalize", True))
    upscale = bool(config.get("upscale", True))
    do_colocate = bool(config.get("llm_colocate", True))
    sample_context = config.get("sample_context", "").strip()
    elem_filter = config.get("elements", None)
    out_stem = output_dir.name

    provider = get_provider(model)
    key, anthropic_url, openai_url = resolve_key(provider)
    key = key or ""

    # Find TIFFs
    tiff_paths = sorted(input_dir.glob("*.tif")) + sorted(input_dir.glob("*.tiff"))
    if elem_filter:
        allowed = {e.lower() for e in elem_filter}
        tiff_paths = [p for p in tiff_paths if p.stem.split("_")[-1].lower() in allowed]
    if not tiff_paths:
        print(f"[ERROR] No TIFF files found in {input_dir}")
        sys.exit(1)

    elem_names = [p.stem.split("_")[-1] for p in tiff_paths]
    N = len(tiff_paths)

    print(f"[Feature Locator] Model: {model}  Provider: {provider}")
    print(f"[Feature Locator] {N} elements: {', '.join(elem_names)}")
    print(f"[Feature Locator] circles={circles}  normalize={normalize}  upscale={upscale}")
    if sample_context:
        preview = sample_context[:80] + ("..." if len(sample_context) > 80 else "")
        print(f"[Feature Locator] Sample context: {preview}")

    # ── Step 1: Per-element VLM locate ───────────────────────────────────────
    print(f"\n[Step 1] Sending {N} element images to VLM simultaneously...")
    t0 = time.time()
    results_by_index: dict = {}

    def _locate_one(i: int, path: pathlib.Path) -> tuple:
        cc = CIRCLE_COLORS_RGB[i % len(CIRCLE_COLORS_RGB)]
        img_b64 = encode_image_for_vlm(str(path), normalize=normalize, upscale=upscale,
                                        red_circles=circles, circle_color=cc)
        features = query_vlm_features(img_b64, LOCATE_SPARSE_XRF_PROMPT, model, provider, key,
                                       anthropic_url=anthropic_url, openai_url=openai_url)
        return i, elem_names[i], features

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(N, 8)) as executor:
        futures = {executor.submit(_locate_one, i, p): i for i, p in enumerate(tiff_paths)}
        for future in concurrent.futures.as_completed(futures):
            i, name, features = future.result()
            results_by_index[i] = (name, features)
            print(f"  {name}: {len(features)} feature(s)  ({len(results_by_index)}/{N} done)")

    elem_features = [results_by_index[i][1] for i in range(N)]
    total_features = sum(len(f) for f in elem_features)
    duration = round(time.time() - t0, 1)
    print(f"[Step 1] Done — {total_features} total features in {duration}s")

    features_data = {
        "elements": elem_names,
        "elem_features": elem_features,
        "total_count": total_features,
        "model": model,
        "duration_seconds": duration,
    }
    features_json = output_dir / f"{out_stem}_multi_features.json"
    features_json.write_text(json.dumps(features_data, indent=2))
    print(f"[Step 1] Saved: {features_json.name}")

    # ── Step 2: Composite images ──────────────────────────────────────────────
    print("\n[Step 2] Building composite images...")
    tiff_strs = [str(p) for p in tiff_paths]
    composite_rgb = _build_composite(tiff_strs, normalize, upscale, circles=False)
    pm_boxes = _draw_element_boxes(composite_rgb, elem_features)
    _save_rgb(pm_boxes, output_dir / f"{out_stem}_multi_elements_composite.png")

    circles_rgb = None
    if circles:
        circles_rgb = _build_composite(tiff_strs, normalize, upscale, circles=True)
        circ_boxes = _draw_element_boxes(circles_rgb, elem_features)
        _save_rgb(circ_boxes, output_dir / f"{out_stem}_multi_elements_circles.png")

    print(f"[Step 2] Composite PNG(s) saved")

    # ── Step 3: LLM co-location ───────────────────────────────────────────────
    if not do_colocate:
        print("\n[Step 3] Skipping LLM co-location (llm_colocate=false)")
        print(f"\n[Done] Outputs in: {output_dir}")
        return

    print(f"\n[Step 3] Sending {total_features} features to LLM for co-location analysis...")
    prompt = build_colocate_prompt(elem_features, elem_names, sample_context)
    t1 = time.time()
    zones = query_llm_colocate(prompt, model, provider, key,
                                anthropic_url=anthropic_url, openai_url=openai_url)

    for i, z in enumerate(zones):
        z["id"] = i + 1
        z.setdefault("llm_suggestion", z.pop("suggestion", ""))
        z.setdefault("confidence", "medium")
        z.setdefault("alternatives", [])
        z.setdefault("reasoning", {})

    print(f"[Step 3] {len(zones)} zone(s) identified in {round(time.time() - t1, 1)}s")

    colocate_data = {
        "elements": elem_names,
        "model": model,
        "method": "llm",
        "mode": "multi",
        "overlap_zones": zones,
        "total_zones": len(zones),
    }
    if sample_context:
        colocate_data["sample_context"] = sample_context

    colocate_json = output_dir / f"{out_stem}_multi_llm_colocate.json"
    colocate_json.write_text(json.dumps(colocate_data, indent=2))
    print(f"[Step 3] Saved: {colocate_json.name}")

    if zones:
        comp_boxes = _draw_zone_boxes(composite_rgb, zones)
        _save_rgb(comp_boxes, output_dir / f"{out_stem}_multi_colocate_composite.png")
        if circles_rgb is not None:
            circ_boxes = _draw_zone_boxes(circles_rgb, zones)
            _save_rgb(circ_boxes, output_dir / f"{out_stem}_multi_colocate_circles.png")
        print(f"[Step 3] Co-location composite PNG(s) saved")

    print(f"\n[Done] All outputs in: {output_dir}")


def main():
    if len(sys.argv) < 2:
        print("Usage: feature-locator-headless <config.json>")
        print()
        print("Config JSON fields:")
        print('  "input_dir"      : folder of element TIFFs (required)')
        print('  "output_dir"     : output folder (default: input_dir)')
        print('  "model"          : model name (default: "claude-sonnet-4-6")')
        print('  "circles"        : colored circles on white bg (default: true)')
        print('  "normalize"      : normalize TIFF intensities (default: true)')
        print('  "upscale"        : upscale to 1092px (default: true)')
        print('  "llm_colocate"   : run LLM co-location analysis (default: true)')
        print('  "sample_context" : optional string describing the sample')
        print('  "elements"       : optional list of element names to include')
        sys.exit(1)

    config_path = pathlib.Path(sys.argv[1])
    if not config_path.exists():
        print(f"[ERROR] Config not found: {config_path}")
        sys.exit(1)

    config = json.loads(config_path.read_text())
    run(config)


if __name__ == "__main__":
    main()
