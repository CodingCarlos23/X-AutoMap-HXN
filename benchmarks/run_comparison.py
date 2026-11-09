"""
Detection + phase comparison run — AutoMap (OpenCV, StarDist) vs VLM/LLM (raw / circle / flower / car).

Usage:
    pixi run python benchmarks/run_comparison.py benchmarks/configs/compare_default.json
    pixi run python benchmarks/run_comparison.py <config> --no-vlm      # AutoMap only, no API cost

Level 1 — per-element detection boxes
    AutoMap: MASTER_raw_detections.json from the real analyze_data_local() run
    VLM:     <name>_multi_features.json from the Feature Locator multi-element workflow
Level 2 — phase / material identification
    AutoMap: phase_detections.json (centroid unions + materials-library match)
    VLM:     <name>_multi_llm_colocate.json (LLM co-location zones + material suggestion)

Each method works on its own copy of the TIFFs inside the run folder, so test_samples/ is never written to.

Output: benchmarks/runs/<timestamp>_<sample_dir_name>/
    <sample>/automap_<method>/ and <sample>/vlm_<variant>/, each containing:
        <folder>_all_blobs.json             every blob per element (same schema for every method)
        <folder>_materials.json             every material region (same schema for every method)
        images/<E>_original[_boxes].png     element image without / with this method's boxes
        images/<E>_model_input[_boxes].png  the exact image the method analysed (AutoMap: normalized ± dilation;
                                            VLM: the image it was sent), without / with boxes
        images/materials_composite[_boxes].png
      plus the method's native outputs (AutoMap *_det/ folder; Feature Locator *_multi_*.json + PNGs)
    results.json                            everything the report shows, in one consistent format
    index.html                              self-contained report (open directly in a browser)
"""

import argparse
import base64
import concurrent.futures
import copy
import datetime
import json
import os
import pathlib
import re
import shutil
import sys
import time
from collections import Counter

os.environ.setdefault("MPLBACKEND", "Agg")  # AutoMap plotting must not open windows

import cv2
import numpy as np
import tifffile

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from automap_hxn import vlm_pipeline  # noqa: E402
from automap_hxn.utils import normalize_and_dilate, resize_if_needed  # noqa: E402
from automap_hxn.feature_locator_headless import (  # noqa: E402
    _build_composite, _draw_element_boxes, _draw_zone_boxes, _save_rgb,
)
from report import write_report, write_runs_index  # noqa: E402
import reconcile  # noqa: E402

VLM_VARIANTS = ("raw", "standard", "circle", "flower", "car")

# USD per 1M tokens (input, output) — Anthropic first-party list prices; override via config "pricing".
DEFAULT_PRICING = {
    "claude-haiku-4-5": [1.0, 5.0],
    "claude-sonnet-4-5": [3.0, 15.0],
    "claude-sonnet-4-6": [3.0, 15.0],
    "claude-opus-4-5": [5.0, 25.0],
    "claude-opus-4-6": [5.0, 25.0],
    "claude-opus-4-7": [5.0, 25.0],
    "claude-opus-4-8": [5.0, 25.0],
    "claude-fable-5": [10.0, 50.0],
}


def call_record(kind, method, element, meta, dt, pricing, model, error=None, raw_path=None):
    """One row of the API-call log: tokens, cost, stop reason, truncation, parse status."""
    tin, tout = meta.get("input_tokens"), meta.get("output_tokens")
    price = pricing.get(model)
    cost = (tin * price[0] + tout * price[1]) / 1e6 if (price and tin is not None and tout is not None) else None
    return {"kind": kind, "method": method, "element": element, "time_s": round(dt, 2),
            "stop_reason": meta.get("stop_reason"), "truncated": bool(meta.get("truncated")),
            "parse_status": "error" if error else meta.get("parse_status"),
            "input_tokens": tin, "output_tokens": tout,
            "cost_usd": round(cost, 5) if cost is not None else None,
            "error": error, "raw_reply": raw_path}
BOX_COLOR_BGR = (0, 161, 237)  # amber #eda100 — visible on the dark element images


# ── Sample discovery ──────────────────────────────────────────────────────────

_TIFF_RE = re.compile(r"^scan_(?P<scan>[^_]+)_(?P<elem>[A-Za-z0-9]+)\.tiff?$")


def _tiffs_in(folder: pathlib.Path) -> list:
    return sorted(p for p in folder.iterdir() if p.is_file() and _TIFF_RE.match(p.name))


def discover_samples(sample_dir: pathlib.Path, elem_filter) -> dict:
    """{sample_name: {"scan_id": str, "tiffs": {elem: path}}}. Accepts a folder of TIFFs or of such folders."""
    folders = [sample_dir] if _tiffs_in(sample_dir) else sorted(
        p for p in sample_dir.iterdir() if p.is_dir() and _tiffs_in(p))
    allowed = {e.lower() for e in elem_filter} if elem_filter else None
    samples = {}
    for f in folders:
        tiffs, scan_ids = {}, set()
        for p in _tiffs_in(f):
            m = _TIFF_RE.match(p.name)
            if allowed and m["elem"].lower() not in allowed:
                continue
            tiffs[m["elem"]] = p
            scan_ids.add(m["scan"])
        if len(scan_ids) > 1:
            sys.exit(f"[ERROR] {f} mixes scan ids {sorted(scan_ids)} — put each scan in its own folder")
        if tiffs:
            samples[f.name] = {"scan_id": scan_ids.pop(), "tiffs": dict(sorted(tiffs.items()))}
    return samples


# ── Box helpers (all boxes are fractional 0–1: x1, y1, x2, y2) ────────────────

def _frac_box(x1, y1, x2, y2, w, h) -> dict:
    return {"x1": max(0.0, x1 / w), "y1": max(0.0, y1 / h),
            "x2": min(1.0, x2 / w), "y2": min(1.0, y2 / h)}


def _clean_vlm_box(f: dict):
    try:
        x1, y1, x2, y2 = (float(f[k]) for k in ("x1", "y1", "x2", "y2"))
    except (KeyError, TypeError, ValueError):
        return None
    if 1.0 < max(x1, y1, x2, y2) <= 100.0:  # some models answer in percent
        x1, y1, x2, y2 = (v / 100.0 for v in (x1, y1, x2, y2))
    x1, x2 = sorted((min(max(x1, 0.0), 1.0), min(max(x2, 0.0), 1.0)))
    y1, y2 = sorted((min(max(y1, 0.0), 1.0), min(max(y2, 0.0), 1.0)))
    if x2 <= x1 or y2 <= y1:
        return None
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _area(b):
    return (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])


def _inter(a, b):
    ix = max(0.0, min(a["x2"], b["x2"]) - max(a["x1"], b["x1"]))
    iy = max(0.0, min(a["y2"], b["y2"]) - max(a["y1"], b["y1"]))
    return ix * iy


def iou(a, b) -> float:
    i = _inter(a, b)
    u = _area(a) + _area(b) - i
    return i / u if u > 0 else 0.0


def overlap_coef(a, b) -> float:
    """Intersection / smaller box. Used for phase regions: AutoMap reports the *union* of its blobs,
    the LLM reports the *intersection* of elements, so plain IoU would under-match."""
    m = min(_area(a), _area(b))
    return _inter(a, b) / m if m > 0 else 0.0


def greedy_match(a: list, b: list, score_fn, thresh: float) -> list:
    """One-to-one matching, best score first. Returns [(i, j, score)]."""
    pairs = sorted(((score_fn(x, y), i, j) for i, x in enumerate(a) for j, y in enumerate(b)),
                   reverse=True)
    used_a, used_b, out = set(), set(), []
    for s, i, j in pairs:
        if s < thresh:
            break
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        out.append((i, j, s))
    return out


def box_agreement(a: list, b: list, thresh: float) -> dict:
    m = greedy_match(a, b, iou, thresh)
    n = len(a) + len(b)
    return {"matched": len(m), "f1": (2 * len(m) / n) if n else 1.0,
            "mean_iou": float(np.mean([s for *_, s in m])) if m else 0.0}


# ── AutoMap baseline (real analyze_data_local run on an isolated copy) ────────

def run_automap(sample: dict, base_cfg: dict, method: str, method_block: dict,
                work: pathlib.Path, dilate: bool = True) -> dict:
    from automap_hxn.analysis import analyze_data_local

    work.mkdir(parents=True, exist_ok=True)
    for p in sample["tiffs"].values():
        shutil.copy2(p, work / p.name)
    params = copy.deepcopy(base_cfg)
    params.setdefault("scan_params", {})["scan_id"] = sample["scan_id"]
    params.setdefault("export_params", {})["out_dir"] = str(work)
    params.setdefault("segmentation_params", {})["blob_detection_method"] = method
    if method_block:
        params.setdefault("detection_methods", {})[method] = method_block
    if not dilate:  # "raw": original normalized image, no dilation
        params.setdefault("morphology_params", {})["dilate_iterations"] = 0
    (work / "automap_config_used.json").write_text(json.dumps(params, indent=2))

    t0 = time.time()
    analyze_data_local(scan_id=sample["scan_id"], params=params)
    det_dir = work / f"{work.name}_det"
    return {
        "time_s": round(time.time() - t0, 2),
        "master": json.loads((det_dir / "MASTER_raw_detections.json").read_text()),
        "phase": (json.loads((det_dir / "phase_detections.json").read_text())
                  if (det_dir / "phase_detections.json").exists() else {"unions": []}),
        "det_dir": det_dir,
    }


# ── VLM ───────────────────────────────────────────────────────────────────────

def encode_variant(path: pathlib.Path, variant: str, elem_idx: int, vcfg: dict) -> str:
    if variant == "raw":  # original normalized image: min-max scaling only
        return vlm_pipeline.encode_image_for_vlm(str(path), normalize=True, upscale=False, clahe=False)
    kw = dict(normalize=vcfg.get("normalize", True), upscale=vcfg.get("upscale", True))
    if variant == "circle":
        # identical to the Feature Locator multi workflow: per-element circle colour
        cc = vlm_pipeline.CIRCLE_COLORS_RGB[elem_idx % len(vlm_pipeline.CIRCLE_COLORS_RGB)]
        kw.update(red_circles=True, circle_color=cc)
    elif variant in ("flower", "car"):
        kw.update(emoji_marker=variant)
    return vlm_pipeline.encode_image_for_vlm(str(path), **kw)


# ── Images ────────────────────────────────────────────────────────────────────

def _display_image(img: np.ndarray, size: int = 512) -> np.ndarray:
    """Upscale (nearest-neighbour, display only) an image to `size`. Float input is min-max scaled to 0–255."""
    if img.dtype != np.uint8:
        img = cv2.normalize(np.nan_to_num(img), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    norm = img
    h, w = norm.shape[:2]
    scale = size / max(h, w)
    norm = cv2.resize(norm, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)


def draw_boxes(base_bgr: np.ndarray, boxes: list, out: pathlib.Path, color=BOX_COLOR_BGR, thickness=1):
    img = base_bgr.copy()
    h, w = img.shape[:2]
    for b in boxes:
        p1 = (int(b["x1"] * w), int(b["y1"] * h))
        p2 = (max(p1[0] + 1, int(b["x2"] * w)), max(p1[1] + 1, int(b["y2"] * h)))
        cv2.rectangle(img, p1, p2, color, thickness)
    cv2.imwrite(str(out), img)


def _composite_bgr(tiff_paths: list, size: int = 640) -> np.ndarray:
    rgb = _build_composite([str(p) for p in tiff_paths], normalize=True, upscale=False, circles=False)
    h, w = rgb.shape[:2]
    scale = size / max(h, w)
    rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def method_dir(s_dir: pathlib.Path, mid: str) -> pathlib.Path:
    return s_dir / (mid.replace("opencv:", "automap_opencv_").replace("stardist:", "automap_stardist_")
                       .replace("vlm:", "vlm_").replace(":", "_"))


def _px(b: dict, w: int, h: int) -> dict:
    return {"x1": round(b["x1"] * w, 1), "y1": round(b["y1"] * h, 1),
            "x2": round(b["x2"] * w, 1), "y2": round(b["y2"] * h, 1)}


def write_method_outputs(mid, s_dir, run_dir, sname, sample, elements, phase,
                         displays, comp, shape, feats_by, raw_imgs=None, morph=None):
    """For one method write:
         <folder>_all_blobs.json   every blob/box per element (same schema for every method)
         <folder>_materials.json   every multi-element material region (same schema for every method)
         images/<E>_original.png + <E>_original_boxes.png            what the element looks like, ± its boxes
         images/<E>_model_input.png + <E>_model_input_boxes.png      the exact image the method analysed, ± boxes
         images/materials_composite.png + materials_composite_boxes.png
    """
    th, tw = shape
    mdir = method_dir(s_dir, mid)
    idir = mdir / "images"
    idir.mkdir(parents=True, exist_ok=True)
    rel = lambda p: str(p.relative_to(run_dir))
    is_vlm = mid.startswith("vlm:")

    all_blobs = {"method": mid, "sample": sname, "scan_id": sample["scan_id"], "image_shape": [th, tw],
                 "coordinates": "x1,y1,x2,y2 fractional 0-1 (0,0 = top-left); 'px' = pixels of the TIFF",
                 "source": (f"{sname}_multi_features.json" if is_vlm
                            else f"{mdir.name}_det/MASTER_raw_detections.json"),
                 "elements": {}}
    for e, d in elements.items():
        boxes = d["boxes"].get(mid, [])
        all_blobs["elements"][e] = {
            "count": len(boxes), "error": d["errors"].get(mid),
            "blobs": [{"id": k + 1, **b, "px": _px(b, tw, th)} for k, b in enumerate(boxes)]}
        imgs = {}
        o = idir / f"{e}_original.png"
        cv2.imwrite(str(o), displays[e])
        ob = idir / f"{e}_original_boxes.png"
        draw_boxes(displays[e], boxes, ob)
        imgs.update(original=rel(o), original_boxes=rel(ob))
        inp = idir / f"{e}_model_input.png"
        if not is_vlm and raw_imgs is not None:
            # rebuild exactly what analyze_data_local handed the detector
            k, it = morph
            _, dilated = normalize_and_dilate(raw_imgs[e], kernel_size=tuple(k),
                                              iterations=0 if mid.endswith(":raw") else it)
            cv2.imwrite(str(inp), _display_image(dilated))
        if inp.exists():
            ib = idir / f"{e}_model_input_boxes.png"
            base = cv2.imread(str(inp))
            draw_boxes(base, boxes, ib, thickness=2 if max(base.shape[:2]) > 600 else 1)
            imgs.update(input=rel(inp), input_boxes=rel(ib))
        d["images"][mid] = imgs
    (mdir / f"{mdir.name}_all_blobs.json").write_text(json.dumps(all_blobs, indent=2))

    regs = phase["regions"].get(mid, [])
    (mdir / f"{mdir.name}_materials.json").write_text(json.dumps({
        "method": mid, "sample": sname, "scan_id": sample["scan_id"], "image_shape": [th, tw],
        "source": (f"{sname}_multi_llm_colocate.json" if is_vlm
                   else f"{mdir.name}_det/phase_detections.json"),
        "note": ("LLM co-location zones (box = where the elements intersect)" if is_vlm else
                 "multi-element unions (box = union of the blobs) with materials-library match"),
        "error": phase["errors"].get(mid),
        "total": len(regs),
        "regions": [{**r, "px": _px(r["box"], tw, th)} for r in regs]}, indent=2))
    cp = idir / "materials_composite.png"
    cv2.imwrite(str(cp), comp)
    cb = idir / "materials_composite_boxes.png"
    draw_boxes(comp, [r["box"] for r in regs], cb, color=(255, 255, 0))  # cyan on the bright composite
    phase["images"][mid] = {"plain": rel(cp), "boxes": rel(cb)}


# ── Phase comparison ──────────────────────────────────────────────────────────

def _automap_regions(phase: dict, w: int, h: int) -> list:
    """Multi-element unions only — co-location zones are ≥2 elements by definition."""
    out = []
    for u in phase.get("unions", []):
        els = u.get("elements_detected", [])
        if len(els) < 2:
            continue
        b = u["union_box"]
        top = (u.get("matches") or [{}])[0]
        out.append({"id": u.get("union_id"), "elements": sorted(els),
                    "box": _frac_box(b["x1"], b["y1"], b["x2"], b["y2"], w, h),
                    "material": top.get("material", "no library match"),
                    "confidence": top.get("confidence", "—"),
                    "ratios": u.get("measured_ratios", {})})
    return out


def _vlm_regions(zones: list) -> list:
    out = []
    for z in zones:
        b = _clean_vlm_box(z)
        if not b:
            continue
        out.append({"id": z.get("id"), "elements": sorted(z.get("elements", [])), "box": b,
                    "material": z.get("llm_suggestion") or z.get("suggestion", ""),
                    "confidence": z.get("confidence", "—"),
                    "reasoning": z.get("reasoning", "")})
    return out


def compare_phase(base: list, vlm: list, thresh: float) -> dict:
    m = greedy_match([r["box"] for r in base], [r["box"] for r in vlm], overlap_coef, thresh)
    rows, jac = [], []
    for i, j, s in m:
        ea, eb = set(base[i]["elements"]), set(vlm[j]["elements"])
        jj = len(ea & eb) / len(ea | eb) if ea | eb else 1.0
        jac.append(jj)
        rows.append({"automap": base[i], "vlm": vlm[j], "overlap": round(s, 3),
                     "element_jaccard": round(jj, 3)})
    n = len(base) + len(vlm)
    return {"n_automap": len(base), "n_vlm": len(vlm), "matched": len(m),
            "f1": (2 * len(m) / n) if n else 1.0,
            "mean_element_jaccard": float(np.mean(jac)) if jac else 0.0,
            "rows": rows}


# ── Main run ──────────────────────────────────────────────────────────────────

def run(config: dict, config_path: pathlib.Path, no_vlm: bool = False,
        runs_root: "pathlib.Path | None" = None) -> pathlib.Path:
    def _resolve(p):
        p = pathlib.Path(p).expanduser()
        if p.is_absolute():
            return p
        return (REPO_ROOT / p) if (REPO_ROOT / p).exists() else (config_path.parent / p)

    sample_dir = _resolve(config["sample_dir"])
    samples = discover_samples(sample_dir, config.get("elements"))
    if not samples:
        sys.exit(f"[ERROR] No scan_<id>_<element>.tiff files found under {sample_dir}")

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (runs_root or REPO_ROOT / "benchmarks" / "runs") / f"{stamp}_{sample_dir.name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    match_iou = float(config.get("match_iou", 0.25))
    phase_thresh = float(config.get("phase_match_overlap", 0.5))

    # AutoMap methods
    base_cfg = json.loads(_resolve(config["automap_config"]).read_text())
    seg = base_cfg.get("segmentation_params", {})
    automap_methods = []  # (method_id, detector, method_block, dilate)
    detectors = []
    opencv_method = seg.get("blob_detection_method", "simple")
    detectors.append((f"opencv:{opencv_method}", opencv_method,
                      base_cfg.get("detection_methods", {}).get(opencv_method, {})))
    sd = config.get("stardist", {})
    if sd.get("enabled", True):
        detectors.append(("stardist", "stardist", {k: v for k, v in sd.items() if k != "enabled"}))
    morph = base_cfg.get("morphology_params", {})
    k = morph.get("normalize_kernel_size") or [3, 3]
    it = morph.get("dilate_iterations", 2)
    model_input_desc = {}
    for base_id, det, block in detectors:
        automap_methods.append((f"{base_id}:standard", det, block, True))
        model_input_desc[f"{base_id}:standard"] = f"min-max normalized + dilated ({k[0]}×{k[1]} kernel, ×{it})"
        if config.get("automap_raw", True):
            automap_methods.append((f"{base_id}:raw", det, block, False))
            model_input_desc[f"{base_id}:raw"] = "original normalized (min-max only, no dilation)"

    # VLM
    vcfg = config.get("vlm", {})
    variants = [] if no_vlm else [v for v in vcfg.get("variants", VLM_VARIANTS) if v in VLM_VARIANTS]
    model = vcfg.get("model", "claude-sonnet-4-6")
    colocate = bool(vcfg.get("llm_colocate", True))
    sample_context = vcfg.get("sample_context", "").strip()
    provider = key = a_url = o_url = None
    if variants:
        provider = vlm_pipeline.get_provider(model)
        key, a_url, o_url = vlm_pipeline.resolve_key(provider)
        if provider != "ollama" and not key:
            sys.exit(f"[ERROR] No API key for {provider}. Load it (see RUN_COMMANDS.txt) or use --no-vlm.")
    suffixes = vcfg.get("prompt_suffix", {})
    prompts = {v: vlm_pipeline.LOCATE_SPARSE_XRF_PROMPT + (" " + suffixes[v] if suffixes.get(v) else "")
               for v in variants}

    up = "upscaled to 1092px + " if vcfg.get("upscale", True) else ""
    vlm_desc = {
        "raw": "original normalized (min-max only, native size)",
        "standard": f"min-max + {up}CLAHE contrast (GUI Feature Locator default)",
        "circle": f"min-max + {up}particles (Otsu) drawn as per-element coloured circles on white",
        "flower": f"min-max + {up}particles (Otsu) drawn as flower emoji on white",
        "car": f"min-max + {up}particles (Otsu) drawn as car emoji on white",
    }
    for v in variants:
        model_input_desc[f"vlm:{v}"] = vlm_desc[v]
    methods = [mid for mid, *_ in automap_methods] + [f"vlm:{v}" for v in variants]
    print(f"[Compare] Samples: {', '.join(samples)}")
    print(f"[Compare] Methods: {', '.join(methods)}")
    if variants:
        n_el = sum(len(s["tiffs"]) for s in samples.values())
        print(f"[Compare] VLM model: {model} ({provider}) — {n_el * len(variants)} image calls"
              + (f" + {len(variants) * len(samples)} co-location calls" if colocate else ""))

    pricing = {**DEFAULT_PRICING, **config.get("pricing", {})}
    calls = []  # every API call: tokens, cost, stop reason, parse status
    results = {"samples": {}}
    for sname, sample in samples.items():
        s_dir = run_dir / sname
        elem_names = list(sample["tiffs"])
        tiff_list = list(sample["tiffs"].values())
        shapes = {e: tifffile.imread(str(p)).shape[:2] for e, p in sample["tiffs"].items()}
        th, tw = Counter(shapes.values()).most_common(1)[0][0]  # AutoMap's target shape

        elements = {e: {"file": p.name, "shape": [th, tw], "boxes": {}, "images": {},
                        "errors": {}} for e, p in sample["tiffs"].items()}
        phase = {"regions": {}, "times": {}, "errors": {}, "images": {}, "comparisons": {}}

        # ── AutoMap (sequential: TensorFlow is not thread-safe) ──
        for mid, det, block, dilate in automap_methods:
            work = method_dir(s_dir, mid)
            print(f"\n[{sname}] AutoMap {mid} …")
            try:
                r = run_automap(sample, base_cfg, det, block, work, dilate=dilate)
            except Exception as err:
                msg = f"{type(err).__name__}: {err}"
                print(f"  [ERROR] {msg}")
                phase["errors"][mid] = msg
                for e in elements:
                    elements[e]["errors"][mid] = msg
                    elements[e]["boxes"][mid] = []
                continue
            for e in elements:
                elements[e]["boxes"][mid] = []
            for blob in r["master"].values():
                e = blob.get("element")
                if e in elements:
                    (cx, cy), rad = blob["image_center"], blob["image_radius"]
                    elements[e]["boxes"][mid].append(_frac_box(cx - rad, cy - rad, cx + rad, cy + rad, tw, th))
            phase["regions"][mid] = _automap_regions(r["phase"], tw, th)
            phase["times"][mid] = r["time_s"]
            for e in elements:
                elements[e].setdefault("time_s", {})[mid] = None  # AutoMap timing is per sample
            print(f"  {sum(len(elements[e]['boxes'][mid]) for e in elements)} blobs, "
                  f"{len(phase['regions'][mid])} multi-element unions in {r['time_s']}s")

        # ── VLM: every element × variant in parallel ──
        feats_by = {v: {} for v in variants}
        if variants:
            print(f"\n[{sname}] VLM: {len(elem_names) * len(variants)} image calls …")

            for v in variants:
                (s_dir / f"vlm_{v}" / "images").mkdir(parents=True, exist_ok=True)

            def _job(i, e, v):
                b64 = encode_variant(sample["tiffs"][e], v, i, vcfg)
                inp = s_dir / f"vlm_{v}" / "images" / f"{e}_model_input.png"
                inp.write_bytes(base64.b64decode(b64))
                meta = {}
                t0 = time.time()
                try:
                    f = vlm_pipeline.query_vlm_features(b64, prompts[v], model, provider, key or "",
                                                        anthropic_url=a_url, openai_url=o_url, meta=meta)
                finally:
                    meta["_dt"] = time.time() - t0
                return f, meta, inp

            for v in variants:
                (s_dir / f"vlm_{v}" / "raw_replies").mkdir(parents=True, exist_ok=True)

            jobs = {(i, e, v) for i, e in enumerate(elem_names) for v in variants}
            with concurrent.futures.ThreadPoolExecutor(max_workers=int(vcfg.get("max_workers", 8))) as ex:
                futs = {ex.submit(_job, i, e, v): (e, v) for i, e, v in jobs}
                for fut in concurrent.futures.as_completed(futs):
                    e, v = futs[fut]
                    mid = f"vlm:{v}"
                    try:
                        f, meta, inp = fut.result()
                        feats_by[v][e] = f
                        raw = s_dir / f"vlm_{v}" / "raw_replies" / f"{e}.txt"
                        raw.write_text(meta.get("text") or "")
                        rec = call_record("locate", mid, e, meta, meta["_dt"], pricing, model,
                                          raw_path=str(raw.relative_to(run_dir)))
                        elements[e].setdefault("time_s", {})[mid] = rec["time_s"]
                        if rec["truncated"] or rec["parse_status"] == "failed":
                            elements[e].setdefault("flags", {})[mid] = (
                                "reply cut off at max_tokens" if rec["truncated"] else "reply could not be parsed")
                    except Exception as err:
                        feats_by[v][e] = []
                        elements[e]["errors"][mid] = f"{type(err).__name__}: {err}"
                        rec = call_record("locate", mid, e, {}, 0.0, pricing, model,
                                          error=elements[e]["errors"][mid])
                    calls.append(rec)
                    elements[e]["boxes"][mid] = [b for b in map(_clean_vlm_box, feats_by[v][e]) if b]
                    print(f"  {e:>3} {mid:<11} {len(elements[e]['boxes'][mid])} box(es)")

            # per-variant Feature Locator outputs (same file names/format as the GUI + headless runner)
            for v in variants:
                vdir = s_dir / f"vlm_{v}"
                vdir.mkdir(parents=True, exist_ok=True)
                ef = [feats_by[v].get(e, []) for e in elem_names]
                (vdir / f"{sname}_multi_features.json").write_text(json.dumps({
                    "elements": elem_names, "elem_features": ef, "total_count": sum(map(len, ef)),
                    "model": model, "variant": v, "system_prompt": prompts[v]}, indent=2))

            # LLM co-location per variant
            if colocate:
                print(f"\n[{sname}] LLM co-location: {len(variants)} call(s) …")

                def _colo(v):
                    ef = [feats_by[v].get(e, []) for e in elem_names]
                    meta = {}
                    t0 = time.time()
                    zones = vlm_pipeline.query_llm_colocate(
                        vlm_pipeline.build_colocate_prompt(ef, elem_names, sample_context),
                        model, provider, key or "", anthropic_url=a_url, openai_url=o_url, meta=meta)
                    (s_dir / f"vlm_{v}" / "raw_replies" / "colocate.txt").write_text(meta.get("text") or "")
                    colo_meta[v] = meta
                    for k, z in enumerate(zones):  # normalise exactly like the headless runner
                        z["id"] = k + 1
                        z.setdefault("llm_suggestion", z.pop("suggestion", ""))
                        z.setdefault("confidence", "medium")
                        z.setdefault("alternatives", [])
                    return zones, time.time() - t0

                colo_meta = {}
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(variants)) as ex:
                    futs = {ex.submit(_colo, v): v for v in variants}
                    for fut in concurrent.futures.as_completed(futs):
                        v = futs[fut]
                        mid = f"vlm:{v}"
                        try:
                            zones, dt = fut.result()
                        except Exception as err:
                            phase["errors"][mid] = f"{type(err).__name__}: {err}"
                            phase["regions"][mid] = []
                            calls.append(call_record("colocate", mid, None, {}, 0.0, pricing, model,
                                                     error=phase["errors"][mid]))
                            continue
                        rec = call_record("colocate", mid, None, colo_meta.get(v, {}), dt, pricing, model,
                                          raw_path=str((s_dir / f"vlm_{v}" / "raw_replies" / "colocate.txt")
                                                       .relative_to(run_dir)))
                        calls.append(rec)
                        if rec["truncated"] or rec["parse_status"] == "failed":
                            phase.setdefault("flags", {})[mid] = (
                                "reply cut off at max_tokens" if rec["truncated"] else "reply could not be parsed")
                        data = {"elements": elem_names, "model": model, "method": "llm", "mode": "multi",
                                "variant": v, "overlap_zones": zones, "total_zones": len(zones)}
                        if sample_context:
                            data["sample_context"] = sample_context
                        (s_dir / f"vlm_{v}" / f"{sname}_multi_llm_colocate.json").write_text(
                            json.dumps(data, indent=2))
                        phase["regions"][mid] = _vlm_regions(zones)
                        phase["times"][mid] = round(dt, 2)
                        print(f"  {mid:<11} {len(zones)} zone(s)")

            # composite PNGs alongside each variant's JSONs (like the GUI)
            comp_rgb = _build_composite([str(p) for p in tiff_list], True, True, circles=False)
            for v in variants:
                vdir = s_dir / f"vlm_{v}"
                ef = [feats_by[v].get(e, []) for e in elem_names]
                _save_rgb(_draw_element_boxes(comp_rgb, ef), vdir / f"{sname}_multi_elements_composite.png")
                zones_v = phase["regions"].get(f"vlm:{v}")
                if zones_v:
                    _save_rgb(_draw_zone_boxes(comp_rgb, [{**r["box"], "llm_suggestion": r["material"],
                                                          "confidence": r["confidence"],
                                                          "elements": r["elements"]} for r in zones_v]),
                              vdir / f"{sname}_multi_colocate_composite.png")

        # ── Per-method outputs: normalised JSONs + with/without-box images ──
        raw_imgs = {}
        for e, p in sample["tiffs"].items():
            a = tifffile.imread(str(p)).astype(np.float32)
            raw_imgs[e] = resize_if_needed(a, p.name, (th, tw)) if a.shape[:2] != (th, tw) else a
        displays = {e: _display_image(a) for e, a in raw_imgs.items()}
        comp = _composite_bgr(tiff_list)
        for mid in methods:
            write_method_outputs(mid, s_dir, run_dir, sname, sample, elements, phase,
                                 displays, comp, (th, tw), feats_by, raw_imgs, (k, it))
        for e in elements:
            elements[e]["agreement"] = {
                f"{a}|{b}": box_agreement(elements[e]["boxes"].get(a, []), elements[e]["boxes"].get(b, []), match_iou)
                for i, a in enumerate(methods) for b in methods[i + 1:]}

        # ── Level 2: AutoMap vs VLM material regions ──
        for mid, *_ in automap_methods:
            for v in variants:
                vm = f"vlm:{v}"
                if mid in phase["regions"] and vm in phase["regions"]:
                    phase["comparisons"][f"{mid}|{vm}"] = compare_phase(
                        phase["regions"][mid], phase["regions"][vm], phase_thresh)

        # ── Level 3: one LLM pass reconciling every method's material regions ──
        recon = None
        rcfg = config.get("reconcile", {})
        if variants and rcfg.get("enabled", True) and phase["regions"]:
            r_model = rcfg.get("model") or model
            r_provider = vlm_pipeline.get_provider(r_model)
            r_key, r_a, r_o = (key, a_url, o_url) if r_provider == provider else vlm_pipeline.resolve_key(r_provider)
            regs = {m: phase["regions"][m] for m in methods if m in phase["regions"]}
            prompt, id_map = reconcile.build_prompt(regs, model_input_desc, sample_context)
            print(f"\n[{sname}] LLM material reconciliation: {len(id_map)} regions from {len(regs)} methods …")
            (s_dir / "material_reconciliation_prompt.txt").write_text(prompt)
            meta, t0 = {}, time.time()
            try:
                text = vlm_pipeline.query_llm_text(reconcile.SYSTEM, prompt, r_model, r_provider, r_key or "",
                                                   anthropic_url=r_a, openai_url=r_o,
                                                   max_tokens=int(rcfg.get("max_tokens", 16384)), meta=meta)
                (s_dir / "material_reconciliation_reply.txt").write_text(text)
                reply = reconcile.parse_reply(text)
                meta["parse_status"] = "ok" if reply.get("groups") else "failed"
                recon = reconcile.postprocess(reply, id_map, regs)
                recon.update(model=r_model, prompt_file="material_reconciliation_prompt.txt",
                             reply_file="material_reconciliation_reply.txt")
                (s_dir / "material_reconciliation.json").write_text(json.dumps(recon, indent=2))
                calls.append(call_record("reconcile", "reconcile", None, meta, time.time() - t0, pricing, r_model,
                                         raw_path=str((s_dir / "material_reconciliation_reply.txt").relative_to(run_dir))))
                print(f"  {len(recon['groups'])} material groups ({recon['n_unassigned']} not assigned by the LLM)")
            except Exception as err:
                msg = f"{type(err).__name__}: {err}"
                print(f"  [ERROR] {msg}")
                recon = {"error": msg}
                calls.append(call_record("reconcile", "reconcile", None, meta, time.time() - t0, pricing, r_model,
                                         error=msg))

        results["samples"][sname] = {"scan_id": sample["scan_id"], "elements": elements, "phase": phase,
                                     "reconcile": recon}

    # overall Level-1 agreement pooled over every element
    overall = {}
    for i, a in enumerate(methods):
        for b in methods[i + 1:]:
            m = n = 0
            ious = []
            for s in results["samples"].values():
                for el in s["elements"].values():
                    r = el["agreement"][f"{a}|{b}"]
                    m += r["matched"]
                    n += len(el["boxes"].get(a, [])) + len(el["boxes"].get(b, []))
                    if r["matched"]:
                        ious.append(r["mean_iou"])
            overall[f"{a}|{b}"] = {"matched": m, "f1": (2 * m / n) if n else 1.0,
                                  "mean_iou": float(np.mean(ious)) if ious else 0.0}

    results.update({
        "run": run_dir.name,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "config": config,
        "automap_config": str(_resolve(config["automap_config"])),
        "methods": methods,
        "automap_methods": [mid for mid, *_ in automap_methods],
        "model_input": model_input_desc,
        "vlm_variants": [f"vlm:{v}" for v in variants],
        "vlm_model": model if variants else None,
        "prompts": prompts,
        "match_iou": match_iou,
        "phase_match_overlap": phase_thresh,
        "overall_agreement": overall,
        "calls": calls,
        "pricing_per_mtok": pricing.get(model) if variants else None,
    })
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    write_report(results, run_dir / "index.html")
    write_runs_index(run_dir.parent)
    print(f"\n[Compare] Done → {run_dir / 'index.html'}")
    return run_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=pathlib.Path)
    ap.add_argument("--no-vlm", action="store_true", help="skip VLM/LLM calls (no API cost)")
    ap.add_argument("--out-dir", type=pathlib.Path, default=None,
                    help="folder to put the run in (default: benchmarks/runs/) — use for test runs")
    args = ap.parse_args()
    run(json.loads(args.config.read_text()), args.config.resolve(), no_vlm=args.no_vlm,
        runs_root=args.out_dir.resolve() if args.out_dir else None)


if __name__ == "__main__":
    main()
