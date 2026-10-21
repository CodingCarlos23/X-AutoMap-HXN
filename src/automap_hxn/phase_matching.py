import json
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import tifffile


_DEFAULT_LIBRARY_PATH = Path(__file__).parents[2] / "configs" / "materials_library.json"


def load_materials_library(path=None):
    p = Path(path) if path else _DEFAULT_LIBRARY_PATH
    if not p.exists():
        print(f"[PHASE] Warning: materials_library.json not found at {p} — phase matching skipped")
        return [], 5.0
    with open(p) as f:
        data = json.load(f)
    return data.get("materials", []), data.get("tolerance_default", 5.0)


# ── Box geometry helpers ──────────────────────────────────────────────────────

def _blob_to_box(blob):
    cx, cy = blob["image_center"]
    r = max(1, int(blob["image_radius"]))
    return [int(cx) - r, int(cy) - r, int(cx) + r, int(cy) + r]


def _boxes_touch(a, b):
    """True if boxes share any area or edge (even 1 pixel)."""
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def _merge_two_boxes(a, b):
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


# ── Step 1: per-element iterative box merging ─────────────────────────────────

def _merge_element_regions(raw_detections, elem):
    """
    For one element: convert all raw blobs to pixel boxes, then iteratively
    merge any touching boxes into one large box (min x1, min y1, max x2, max y2).
    Returns list of regions: {'box': [x1,y1,x2,y2], 'blobs': [blob, ...]}
    """
    blobs = [b for b in raw_detections.values() if b["element"] == elem]
    if not blobs:
        return []

    regions = [{"box": _blob_to_box(b), "blobs": [b]} for b in blobs]

    changed = True
    while changed:
        changed = False
        new_regions = []
        used = [False] * len(regions)
        for i in range(len(regions)):
            if used[i]:
                continue
            cur_box = regions[i]["box"][:]
            cur_blobs = list(regions[i]["blobs"])
            for j in range(i + 1, len(regions)):
                if used[j]:
                    continue
                if _boxes_touch(cur_box, regions[j]["box"]):
                    cur_box = _merge_two_boxes(cur_box, regions[j]["box"])
                    cur_blobs.extend(regions[j]["blobs"])
                    used[j] = True
                    changed = True
            new_regions.append({"box": cur_box, "blobs": cur_blobs})
        regions = new_regions

    return regions


# ── Step 2: centroid-distance cross-element union clustering ──────────────────

def _build_centroid_union_clusters(raw_detections, proximity_px, id_offset=0):
    """
    Cluster blobs into union groups using centroid distance instead of box overlap.
    Only cross-element pairs within proximity_px pixels form edges — same-element
    blobs never directly connect to each other, preventing chain-merge inflation.
    Each BFS connected component becomes one union cluster.
    """
    items = list(raw_detections.items())  # [(name, blob_dict), ...]
    n = len(items)

    adj = defaultdict(list)
    for i in range(n):
        _, blob_i = items[i]
        cx_i, cy_i = blob_i["image_center"]
        elem_i = blob_i["element"]
        for j in range(i + 1, n):
            _, blob_j = items[j]
            if blob_j["element"] == elem_i:
                continue  # only cross-element edges
            cx_j, cy_j = blob_j["image_center"]
            dist = ((cx_i - cx_j) ** 2 + (cy_i - cy_j) ** 2) ** 0.5
            if dist <= proximity_px:
                adj[i].append(j)
                adj[j].append(i)

    visited = [False] * n
    clusters = []
    uid = id_offset
    for start in range(n):
        if visited[start]:
            continue
        component = []
        q = deque([start])
        visited[start] = True
        while q:
            node = q.popleft()
            component.append(node)
            for nb in adj[node]:
                if not visited[nb]:
                    visited[nb] = True
                    q.append(nb)

        by_elem = defaultdict(lambda: {"boxes": [], "blobs": []})
        for idx in component:
            _, blob = items[idx]
            elem = blob["element"]
            box = _blob_to_box(blob)
            by_elem[elem]["boxes"].append(box)
            by_elem[elem]["blobs"].append(blob)

        all_boxes = [_blob_to_box(items[idx][1]) for idx in component]
        union_box = [
            min(b[0] for b in all_boxes),
            min(b[1] for b in all_boxes),
            max(b[2] for b in all_boxes),
            max(b[3] for b in all_boxes),
        ]

        uid += 1
        clusters.append({
            "union_id": f"centroid_union_{uid:03d}",
            "group": "all",
            "union_box": union_box,
            "by_elem": {e: d for e, d in by_elem.items()},
        })

    return clusters


# ── Step 3: intensity ratios from blobs in each cluster ───────────────────────

def _compute_ratios(cluster):
    averages = {}
    norm_fallback_blobs = []  # (elem, blob_name) pairs that lacked raw intensity
    for elem, data in cluster["by_elem"].items():
        blobs = data["blobs"]
        vals = []
        for b in blobs:
            if "mean_intensity_raw" in b:
                vals.append(b["mean_intensity_raw"])
            else:
                vals.append(b.get("mean_intensity_norm", 0))
                norm_fallback_blobs.append((elem, b.get("text", b.get("Box", "unknown"))))
        averages[elem] = sum(vals) / len(vals) if vals else 0.0
    grand = sum(averages.values())
    ratios = {e: 0.0 for e in averages} if grand == 0 else {e: round(v / grand * 100, 1) for e, v in averages.items()}
    return ratios, norm_fallback_blobs


# ── Library scoring ───────────────────────────────────────────────────────────

def _confidence_label(avg_dev):
    if avg_dev < 3:
        return "high"
    if avg_dev <= 6:
        return "medium"
    return "low"


def score_against_library(ratios, library, tolerance_default):
    matches = []
    for mat in library:
        tol = mat.get("tolerance", tolerance_default)
        deviations = []
        passed = True
        for elem, lib_val in mat["elements"].items():
            dev = abs(ratios.get(elem, 0.0) - lib_val)
            deviations.append(dev)
            if dev > tol:
                passed = False
                break
        if passed:
            avg_dev = round(sum(deviations) / len(deviations), 2) if deviations else 0.0
            matches.append({
                "material": mat["name"],
                "elements_in_material": " ".join(sorted(mat["elements"].keys())),
                "avg_deviation_pts": avg_dev,
                "confidence": _confidence_label(avg_dev),
            })

    if not matches:
        return []

    matches.sort(key=lambda m: m["avg_deviation_pts"])

    if len(ratios) == 1 and len(matches) >= 2:
        names = ", ".join(m["material"] for m in matches)
        elem = next(iter(ratios))
        return [{
            "material": f"ambiguous — could be: {names}",
            "avg_deviation_pts": 0.0,
            "confidence": f"low — single element detected ({elem})",
        }]

    return matches


# ── PNG helpers ───────────────────────────────────────────────────────────────

def _find_all_tiffs(results_dir):
    tile_dir = Path(results_dir).parent
    return sorted(tile_dir.glob("*.tiff")) + sorted(tile_dir.glob("*.tif"))


def _composite_bgr(results_dir):
    """
    Build a grayscale background by normalizing each element TIFF to 0-1
    then summing them and clipping to 255. Co-located elements appear brighter.
    """
    tiff_paths = _find_all_tiffs(results_dir)
    if not tiff_paths:
        return None

    composite = None
    for path in tiff_paths:
        img = tifffile.imread(str(path))
        if img.ndim == 3:
            img = img[0]
        img_f = img.astype(np.float32)
        vmin, vmax = img_f.min(), img_f.max()
        if vmax > vmin:
            img_f = (img_f - vmin) / (vmax - vmin)
        else:
            img_f = np.zeros_like(img_f)
        composite = img_f if composite is None else composite + img_f

    composite = np.clip(composite * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(composite, cv2.COLOR_GRAY2BGR)


def _draw_boxes(bgr, boxes, color=(255, 255, 255), thickness=1):
    h, w = bgr.shape[:2]
    for box in boxes:
        x1, y1 = max(0, box[0]), max(0, box[1])
        x2, y2 = min(w - 1, box[2]), min(h - 1, box[3])
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, thickness)


def _save_pngs(raw_detections, all_clusters, unions_out, results_dir):
    bgr = _composite_bgr(results_dir)
    if bgr is None:
        print("[PHASE] No TIFFs found — skipping PNGs")
        return

    # PNG 1: all raw detection boxes (individual blobs)
    img1 = bgr.copy()
    raw_boxes = [_blob_to_box(b) for b in raw_detections.values()]
    _draw_boxes(img1, raw_boxes)
    png1 = Path(results_dir) / "phase_all_boxes.png"
    cv2.imwrite(str(png1), img1)

    # PNG 2: matched union boxes only
    img2 = bgr.copy()
    matched_boxes = [
        [u["union_box"]["x1"], u["union_box"]["y1"], u["union_box"]["x2"], u["union_box"]["y2"]]
        for u in unions_out if u["matches"]
    ]
    _draw_boxes(img2, matched_boxes)
    png2 = Path(results_dir) / "phase_matches.png"
    cv2.imwrite(str(png2), img2)

    # PNG 3: all union boxes (matched + unmatched) — everything the union step found
    img3 = bgr.copy()
    all_union_boxes = [
        [u["union_box"]["x1"], u["union_box"]["y1"], u["union_box"]["x2"], u["union_box"]["y2"]]
        for u in unions_out
    ]
    unmatched_boxes = [
        [u["union_box"]["x1"], u["union_box"]["y1"], u["union_box"]["x2"], u["union_box"]["y2"]]
        for u in unions_out if not u["matches"]
    ]
    _draw_boxes(img3, matched_boxes, color=(0, 255, 0))    # green = matched
    _draw_boxes(img3, unmatched_boxes, color=(0, 0, 255))  # red = no match
    png3 = Path(results_dir) / "phase_all_unions.png"
    cv2.imwrite(str(png3), img3)

    print(f"[PHASE] PNGs → {png1.name} ({len(raw_boxes)} raw boxes)  |  "
          f"{png2.name} ({len(matched_boxes)} matched)  |  "
          f"{png3.name} ({len(all_union_boxes)} total unions: {len(matched_boxes)} green matched, {len(unmatched_boxes)} red unmatched)")


# ── Main entry point ──────────────────────────────────────────────────────────

def run_phase_matching(raw_detections, results_dir, library_path=None, groups=None, proximity_px=5):
    # proximity_px is arbitrary — needs beamline testing to calibrate against real particle sizes
    library, tol_default = load_materials_library(library_path)
    if not library:
        return

    all_elements = sorted({v["element"] for v in raw_detections.values()})

    # Report image size so threshold-to-image ratio is visible
    _tiff_paths = _find_all_tiffs(results_dir)
    if _tiff_paths:
        _img = tifffile.imread(str(_tiff_paths[0]))
        _h, _w = (_img.shape[-2], _img.shape[-1]) if _img.ndim >= 2 else (_img.shape[0], _img.shape[0])
        print(f"[PHASE] Image size: {_w}×{_h}px | proximity threshold: {proximity_px}px ({proximity_px/_w*100:.1f}% of width) | composite background from {len(_tiff_paths)} TIFFs")
    else:
        print(f"[PHASE] proximity threshold: {proximity_px}px (no TIFFs found for size reference)")

    print(f"[PHASE] {len(raw_detections)} raw blobs across {len(all_elements)} elements — "
          f"clustering by centroid proximity ({proximity_px}px threshold)")

    all_clusters = _build_centroid_union_clusters(raw_detections, proximity_px, id_offset=0)
    n_multi = sum(1 for c in all_clusters if len(c["by_elem"]) > 1)
    n_single = len(all_clusters) - n_multi
    print(f"[PHASE] {len(all_clusters)} clusters found ({n_multi} multi-element, {n_single} single-element)")

    # Step 3: ratio computation + library scoring
    n_matched = n_ambiguous = n_unmatched = 0
    unions_out = []
    all_norm_fallbacks = []  # collect blobs that used norm instead of raw

    for cluster in all_clusters:
        ratios, norm_fallbacks = _compute_ratios(cluster)
        if norm_fallbacks:
            all_norm_fallbacks.extend(norm_fallbacks)
        matches = score_against_library(ratios, library, tol_default)

        if not matches:
            n_unmatched += 1
        elif "ambiguous" in matches[0]["material"]:
            n_ambiguous += 1
        else:
            n_matched += 1

        # Build pixel_blobs for JSON self-containment
        pixel_blobs = []
        for elem, data in cluster["by_elem"].items():
            for b in data["blobs"]:
                pixel_blobs.append({
                    "element": elem,
                    "image_center": b["image_center"],
                    "image_radius": b["image_radius"],
                })

        unions_out.append({
            "union_id": cluster["union_id"],
            "group": cluster["group"],
            "elements_detected": sorted(cluster["by_elem"].keys()),
            "measured_ratios": ratios,
            "union_box": {"x1": cluster["union_box"][0], "y1": cluster["union_box"][1],
                          "x2": cluster["union_box"][2], "y2": cluster["union_box"][3]},
            "pixel_blobs": pixel_blobs,
            "matches": matches,
        })

    # Warn about any blobs that fell back to norm intensity (missing raw — check for errors)
    if all_norm_fallbacks:
        print(f"[PHASE] WARNING: {len(all_norm_fallbacks)} blob(s) used normalized intensity (no raw data) — ratios may be inaccurate. Check these for errors:")
        for elem, name in all_norm_fallbacks:
            print(f"  [{elem}] {name}")
    else:
        print("[PHASE] All blobs used raw intensity for ratio computation.")

    output = {
        "generated_at": datetime.now().isoformat(),
        "library_version": "1.0",
        "total_raw_unions": len(all_clusters),
        "unions": unions_out,
    }

    out_path = Path(results_dir) / "phase_detections.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"[PHASE] Phase calculation complete — {len(all_clusters)} union clusters → "
          f"{n_matched} matched, {n_ambiguous} ambiguous, {n_unmatched} unmatched | "
          f"{n_matched + n_ambiguous} possible phase matches total → {out_path.name}")

    try:
        _save_pngs(raw_detections, all_clusters, unions_out, results_dir)
    except Exception as e:
        print(f"[PHASE] PNG generation failed (non-fatal): {e}")
