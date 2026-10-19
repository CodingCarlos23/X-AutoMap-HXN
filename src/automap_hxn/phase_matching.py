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


# ── Step 2: cross-element union boxes ─────────────────────────────────────────

def _build_union_clusters(element_regions, group_name, id_offset=0):
    """
    Given per-element merged regions, find clusters where element boxes touch
    across different elements (BFS on the touching graph). Each connected
    component becomes one union box.
    Returns list of union cluster dicts.
    """
    # Flatten to a list of (elem, region) pairs
    items = []
    for elem, regions in element_regions.items():
        for region in regions:
            items.append({"elem": elem, "box": region["box"], "blobs": region["blobs"]})

    n = len(items)
    adj = defaultdict(list)
    for i in range(n):
        for j in range(i + 1, n):
            if items[i]["elem"] != items[j]["elem"] and _boxes_touch(items[i]["box"], items[j]["box"]):
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
            elem = items[idx]["elem"]
            by_elem[elem]["boxes"].append(items[idx]["box"])
            by_elem[elem]["blobs"].extend(items[idx]["blobs"])

        all_boxes = [items[idx]["box"] for idx in component]
        union_box = [
            min(b[0] for b in all_boxes),
            min(b[1] for b in all_boxes),
            max(b[2] for b in all_boxes),
            max(b[3] for b in all_boxes),
        ]

        uid += 1
        clusters.append({
            "union_id": f"raw_union_{uid:03d}",
            "group": group_name,
            "union_box": union_box,
            "by_elem": {e: d for e, d in by_elem.items()},
        })

    return clusters


# ── Step 3: intensity ratios from blobs in each cluster ───────────────────────

def _compute_ratios(cluster):
    averages = {}
    for elem, data in cluster["by_elem"].items():
        blobs = data["blobs"]
        averages[elem] = sum(b.get("mean_intensity", 0) for b in blobs) / len(blobs) if blobs else 0.0
    grand = sum(averages.values())
    if grand == 0:
        return {e: 0.0 for e in averages}
    return {e: round(v / grand * 100, 1) for e, v in averages.items()}


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

def _find_first_tiff(results_dir):
    tile_dir = Path(results_dir).parent
    tiffs = sorted(tile_dir.glob("*.tiff")) + sorted(tile_dir.glob("*.tif"))
    return tiffs[0] if tiffs else None


def _tiff_to_bgr(tiff_path):
    img = tifffile.imread(str(tiff_path))
    if img.ndim == 3:
        img = img[0]
    img_f = img.astype(np.float32)
    vmin, vmax = img_f.min(), img_f.max()
    img_u8 = ((img_f - vmin) / (vmax - vmin) * 255).astype(np.uint8) if vmax > vmin else np.zeros_like(img_f, dtype=np.uint8)
    return cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)


def _draw_boxes(bgr, boxes, color=(255, 255, 255), thickness=1):
    h, w = bgr.shape[:2]
    for box in boxes:
        x1, y1 = max(0, box[0]), max(0, box[1])
        x2, y2 = min(w - 1, box[2]), min(h - 1, box[3])
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, thickness)


def _save_pngs(raw_detections, all_clusters, unions_out, results_dir):
    tiff_path = _find_first_tiff(results_dir)
    if tiff_path is None:
        print("[PHASE] No TIFF found — skipping PNGs")
        return

    bgr = _tiff_to_bgr(tiff_path)

    # PNG 1: all raw detection boxes (individual blobs)
    img1 = bgr.copy()
    raw_boxes = [_blob_to_box(b) for b in raw_detections.values()]
    _draw_boxes(img1, raw_boxes)
    png1 = Path(results_dir) / "detected_material_on_raw_master_data_all_boxes.png"
    cv2.imwrite(str(png1), img1)

    # PNG 2: matched union boxes only
    img2 = bgr.copy()
    matched_boxes = [
        [u["union_box"]["x1"], u["union_box"]["y1"], u["union_box"]["x2"], u["union_box"]["y2"]]
        for u in unions_out if u["matches"]
    ]
    _draw_boxes(img2, matched_boxes)
    png2 = Path(results_dir) / "detected_material_on_raw_master_data_matches.png"
    cv2.imwrite(str(png2), img2)

    print(f"[PHASE] PNGs → {png1.name} ({len(raw_boxes)} raw boxes)  |  {png2.name} ({len(matched_boxes)} matched unions)")


# ── Main entry point ──────────────────────────────────────────────────────────

def run_phase_matching(raw_detections, results_dir, library_path=None, groups=None):
    library, tol_default = load_materials_library(library_path)
    if not library:
        return

    # Step 1: per-element merging across ALL elements (no group separation)
    all_elements = sorted({v["element"] for v in raw_detections.values()})
    element_regions = {
        elem: _merge_element_regions(raw_detections, elem)
        for elem in all_elements
    }
    total_regions = sum(len(r) for r in element_regions.values())
    print(f"[PHASE] {len(raw_detections)} raw blobs → {total_regions} merged element regions ({len(all_elements)} elements)")

    # Step 2: one cross-all-elements union pass — any element can union with any other
    all_clusters = _build_union_clusters(element_regions, "all", id_offset=0)
    print(f"[PHASE] {total_regions} merged regions → {len(all_clusters)} union clusters")

    # Step 3: ratio computation + library scoring
    n_matched = n_ambiguous = n_unmatched = 0
    unions_out = []

    for cluster in all_clusters:
        ratios = _compute_ratios(cluster)
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

    output = {
        "generated_at": datetime.now().isoformat(),
        "library_version": "1.0",
        "total_raw_unions": len(all_clusters),
        "unions": unions_out,
    }

    out_path = Path(results_dir) / "detected_material_on_raw_master_data.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"[PHASE] {len(all_clusters)} total union clusters → "
          f"{n_matched} matched, {n_ambiguous} ambiguous, {n_unmatched} unmatched → {out_path.name}")

    try:
        _save_pngs(raw_detections, all_clusters, unions_out, results_dir)
    except Exception as e:
        print(f"[PHASE] PNG generation failed (non-fatal): {e}")
