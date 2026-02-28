import time
import numpy as np
import cv2
from skimage.measure import shannon_entropy
from pathlib import Path

def wait_for_element_tiffs(element_list, watch_dir):
    tiff_paths = {}
    print(watch_dir)
    print("\nWaiting for TIFF files for all elements:", element_list)
    missing_reported = set()
    while True:
        all_found = True
        tiff_paths.clear()
        missing_now = set()
        for element in element_list:
            pattern = f"scan_*_{element}.tiff"
            watch_dir = Path(watch_dir)
            matches = list(watch_dir.glob(pattern))
            if matches:
                tiff_paths[element] = matches[0]
            else:
                all_found = False
                missing_now.add(element)
        # Only print for elements that are newly missing
        for element in missing_now - missing_reported:
            print(f"Waiting for TIFF file for element: {element}")
        missing_reported = missing_now
        if all_found:
            break
        time.sleep(2)
    print("\n✅ Found TIFF files for all elements:")
    for element in element_list:
        print(f"{element}: {tiff_paths[element].name}")
    return tiff_paths

def table_to_individual_scans(df, output_dir="scans"):
    """
    Convert fine scan table (DataFrame) to individual scan JSON files.
    This allows fine scans to be created from a table instead of directly from formatted_unions.
    
    Args:
        df: pandas DataFrame with columns: label, cx, cy, num_x, num_y (minimum required)
        output_dir: directory to save individual JSON files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    required_cols = ['label', 'cx', 'cy', 'num_x', 'num_y']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")
    
    for _, row in df.iterrows():
        label = row['label']
        scan_data = {
            label: {
                "cx": float(row['cx']),
                "cy": float(row['cy']),
                "num_x": float(row['num_x']),
                "num_y": float(row['num_y'])
            }
        }
        
        file_path = output_dir / f"{label}.json"
        with open(file_path, "w") as f:
            json.dump(make_json_serializable(scan_data), f, indent=4)
    
    print(f"✅ Created {len(df)} individual scan JSON files in {output_dir}")

def is_featureless(img):
    img = np.nan_to_num(img)
    ent = shannon_entropy(img)
    pnr = (img.max() - img.mean()) / (img.std() + 1e-5)
    edge_map = cv2.Canny(img.astype(np.uint8), 50, 150)
    edge_ratio = np.count_nonzero(edge_map) / img.size

    return (ent < 2.5) and (pnr < 2.5) and (edge_ratio < 0.01)

def make_json_serializable(obj):
    """
    Recursively convert numpy types and other non-JSON-serializable objects to JSON-safe types.
    Handles numpy integers (uint8, int32, int64, etc.), floats, arrays, and nested structures.
    """
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(i) for i in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()

    # Catch all numpy scalar types (uint8, int32, int64, float32, float64, etc.)
    elif isinstance(obj, (np.integer, np.uint8, np.int8, np.int16, np.int32, np.int64, 
                         np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        # Fallback: try to convert to string for unknown types
        return str(obj)

def formatted_unions_to_table(formatted_unions, save_to=None):
    """
    Convert formatted_unions dict to a pandas DataFrame with fine scan parameters.
    
    Args:
        formatted_unions: dict with keys like "Box #1", values with cx, cy, num_x, num_y
        save_to: optional path to save as CSV (e.g., "fine_scans.csv")
    
    Returns:
        pandas DataFrame with columns: label, cx, cy, num_x, num_y (only what's needed for fine scans)
    """
    if not formatted_unions:
        print("[TABLE] Warning: formatted_unions is empty, creating empty DataFrame")
        return pd.DataFrame(columns=['label', 'cx', 'cy', 'num_x', 'num_y'])
    
    rows = []
    for label, info in formatted_unions.items():
        # Validate required keys
        if not all(key in info for key in ['cx', 'cy', 'num_x', 'num_y']):
            missing = [key for key in ['cx', 'cy', 'num_x', 'num_y'] if key not in info]
            print(f"[TABLE WARNING] Box '{label}' missing keys: {missing}, skipping or using defaults")
        
        # Only keep essential fine scan parameters
        row = {
            'label': label,
            'cx': info.get('cx', 0),
            'cy': info.get('cy', 0),
            'num_x': info.get('num_x', 0),
            'num_y': info.get('num_y', 0),
        }
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Ensure numeric columns
    for col in ['cx', 'cy', 'num_x', 'num_y']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    if save_to:
        os.makedirs(os.path.dirname(save_to) if os.path.dirname(save_to) else '.', exist_ok=True)
        df.to_csv(save_to, index=False)
        print(f"✅ Fine scan table saved to: {save_to}")
    
    print(f"[TABLE] Created table with {len(df)} rows: {list(df.columns)}")
    return df

def resize_if_needed(img, name, target_shape):
        if img.shape != target_shape:
            # print(f"Resizing {name} from {img.shape} → {target_shape}")
            return cv2.resize(img, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_AREA)
        return img

def normalize_and_dilate(img, kernel_size=None, iterations=None):
    img = np.nan_to_num(img)

    if is_featureless(img):
        print("[normalize_and_dilate] Skipped — no signal detected (entropy+pnr+edges)")
        return np.zeros_like(img, dtype=np.uint8), np.zeros_like(img, dtype=np.uint8)
    
    norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Use defaults if parameters not provided (backwards compatibility)
    if kernel_size is None:
        kernel_size = (3, 3)
    if iterations is None:
        iterations = 2
    
    kernel = np.ones(kernel_size, np.uint8) if isinstance(kernel_size, tuple) else np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(norm, kernel, iterations=iterations)
    return norm, dilated

def box_area(top_left, bottom_right):
    w = bottom_right[0] - top_left[0]
    h = bottom_right[1] - top_left[1]
    return max(0, w) * max(0, h)

def intersection_area(box1, box2):
    x1 = max(box1["real_top_left_um"][0], box2["real_top_left_um"][0])
    y1 = max(box1["real_top_left_um"][1], box2["real_top_left_um"][1])
    x2 = min(box1["real_bottom_right_um"][0], box2["real_bottom_right_um"][0])
    y2 = min(box1["real_bottom_right_um"][1], box2["real_bottom_right_um"][1])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)

def boxes_overlap(box1, box2, overlap_thresh=0.5):
    inter_area = intersection_area(box1, box2)
    if inter_area <= 0:
        return False
    area1 = box_area(box1["real_top_left_um"], box1["real_bottom_right_um"])
    area2 = box_area(box2["real_top_left_um"], box2["real_bottom_right_um"])
    smaller_area = min(area1, area2)
    return (inter_area / smaller_area) >= overlap_thresh

def compute_px_per_um(box):
    """Derive px/um from any one box that contains both image_length and real_size_um."""
    if "image_length" in box and "real_size_um" in box:
        # image_length is pixel side; real_size_um is [w_um, h_um]
        real_len_um = max(float(box["real_size_um"][0]), float(box["real_size_um"][1]))
        img_len_px  = float(box["image_length"])
        if real_len_um > 0:
            return img_len_px / real_len_um
    return None

def add_compatibility_keys(box):
    """Ensure 'center', 'length', 'area' keys exist (pixel-based) and duplicate real_area key."""
    # area key duplication for safety with both encodings
    if "real_area_um²" in box and "real_area_um\u00b2" not in box:
        box["real_area_um\u00b2"] = box["real_area_um²"]
    if "real_area_um\u00b2" in box and "real_area_um²" not in box:
        box["real_area_um²"] = box["real_area_um\u00b2"]

    # provide px/um if computable from the box itself
    px_per_um = compute_px_per_um(box)
    if px_per_um is not None:
        box["px_per_um"] = px_per_um  # optional, can be handy later

    # center (pixels)
    if "center" not in box:
        if "image_center" in box:
            box["center"] = box["image_center"]
        elif px_per_um is not None and "real_center_um" in box:
            rc = box["real_center_um"]
            box["center"] = [int(round(rc[0] * px_per_um)), int(round(rc[1] * px_per_um))]

    # length (pixels)
    if "length" not in box:
        if "image_length" in box:
            box["length"] = box["image_length"]
        elif px_per_um is not None and "real_size_um" in box:
            sx_um, sy_um = box["real_size_um"]
            box["length"] = int(round(max(sx_um, sy_um) * px_per_um))

    # area (pixels^2)
    if "area" not in box:
        if "image_area_px²" in box:
            box["area"] = box["image_area_px²"]
        elif "length" in box:
            L = int(round(box["length"]))
            box["area"] = int(L * L)

    return box

# ---------- merging ----------
def merge_boxes_strict(box1, box2, new_label):
    """Merge two boxes -> union in real units, then recalc image fields via px_per_um if available."""
    # union in real coordinates
    x1 = min(box1["real_top_left_um"][0],  box2["real_top_left_um"][0])
    y1 = min(box1["real_top_left_um"][1],  box2["real_top_left_um"][1])
    x2 = max(box1["real_bottom_right_um"][0], box2["real_bottom_right_um"][0])
    y2 = max(box1["real_bottom_right_um"][1], box2["real_bottom_right_um"][1])

    size_x_um = x2 - x1
    size_y_um = y2 - y1
    center_um = [(x1 + x2) / 2, (y1 + y2) / 2]

    merged = {
        "text": new_label,
        "real_top_left_um": [x1, y1],
        "real_bottom_right_um": [x2, y2],
        "real_center_um": center_um,
        "real_size_um": [size_x_um, size_y_um],
        "real_area_um²": size_x_um * size_y_um,
        "merged_from": [box1.get("text", ""), box2.get("text", "")]
    }
    # duplicate area key with \u00b2 for robustness
    merged["real_area_um\u00b2"] = merged["real_area_um²"]

    # Try to get a px/um from either input
    px_per_um = compute_px_per_um(box1) or compute_px_per_um(box2)

    if px_per_um is not None:
        size_x_px = int(round(size_x_um * px_per_um))
        size_y_px = int(round(size_y_um * px_per_um))
        center_px = [int(round(center_um[0] * px_per_um)),
                     int(round(center_um[1] * px_per_um))]
        merged["image_center"]   = center_px
        merged["image_length"]   = int(max(size_x_px, size_y_px))
        merged["image_area_px²"] = int(size_x_px * size_y_px)
        merged["px_per_um"]      = float(px_per_um)

    # add shorthand compatibility keys
    return add_compatibility_keys(merged)

def merge_overlapping_boxes_dict(data: dict, overlap_thresh=0.5) -> dict:
    """
    Repeatedly merge overlapping boxes; recalc real+image geometry;
    add compatibility keys ('center','length','area').
    """
    boxes = list(data.values())
    merged_any = True
    counter = 1

    while merged_any:
        merged_any = False
        new_boxes = []
        used = set()

        for i in range(len(boxes)):
            if i in used:
                continue
            current = boxes[i]
            for j in range(i + 1, len(boxes)):
                if j in used:
                    continue
                if boxes_overlap(current, boxes[j], overlap_thresh):
                    current = merge_boxes_strict(current, boxes[j], f"Merged Box #{counter}")
                    used.add(j)
                    merged_any = True
            used.add(i)
            new_boxes.append(current)
            counter += 1
        boxes = new_boxes

    # Ensure non-merged boxes also have compat keys
    boxes = [add_compatibility_keys(b) for b in boxes]

    return {f"Final Box #{i+1}": b for i, b in enumerate(boxes)}
