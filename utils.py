from curses import meta
import os
import sys
import re
import time
import copy
import json
import pickle
import threading
import multiprocessing
import traceback
from collections import Counter
from pathlib import Path
import traceback as trackback
import inspect
from skimage.measure import shannon_entropy
from scipy import ndimage
from skimage.segmentation import watershed  
from skimage.feature import peak_local_max
import warnings

# Cellpose imports (optional - will gracefully handle if not installed)
try:
    from cellpose import models
    from PIL import Image
    CELLPOSE_AVAILABLE = True
except ImportError:
    CELLPOSE_AVAILABLE = False
    models = None
    Image = None
import tqdm

import cv2
import numpy as np
import tifffile as tiff
import time
import pandas as pd

from hxntools.CompositeBroker import db
from bluesky_queueserver_api import BPlan
from bluesky_queueserver_api.zmq import REManagerAPI
RM = REManagerAPI()
from tiled.client import from_uri
c = from_uri('https://tiled.nsls2.bnl.gov')
container = c["tst/sandbox/synaps/reconstructions"]

# Suppress DataFrame fragmentation warnings from databroker
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning, message='.*DataFrame is highly fragmented.*')

from PyQt5.QtWidgets import (
    QApplication, QLabel, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QLineEdit, QCheckBox, QSlider, QFileDialog, QListWidget, QListWidgetItem,
    QFrame, QMessageBox, QDoubleSpinBox, QProgressBar, QScrollArea, QSizePolicy,
    QGraphicsEllipseItem
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen
from PyQt5.QtCore import Qt, QRect, QTimer

from remote_segmentation import RemoteSegmentationSender, RemoteSegmentationReceiver
# Create a global instance of the remote sender
remote_sender = RemoteSegmentationSender() 


def save_each_blob_as_individual_scan(json_safe_data, output_dir="scans"):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    for idx, info in json_safe_data.items():
        # Handle both old format (real_center_um, real_size_um) and new format (cx, cy, num_x, num_y)
        if "real_center_um" in info and "real_size_um" in info:
            cx, cy = info["real_center_um"]
            sx, sy = info["real_size_um"]
        elif "cx" in info and "cy" in info and "num_x" in info and "num_y" in info:
            cx, cy = info["cx"], info["cy"]
            sx, sy = info["num_x"], info["num_y"]
        else:
            print(f"⚠️ Skipping {idx}: missing required keys (cx/cy or real_center_um)")
            continue

        scan_data = {
            idx: {  # Use the union box title as the key
                "cx": float(cx),  # Ensure float conversion for JSON serialization
                "cy": float(cy),
                "num_x": float(sx),
                "num_y": float(sy)
            }
        }

        file_path = output_dir / f"{idx}.json"
        with open(file_path, "w") as f:
            json.dump(make_json_serializable(scan_data), f, indent=4)


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


def load_fine_scans_table(csv_path):
    """
    Load a fine scans table from CSV file (for use with remote servers).
    
    Args:
        csv_path: path to CSV file with fine scan parameters
    
    Returns:
        pandas DataFrame with fine scan parameters
    """
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded fine scans table: {len(df)} scans")
    print(f"   Columns: {list(df.columns)}")
    
    return df

def headless_send_queue_coarse_scan(params_path, remote_seg=True):
    """
    Performs coarse scan using parameters from a single JSON config file.
    
    Args:
        params_path: Path to JSON config file containing:
                     - all beamline parameters (det_name, mot1, mot2, mot1_s, mot1_e, mot2_s, mot2_e, etc.)
                     - scan_id: Scan ID (optional, default: null)
                     - proceed_with_fine_scan: Whether to proceed with fine scans after coarse (optional, default: false)
        remote_seg: Whether to use remote segmentation (default: True)
    
    Example:
        headless_send_queue_coarse_scan('initial_scan_sim.json', remote_seg=True)
    """ 
    
    with open(params_path, 'r') as f:
        params = json.load(f)

    # Read optional parameters from JSON with nested access
    scan_id = params.get("scan_params", {}).get("scan_id")
    proceed_with_fine_scan = params.get("execution_params", {}).get("proceed_with_fine_scan", False)

    dets = params.get("scan_params", {}).get("det_name", "dets_fast")
    x_motor = params.get("scan_params", {}).get("mot1", "zpssx")
    y_motor = params.get("scan_params", {}).get("mot2", "zpssy")

    x_start = params.get("scan_params", {}).get("mot1_s", 0)
    x_end = params.get("scan_params", {}).get("mot1_e", 0)
    y_start = params.get("scan_params", {}).get("mot2_s", 0)
    y_end = params.get("scan_params", {}).get("mot2_e", 0)

    # step_size_coarse might not exist in new format, try nested access first, then fallback
    # Also try 'step_size' in scan_params as fallback
    step_size = (
        params.get("scan_params", {}).get("step_size_coarse") or 
        params.get("scan_params", {}).get("step_size") or 
        params.get("step_size_coarse", 0.25)
    )
    mot1_n = int(abs(x_end-x_start)/step_size)
    mot2_n = int(abs(y_end-y_start)/step_size)
    
    # Validate step counts
    if mot1_n == 0 or mot2_n == 0:
        raise ValueError(
            f"Coarse scan has zero steps! "
            f"mot1: {x_start} to {x_end} (n={mot1_n}), "
            f"mot2: {y_start} to {y_end} (n={mot2_n}), "
            f"step_size={step_size:.3f}. "
            f"Check scan_params in JSON config."
        )
    
    # exp_t_coarse might not exist in new format, try nested access first, then fallback
    exp_time = params.get("scan_params", {}).get("exp_t_coarse") or params.get("scan_params", {}).get("exp_t") or params.get("exp_t_coarse", 0.01)

    # Calculate center as midpoint
    cx = (x_start + x_end) / 2
    cy = (y_start + y_end) / 2
    
    print(f"[COARSE_SCAN] Range: [{x_start:.2f} to {x_end:.2f}] x [{y_start:.2f} to {y_end:.2f}]")
    print(f"[COARSE_SCAN] Step size: {step_size:.3f} μm, Points: {mot1_n} x {mot2_n}")
    print(f"[COARSE_SCAN] Center: ({cx:.2f}, {cy:.2f}), Exp time: {exp_time}s")
    
    roi = {x_motor: cx, y_motor: cy}

    RM.item_add(BPlan("piezos_to_zero"))
    
    # Pass the same config file to load_and_queue
    load_and_queue(params_path, 
                   target_id=scan_id, 
                   remote_seg=remote_seg, 
                   proceed_fine_scans=proceed_with_fine_scan)

def headless_send_queue_fine_scan(json_path, fine_scans_table=None):
    """
    Performs fine scans from a fine_scans_table (DataFrame or CSV path).
    Reads all configuration from a single JSON config file with nested structure.
    
    Args:
        json_path: Path to JSON config file containing:
                   - execution_params (mode, etc.)
                   - scan_params (mot1, mot2, exp_t, step_size_fine, etc.)
                   - fine_scans_table_path (optional, path to CSV with fine scan parameters)
        fine_scans_table: Optional pandas DataFrame or CSV path with fine scan parameters
                         Columns required: label, cx, cy, num_x, num_y
                         If not provided, tries to load from JSON config
    
    Example:
        headless_send_queue_fine_scan('initial_scan_sim.json', fine_scans_table='fine_scans_table_RGB.csv')
    """
    
    # Load JSON config
    with open(json_path, 'r') as f:
        params = json.load(f)
    
    # Extract parameters from nested structure
    execution_params = params.get('execution_params', {})
    scan_params = params.get('scan_params', {})
    fine_scan_params = params.get('fine_scan_params', {})
    
    # Get mode
    mode = str(execution_params.get('mode', 'simulation')).lower()
    is_real = (mode == 'real')
    is_offline = (mode == 'offline')
    is_sim = (mode == 'simulation')
    
    # Extract beamline parameters from scan_params
    dets = scan_params.get('dets', 'dets_fast')
    # Get detector names list from config, with fallback to default
    det_names = scan_params.get('det_names', ['fs', 'eiger2', 'xspress3'])
    
    x_motor = scan_params.get('mot1', 'zpssx')
    y_motor = scan_params.get('mot2', 'zpssy')
    exp_t = fine_scan_params.get('exp_t_fine', scan_params.get('exp_t', 0.01))
    step_size = fine_scan_params.get('step_size_fine', 0.1)
    fine_scan_pad_ratio = fine_scan_params.get('fine_scan_pad_ratio', 0.25)
    
    # Additional parameters for fly2d_qserver_scan_export
    zp_move_flag = scan_params.get('zp_move_flag', 0)
    smar_move_flag = scan_params.get('smar_move_flag', 0)
    ic1_count = scan_params.get('ic1_count', 55000)
    
    # Export parameters
    export_params = params.get('export_params', {})
    elem_list = export_params.get('elem_list', [])
    # Flatten nested list if needed
    if elem_list and isinstance(elem_list[0], list):
        elem_list = list(set(elem for sublist in elem_list for elem in sublist))
    export_norm = export_params.get('export_norm', 'sclr1_ch4')
    data_wd = export_params.get('data_wd', '/data/users/current_user')
    
    # Determine which table to use
    if fine_scans_table is None:
        # Try to load from JSON config
        table_path = params.get('fine_scans_table_path')
        if table_path:
            print(f"[FINE_SCANS] Loading table from JSON config: {table_path}")
            fine_scans_table = load_fine_scans_table(table_path)
        else:
            print(f"[FINE_SCANS] No fine_scans_table provided and no fine_scans_table_path in JSON")
            return
    elif isinstance(fine_scans_table, str):
        # Load from CSV path
        print(f"[FINE_SCANS] Loading table from CSV: {fine_scans_table}")
        fine_scans_table = load_fine_scans_table(fine_scans_table)
    
    # Process each fine scan from the table
    print(f"\n[FINE_SCANS] Processing {len(fine_scans_table)} scans from table (Mode: {mode.upper()})")
    
    for idx, row in fine_scans_table.iterrows():
        time.sleep(0.5)
        label = row['label']
        cx = row['cx']
        cy = row['cy']
        sx = row['num_x']
        sy = row['num_y']
        
        # Expand scan size by padding ratio
        sx_padded = sx * (1 + fine_scan_pad_ratio)
        sy_padded = sy * (1 + fine_scan_pad_ratio)

        # Define relative scan range around center
        x_start = -sx_padded / 2
        x_end = sx_padded / 2
        y_start = -sy_padded / 2
        y_end = sy_padded / 2

        # Step counts based on padded size
        num_steps_x = int(sx_padded / step_size)
        num_steps_y = int(sy_padded / step_size)
        
        # Validate step counts
        if num_steps_x == 0 or num_steps_y == 0:
            print(f"⚠️ WARNING: {label} has zero steps! sx_padded={sx_padded:.3f}, sy_padded={sy_padded:.3f}, step_size={step_size:.3f}")
            print(f"⚠️ This likely indicates a unit mismatch or incorrect step_size_fine value.")
            print(f"⚠️ Skipping this scan to avoid errors.")
            continue

        # ROI centered on original center
        roi = {x_motor: cx, y_motor: cy}
        roi_json = json.dumps(roi)

        if is_real:
            print(f"[FINE_SCANS] Queuing: {label} (cx={cx:.2f}, cy={cy:.2f}, sx={sx:.2f}, sy={sy:.2f})")
            print(f"[FINE_SCANS]   → Padded size: {sx_padded:.2f} x {sy_padded:.2f} μm, step: {step_size:.3f} μm")
            print(f"[FINE_SCANS]   → Points: {num_steps_x} x {num_steps_y}, range: [{x_start:.2f} to {x_end:.2f}] x [{y_start:.2f} to {y_end:.2f}]")
            RM.item_add(BPlan(
                "fly2d_qserver_scan_export",
                label,
                det_names,  # Use detector names list, not string
                x_motor,
                x_start,
                x_end,
                num_steps_x,
                y_motor,
                y_start,
                y_end,
                num_steps_y,
                exp_t,
                roi_json,
                "",  # scan_id (empty for fine scans)
                zp_move_flag,
                smar_move_flag,
                ic1_count,
                json.dumps(elem_list),
                export_norm,
                data_wd
            ))
        else:
            print(f"[{mode.upper()}] Would queue: {label} (cx={cx:.2f}, cy={cy:.2f})")
    
    print(f"[FINE_SCANS] ✅ All {len(fine_scans_table)} fine scans {'queued' if is_real else 'prepared'}")



def create_rgb_tiff(tiff_paths, output_dir, element_list, group_name=None):
    """
    Merges the first three element TIFFs into a single RGB TIFF file,
    and draws the union boxes on it.
    """
    if len(element_list) < 3:
        print("⚠️ Not enough elements to create an RGB TIFF (need at least 3).")
        return

    rgb_elements = element_list[:3]
    print(f"Creating RGB TIFF from elements (R, G, B): {rgb_elements[0]}, {rgb_elements[1]}, {rgb_elements[2]}")

    try:
        # Read the three images
        img_r = tiff.imread(tiff_paths[rgb_elements[0]])
        img_g = tiff.imread(tiff_paths[rgb_elements[1]])
        img_b = tiff.imread(tiff_paths[rgb_elements[2]])

        # Determine target shape and resize if needed
        shapes = [img.shape for img in (img_r, img_g, img_b)]
        target_shape = Counter(shapes).most_common(1)[0][0]

        img_r = resize_if_needed(img_r, rgb_elements[0], target_shape)
        img_g = resize_if_needed(img_g, rgb_elements[1], target_shape)
        img_b = resize_if_needed(img_b, rgb_elements[2], target_shape)

        # Normalize each channel to 0-255
        norm_r = cv2.normalize(np.nan_to_num(img_r), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        norm_g = cv2.normalize(np.nan_to_num(img_g), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        norm_b = cv2.normalize(np.nan_to_num(img_b), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Merge channels
        merged_rgb = cv2.merge([norm_r, norm_g, norm_b])

        # Draw union boxes
        unions_json_filename = "unions_output.json"
        if group_name:
            unions_json_filename = f"unions_output_{group_name}.json"
        unions_json_path = Path(output_dir) / unions_json_filename
        
        if unions_json_path.exists():
            merged_unions_path = process_and_save_json(unions_json_path)
            if merged_unions_path and Path(merged_unions_path).exists():
                print(f"Drawing union boxes from {merged_unions_path}...")
                with open(merged_unions_path, "r") as f:
                    unions_data = json.load(f)
                
                for union_info in unions_data.values():
                    center = union_info.get("image_center")
                    length = union_info.get("image_length")

                    if center and length:
                        x, y = center[0], center[1]
                        half_len = length / 2
                        top_left = (int(x - half_len), int(y - half_len))
                        bottom_right = (int(x + half_len), int(y + half_len))
                        cv2.rectangle(merged_rgb, top_left, bottom_right, (255, 255, 255), 1) # White box, thickness 1
            else:
                print(f"⚠️ Could not find merged unions file from {unions_json_path} to draw boxes.")
        else:
            print(f"⚠️ Could not find {unions_json_path} to draw boxes.")

        # Save the final image
        output_filename = "Union of elements.tiff"
        if group_name:
            output_filename = f"Union of elements {group_name}.tiff"
        output_path = Path(output_dir) / output_filename
        tiff.imwrite(output_path, merged_rgb)
        print(f"✅ Saved merged RGB image with boxes to: {output_path}")

    except KeyError as e:
        print(f"❌ Could not create RGB TIFF. Missing element TIFF: {e}")
    except Exception as e:
        print(f"❌ An error occurred during RGB TIFF creation: {e}")
        trackback.print_exc()


def create_all_elements_tiff(tiff_paths, output_dir, element_list, precomputed_blobs, group_name=None):
    """
    Creates a TIFF image with individual blob boxes for each element, named All_of_elements.tiff.
    The base image is an RGB composite of the first up to 3 elements.
    """
    import traceback
    from pathlib import Path
    import tifffile as tiff
    import numpy as np
    import cv2

    try:
        # --- Create a base RGB image ---
        if not element_list or not tiff_paths:
            print("⚠️ Not enough elements or TIFF paths to create an image.")
            return

        # Determine a consistent shape from the first element's tiff
        first_element = element_list[0]
        first_path = tiff_paths.get(first_element)
        if not first_path:
            print(f"⚠️ Cannot find TIFF for base element {first_element}.")
            return
        
        base_img = tiff.imread(first_path)
        target_shape = base_img.shape

        # Prepare channels based on number of elements
        if len(element_list) >= 3:
            elements_to_use = element_list[:3]
            print(f"Creating RGB base from elements (R, G, B): {', '.join(elements_to_use)}")
            img_r = tiff.imread(tiff_paths[elements_to_use[0]])
            img_g = tiff.imread(tiff_paths[elements_to_use[1]])
            img_b = tiff.imread(tiff_paths[elements_to_use[2]])
        elif len(element_list) == 2:
            elements_to_use = element_list[:2]
            print(f"Creating RG base from elements (R, G): {', '.join(elements_to_use)}")
            img_r = tiff.imread(tiff_paths[elements_to_use[0]])
            img_g = tiff.imread(tiff_paths[elements_to_use[1]])
            img_b = np.zeros(target_shape, dtype=base_img.dtype)
        else: # 1 element
            element_to_use = element_list[0]
            print(f"Creating grayscale base from element: {element_to_use}")
            img_r = tiff.imread(tiff_paths[element_to_use])
            img_g = img_r
            img_b = img_r

        # Resize all to target shape
        img_r = resize_if_needed(img_r, 'R channel', target_shape)
        img_g = resize_if_needed(img_g, 'G channel', target_shape)
        img_b = resize_if_needed(img_b, 'B channel', target_shape)

        # Normalize and merge (BGR for OpenCV drawing)
        norm_r = cv2.normalize(np.nan_to_num(img_r), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        norm_g = cv2.normalize(np.nan_to_num(img_g), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        norm_b = cv2.normalize(np.nan_to_num(img_b), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        merged_bgr = cv2.merge([norm_b, norm_g, norm_r])

        # --- Draw individual blob boxes ---
        color_map = {
            'red':    (0, 0, 255),   # Red
            'green':  (0, 255, 0),   # Green
            'blue':   (255, 0, 0),   # Blue
            'orange': (0, 165, 255),
            'purple': (128, 0, 128),
            'cyan':   (255, 255, 0),
            'olive':  (0, 128, 128),
            'yellow': (0, 255, 255),
            'brown':  (42, 42, 165),
            'pink':   (203, 192, 255)
        }

        print("Drawing individual element boxes...")
        for color_name, blob_data in precomputed_blobs.items():
            if color_name not in color_map:
                continue
            
            box_color = color_map[color_name]
            
            for (thresh, area), blobs in blob_data.items():
                for blob in blobs:
                    x = blob.get('box_x')
                    y = blob.get('box_y')
                    size = blob.get('box_size')

                    if x is not None and y is not None and size is not None:
                        top_left = (int(x), int(y))
                        bottom_right = (int(x + size), int(y + size))
                        cv2.rectangle(merged_bgr, top_left, bottom_right, box_color, 2)

        # --- Save the final image ---
        merged_rgb_for_save = cv2.cvtColor(merged_bgr, cv2.COLOR_BGR2RGB)
        output_filename = "All_of_elements.tiff"
        if group_name:
            output_filename = f"All_of_elements {group_name}.tiff"
        output_path = Path(output_dir) / output_filename
        tiff.imwrite(str(output_path), merged_rgb_for_save)
        print(f"✅ Saved image with individual boxes to: {output_path}")

    except KeyError as e:
        print(f"❌ Could not create image. Missing element TIFF: {e}")
    except Exception as e:
        print(f"❌ An error occurred during image creation: {e}")
        traceback.print_exc()

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

#merge boxes option 

# ---------- helpers ----------
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


# ---------------- File wrapper ----------------
def process_and_save_json(input_path, overlap_thresh=0.5):
    """Load JSON file, merge overlapping boxes, save as *_merged.json."""
    with open(input_path, "r") as f:
        data = json.load(f)

    merged = merge_overlapping_boxes_dict(data, overlap_thresh=overlap_thresh)

    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_merged.json"

    with open(output_path, "w") as f:
        json.dump(make_json_serializable(merged), f, indent=2)

    print(f"✅ Merged JSON saved to: {output_path}")
    return output_path

def _detect_blobs_simple(img_norm, img_orig, min_thresh, min_area, **kwargs):
    """Simple blob detector method (OpenCV SimpleBlobDetector)"""
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = min_thresh
    params.maxThreshold = kwargs.get('max_threshold', 255)
    params.filterByArea = True
    params.minArea = min_area
    params.maxArea = kwargs.get('max_area', 1600)
    params.thresholdStep = kwargs.get('threshold_step', 2)

    params.filterByColor = kwargs.get('filter_by_color', False)
    params.filterByCircularity = kwargs.get('filter_by_circularity', False)
    params.filterByInertia = kwargs.get('filter_by_inertia', False)
    params.filterByConvexity = kwargs.get('filter_by_convexity', False)
    params.minRepeatability = kwargs.get('min_repeatability', 1)
    
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img_norm)
    
    detections = []
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        radius = int(kp.size / 2)
        detections.append({'center': (x, y), 'radius': radius})
    
    return detections

def _detect_blobs_contours(img_norm, img_orig, min_thresh, min_area, **kwargs):
    """Contour-based blob detection"""
    # Apply threshold
    _, binary = cv2.threshold(img_norm, min_thresh, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            # Get bounding circle
            (x, y), radius = cv2.minEnclosingCircle(contour)
            detections.append({'center': (int(x), int(y)), 'radius': int(radius)})
    
    return detections

def _detect_blobs_hough_circles(img_norm, img_orig, min_thresh, min_area, **kwargs):
    """Hough circle detection for circular blobs"""
    # Convert min_area to min_radius (assuming circular blobs)
    min_radius = int(np.sqrt(min_area / np.pi))
    max_radius = kwargs.get('max_radius', 40)
    
    circles = cv2.HoughCircles(
        img_norm,
        cv2.HOUGH_GRADIENT,
        dp=kwargs.get('dp', 1),
        minDist=kwargs.get('min_dist', min_radius * 2),
        param1=kwargs.get('param1', 50),
        param2=kwargs.get('param2', 30),
        minRadius=min_radius,
        maxRadius=max_radius
    )
    
    detections = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            detections.append({'center': (x, y), 'radius': r})
    
    return detections

def _detect_blobs_connected_components(img_norm, img_orig, min_thresh, min_area, **kwargs):
    """Connected components labeling for blob detection"""
    # Apply threshold
    _, binary = cv2.threshold(img_norm, min_thresh, 255, cv2.THRESH_BINARY)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    detections = []
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            x, y = int(centroids[i][0]), int(centroids[i][1])
            # Estimate radius from area
            radius = int(np.sqrt(area / np.pi))
            detections.append({'center': (x, y), 'radius': radius})
    
    return detections

def _masks_to_boxes_and_areas(masks):
    """
    Convert Cellpose masks to bounding boxes and areas.
    
    Returns:
        boxes: list of (x1, y1, x2, y2)
        areas: list of mask pixel areas (same order as boxes)
    """
    boxes, areas = [], []
    ids = np.unique(masks)
    ids = ids[ids != 0]  # Skip background
    
    for i in ids:
        ys, xs = np.where(masks == i)
        if xs.size == 0:
            continue
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        boxes.append((x1, y1, x2, y2))
        areas.append(int(xs.size))
        
    return boxes, areas


def _area_to_equiv_diameter(area_px):
    """Convert area to equivalent circle diameter: A = π (d/2)^2  -> d = 2*sqrt(A/π)"""
    return 2.0 * np.sqrt(area_px / np.pi)


def _detect_blobs_cellpose(img_norm, img_orig, min_thresh, min_area, **kwargs):
    """Cellpose-based blob detection for cell/particle segmentation"""
    if not CELLPOSE_AVAILABLE:
        raise ImportError("Cellpose not available. Install with: pip install cellpose")
    
    # Use img_orig (normalized but NOT dilated) because Cellpose is a deep learning model
    # trained on raw images. Morphological dilation can destroy fine details.
    # img_norm = dilated image (used for simple/contour methods)
    # img_orig = normalized but not dilated (better for deep learning models)
    cellpose_input = img_orig
    
    # Convert to format expected by Cellpose
    if len(cellpose_input.shape) == 2:
        # Convert grayscale to RGB format for Cellpose
        img_rgb = np.stack([cellpose_input, cellpose_input, cellpose_input], axis=2)
    else:
        img_rgb = cellpose_input.copy()
    
    # Normalize to [0,1] range
    img_min, img_max = float(img_rgb.min()), float(img_rgb.max())
    if img_max > img_min:
        img_rgb = (img_rgb - img_min) / (img_max - img_min)
    else:
        # Handle constant image
        return []
    
    # Cellpose parameters
    diameter_guess = kwargs.get('diameter', 60)
    model_type = kwargs.get('model_type', 'cyto3')
    gpu = kwargs.get('gpu', False)
    flow_threshold = kwargs.get('flow_threshold', 0.4)
    cellprob_threshold = kwargs.get('cellprob_threshold', 0.0)
    channels = kwargs.get('channels', [0, 0])  # [cytoplasm, nucleus] channels
    
    # Initialize model
    model = models.CellposeModel(pretrained_model=model_type, gpu=gpu)
    
    # Run detection
    try:
        res = model.eval(
            img_rgb,
            channels=channels,
            diameter=diameter_guess,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold
        )
        
        # Handle different return formats
        if len(res) == 4:
            masks, flows, styles, diams = res
        else:
            masks, flows, styles = res
            
    except Exception as e:
        print(f"Cellpose detection failed: {e}")
        return []
    
    # Convert masks to boxes and areas
    boxes, areas = _masks_to_boxes_and_areas(masks)
    
    # Filter by diameter range if specified
    min_diameter = kwargs.get('min_diameter', 0)
    max_diameter = kwargs.get('max_diameter', float('inf'))
    
    detections = []
    for box, area in zip(boxes, areas):
        # Check area threshold
        if area < min_area:
            continue
            
        # Check diameter threshold
        equiv_diameter = _area_to_equiv_diameter(area)
        if not (min_diameter <= equiv_diameter <= max_diameter):
            continue
        
        # Calculate center and radius from bounding box
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Use equivalent radius from area for consistency
        radius = equiv_diameter / 2
        
        detections.append({
            'center': (int(center_x), int(center_y)),
            'radius': int(radius),
            'area': area,
            'equiv_diameter': equiv_diameter,
            'bbox': box
        })
    
    return detections


def _detect_blobs_watershed(img_norm, img_orig, min_thresh, min_area, **kwargs):
    """Watershed segmentation for blob detection"""
    from scipy import ndimage
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max
    
    # Apply threshold
    _, binary = cv2.threshold(img_norm, min_thresh, 255, cv2.THRESH_BINARY)
    
    # Distance transform
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    
    # Find local maxima as markers
    local_max_coords = peak_local_max(
        dist_transform, 
        min_distance=kwargs.get('min_distance', 10),
        threshold_abs=kwargs.get('threshold_abs', 0.3 * dist_transform.max())
    )
    
    # Create markers
    markers = np.zeros_like(binary, dtype=np.int32)
    for i, (y, x) in enumerate(local_max_coords):
        markers[y, x] = i + 1
    
    # Apply watershed
    labels = watershed(-dist_transform, markers, mask=binary)
    
    detections = []
    for label_id in np.unique(labels):
        if label_id == 0:  # Skip background
            continue
        
        mask = labels == label_id
        area = np.sum(mask)
        
        if area >= min_area:
            # Calculate centroid
            y_coords, x_coords = np.where(mask)
            x = int(np.mean(x_coords))
            y = int(np.mean(y_coords))
            radius = int(np.sqrt(area / np.pi))
            detections.append({'center': (x, y), 'radius': radius})
    
    return detections

def detect_blobs(img_norm, img_orig, min_thresh, min_area, color, 
                 file_name, method='simple', 
                 include_method_info=False, **kwargs):
    """
    General blob detection function that supports multiple detection methods.
    
    Parameters:
    -----------
    img_norm : np.ndarray
        Normalized image for detection
    img_orig : np.ndarray  
        Original image for intensity calculations
    min_thresh : float
        Minimum threshold for detection
    min_area : float
        Minimum area for blob filtering
    color : str
        Color label for the blobs
    file_name : str
        Name of the file being processed
    method : str
        Detection method to use. Options:
        - 'simple': OpenCV SimpleBlobDetector (default) - Good for general circular/elliptical blobs
        - 'contours': Contour-based detection - Good for irregular shapes
        - 'hough': Hough circle detection - Best for perfect circles
        - 'connected_components': Connected components labeling - Fast, good for well-separated objects
        - 'watershed': Watershed segmentation - Good for touching/overlapping objects
        - 'cellpose': Cellpose deep learning segmentation - Best for cells and complex biological objects
    include_method_info : bool
        If True, includes 'method' key in output for compatibility (default: False)
    **kwargs : dict
        Additional method-specific parameters:
        
        For 'simple' method:
            max_threshold=255, max_area=1600, threshold_step=2,
            filter_by_color=False, filter_by_circularity=False, etc.
            
        For 'hough' method:
            max_radius=40, dp=1, min_dist=20, param1=50, param2=30
            
        For 'watershed' method:
            min_distance=10, threshold_abs=0.3
            
        For 'cellpose' method:
            diameter=60, model_type='cyto3', gpu=False, flow_threshold=0.4,
            cellprob_threshold=0.0, channels=[0,0], min_diameter=0, max_diameter=inf
        
    Returns:
    --------
    list : List of detected blob dictionaries with keys:
        'Box', 'center', 'radius', 'color', 'file', 
        'max_intensity', 'mean_intensity', 'mean_dilation',
        'box_x', 'box_y', 'box_size'
        (plus 'method' key if include_method_info=True)
        
    Examples:
    ---------
    # Basic usage (default simple method) - SAME OUTPUT FORMAT AS BEFORE
    blobs = detect_blobs(img_norm, img_orig, 50, 100, 'red', 'test.tiff')
    
    # Use contour detection for irregular shapes
    blobs = detect_blobs(img_norm, img_orig, 50, 100, 'red', 'test.tiff', method='contours')
    
    # Use Hough circles with custom parameters 
    blobs = detect_blobs(img_norm, img_orig, 50, 100, 'red', 'test.tiff', 
                        method='hough', max_radius=50, min_dist=30)
                        
    # Use Cellpose for biological samples
    blobs = detect_blobs(img_norm, img_orig, 50, 100, 'red', 'test.tiff',
                        method='cellpose', diameter=60, model_type='cyto3')
                        method='contours', include_method_info=True)
                        
    # Compare multiple methods (automatically includes method info)
    results = detect_blobs_multi_method(img_norm, img_orig, 50, 100, 'red', 'test.tiff',
                                       methods=['simple', 'contours', 'hough'])
    """
    
    # Method dispatch
    method_map = {
        'simple': _detect_blobs_simple,
        'contours': _detect_blobs_contours, 
        'hough': _detect_blobs_hough_circles,
        'connected_components': _detect_blobs_connected_components,
        'watershed': _detect_blobs_watershed,
        'cellpose': _detect_blobs_cellpose
    }
    
    if method not in method_map:
        raise ValueError(f"Unknown detection method: {method}. Available: {list(method_map.keys())}")
    
    # Special check for Cellpose availability
    if method == 'cellpose' and not CELLPOSE_AVAILABLE:
        raise ImportError(f"Cellpose not available. Install with: pip install cellpose[gui]")
    
    # Apply morphological preprocessing (normalize and dilate)
    # EXCEPTION: Skip for cellpose - deep learning models need raw/original images
    # Morphological dilation can destroy fine details that cellpose was trained to recognize
    if method == 'cellpose': #TODO not clean fix later
        # Use original images for cellpose (no morphological preprocessing)
        processed_norm = img_norm
        processed_dilated = img_orig
    else:
        # Apply morphological preprocessing for all other methods
        processed_norm, processed_dilated = normalize_and_dilate(img_orig, 
                                                                 kernel_size=3, 
                                                                 iterations=1)
    
    # Detect blobs using the selected method
    detections = method_map[method](processed_dilated, processed_norm, min_thresh, min_area, **kwargs)
    
    # Convert detections to standard format
    blobs = []
    for idx, detection in enumerate(detections, start=1):
        x, y = detection['center']
        radius = detection['radius']
        box_size = 2 * radius
        box_x, box_y = x - radius, y - radius

        x1, y1 = max(0, box_x), max(0, box_y)
        x2, y2 = min(processed_norm.shape[1], x + radius), min(processed_norm.shape[0], y + radius)
        roi_orig = processed_norm[y1:y2, x1:x2]
        roi_dilated = processed_dilated[y1:y2, x1:x2]

        if roi_orig.size > 0:
            blob_dict = {
                'Box': f"{file_name} Box #{idx}",
                'center': (x, y),
                'radius': radius,
                'color': color,
                'file': file_name,
                'max_intensity': roi_orig.max(),
                'mean_intensity': roi_orig.mean(),
                'mean_dilation': float(roi_dilated.mean()),
                'box_x': box_x,
                'box_y': box_y,
                'box_size': box_size
            }
            
            # Only add method info if requested for backward compatibility
            if include_method_info:
                blob_dict['method'] = method
                
            blobs.append(blob_dict)
    
    return blobs


# Helper functions for convenient method-specific detection

def detect_blobs_simple(img_norm, img_orig, min_thresh, min_area, color, file_name, include_method_info=False, **kwargs):
    """Convenient wrapper for simple blob detection"""
    return detect_blobs(img_norm, img_orig, min_thresh, min_area, color, file_name, 
                       method='simple', include_method_info=include_method_info, **kwargs)

def detect_blobs_contours(img_norm, img_orig, min_thresh, min_area, color, file_name, include_method_info=False, **kwargs):
    """Convenient wrapper for contour-based blob detection"""
    return detect_blobs(img_norm, img_orig, min_thresh, min_area, color, file_name,
                       method='contours', include_method_info=include_method_info, **kwargs)

def detect_blobs_hough(img_norm, img_orig, min_thresh, min_area, color, file_name, include_method_info=False, **kwargs):  
    """Convenient wrapper for Hough circle detection"""
    return detect_blobs(img_norm, img_orig, min_thresh, min_area, color, file_name,
                       method='hough', include_method_info=include_method_info, **kwargs)

def detect_blobs_connected_components(img_norm, img_orig, min_thresh, min_area, color, file_name, include_method_info=False, **kwargs):
    """Convenient wrapper for connected components detection"""
    return detect_blobs(img_norm, img_orig, min_thresh, min_area, color, file_name,
                       method='connected_components', include_method_info=include_method_info, **kwargs)

def detect_blobs_watershed(img_norm, img_orig, min_thresh, min_area, color, file_name, include_method_info=False, **kwargs):
    """Convenient wrapper for watershed segmentation detection"""
    return detect_blobs(img_norm, img_orig, min_thresh, min_area, color, file_name,
                       method='watershed', include_method_info=include_method_info, **kwargs)

def detect_blobs_cellpose(img_norm, img_orig, min_thresh, min_area, color, file_name, include_method_info=False, **kwargs):
    """Convenient wrapper for Cellpose deep learning segmentation"""
    return detect_blobs(img_norm, img_orig, min_thresh, min_area, color, file_name,
                       method='cellpose', include_method_info=include_method_info, **kwargs)


def get_available_detection_methods():
    """Returns list of available detection methods"""
    methods = ['simple', 'contours', 'hough', 'connected_components', 'watershed']
    if CELLPOSE_AVAILABLE:
        methods.append('cellpose')
    return methods


def detect_blobs_multi_method(img_norm, img_orig, min_thresh, min_area, color, file_name, 
                             methods=['simple'], combine_results=True, **kwargs):
    """
    Apply multiple detection methods and optionally combine results.
    
    Parameters:
    -----------
    methods : list
        List of detection methods to apply
    combine_results : bool  
        If True, combine all results into single list. If False, return dict by method.
    **kwargs : dict
        Additional parameters for detection methods
        
    Returns:
    --------
    list or dict : Combined results or dict of results by method
    """
    all_results = {}
    
    for method in methods:
        try:
            blobs = detect_blobs(img_norm, img_orig, min_thresh, min_area, color, file_name,
                               method=method, include_method_info=True, **kwargs)
            all_results[method] = blobs
            print(f"Method '{method}': Found {len(blobs)} blobs")
        except Exception as e:
            print(f"Error with method '{method}': {e}")
            all_results[method] = []
    
    if combine_results:
        # Combine all results (method info already included via include_method_info=True)
        combined_blobs = []
        for method, blobs in all_results.items():
            combined_blobs.extend(blobs)
        return combined_blobs
    
    return all_results



#not used
def first_scan_detect_blobs():
    COLOR_ORDER = [
        'red', 'green', 'blue', 'orange', 'purple',
        'cyan', 'olive', 'yellow', 'brown', 'pink'
    ]
    watch_dir = Path(os.getcwd())
    json_path = watch_dir / "first_scan.json"
    precomputed_blobs = {color: {} for color in COLOR_ORDER}

    # --- STEP 1: Load JSON ---
    with open(json_path, "r") as f:
        data = json.load(f)
        if isinstance(data, dict):
            json_items = [[key, value] for key, value in data.items()]
        else:
            json_items = [data]

    print("\n✅ Loaded JSON:")
    for pair in json_items:
        print(pair)

    # --- STEP 2: Get expected number of TIFF files ---
    try:
        expected_tiff_count = int(json_items[6][1])
    except (IndexError, ValueError) as e:
        print(f"❌ Error extracting expected TIFF count from JSON: {e}")
        return None

    if expected_tiff_count > len(COLOR_ORDER):
        print(f"❌ Too many TIFF files requested. Max supported: {len(COLOR_ORDER)}")
        return None

    # --- STEP 3: Wait for TIFF files ---
    print(f"\n🔍 Waiting for {expected_tiff_count} unique .tiff files...")
    while True:
        tiff_files = sorted({f for f in os.listdir(watch_dir) if f.endswith(".tiff")})
        if len(tiff_files) >= expected_tiff_count:
            selected_tiffs = tiff_files[:expected_tiff_count]
            break
        time.sleep(1)

    print("\n✅ Found required TIFF files:")
    for idx, fname in enumerate(selected_tiffs):
        print(f"{COLOR_ORDER[idx].capitalize()}: {fname}")

    # --- STEP 4: Process TIFF files ---
    for idx, tiff_name in enumerate(selected_tiffs):
        color = COLOR_ORDER[idx]
        tiff_path = watch_dir / tiff_name
        try:
            tiff_img = tiff.imread(str(tiff_path)).astype(np.float32)
            norm, dilated = normalize_and_dilate(tiff_img)
            threshold = json_items[0][1]
            min_area = json_items[1][1]
            blobs = detect_blobs(dilated, norm, threshold, min_area, color, tiff_name)
            precomputed_blobs[color][(threshold, min_area)] = blobs
        except Exception as e:
            print(f"❌ Error processing {tiff_name}: {e}")

    return precomputed_blobs

def structure_blob_tooltips(json_path):
    """
    Reads a JSON file containing blobs with HTML tooltips,
    extracts and structures the data, and writes it back to the same file.
    """
    
    def extract_numbers(s):
        """Extract all integers/floats from a string as a list."""
        return [float(x) if '.' in x else int(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", s)]

    with open(json_path, "r") as f:
        original_data = json.load(f)

    structured_data = {}

    for key, blob in original_data.items():
        text = blob.get("text", "")
        tooltip = blob.get("tooltip", "")

        fields = tooltip.replace("<b>", "").replace("</b>", "").split("<br>")
        fields = [line.strip() for line in fields if line.strip()]  # clean empty lines

        structured = {"text": text}

        for line in fields:
            if line.startswith("Center:"):
                structured["image_center"] = extract_numbers(line)
            elif "Length:" in line:
                structured["image_length"] = extract_numbers(line)[0]
            elif "Area:" in line and "px²" in line:
                structured["image_area_px²"] = extract_numbers(line)[0]
            elif "Box area:" in line:
                structured["image_area_px²"] = extract_numbers(line)[0]
            elif "Real Center location" in line or "Real Center:" in line:
                structured["real_center_um"] = extract_numbers(line)
            elif "Real box size" in line or "Real Size:" in line:
                structured["real_size_um"] = extract_numbers(line)
            elif "Real box area" in line or "Real Area:" in line:
                structured["real_area_um²"] = extract_numbers(line)[0]
            elif "Real Top-Left:" in line:
                structured["real_top_left_um"] = extract_numbers(line)
            elif "Real Bottom-Right:" in line:
                structured["real_bottom_right_um"] = extract_numbers(line)
            elif "Max intensity" in line:
                structured["max_intensity"] = extract_numbers(line)[0]
            elif "Mean intensity" in line:
                structured["mean_intensity"] = extract_numbers(line)[0]
            elif "Mean dilation intensity" in line:
                structured["mean_dilation_intensity"] = extract_numbers(line)[0]

        structured_data[key] = structured

    # Overwrite original file with structured data
    with open(json_path, "w") as f:
        json.dump(structured_data, f, indent=4)

def resize_if_needed(img, name, target_shape):
        if img.shape != target_shape:
            # print(f"Resizing {name} from {img.shape} → {target_shape}")
            return cv2.resize(img, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_AREA)
        return img

def is_featureless(img):
    img = np.nan_to_num(img)
    ent = shannon_entropy(img)
    pnr = (img.max() - img.mean()) / (img.std() + 1e-5)
    edge_map = cv2.Canny(img.astype(np.uint8), 50, 150)
    edge_ratio = np.count_nonzero(edge_map) / img.size


    return (ent < 2.5) and (pnr < 2.5) and (edge_ratio < 0.01)


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

def boxes_intersect(b1, b2):
    x1_min, y1_min = b1['box_x'], b1['box_y']
    x1_max, y1_max = x1_min + b1['box_size'], y1_min + b1['box_size']

    x2_min, y2_min = b2['box_x'], b2['box_y']
    x2_max, y2_max = x2_min + b2['box_size'], y2_min + b2['box_size']

    return not (x1_max < x2_min or x1_min > x2_max or y1_max < y2_min or y1_min > y2_max)


def union_box_dimensions(b1, b2, b3):
    """
    Computes the union box of three blobs using their box_x, box_y, and box_size.
    The union box is defined by the min bottom-left and max top-right corners.
    Returns:
        center (tuple): (x, y) of union box center
        length (float): side length of union box
        area (float): area of union box
    """
    # bottom-left corners
    bl_x = [b1['box_x'], b2['box_x'], b3['box_x']]
    bl_y = [b1['box_y'], b2['box_y'], b3['box_y']]
   
    # top-right corners
    tr_x = [b1['box_x'] + b1['box_size'], b2['box_x'] + b2['box_size'], b3['box_x'] + b3['box_size']]
    tr_y = [b1['box_y'] + b1['box_size'], b2['box_y'] + b2['box_size'], b3['box_y'] + b3['box_size']]
   
    # union box bounds
    min_x = min(bl_x)
    min_y = min(bl_y)
    max_x = max(tr_x)
    max_y = max(tr_y)
   
    # center of union box
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
   
    # side length and area
    width = max_x - min_x
    height = max_y - min_y
    length = max(width, height)  # make it square
    area = length * length
   
    return (center_x, center_y), float(length), float(area)


def union_center(b1, b2, b3):
    """
    Computes the center of the union box of three blobs.
    Uses the union_box_dimensions function to avoid repeating logic.
    """
    center, _, _ = union_box_dimensions(b1, b2, b3)
    return center


def find_union_blobs(blobs, microns_per_pixel_x, microns_per_pixel_y, true_origin_x, true_origin_y):
    blobs_by_color = {color: [] for color in blobs}

    for color, blob_dict in blobs.items():
        for coord_key, blob_list in blob_dict.items():
            blobs_by_color[color].extend(blob_list)

    union_objects = {}
    union_index = 1
    reds = blobs_by_color.get('red', [])
    greens = blobs_by_color.get('green', [])
    blues = blobs_by_color.get('blue', [])

    for r in reds:
        for g in greens:
            if not boxes_intersect(r, g):
                continue
            for b in blues:
                if boxes_intersect(r, b) and boxes_intersect(g, b):
                    cx, cy = union_center(r, g, b)
                    _, length, area = union_box_dimensions(r, g, b)
                    top_left_x = cx - length // 2
                    top_left_y = cy - length // 2
                    bottom_right_x = top_left_x + length
                    bottom_right_y = top_left_y + length

                    real_cx = (cx * microns_per_pixel_x) + true_origin_x
                    real_cy = (cy * microns_per_pixel_y) + true_origin_y
                    real_length_x = length * microns_per_pixel_x
                    real_length_y = length * microns_per_pixel_y
                    real_area = real_length_x * real_length_y

                    real_top_left = (
                        (top_left_x * microns_per_pixel_x) + true_origin_x,
                        (top_left_y * microns_per_pixel_y) + true_origin_y
                    )
                    real_bottom_right = (
                        (bottom_right_x * microns_per_pixel_x) + true_origin_x,
                        (bottom_right_y * microns_per_pixel_y) + true_origin_y
                    )

                    union_obj = {
                        # Original fields (used by formatter)
                        'center': [cx, cy],
                        'length': length,
                        'area': area,

                        # Alias for compatibility with merge logic
                        'image_center': [cx, cy],
                        'image_length': length,
                        'image_area_px²': area,

                        # Real-world
                        'real_center_um': [real_cx, real_cy],
                        'real_size_um': [real_length_x, real_length_y],
                        'real_area_um\u00b2': real_area,
                        'real_top_left_um': list(real_top_left),
                        'real_bottom_right_um': list(real_bottom_right),
                    }

                    union_objects[union_index] = union_obj
                    union_index += 1

    return union_objects


def find_union_blobs_(blobs, microns_per_pixel_x, microns_per_pixel_y, true_origin_x, true_origin_y):
    blobs_by_color = {color: [] for color in blobs}

    for color, blob_dict in blobs.items():
        for coord_key, blob_list in blob_dict.items():
            blobs_by_color[color].extend(blob_list)
    union_objects = {}
    union_index = 1
    reds = blobs_by_color.get('red', [])
    greens = blobs_by_color.get('green', [])
    blues = blobs_by_color.get('blue', [])
    for r in reds:
        for g in greens:
            if not boxes_intersect(r, g):
                continue
            for b in blues:
                if boxes_intersect(r, b) and boxes_intersect(g, b):
                    cx, cy = union_center(r, g, b)
                    length, area = union_box_dimensions(r, g, b)
                    top_left_x = cx - length // 2
                    top_left_y = cy - length // 2
                    bottom_right_x = top_left_x + length
                    bottom_right_y = top_left_y + length

                    real_cx = (cx * microns_per_pixel_x) + true_origin_x
                    real_cy = (cy * microns_per_pixel_y) + true_origin_y
                    real_length_x = length * microns_per_pixel_x
                    real_length_y = length * microns_per_pixel_y
                    real_area = real_length_x * real_length_y

                    real_top_left = (
                        (top_left_x * microns_per_pixel_x) + true_origin_x,
                        (top_left_y * microns_per_pixel_y) + true_origin_y
                    )
                    real_bottom_right = (
                        (bottom_right_x * microns_per_pixel_x) + true_origin_x,
                        (bottom_right_y * microns_per_pixel_y) + true_origin_y
                    )

                    # union_obj = {
                    #     'center': (cx, cy),
                    #     'length': length,
                    #     'area': area,
                    #     'real_center_um': (real_cx, real_cy),
                    #     'real_size_um': (real_length_x, real_length_y),
                    #     'real_area_um\u00b2': real_area,
                    #     'real_top_left_um': real_top_left,
                    #     'real_bottom_right_um': real_bottom_right
                    # }

                    union_obj = {
                            # Original (pixel-space)
                            'center': (cx, cy),
                            'length': length,
                            'area': area,

                            # Synonyms for downstream merger + formatter
                            'image_center': [cx, cy],
                            'image_length': length,
                            'image_area_px²': area,

                            # Real-world units
                            'real_center_um': (real_cx, real_cy),
                            'real_size_um': (real_length_x, real_length_y),
                            'real_area_um\u00b2': real_area,
                            'real_top_left_um': real_top_left,
                            'real_bottom_right_um': real_bottom_right
}


                    union_objects[union_index] = union_obj
                    union_index += 1

    return union_objects
 
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

def _get_flyscan_dimensions(hdr):
    start_doc = hdr.start
    # 2D_FLY_PANDA: prefer 'dimensions', fallback to 'shape'
    if 'scan' in start_doc and start_doc['scan'].get('type') == '2D_FLY_PANDA':
        if 'dimensions' in start_doc:
            return start_doc['dimensions']
        elif 'shape' in start_doc:
            return start_doc['shape']
        else:
            raise ValueError("No dimensions or shape found for 2D_FLY_PANDA scan")
    # rel_scan: use 'shape' or 'num_points'
    elif start_doc.get('plan_name') == 'rel_scan':
        if 'shape' in start_doc:
            return start_doc['shape']
        elif 'num_points' in start_doc:
            return [start_doc['num_points']]
        else:
            raise ValueError("No shape or num_points found for rel_scan")
    else:
        raise ValueError("Unknown scan type for _get_flyscan_dimensions")

def _pad_scalar_to_expected_length(scalar, expected_length):
    """
    Pad scalar array to expected length using the last collected point.
    Handles cases where scalar data has dropped points.
    
    Args:
        scalar: numpy array of scalar values
        expected_length: expected total number of points
    
    Returns:
        padded_scalar: numpy array padded to expected length
    """
    if len(scalar) == expected_length:
        return scalar
    
    if len(scalar) > expected_length:
        print(f"[SCALAR] Warning: scalar length ({len(scalar)}) > expected ({expected_length}), truncating")
        return scalar[:expected_length]
    
    # Pad with last point
    padding_needed = expected_length - len(scalar)
    last_point = scalar[-1] if len(scalar) > 0 else 1.0  # fallback to 1.0 if empty
    padded_values = np.full(padding_needed, last_point)
    padded_scalar = np.concatenate([scalar, padded_values])
    
    print(f"[SCALAR] Padded scalar from {len(scalar)} to {len(padded_scalar)} points using last value {last_point}")
    return padded_scalar

def _export_xrf_remote(scan_id, norm='sclr1_ch4', elem_list=[]):
    """
    Export XRF data to remote handler for remote segmentation.
    
    Args:
        scan_id: Scan ID to export
        norm: Normalization channel (default: 'sclr1_ch4')
        elem_list: List of elements to export
    """
    if not scan_id:
        print("[EXPORT] Skipping remote XRF export - no scan ID provided.")
        return

    hdr = db[int(scan_id)]
    scan_id = hdr.start["scan_id"]
    
    channels = [1, 2, 3]
    print(f"[REMOTE] {elem_list = }")
    print(f"[REMOTE] fetching XRF ROIs")
    scan_dim = _get_flyscan_dimensions(hdr)
    print(f"[REMOTE] fetching scalar values")

    scalar = np.array(list(hdr.data(norm))).squeeze()
    print(f"[REMOTE] fetching scalar {norm} values done")
    
    # Calculate expected length from scan dimensions
    expected_length = np.prod(scan_dim)
    
    for elem in sorted(elem_list):
        if elem not in remote_sender.get_cache():
            remote_sender.append_cache(elem)
            roi_keys = [f'Det{chan}_{elem}' for chan in channels]
            spectrum = np.sum([np.array(list(hdr.data(roi)), dtype=np.float32).squeeze() for roi in roi_keys], axis=0)
            
            # Pad scalar if needed to match spectrum length
            if norm is not None:
                padded_scalar = _pad_scalar_to_expected_length(scalar, len(spectrum))
                spectrum = spectrum / padded_scalar
            
            xrf_img = spectrum.reshape(scan_dim)
            remote_sender.write(xrf_img)

def _export_xrf_remote_container(scan_id, norm='sclr1_ch4', elem_list=[],
                                 append_meta_with = {}):
    """
    Export XRF data to remote handler for remote segmentation.
    
    Args:
        scan_id: Scan ID to export
        norm: Normalization channel (default: 'sclr1_ch4')
        elem_list: List of elements to export
    """

    
    if not scan_id:
        print("[EXPORT] Skipping remote XRF export - no scan ID provided.")
        return

    hdr = db[int(scan_id)]
    scan_id = hdr.start["scan_id"]

    meta = export_scan_params(sid=scan_id)
    
    # Append additional metadata if provided
    if append_meta_with:
        meta.update(append_meta_with)

    import time
    timestamp = int(time.time())
    scan_container = container.create_container(f"automap_{scan_id}_{timestamp}", 
                                                metadata=meta, 
                                                access_tags=["synaps_project"])
    
    channels = [1, 2, 3]
    print(f"[REMOTE] {elem_list = }")
    print(f"[REMOTE] fetching XRF ROIs")
    scan_dim = _get_flyscan_dimensions(hdr)
    print(f"[REMOTE] fetching scalar values")

    scalar = np.array(list(hdr.data(norm))).squeeze()
    print(f"[REMOTE] fetching scalar {norm} values done")

    # Calculate expected length from scan dimensions
    expected_length = np.prod(scan_dim)
    
    # Collect all normalized XRF images for stacking
    xrf_images = []
    element_names = []

    if elem_list and isinstance(elem_list[0], list):
        elem_list = list(set(elem for sublist in elem_list for elem in sublist))
    else:
        elem_list = list(set(elem_list)) if elem_list else []
    
    for elem in sorted(elem_list):
        try:
            roi_keys = [f'Det{chan}_{elem}' for chan in channels]
            spectrum = np.sum([np.array(list(hdr.data(roi)), dtype=np.float32).squeeze() for roi in roi_keys], axis=0)
            
            # Pad scalar if needed to match spectrum length
            if norm is not None:
                padded_scalar = _pad_scalar_to_expected_length(scalar, len(spectrum))
                spectrum = spectrum / padded_scalar
            
            xrf_img = spectrum.reshape(scan_dim)
            xrf_images.append(xrf_img)
            element_names.append(elem)
            print(f"[REMOTE] Processed element {elem} for stacking")
        except Exception as e:
            print(f"[REMOTE ERROR] Failed to process element {elem} for scan {scan_id}: {e}")
    
    # Stack all images and send as single array
    if xrf_images:
        try:
            # Stack along first axis: (n_elements, height, width)
            stacked_array = np.stack(xrf_images, axis=0)
            
            # Create compound key name from all elements
            compound_key = "".join(element_names)
            
            # Send stacked array with compound key
            result = scan_container.write_array(stacked_array, key=compound_key, access_tags=["synaps_project"])
            print(f"[REMOTE] Successfully exported stacked array for elements {element_names} as key '{compound_key}', shape: {stacked_array.shape}, result: {result}")
        except Exception as e:
            print(f"[REMOTE ERROR] Failed to export stacked array for scan {scan_id}: {e}")

        print(f"[REMOTE] meta for scan {scan_id}: {meta}") 
    else:
        print(f"[REMOTE WARNING] No XRF images processed for scan {scan_id}")

        #remote_sender.write(xrf_img)

def _export_xrf_local(scan_id, norm='sclr1_ch4', elem_list=[], wd='.'):
    """
    Export XRF data as local TIFF files.
    
    Args:
        scan_id: Scan ID to export
        norm: Normalization channel (default: 'sclr1_ch4')
        elem_list: List of elements to export
        wd: Working directory for output files
    """
    if not scan_id:
        print("[EXPORT] Skipping local XRF export - no scan ID provided.")
        return

    hdr = db[int(scan_id)]
    scan_id = hdr.start["scan_id"]
    
    channels = [1, 2, 3]
    print(f"[LOCAL] {elem_list = }")
    print(f"[LOCAL] fetching XRF ROIs")
    scan_dim = _get_flyscan_dimensions(hdr)
    print(f"[LOCAL] fetching scalar values")

    scalar = np.array(list(hdr.data(norm))).squeeze()
    print(f"[LOCAL] fetching scalar {norm} values done")
    
    # Calculate expected length from scan dimensions
    expected_length = np.prod(scan_dim)
    
    for elem in sorted(elem_list):
        roi_keys = [f'Det{chan}_{elem}' for chan in channels]
        spectrum = np.sum([np.array(list(hdr.data(roi)), dtype=np.float32).squeeze() for roi in roi_keys], axis=0)
        
        # Pad scalar if needed to match spectrum length
        if norm is not None:
            padded_scalar = _pad_scalar_to_expected_length(scalar, len(spectrum))
            spectrum = spectrum / padded_scalar
        
        xrf_img = spectrum.reshape(scan_dim)
        tiff.imwrite(os.path.join(wd, f"scan_{scan_id}_{elem}.tiff"), xrf_img)


def export_xrf_roi_data(scan_id, norm='sclr1_ch4', elem_list=[], 
                        wd='.', remote_seg=False, append_meta_with={}):
    """
    Export XRF ROI data either remotely or as local TIFF files.
    
    Args:
        scan_id: Scan ID to export
        norm: Normalization channel (default: 'sclr1_ch4')
        elem_list: List of elements to export
        wd: Working directory for local export
        remote_seg: If True, use remote handler; if False, write local TIFFs
        append_meta_with: Additional metadata to append (default: empty dict)
    """
    if remote_seg:
       # _export_xrf_remote(scan_id, norm, elem_list)
       _export_xrf_remote_container(scan_id, 
                                    norm=norm, 
                                    elem_list=elem_list, 
                                    append_meta_with=append_meta_with)
    else:
        _export_xrf_local(scan_id, norm, elem_list, wd)


def export_scan_params(sid=-1, zp_flag=True, save_to=None):
    """
    Fetch scan parameters, ROI positions, step size, and the full start_doc
    for scan `sid`.  Optionally write them out as JSON.

    Returns a dict with:
      - scan_id
      - start_doc
      - roi_positions
      - step_size (computed from scan_input for 2D_FLY_PANDA)
    """
    if sid == -1:
        print("[EXPORT] Skipping scan params export - no valid scan ID provided.")
        return
    # 1) Pull the header
    hdr = db[int(sid)]
    start_doc = dict(hdr.start)  # cast to plain dict

    # 2) Grab the baseline table and build the ROI dict
    tbl = db.get_table(hdr, stream_name='baseline')
    row = tbl.iloc[0]
    if zp_flag:
        roi = {
            "zpssx":    float(row["zpssx"]),
            "zpssy":    float(row["zpssy"]),
            "zpssz":    float(row["zpssz"]),
            "smarx":    float(row["smarx"]),
            "smary":    float(row["smary"]),
            "smarz":    float(row["smarz"]),
            "zp.zpz1":  float(row["zpz1"]),
            "zpsth":    float(row["zpsth"]),
            "zps.zpsx": float(row["zpsx"]),
            "zps.zpsz": float(row["zpsz"]),
        }
    else:
        roi = {
            "dssx":  float(row["dssx"]),
            "dssy":  float(row["dssy"]),
            "dssz":  float(row["dssz"]),
            "dsx":   float(row["dsx"]),
            "dsy":   float(row["dsy"]),
            "dsz":   float(row["dsz"]),
            "sbz":   float(row["sbz"]),
            "dsth":  float(row["dsth"]),
        }

    # 3) Compute unified step_size from scan_input
    scan_info = start_doc.get("scan", {})
    si = scan_info.get("scan_input", [])
    if scan_info.get("type") == "2D_FLY_PANDA" and len(si) >= 3:
        fast_start, fast_end, fast_N = si[0], si[1], si[2]
        step_size = abs(fast_end - fast_start) / fast_N
    else:
        raise ValueError(f"Cannot compute step_size for scan type {scan_info.get('type')}")

    # 4) Assemble the result dict
    result = {
        "scan_id":       int(sid),
        "start_doc":     start_doc,
        "roi_positions": roi,
        "step_size":     float(step_size),
    }

    # 5) Optionally write out JSON
    if save_to:
        if os.path.isdir(save_to):
            filename = os.path.join(save_to, f"scan_{sid}_params.json")
        else:
            filename = save_to if save_to.lower().endswith(".json") else save_to + ".json"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            json.dump(make_json_serializable(result), f, indent=2)

    return result


def export_batch_scan_params(scan_ids, zp_flag=True, save_to=None):
    """
    Export scan parameters for a batch of scan IDs.
    
    Args:
        scan_ids (list): List of scan IDs to export
        zp_flag (bool): Whether to use ZP motors or DS motors
        save_to (str): Directory or base filename to save to
    
    Returns:
        dict: Dictionary mapping scan_id to exported parameters
    """
    if not scan_ids:
        print(f"[EXPORT] Skipping batch scan params export - no scan IDs provided.")
        return {}
    
    results = {}
    
    for i, sid in enumerate(scan_ids):
        print(f"[BATCH] Exporting scan {sid} ({i+1}/{len(scan_ids)})")
        try:
            # Determine save path for this scan
            scan_save_to = None
            if save_to:
                if os.path.isdir(save_to):
                    scan_save_to = save_to
                else:
                    # If save_to is a filename, create directory structure
                    base_dir = os.path.dirname(save_to) or "."
                    scan_save_to = base_dir
            
            result = export_scan_params(
                sid=sid,
                zp_flag=zp_flag,
                save_to=scan_save_to
            )
            
            if result:
                results[sid] = result
                print(f"[BATCH] ✅ Exported scan {sid}")
            else:
                print(f"[BATCH] ⚠️ No data returned for scan {sid}")
                
        except Exception as e:
            print(f"[BATCH] ❌ Error exporting scan {sid}: {e}")
            results[sid] = {"error": str(e)}
    
    # Optionally save a summary file
    if save_to and results:
        summary_path = os.path.join(save_to if os.path.isdir(save_to) else os.path.dirname(save_to), 
                                   "batch_export_summary.json")
        try:
            with open(summary_path, "w") as f:
                json.dump({
                    "exported_scans": list(results.keys()),
                    "total_scans": len(scan_ids),
                    "successful_exports": len([r for r in results.values() if "error" not in r]),
                    "failed_exports": len([r for r in results.values() if "error" in r]),
                    "export_timestamp": time.time()
                }, f, indent=2)
            print(f"[BATCH] Summary saved to: {summary_path}")
        except Exception as e:
            print(f"[BATCH] ⚠️ Could not save summary: {e}")
    
    print(f"[BATCH] Completed batch export: {len(results)} scans processed")
    return results

def _fly2d_qserver_scan_export(label,
                           dets,
                           mot1, mot1_s, mot1_e, mot1_n,
                           mot2, mot2_s, mot2_e, mot2_n,
                           exp_t,
                           roi_positions=None,
                           scan_id=None,
                           zp_move_flag=1,
                           smar_move_flag=1,
                           ic1_count=55000,
                           # **POST-SCAN EXPORTS**
                           elem_list=None,           # list of elements for XRF
                           export_norm='sclr1_ch4',  # channel to normalize by
                           data_wd='.'):             # where to write TIFFs
    """
    1) Optionally recover a previous scan or ROI dict
    2) Do beam/flux checks
    3) Run fly2dpd
    4) Export XRF-ROI data TIFFs
    5) Save final ROI positions JSON
    """
    print(f"{label} starting…")
    RE.md["scan_name"] = str(label)

    # — 1) RECOVERY —
    moved = False
    # If a valid scan_id is provided (truthy), recover from that scan

    if scan_id:
        yield from recover_zp_scan_pos(scan_id,
                                       zp_move_flag=zp_move_flag,
                                       smar_move_flag=smar_move_flag,
                                       move_base=1)
        moved = True

    #Else if ROI positions dict/string provided, and not all values None
    elif roi_positions:
        if isinstance(roi_positions, str):
            roi_positions = json.loads(roi_positions)
        # Filter out keys with None values
        non_null = {k: v for k, v in roi_positions.items() if v is not None}
        if non_null:
            for key, val in non_null.items():
                if key != "zp.zpz1":
                    yield from bps.mov(eval(key), val)
                else:
                    yield from mov_zpz1(val)
                print(f"  → {key} @ {val:.3f}")
            yield from check_for_beam_dump(threshold=5000)
            if sclr2_ch2.get() < ic1_count * 0.9:
                yield from peak_the_flux()
            moved = True

    if not moved:
        print("[RECOVERY] no ROI recovery requested; skipping motor moves.")

    # — 2) FLY SCAN —
    yield from fly2dpd(dets,
                       mot1, mot1_s, mot1_e, mot1_n,
                       mot2, mot2_s, mot2_e, mot2_n,
                       exp_t)
    # produce a zmq message with scan id?

    # — 3) POST-SCAN EXPORTS —
    # hdr = db[-1]
    # last_id = hdr.start["scan_id"]
    # print(f"[POST] exporting XRF ROI data for scan {last_id}…")
    # export_xrf_roi_data(last_id,
    #                     norm=export_norm,
    #                     elem_list=elem_list or [],
    #                     wd=data_wd)

    # if pos_save_to:
    #     print(f"[POST] saving ROI positions JSON to {pos_save_to}…")
    #     export_scan_params(sid=last_id, zp_flag=True, save_to=pos_save_to)

    # print("[POST] done.")


def send_fly2d_to_queue(label,
                        dets,
                        det_names,
                        mot1, mot1_s, mot1_e, mot1_n,
                        mot2, mot2_s, mot2_e, mot2_n,
                        exp_t,
                        roi_positions=None,
                        scan_id=None,
                        zp_move_flag=1,
                        smar_move_flag=1,
                        ic1_count = 55000,
                        elem_list=None,
                        export_norm='sclr1_ch4',
                        data_wd='.',
                        real_test=0):
    # Use provided det_names or fallback to default
    if not det_names:
        det_names = ['fs', 'eiger2', 'xspress3']

    roi_json = ""
    if isinstance(roi_positions, dict):
        roi_json = json.dumps(roi_positions)
    elif isinstance(roi_positions, str):
        roi_json = roi_positions

    print("Coarse scan - submitting to queue...")
    RM.item_add(BPlan("fly2d_qserver_scan_export",
                      label,
                      det_names,
                      mot1, mot1_s, mot1_e, mot1_n,
                      mot2, mot2_s, mot2_e, mot2_n,
                      exp_t,
                      roi_json,
                      scan_id or "",
                      zp_move_flag,
                      smar_move_flag,
                      ic1_count,
                      json.dumps(elem_list or []),
                      export_norm,
                      data_wd))
    print("Coarse scan sent to queue.")

def wait_for_queue_done(poll_interval=5.0, idle_timeout=3600, auto_restart=True):
    """
    Wait until QServer queue is empty and manager is idle.
    Optionally restart the queue if stuck in idle with items remaining.

    Args:
        poll_interval (float): Seconds between polls.
        idle_timeout (float): How long to wait in idle with items before triggering restart.
        auto_restart (bool): If True, will automatically call RM.queue_start() after timeout.
        
    Returns:
        bool: True if queue completed normally, False if timed out
    """
    import time

    print("[WAIT] polling queue status...", end="", flush=True)
    idle_stuck_start = None

    while True:
        st = RM.status()
        items = st.get("items_in_queue", 0)
        state = st.get("manager_state", "")

        if items == 0 and state == "idle":
            print(" done.")
            return True

        if items > 0 and state == "idle":
            if idle_stuck_start is None:
                idle_stuck_start = time.time()
            elif time.time() - idle_stuck_start > idle_timeout:
                if auto_restart:
                    print("\n⚠️ Queue is idle with items still in queue.")
                    print("🔁 Automatically restarting queue with RM.queue_start()...")
                    RM.queue_start()
                else:
                    print("\n⚠️ Queue is idle with items still in queue.")
                    print("🔁 Consider running: RM.queue_start() to resume.")
                return False
        else:
            idle_stuck_start = None  # reset if queue becomes active again

        print(".", end="", flush=True)
        time.sleep(poll_interval)

def submit_and_export(execution_params, scan_params, export_params, segmentation_params=None):
    """
    Step 1: Enqueue scan (if real), wait (if real), export data (real/offline).
    
    Args:
        execution_params (dict): Execution mode and flags
        scan_params (dict): Scan parameters (motors, dets, positions, etc)
        export_params (dict): Export settings (elem_list, data_wd, etc)
        segmentation_params (dict): Segmentation settings (optional)
    
    Returns:
        tuple: (last_id, out_dir)
    """
    if segmentation_params is None:
        segmentation_params = {}
    
    # Get mode from execution_params
    mode = str(execution_params.get('mode', 'simulation')).lower()
    is_real = (mode == 'real')
    is_sim  = (mode == 'simulation')
    is_offline = (mode == 'offline')
    
    # Get remote_seg flag
    is_remote = segmentation_params.get('remote_seg', False)

    # --- 1. Enqueue (Real Only) ---
    label = scan_params.get('label', '')
    
    if is_real:
        print(f"[REAL] [SUBMIT] Queueing scan '{label}'...")
        
        # Build flat parameter dict for send_fly2d_to_queue
        flat_params = {
            'label': label,
            'dets': scan_params.get('dets', 'dets_fast'),
            'det_names': scan_params.get('det_names', ['fs', 'eiger2', 'xspress3']),
            'mot1': scan_params.get('mot1', 'zpssx'),
            'mot1_s': scan_params.get('mot1_s', 0),
            'mot1_e': scan_params.get('mot1_e', 0),
            'mot2': scan_params.get('mot2', 'zpssy'),
            'mot2_s': scan_params.get('mot2_s', 0),
            'mot2_e': scan_params.get('mot2_e', 0),
            'exp_t': scan_params.get('exp_t', 0.01),
            'roi_positions': scan_params.get('roi_positions_file'),
            'scan_id': scan_params.get('scan_id'),
            'zp_move_flag': scan_params.get('zp_move_flag', 1),
            'smar_move_flag': scan_params.get('smar_move_flag', 1),
            'elem_list': export_params.get('elem_list', []),
            'export_norm': export_params.get('export_norm', 'sclr1_ch4'),
            'data_wd': export_params.get('data_wd', '.'),
        }
        
        # Calculate mot1_n and mot2_n from step_size
        step_size = scan_params.get('step_size', 1.0)
        flat_params['mot1_n'] = int(abs(flat_params['mot1_e'] - flat_params['mot1_s']) / step_size) if step_size > 0 else 1
        flat_params['mot2_n'] = int(abs(flat_params['mot2_e'] - flat_params['mot2_s']) / step_size) if step_size > 0 else 1
        
        send_fly2d_to_queue(**flat_params)
        RM.queue_start()
        time.sleep(1)
        
    elif is_offline:
        print(f"[OFFLINE] Skipping submission. Targeting existing/past scan.")
        
    else: # Sim
        print(f"[SIM] Would call: send_fly2d_to_queue(...)")
        time.sleep(1)

    # --- 2. Wait for Completion & Get ID ---
    data_wd = export_params.get('data_wd', '/data/users/current_user')
    
    if is_real:
        queue_success = wait_for_queue_done(poll_interval=1.0, idle_timeout=60, auto_restart=True)
        
        if not queue_success:
            raise RuntimeError("❌ Coarse scan queue timed out or failed to complete!")
        
        # Verify scan completed successfully
        try:
            hdr = db[-1]
            last_id = hdr.start['scan_id']
            
            # Check if scan has a stop document (completed)
            if not hasattr(hdr, 'stop') or hdr.stop is None:
                raise RuntimeError(f"❌ Scan {last_id} did not complete - no stop document found!")
            
            # Check exit_status if available
            exit_status = hdr.stop.get('exit_status', 'unknown')
            if exit_status not in ['success', 'unknown']:
                raise RuntimeError(f"❌ Scan {last_id} exit status: {exit_status}")
            
            print(f"✅ Coarse scan {last_id} completed successfully")
            
        except IndexError:
            raise RuntimeError("❌ No scan found in database after queue completion!")
    elif is_offline:
        last_id = scan_params.get('scan_id')
        if last_id is None:
            raise ValueError("Mode is Offline but no 'scan_id' provided in export_params!")
        print(f"[OFFLINE] Using Target ID: {last_id}")
    else:
        last_id = 111111 
        print(f"[SIM] Using dummy ID: {last_id}")

    out_dir = os.path.join(data_wd, f"automap_{last_id}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[EXPORT] Output directory: {out_dir}")

    # --- 3. Export Data ---
    all_elem_list = export_params.get('elem_list', [])
    
    # Flatten nested list and remove duplicates
    if all_elem_list and isinstance(all_elem_list[0], list):
        all_elem_list = list(set(elem for sublist in all_elem_list for elem in sublist))
    else:
        all_elem_list = list(set(all_elem_list)) if all_elem_list else []

    if is_real or is_offline:
        # Both Real and Offline modes trigger the export logic
        print(f"[{'REAL' if is_real else 'OFFLINE'}] Exporting data (remote_seg={is_remote})...")
        export_xrf_roi_data(
            last_id,
            norm=export_params.get('export_norm', 'sclr1_ch4'),
            elem_list=all_elem_list,
            wd=out_dir,
            remote_seg=is_remote, # Pass the remote flag,
            append_meta_with=segmentation_params
        )
        export_scan_params(
            sid=last_id,
            zp_flag=bool(scan_params.get('zp_move_flag', True)),
            save_to=out_dir
        )
    else:
        # Sim Mode: Manual Copy
        params_file_name = f"scan_{last_id}_params.json"
        print("\n" + "!"*60)
        print(f"[SIMULATION] Waiting for files in: {out_dir}")
        print(f"Copy TIFFs and '{params_file_name}' here.")
        print("!"*60)

        while True:
            tiffs_in_dir = list(Path(out_dir).glob("*.tiff")) + list(Path(out_dir).glob("*.tif"))
            if tiffs_in_dir:
                print(f"[SIM] Found {len(tiffs_in_dir)} TIFFs. Resuming...")
                break
            time.sleep(3)

    return last_id, out_dir

def analyze_data_remote(np_array, scan_metadata):
    """
    Placeholder for remote analysis function.
    In a real implementation, this would send data to a remote server
    for analysis and return the results.
    """
    print("[REMOTE ANALYSIS] This is a placeholder function.")
    # Implement remote analysis logic here
    return np_array, scan_metadata

def analyze_data_local(scan_id=None, 
                       return_results=False, 
                       params=None):
    """
    Step 2: Analysis. 
    Iterates through element groups, calculates unions, and saves individual 
    blob JSONs into 'out_dir' for the headless scanner to find.
    
    Args:
        scan_id: Scan ID for analysis (can also be in params)
        return_results: Deprecated. Always saves and returns results.
        params: Dictionary of analysis parameters from JSON config (must include 'out_dir' and 'scan_id')
    
    Returns:
        dict: Analysis results with scan data, blobs, and groups
    """
    # Initialize params if not provided
    if params is None:
        params = {}
    
    # Handle keyword-only arguments
    if scan_id is None:
        scan_id = params.get('scan_params', {}).get('scan_id') or params.get('scan_id')
    out_dir = params.get('export_params', {}).get('out_dir') or params.get('out_dir')
    print(f"\n[ANALYSIS] Starting analysis for Scan {scan_id} in {out_dir}")
    
    # Skip analysis if remote_seg is True (data sent to remote port, no TIFFs)
    remote_seg = params.get('remote_seg') or params.get('segmentation_params', {}).get('remote_seg', False)
    if remote_seg:
        print("[ANALYSIS] remote_seg=True, skipping local analysis (handled remotely)...")
        return {'error': 'Remote segmentation requested - no local results available'}
    
    # --- 1. Read Scan Parameters ---
    params_json_path = os.path.join(out_dir, f"scan_{scan_id}_params.json")
    step_size = 1.0
    x_start = 0.0
    y_start = 0.0

    if os.path.exists(params_json_path):
        with open(params_json_path, 'r') as f:
            params_data = json.load(f)
            step_size = params_data.get('step_size', 1.0)
            scan_input = params_data.get('start_doc', {}).get('scan', {}).get('scan_input', [])
            if len(scan_input) >= 4:
                x_start = scan_input[0]
                y_start = scan_input[3]

    # --- 2. Prepare Elements ---
    elem_list_of_lists = params.get("export_params", {}).get("elem_list", []) or params.get("elem_list", [])
    if not elem_list_of_lists:
        print("elem_list is empty.")
        return

    if isinstance(elem_list_of_lists[0], str):
        elem_list_of_lists = [elem_list_of_lists]

    # Flatten to get unique elements for loading
    all_elements = sorted(list(set(elem for sublist in elem_list_of_lists for elem in sublist)))
    
    # Load Tiff Paths
    tiff_paths = wait_for_element_tiffs(all_elements, out_dir)

    COLOR_ORDER = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'olive', 'yellow', 'brown', 'pink']
    precomputed_blobs = {color: {} for color in COLOR_ORDER}
    element_to_color = {element: COLOR_ORDER[i] for i, element in enumerate(all_elements) if i < len(COLOR_ORDER)}
    
    segmentation = params.get("segmentation_params", {})
    min_thresh = segmentation.get("min_threshold_intensity") or params.get("min_threshold_intensity")
    min_area = segmentation.get("min_threshold_area") or params.get("min_threshold_area")
    detection_method = segmentation.get("blob_detection_method") or params.get("blob_detection_method")
    
    # Method-specific parameters from JSON config
    detection_methods = params.get("detection_methods", {})
    simple_methods = detection_methods.get("simple", {})
    hough_methods = detection_methods.get("hough", {})
    watershed_methods = detection_methods.get("watershed", {})
    cellpose_methods = detection_methods.get("cellpose", {})
    connected_components_methods = detection_methods.get("connected_components", {})
    
    method_params = {
        # Simple blob detector parameters
        'max_threshold': simple_methods.get('max_threshold') or params.get('simple_max_threshold'),
        'max_area': simple_methods.get('max_area') or params.get('simple_max_area'),
        'threshold_step': simple_methods.get('threshold_step') or params.get('simple_threshold_step'),
        'filter_by_color': simple_methods.get('filter_by_color') or params.get('simple_filter_by_color'),
        'filter_by_circularity': simple_methods.get('filter_by_circularity') or params.get('simple_filter_by_circularity'),
        
        # Hough circle parameters
        'max_radius': hough_methods.get('max_radius') or params.get('hough_max_radius'),
        'dp': hough_methods.get('dp') or params.get('hough_dp'),
        'min_dist': hough_methods.get('min_dist') or params.get('hough_min_dist'),
        'param1': hough_methods.get('param1') or params.get('hough_param1'),
        'param2': hough_methods.get('param2') or params.get('hough_param2'),
        
        # Watershed parameters
        'min_distance': watershed_methods.get('min_distance') or params.get('watershed_min_distance'),
        'threshold_abs': watershed_methods.get('threshold_abs') or params.get('watershed_threshold_abs'),
        
        # Cellpose parameters
        'diameter': cellpose_methods.get('diameter') or params.get('cellpose_diameter'),
        'model_type': cellpose_methods.get('model_type') or params.get('cellpose_model_type'),
        'gpu': cellpose_methods.get('gpu') or params.get('cellpose_gpu'),
        'flow_threshold': cellpose_methods.get('flow_threshold') or params.get('cellpose_flow_threshold'),
        'cellprob_threshold': cellpose_methods.get('cellprob_threshold') or params.get('cellpose_cellprob_threshold'),
        'channels': cellpose_methods.get('channels') or params.get('cellpose_channels'),
        'min_diameter': cellpose_methods.get('min_diameter') or params.get('cellpose_min_diameter'),
        'max_diameter': cellpose_methods.get('max_diameter') or params.get('cellpose_max_diameter'),
        
        # Connected components parameters
        'connectivity': connected_components_methods.get('connectivity') or params.get('connected_components_connectivity')
    }
    
    # Filter out None values to avoid overriding method defaults
    method_params = {k: v for k, v in method_params.items() if v is not None}

    # --- 3. Blob Detection Loop ---
    for element in all_elements:
        if element not in tiff_paths: continue
        
        color = element_to_color.get(element)
        if not color: continue

        tiff_path = tiff_paths[element]
        print(f"Processing {tiff_path.name} ({color})")
        try:
            tiff_img = tiff.imread(str(tiff_path)).astype(np.float32)
            
            # Use configurable normalization and dilation parameters
            morphology = params.get('morphology_params', {})
            kernel_size = tuple(morphology.get('normalize_kernel_size') or params.get('normalize_kernel_size', [3, 3]))
            iterations = morphology.get('dilate_iterations') or params.get('dilate_iterations', 2)
            tiff_norm, tiff_dilated = normalize_and_dilate(tiff_img, kernel_size=kernel_size, iterations=iterations)

            b = detect_blobs(tiff_dilated, 
                             tiff_norm, min_thresh,
                             min_area, color, 
                             tiff_path.name, 
                             method=detection_method,
                             **method_params)
            
            precomputed_blobs[color][(min_thresh, min_area)] = b
        except Exception as e:
            print(f"❌ Error processing {tiff_path.name}: {e}")
            traceback.print_exc()

    # --- 4. Union & Export Loop ---
    all_results = {
        'scan_id': scan_id,
        'precomputed_blobs': precomputed_blobs,
        'groups': {},
        'tiff_paths': tiff_paths
    }
    
    for elem_list in elem_list_of_lists:
        group_name = "".join(elem_list)
        print(f"\n--- Processing Group: {group_name} (Elements: {len(elem_list)}) ---")

        group_blobs_for_union = {}
        for i, element in enumerate(elem_list):
            if i >= 3: break
            original_color = element_to_color.get(element)
            if not original_color: continue
            
            new_color = ['red', 'green', 'blue'][i]
            if original_color in precomputed_blobs:
                group_blobs_for_union[new_color] = precomputed_blobs[original_color]

        formatted_unions = {}
        
        if len(group_blobs_for_union) == 1:
            # Single element: process individual blobs without union formation
            print(f"[SINGLE ELEMENT] Processing individual blobs for {group_name}")
            color = list(group_blobs_for_union.keys())[0]
            blob_data = group_blobs_for_union[color]
            
            # Get blobs from the (min_thresh, min_area) key
            individual_blobs = list(blob_data.values())
            if individual_blobs:
                individual_blobs = individual_blobs[0]  # Get the blob list
                
                for idx, blob in enumerate(individual_blobs, start=1):
                    # Convert blob coordinates to real-world coordinates
                    image_center_x = blob['center'][0]
                    image_center_y = blob['center'][1]
                    real_center_x = x_start + (image_center_x * step_size)
                    real_center_y = y_start + (image_center_y * step_size)
                    
                    # Use blob size or default size
                    blob_size_um = blob.get('box_size', blob['radius'] * 2) * step_size
                    
                    box_name = f"Individual Blob {group_name} #{idx}"
                    formatted_unions[box_name] = {
                        "text": box_name,
                        "cx": real_center_x,
                        "cy": real_center_y,
                        "num_x": blob_size_um,
                        "num_y": blob_size_um,
                        # Preserve original blob info
                        "image_center": blob['center'],
                        "image_radius": blob['radius'],
                        "color": blob['color'],
                        "max_intensity": blob.get('max_intensity', 0),
                        "mean_intensity": blob.get('mean_intensity', 0)
                    }
                    
        elif len(group_blobs_for_union) >= 2:
            # Multiple elements: create union boxes
            print(f"[UNION MODE] Creating union boxes for {group_name}")
            unions = find_union_blobs(group_blobs_for_union, step_size, step_size, x_start, y_start)
            unions = merge_overlapping_boxes_dict(unions, overlap_thresh=segmentation.get('overlap_thresh', 0.5) or params.get('overlap_thresh', 0.5))

            for idx, union in unions.items():
                box_name = f"Union Box {group_name} #{idx.split('#')[-1].strip()}"
                formatted_unions[box_name] = {
                    "text": box_name,
                    "cx": union["real_center_um"][0], # Ensuring keys match headless expectations
                    "cy": union["real_center_um"][1],
                    "num_x": union["real_size_um"][0],
                    "num_y": union["real_size_um"][1],
                    # Preserve original verbose keys if needed for other logs
                    "image_center": union["center"],
                    "image_length": union["length"],
                    "real_center_um": union["real_center_um"],
                    "real_size_um": union["real_size_um"],
                }
        else:
            print(f"[SKIP] No valid blobs found for group {group_name}")
            continue

        # Save results if we have any formatted unions/blobs
        if formatted_unions:
            # Save the "Master" output JSON (Headless ignores this via startswith("unions_output"))
            out_json = Path(out_dir) / f"unions_output_{group_name}.json"
            # Convert to JSON-serializable format
            serializable_unions = make_json_serializable(formatted_unions)
            with open(out_json, "w") as f:
                json.dump(serializable_unions, f, indent=2)
            
            # Save the INDIVIDUAL JSONs (Headless finds these)
            # This function must create files that do NOT start with "unions_output"  
            save_each_blob_as_individual_scan(formatted_unions, out_dir)
            
            # Initialize results dictionary for this group
            all_results['groups'][group_name] = {
                'formatted_unions': formatted_unions,
                'group_blobs_for_union': group_blobs_for_union,
                'element_count': len(elem_list),
                'processing_mode': 'individual' if len(group_blobs_for_union) == 1 else 'union'
            }
            
            # Create and save fine scans table (for remote server compatibility)
            try:
                fine_scans_table_path = Path(out_dir) / f"fine_scans_table_{group_name}.csv"
                print(f"[TABLE] Creating fine scans table from {len(formatted_unions)} formatted unions...")
                table = formatted_unions_to_table(formatted_unions, save_to=str(fine_scans_table_path))
                if not table.empty:
                    # Store table for passing to fine scans submission
                    if 'fine_scans_tables' not in all_results:
                        all_results['fine_scans_tables'] = {}
                    all_results['fine_scans_tables'][group_name] = table
                    all_results['groups'][group_name]['fine_scans_table'] = table.to_dict()
                    print(f"[TABLE] ✅ Table saved and stored in results")
                else:
                    print(f"[TABLE] ⚠️ Table is empty, skipping storage in results")
                    all_results['groups'][group_name]['fine_scans_table'] = {}
            except Exception as e:
                print(f"⚠️ Error creating fine scans table for {group_name}: {type(e).__name__}: {e}")
                traceback.print_exc()
                all_results['groups'][group_name]['fine_scans_table'] = {}
            # Add union data for multi-element groups
            if len(group_blobs_for_union) >= 2:
                all_results['groups'][group_name]['unions'] = unions

    # --- 5. Visualization ---
    if tiff_paths:
        group_blobs_vis = {}
        for i, element in enumerate(elem_list):
            if i >= len(COLOR_ORDER): break
            orig = element_to_color.get(element)
            if orig: group_blobs_vis[COLOR_ORDER[i]] = precomputed_blobs[orig]

        create_rgb_tiff(tiff_paths, out_dir, elem_list, group_name)
        create_all_elements_tiff(tiff_paths, out_dir, elem_list, group_blobs_vis, group_name)
        
        # Plot analysis results with bounding boxes
        # Collect formatted unions for plotting
        formatted_unions_dict = {}
        for elem_list in elem_list_of_lists:
            group_name_plot = "".join(elem_list)
            # Get formatted_unions from all_results
            if 'groups' in all_results and group_name_plot in all_results['groups']:
                formatted_unions_dict[group_name_plot] = all_results['groups'][group_name_plot]['formatted_unions']
        
        if formatted_unions_dict:
            plot_analysis_results(tiff_paths, elem_list, formatted_unions_dict, out_dir)

    print("[ANALYSIS] Done.")
    
    return all_results


def analyze_data_from_arrays(element_arrays, params):
    """
    Analyze XRF data from a 3D numpy array instead of loading TIFF files.
    Performs equivalent processing to analyze_data_local() but with in-memory arrays.
    
    Args:
        element_arrays (np.ndarray): 3D array of shape (n_elements, height, width) containing XRF images
        params (dict): Analysis parameters dictionary with element order info.
                      Must contain 'elem_list' or 'export_params.elem_list' specifying element names/order
                      and segmentation/detection parameters
    
    Returns:
        dict: Analysis results containing:
              - scan_id: Scan ID used for analysis
              - precomputed_blobs: Detected blobs by color
              - groups: Analysis results grouped by element combinations
              - element_arrays: Reference to input arrays used
              
    Example:
    --------
    >>> # 3D array: (n_elements, height, width)
    >>> element_arrays = np.random.rand(3, 100, 100).astype(np.float32) * 1000
    >>> params = {
    ...     'scan_id': 12345,
    ...     'elem_list': ['Fe', 'Cu', 'Ni'],  # Element order matching array indices
    ...     'segmentation_params': {...}
    ... }
    >>> results = analyze_data_from_arrays(element_arrays, params)
    """
    # Extract parameters
    scan_id = params.get('scan_params', {}).get('scan_id') or params.get('scan_id')
    
    print(f"\n[ANALYSIS-ARRAYS] Analyzing 3D array for Scan {scan_id}")
    print(f"[ANALYSIS-ARRAYS] Array shape: {element_arrays.shape}")
    
    # Check input array
    if element_arrays is None or element_arrays.size == 0:
        print("[ANALYSIS-ARRAYS] Error: element_arrays is empty")
        return {'error': 'No arrays provided'}
    
    if len(element_arrays.shape) != 3:
        print(f"[ANALYSIS-ARRAYS] Error: expected 3D array, got shape {element_arrays.shape}")
        return {'error': 'Array must be 3D (n_elements, height, width)'}
    
    # Get element order from params
    elem_list_of_lists = params.get("export_params", {}).get("elem_list", []) or params.get("elem_list", [])
    if not elem_list_of_lists:
        print("[ANALYSIS-ARRAYS] Error: elem_list not found in params")
        return {'error': 'Element list not provided'}
    
    if isinstance(elem_list_of_lists[0], str):
        elem_list_of_lists = [elem_list_of_lists]
    
    all_elements = sorted(list(set(elem for sublist in elem_list_of_lists for elem in sublist)))
    print(f"[ANALYSIS-ARRAYS] Element order from params: {all_elements}")
    
    # Verify array has correct number of elements
    if element_arrays.shape[0] != len(all_elements):
        print(f"[ANALYSIS-ARRAYS] Warning: array has {element_arrays.shape[0]} elements but {len(all_elements)} expected")
        return {'error': f'Array dimension mismatch: {element_arrays.shape[0]} != {len(all_elements)}'}
    
    # Create dict mapping element names to 2D arrays
    element_array_dict = {elem: element_arrays[i].astype(np.float32) 
                         for i, elem in enumerate(all_elements)}
    
    # --- 1. Extract Analysis Parameters ---
    segmentation = params.get("segmentation_params", {})
    min_thresh = segmentation.get("min_threshold_intensity") or params.get("min_threshold_intensity")
    min_area = segmentation.get("min_threshold_area") or params.get("min_threshold_area")
    detection_method = segmentation.get("blob_detection_method") or params.get("blob_detection_method")
    
    # Spatial parameters (with defaults)
    step_size = params.get('step_size', 1.0)
    x_start = params.get('x_start', 0.0)
    y_start = params.get('y_start', 0.0)
    
    print(f"[ANALYSIS-ARRAYS] Detection method: {detection_method}, threshold: {min_thresh}, area: {min_area}")
    
    # Method-specific parameters
    detection_methods = params.get("detection_methods", {})
    simple_methods = detection_methods.get("simple", {})
    hough_methods = detection_methods.get("hough", {})
    watershed_methods = detection_methods.get("watershed", {})
    cellpose_methods = detection_methods.get("cellpose", {})
    connected_components_methods = detection_methods.get("connected_components", {})
    
    method_params = {
        'max_threshold': simple_methods.get('max_threshold') or params.get('simple_max_threshold'),
        'max_area': simple_methods.get('max_area') or params.get('simple_max_area'),
        'threshold_step': simple_methods.get('threshold_step') or params.get('simple_threshold_step'),
        'filter_by_color': simple_methods.get('filter_by_color') or params.get('simple_filter_by_color'),
        'filter_by_circularity': simple_methods.get('filter_by_circularity') or params.get('simple_filter_by_circularity'),
        'max_radius': hough_methods.get('max_radius') or params.get('hough_max_radius'),
        'dp': hough_methods.get('dp') or params.get('hough_dp'),
        'min_dist': hough_methods.get('min_dist') or params.get('hough_min_dist'),
        'param1': hough_methods.get('param1') or params.get('hough_param1'),
        'param2': hough_methods.get('param2') or params.get('hough_param2'),
        'min_distance': watershed_methods.get('min_distance') or params.get('watershed_min_distance'),
        'threshold_abs': watershed_methods.get('threshold_abs') or params.get('watershed_threshold_abs'),
        'diameter': cellpose_methods.get('diameter') or params.get('cellpose_diameter'),
        'model_type': cellpose_methods.get('model_type') or params.get('cellpose_model_type'),
        'gpu': cellpose_methods.get('gpu') or params.get('cellpose_gpu'),
        'flow_threshold': cellpose_methods.get('flow_threshold') or params.get('cellpose_flow_threshold'),
        'cellprob_threshold': cellpose_methods.get('cellprob_threshold') or params.get('cellpose_cellprob_threshold'),
        'channels': cellpose_methods.get('channels') or params.get('cellpose_channels'),
        'min_diameter': cellpose_methods.get('min_diameter') or params.get('cellpose_min_diameter'),
        'max_diameter': cellpose_methods.get('max_diameter') or params.get('cellpose_max_diameter'),
        'connectivity': connected_components_methods.get('connectivity') or params.get('connected_components_connectivity')
    }
    method_params = {k: v for k, v in method_params.items() if v is not None}
    
    # Prepare colors
    COLOR_ORDER = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'olive', 'yellow', 'brown', 'pink']
    precomputed_blobs = {color: {} for color in COLOR_ORDER}
    element_to_color = {element: COLOR_ORDER[i] for i, element in enumerate(all_elements) if i < len(COLOR_ORDER)}
    
    # --- 2. Blob Detection from Arrays ---
    morphology = params.get('morphology_params', {})
    kernel_size = tuple(morphology.get('normalize_kernel_size') or params.get('normalize_kernel_size', [3, 3]))
    iterations = morphology.get('dilate_iterations') or params.get('dilate_iterations', 2)
    
    for element in all_elements:
        color = element_to_color.get(element)
        if not color:
            continue
        
        print(f"[ANALYSIS-ARRAYS] Processing {element} ({color})")
        try:
            img = element_array_dict[element]
            
            
            # Detect blobs
            b = detect_blobs(img, 
                             img, 
                             min_thresh,
                             min_area, 
                             color, 
                             f"{element}_array", 
                             method=detection_method,
                             **method_params)
            
            precomputed_blobs[color][(min_thresh, min_area)] = b
            print(f"[ANALYSIS-ARRAYS] Found {len(b)} blobs for {element}")
        except Exception as e:
            print(f"[ANALYSIS-ARRAYS] ❌ Error processing {element}: {e}")
            traceback.print_exc()
    
    # --- 3. Union & Table Generation Loop ---
    fine_scans_tables = {}
    
    # --- 3. Union & Table Generation Loop ---
    fine_scans_tables = {}
    
    for elem_list in elem_list_of_lists:
        group_name = "".join(elem_list)
        print(f"\n[ANALYSIS-ARRAYS] Processing group: {group_name}")
        
        group_blobs_for_union = {}
        for i, element in enumerate(elem_list):
            if i >= 3:
                break
            original_color = element_to_color.get(element)
            if not original_color:
                continue
            
            new_color = ['red', 'green', 'blue'][i]
            if original_color in precomputed_blobs:
                group_blobs_for_union[new_color] = precomputed_blobs[original_color]
        
        formatted_unions = {}
        
        if len(group_blobs_for_union) == 1:
            # Single element: process individual blobs
            print(f"[ANALYSIS-ARRAYS] Single element mode for {group_name}")
            color = list(group_blobs_for_union.keys())[0]
            blob_data = group_blobs_for_union[color]
            
            individual_blobs = list(blob_data.values())
            if individual_blobs:
                individual_blobs = individual_blobs[0]
                
                for idx, blob in enumerate(individual_blobs, start=1):
                    image_center_x = blob['center'][0]
                    image_center_y = blob['center'][1]
                    real_center_x = x_start + (image_center_x * step_size)
                    real_center_y = y_start + (image_center_y * step_size)
                    blob_size_um = blob.get('box_size', blob['radius'] * 2) * step_size
                    
                    box_name = f"Individual Blob {group_name} #{idx}"
                    formatted_unions[box_name] = {
                        "text": box_name,
                        "cx": real_center_x,
                        "cy": real_center_y,
                        "num_x": blob_size_um,
                        "num_y": blob_size_um,
                        "image_center": blob['center'],
                        "image_radius": blob['radius'],
                        "color": blob['color'],
                        "max_intensity": blob.get('max_intensity', 0),
                        "mean_intensity": blob.get('mean_intensity', 0)
                    }
        
        elif len(group_blobs_for_union) >= 2:
            # Multiple elements: create union boxes
            print(f"[ANALYSIS-ARRAYS] Union mode for {group_name}")
            unions = find_union_blobs(group_blobs_for_union, step_size, step_size, x_start, y_start)
            unions = merge_overlapping_boxes_dict(unions, overlap_thresh=segmentation.get('overlap_thresh', 0.5) or params.get('overlap_thresh', 0.5))
            
            for idx, union in unions.items():
                box_name = f"Union Box {group_name} #{idx.split('#')[-1].strip()}"
                formatted_unions[box_name] = {
                    "text": box_name,
                    "cx": union["real_center_um"][0],
                    "cy": union["real_center_um"][1],
                    "num_x": union["real_size_um"][0],
                    "num_y": union["real_size_um"][1],
                    "image_center": union["center"],
                    "image_length": union["length"],
                    "real_center_um": union["real_center_um"],
                    "real_size_um": union["real_size_um"],
                }
        else:
            print(f"[ANALYSIS-ARRAYS] Skipping {group_name} - no blobs found")
            continue
        
        # Create fine scans table for this group
        if formatted_unions:
            try:
                table = formatted_unions_to_table(formatted_unions, save_to=None)
                if not table.empty:
                    fine_scans_tables[group_name] = table
                    print(f"[ANALYSIS-ARRAYS] Created fine scans table with {len(table)} rows")
            except Exception as e:
                print(f"[ANALYSIS-ARRAYS] Warning: Could not create fine scans table: {e}")
    
    print("[ANALYSIS-ARRAYS] Complete.")
    return fine_scans_tables


def analyze_data_get_fine_scans_table(scan_id=None, 
                                      params=None):
    """
    Step 2B: Analysis with table return.
    Same analysis as analyze_data_local but returns fine scan tables for follow-up
    without saving files or generating plots.
    
    Args:
        scan_id: Scan ID for this analysis
        params: Dictionary of analysis parameters from JSON config (must include 'out_dir' and 'scan_id')
    
    Returns:
        dict: Mapping of group_name -> pandas DataFrame with fine scan parameters
              Columns: label, cx, cy, num_x, num_y, color, element, max_intensity, mean_intensity
    """
    # Initialize params if not provided
    if params is None:
        params = {}
    
    # Handle keyword-only arguments
    if scan_id is None:
        scan_id = params.get('scan_params', {}).get('scan_id') or params.get('scan_id')
    out_dir = params.get('export_params', {}).get('out_dir') or params.get('out_dir')
    print(f"\n[ANALYSIS-TABLE] Starting analysis for Scan {scan_id} in {out_dir}")
    
    # Skip analysis if remote_seg is True
    remote_seg = params.get('remote_seg') or params.get('segmentation_params', {}).get('remote_seg', False)
    if remote_seg:
        print("[ANALYSIS-TABLE] remote_seg=True, skipping (no TIFFs available)...")
        return {}
    
    # --- 1. Read Scan Parameters ---
    params_json_path = os.path.join(out_dir, f"scan_{scan_id}_params.json")
    step_size = 1.0
    x_start = 0.0
    y_start = 0.0

    if os.path.exists(params_json_path):
        with open(params_json_path, 'r') as f:
            params_data = json.load(f)
            step_size = params_data.get('step_size', 1.0)
            scan_input = params_data.get('start_doc', {}).get('scan', {}).get('scan_input', [])
            if len(scan_input) >= 4:
                x_start = scan_input[0]
                y_start = scan_input[3]

    # --- 2. Prepare Elements ---
    elem_list_of_lists = params.get("export_params", {}).get("elem_list", []) or params.get("elem_list", [])
    if not elem_list_of_lists:
        print("elem_list is empty.")
        return {}

    if isinstance(elem_list_of_lists[0], str):
        elem_list_of_lists = [elem_list_of_lists]

    # Flatten to get unique elements for loading
    all_elements = sorted(list(set(elem for sublist in elem_list_of_lists for elem in sublist)))
    
    # Load Tiff Paths
    tiff_paths = wait_for_element_tiffs(all_elements, out_dir)

    COLOR_ORDER = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'olive', 'yellow', 'brown', 'pink']
    precomputed_blobs = {color: {} for color in COLOR_ORDER}
    element_to_color = {element: COLOR_ORDER[i] for i, element in enumerate(all_elements) if i < len(COLOR_ORDER)}
    
    segmentation = params.get("segmentation_params", {})
    detection_methods = params.get("detection_methods", {})
    simple_methods = detection_methods.get("simple", {})
    hough_methods = detection_methods.get("hough", {})
    watershed_methods = detection_methods.get("watershed", {})
    cellpose_methods = detection_methods.get("cellpose", {})
    connected_components_methods = detection_methods.get("connected_components", {})
    
    min_thresh = segmentation.get("min_threshold_intensity") or params.get("min_threshold_intensity")
    min_area = segmentation.get("min_threshold_area") or params.get("min_threshold_area")
    detection_method = segmentation.get("blob_detection_method") or params.get("blob_detection_method")
    
    # Method-specific parameters from JSON config
    method_params = {
        # Simple blob detector parameters
        'max_threshold': simple_methods.get('max_threshold') or params.get('simple_max_threshold'),
        'max_area': simple_methods.get('max_area') or params.get('simple_max_area'),
        'threshold_step': simple_methods.get('threshold_step') or params.get('simple_threshold_step'),
        'filter_by_color': simple_methods.get('filter_by_color') or params.get('simple_filter_by_color'),
        'filter_by_circularity': simple_methods.get('filter_by_circularity') or params.get('simple_filter_by_circularity'),
        
        # Hough circle parameters
        'max_radius': hough_methods.get('max_radius') or params.get('hough_max_radius'),
        'dp': hough_methods.get('dp') or params.get('hough_dp'),
        'min_dist': hough_methods.get('min_dist') or params.get('hough_min_dist'),
        'param1': hough_methods.get('param1') or params.get('hough_param1'),
        'param2': hough_methods.get('param2') or params.get('hough_param2'),
        
        # Watershed parameters
        'min_distance': watershed_methods.get('min_distance') or params.get('watershed_min_distance'),
        'threshold_abs': watershed_methods.get('threshold_abs') or params.get('watershed_threshold_abs'),
        
        # Cellpose parameters
        'diameter': cellpose_methods.get('diameter') or params.get('cellpose_diameter'),
        'model_type': cellpose_methods.get('model_type') or params.get('cellpose_model_type'),
        'gpu': cellpose_methods.get('gpu') or params.get('cellpose_gpu'),
        'flow_threshold': cellpose_methods.get('flow_threshold') or params.get('cellpose_flow_threshold'),
        'cellprob_threshold': cellpose_methods.get('cellprob_threshold') or params.get('cellpose_cellprob_threshold'),
        'channels': cellpose_methods.get('channels') or params.get('cellpose_channels'),
        'min_diameter': cellpose_methods.get('min_diameter') or params.get('cellpose_min_diameter'),
        'max_diameter': cellpose_methods.get('max_diameter') or params.get('cellpose_max_diameter'),
        
        # Connected components parameters
        'connectivity': connected_components_methods.get('connectivity') or params.get('connected_components_connectivity')
    }
    
    # Filter out None values to avoid overriding method defaults
    method_params = {k: v for k, v in method_params.items() if v is not None}

    # --- 3. Blob Detection Loop ---
    for element in all_elements:
        if element not in tiff_paths: continue
        
        color = element_to_color.get(element)
        if not color: continue

        tiff_path = tiff_paths[element]
        print(f"Processing {tiff_path.name} ({color})")
        try:
            tiff_img = tiff.imread(str(tiff_path)).astype(np.float32)
            
            # Use configurable normalization and dilation parameters
            morphology = params.get('morphology_params', {})
            kernel_size = tuple(morphology.get('normalize_kernel_size') or params.get('normalize_kernel_size', [3, 3]))
            iterations = morphology.get('dilate_iterations') or params.get('dilate_iterations', 2)
            tiff_norm, tiff_dilated = normalize_and_dilate(tiff_img, 
                                                           kernel_size=kernel_size, iterations=iterations)

            b = detect_blobs(tiff_dilated, 
                             tiff_norm, min_thresh,
                             min_area, color, 
                             tiff_path.name, 
                             method=detection_method,
                             **method_params)
            
            precomputed_blobs[color][(min_thresh, min_area)] = b
        except Exception as e:
            print(f"❌ Error processing {tiff_path.name}: {e}")
            traceback.print_exc()

    # --- 4. Union & Table Generation Loop ---
    fine_scans_tables = {}
    
    for elem_list in elem_list_of_lists:
        group_name = "".join(elem_list)
        print(f"\n--- Processing Group: {group_name} (Elements: {len(elem_list)}) ---")

        group_blobs_for_union = {}
        for i, element in enumerate(elem_list):
            if i >= 3: break
            original_color = element_to_color.get(element)
            if not original_color: continue
            
            new_color = ['red', 'green', 'blue'][i]
            if original_color in precomputed_blobs:
                group_blobs_for_union[new_color] = precomputed_blobs[original_color]

        formatted_unions = {}
        
        if len(group_blobs_for_union) == 1:
            # Single element: process individual blobs without union formation
            print(f"[SINGLE ELEMENT] Processing individual blobs for {group_name}")
            color = list(group_blobs_for_union.keys())[0]
            blob_data = group_blobs_for_union[color]
            
            # Get blobs from the (min_thresh, min_area) key
            individual_blobs = list(blob_data.values())
            if individual_blobs:
                individual_blobs = individual_blobs[0]  # Get the blob list
                
                for idx, blob in enumerate(individual_blobs, start=1):
                    # Convert blob coordinates to real-world coordinates
                    image_center_x = blob['center'][0]
                    image_center_y = blob['center'][1]
                    real_center_x = x_start + (image_center_x * step_size)
                    real_center_y = y_start + (image_center_y * step_size)
                    
                    # Use blob size or default size
                    blob_size_um = blob.get('box_size', blob['radius'] * 2) * step_size
                    
                    box_name = f"Individual Blob {group_name} #{idx}"
                    formatted_unions[box_name] = {
                        "text": box_name,
                        "cx": real_center_x,
                        "cy": real_center_y,
                        "num_x": blob_size_um,
                        "num_y": blob_size_um,
                        # Preserve original blob info
                        "image_center": blob['center'],
                        "image_radius": blob['radius'],
                        "color": blob['color'],
                        "max_intensity": blob.get('max_intensity', 0),
                        "mean_intensity": blob.get('mean_intensity', 0)
                    }
                    
        elif len(group_blobs_for_union) >= 2:
            # Multiple elements: create union boxes
            print(f"[UNION MODE] Creating union boxes for {group_name}")
            unions = find_union_blobs(group_blobs_for_union, step_size, step_size, x_start, y_start)
            unions = merge_overlapping_boxes_dict(unions, overlap_thresh=segmentation.get('overlap_thresh', 0.5) or params.get('overlap_thresh', 0.5))

            for idx, union in unions.items():
                box_name = f"Union Box {group_name} #{idx.split('#')[-1].strip()}"
                formatted_unions[box_name] = {
                    "text": box_name,
                    "cx": union["real_center_um"][0],
                    "cy": union["real_center_um"][1],
                    "num_x": union["real_size_um"][0],
                    "num_y": union["real_size_um"][1],
                    # Preserve original verbose keys if needed for other logs
                    "image_center": union["center"],
                    "image_length": union["length"],
                    "real_center_um": union["real_center_um"],
                    "real_size_um": union["real_size_um"],
                }
        else:
            print(f"[SKIP] No valid blobs found for group {group_name}")
            continue

        # Convert to table if we have formatted unions
        if formatted_unions:
            try:
                table = formatted_unions_to_table(formatted_unions)
                fine_scans_tables[group_name] = table
                print(f"✅ Created fine scans table for {group_name}: {len(table)} rows")
            except Exception as e:
                print(f"❌ Error creating table for {group_name}: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                fine_scans_tables[group_name] = pd.DataFrame()  # Empty dataframe as fallback

    print("[ANALYSIS-TABLE] Done.")
    
    # Return all tables
    return fine_scans_tables


def plot_image_with_boxes(image, formatted_unions, title="Analysis Results", save_path=None, show_plot=False):
    """
    Plot image with bounding boxes overlay.
    
    Args:
        image: numpy array of the image
        formatted_unions: dict with union/blob data containing 'image_center' and 'image_length' (or 'box_x', 'box_y', 'box_size')
        title: plot title
        save_path: optional path to save the figure
        show_plot: whether to display the plot (default: False for headless mode)
    """
    import matplotlib
    # Use non-interactive backend if not displaying
    if not show_plot:
        matplotlib.use('Agg')
    
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Display image
    if len(image.shape) == 2:
        ax.imshow(image, cmap='gray')
    else:
        ax.imshow(image)
    
    # Draw bounding boxes
    color_cycle = plt.cm.tab20(range(len(formatted_unions)))
    for idx, (name, info) in enumerate(formatted_unions.items()):
        color = color_cycle[idx % len(color_cycle)]
        
        # Try different key formats
        if 'image_center' in info and 'image_length' in info:
            cx, cy = info['image_center']
            size = info['image_length']
            x = cx - size / 2
            y = cy - size / 2
        elif 'box_x' in info and 'box_y' in info and 'box_size' in info:
            x = info['box_x']
            y = info['box_y']
            size = info['box_size']
        else:
            continue
        
        # Draw rectangle
        rect = patches.Rectangle((x, y), size, size, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        # Add label
        ax.text(x, y - 5, name, fontsize=8, color=color, weight='bold', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax.set_title(title, fontsize=14, weight='bold')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Plot saved to: {save_path}")
    
    if show_plot:
        plt.tight_layout()
        plt.show()
    else:
        plt.close(fig)


def plot_analysis_results(tiff_paths, elem_list, formatted_unions_dict, out_dir, group_name=None):
    """
    Plot analysis results with bounding boxes for each element.
    
    Args:
        tiff_paths: dict of element -> TIFF path
        elem_list: list of elements
        formatted_unions_dict: dict of group_name -> formatted_unions
        out_dir: output directory for saving plots
        group_name: specific group to plot (if None, plots all groups)
    """
    import matplotlib.pyplot as plt
    
    if group_name:
        groups_to_plot = {group_name: formatted_unions_dict.get(group_name, {})}
    else:
        groups_to_plot = formatted_unions_dict
    
    for gname, formatted_unions in groups_to_plot.items():
        if not formatted_unions:
            print(f"⏭️ Skipping {gname}: no unions/blobs found")
            continue
        
        # Get the first element's image for visualization
        first_element = None
        for elem in elem_list:
            if elem in tiff_paths:
                first_element = elem
                break
        
        if not first_element:
            print(f"❌ No TIFF found for visualization in {gname}")
            continue
        
        try:
            tiff_path = tiff_paths[first_element]
            image = tiff.imread(str(tiff_path)).astype(np.float32)
            
            # Normalize for display
            image_norm = cv2.normalize(np.nan_to_num(image), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # Create plot
            title = f"Analysis Results - Group {gname} (Element: {first_element})"
            save_path = Path(out_dir) / f"analysis_plot_{gname}.png"
            
            plot_image_with_boxes(image_norm, formatted_unions, title=title, save_path=str(save_path))
            
        except Exception as e:
            print(f"❌ Error plotting {gname}: {e}")
            traceback.print_exc()


def submit_fine_scans_to_queue(json_path, scan_id, out_dir, execution_params, fine_scans_tables=None):
    """
    Step 3: Queue Submission.
    Only actually queues if mode == 'real'. 
    Offline and Sim will just print.
    
    Args:
        json_path (str): Path to JSON config file
        scan_id (int): Scan ID for fine scans
        out_dir (str): Output directory
        execution_params (dict): Execution mode and flags
        fine_scans_tables (dict): Pre-computed fine scans tables by group_name (optional)
    """
    # Get mode from execution_params
    mode = str(execution_params.get('mode', 'simulation')).lower()
    is_real = (mode == 'real')
    
    print(f"\n[QUEUE] Processing fine scans in: {out_dir}")
    
    if is_real:
        # Process each table if provided
        if fine_scans_tables:
            for group_name, table in fine_scans_tables.items():
                print(f"[QUEUE] Submitting {len(table)} fine scans for group '{group_name}'")
                headless_send_queue_fine_scan(json_path, fine_scans_table=table)
        else:
            # Fallback: load from JSON config or CSV files
            headless_send_queue_fine_scan(json_path)
    else:
        # Covers both Sim and Offline
        print(f"[{'OFFLINE' if mode=='offline' else 'SIM'}] Skipping actual queue submission.")
        if fine_scans_tables:
            print(f"[{'OFFLINE' if mode=='offline' else 'SIM'}] Would queue {sum(len(t) for t in fine_scans_tables.values())} fine scans from {len(fine_scans_tables)} groups")
        print(f"Would call: headless_send_queue_fine_scan('{json_path}')")

def run_fine_scans(is_real): 
    """
    Step 4: Start the Queue.
    """
    if is_real:
        st = RM.status()
        if st['items_in_queue'] != 0 and st['manager_state'] == 'idle':
            RM.queue_start()
            print('[QSERVER] Queue started')
        else: 
            print('[QSERVER] Queue waiting or already running')
        
        wait_for_queue_done()
    else:
        print("[SIM] Would check RM.status() and start queue.")




def load_and_queue(json_path, target_id=None, 
                   remote_seg=False, proceed_fine_scans=True):
    """
    Main workflow function supporting multiple modes.
    Mode is specified in JSON config file as 'mode' key:
    - 'simulation': Simulation mode
    - 'real': Real scanning mode 
    - 'offline': Offline mode (use existing scan)
    - 'analysis-only': Analysis-only mode (use existing scan, return results)
    """
    
    # 0) Clear caches
    if 'remote_handler' in globals():
        remote_handler.clear_cache() 

    # 1) Load JSON
    with open(json_path, 'r') as f:
        params = json.load(f)

    # 2) ROI & Calc Logic
    roi_file = params.pop('roi_positions_file', None)
    if roi_file:
        if not os.path.isfile(roi_file):
            raise FileNotFoundError(f"ROI file not found: {roi_file}")
        with open(roi_file, 'r') as rf:
            params['roi_positions'] = json.load(rf)
    elif isinstance(params.get('roi_positions') or params.get('scan_params', {}).get('roi_positions'), str) and os.path.isfile(params.get('roi_positions') or params.get('scan_params', {}).get('roi_positions', '')):
        with open(params['roi_positions'], 'r') as rf:
            params['roi_positions'] = json.load(rf)

    if 'step_size' in params:
        step = params.pop('step_size')
        params['mot1_n'] = int(abs(params['mot1_e'] - params['mot1_s']) / step)
        params['mot2_n'] = int(abs(params['mot2_e'] - params['mot2_s']) / step)

    # 3) Get mode from JSON config
    mode = str(params.get('execution_params', {}).get('mode', 'simulation')).lower()
    is_real = (mode == 'real')
    is_sim  = (mode == 'simulation')
    is_offline = (mode == 'offline')
    is_analysis_only = (mode == 'analysis-only')
    
    # Map mode to legacy real_test for backward compatibility with other functions
    mode_map = {'simulation': 0, 'real': 1, 'offline': 2, 'analysis-only': 3}
    params['real_test'] = mode_map.get(mode, 0)
    params['remote_seg'] = remote_seg
    
    # 3.1) Add default segmentation parameters if not present
    segmentation = params.get('segmentation_params', {})
    morphology = params.get('morphology_params', {})
    detection_methods = params.get('detection_methods', {})
    simple_methods = detection_methods.get('simple', {})
    hough_methods = detection_methods.get('hough', {})
    watershed_methods = detection_methods.get('watershed', {})
    cellpose_methods = detection_methods.get('cellpose', {})
    connected_components_methods = detection_methods.get('connected_components', {})
    contours_methods = detection_methods.get('contours', {})
    
    segmentation_defaults = {
        # Basic detection parameters
        'min_threshold_intensity': segmentation.get('min_threshold_intensity', params.get('min_threshold_intensity', 50)),
        'min_threshold_area': segmentation.get('min_threshold_area', params.get('min_threshold_area', 100)),
        'blob_detection_method': segmentation.get('blob_detection_method', params.get('blob_detection_method', 'simple')),
        'overlap_thresh': segmentation.get('overlap_thresh', params.get('overlap_thresh', 0.5)),
        
        # Normalization and morphology parameters
        'normalize_kernel_size': morphology.get('normalize_kernel_size', params.get('normalize_kernel_size', [3, 3])),
        'dilate_iterations': morphology.get('dilate_iterations', params.get('dilate_iterations', 2)),
        'blur_kernel': morphology.get('blur_kernel', params.get('blur_kernel', [3, 3])),
        
        # Method-specific parameters for simple detection
        'simple_max_threshold': simple_methods.get('max_threshold', params.get('simple_max_threshold', 255)),
        'simple_max_area': simple_methods.get('max_area', params.get('simple_max_area', 1600)),
        'simple_threshold_step': simple_methods.get('threshold_step', params.get('simple_threshold_step', 2)),
        'simple_filter_by_color': simple_methods.get('filter_by_color', params.get('simple_filter_by_color', False)),
        'simple_filter_by_circularity': simple_methods.get('filter_by_circularity', params.get('simple_filter_by_circularity', False)),
        
        # Hough circle detection parameters
        'hough_max_radius': hough_methods.get('max_radius', params.get('hough_max_radius', 40)),
        'hough_dp': hough_methods.get('dp', params.get('hough_dp', 1)),
        'hough_min_dist': hough_methods.get('min_dist', params.get('hough_min_dist', 20)),
        'hough_param1': hough_methods.get('param1', params.get('hough_param1', 50)),
        'hough_param2': hough_methods.get('param2', params.get('hough_param2', 30)),
        
        # Watershed segmentation parameters
        'watershed_min_distance': watershed_methods.get('min_distance', params.get('watershed_min_distance', 10)),
        'watershed_threshold_abs': watershed_methods.get('threshold_abs', params.get('watershed_threshold_abs', 0.3)),
        
        # Cellpose parameters
        'cellpose_diameter': cellpose_methods.get('diameter', params.get('cellpose_diameter', 8)),
        'cellpose_model_type': cellpose_methods.get('model_type', params.get('cellpose_model_type', 'cyto3')),
        'cellpose_gpu': cellpose_methods.get('gpu', params.get('cellpose_gpu', False)),
        'cellpose_flow_threshold': cellpose_methods.get('flow_threshold', params.get('cellpose_flow_threshold', 0.4)),
        'cellpose_cellprob_threshold': cellpose_methods.get('cellprob_threshold', params.get('cellpose_cellprob_threshold', 0.0)),
        'cellpose_channels': cellpose_methods.get('channels', params.get('cellpose_channels', [0, 0])),
        'cellpose_min_diameter': cellpose_methods.get('min_diameter', params.get('cellpose_min_diameter', 2)),
        'cellpose_max_diameter': cellpose_methods.get('max_diameter', params.get('cellpose_max_diameter', float('100'))),
        
        # Connected components parameters
        'connected_components_connectivity': connected_components_methods.get('connectivity', params.get('connected_components_connectivity', 8)),
        
        # Contour detection parameters
        'contours_mode': contours_methods.get('mode', params.get('contours_mode', 'external')),
        'contours_method': contours_methods.get('method', params.get('contours_method', 'simple'))
    }
    
    # Update params with segmentation defaults
    for key, default_value in segmentation_defaults.items():
        if key not in params:
            params[key] = default_value
    
    # IMPORTANT: If offline or analysis-only mode, scan_id is mandatory.
    if target_id is not None:
        params['scan_id'] = target_id

    elif (is_offline or is_analysis_only) and 'scan_id' not in params['scan_params']:
        print(f"[WARNING] Running in '{mode}' mode but no scan_id provided.")
        # You might want to raise an error or rely on it being in the JSON
    
    # For analysis-only mode, force local analysis and skip fine scans
    if is_analysis_only:
        remote_seg = False
        params['remote_seg'] = False
        proceed_fine_scans = False
    
    # 4) EXECUTE
    print(f"--- Workflow: {os.path.basename(json_path)} (Mode: {mode.capitalize()}) ---")

    # A. Submit / Export (skip for analysis-only mode)
    if is_analysis_only:
        # For analysis-only mode, use the target_id directly and create output directory
        scan_id = target_id
        data_wd = params.get('data_wd', '/data/users/current_user')
        out_dir = os.path.join(data_wd, f"automap_{scan_id}")
        os.makedirs(out_dir, exist_ok=True)
        print(f"[ANALYSIS-ONLY] Using existing scan {scan_id}, output dir: {out_dir}")
    else:
        scan_id, out_dir = submit_and_export(
            params['execution_params'],
            params['scan_params'],
            params['export_params'],
            params.get('segmentation_params')
        )
    
    # Update params with scan_id and out_dir
    params['scan_id'] = scan_id
    params['out_dir'] = out_dir
    
    # B. Analyze
    analysis_results = None
    fine_scans_tables = None
    if remote_seg:
        elem_list=params['export_params']['elem_list']
        export_xrf_roi_data(scan_id, 
                            norm=params['export_params']['export_norm'],
                            elem_list=elem_list, 
                            remote_seg=remote_seg, 
                            append_meta_with=params)  # Ensure ROI data is exported for remote analysis
        #print(f"{elem_list=}")
        print(f"[DATA], Exported ROI data for remote analysis {scan_id = }.")


        #print("no reciever implemented yet, skipping remote analysis...")
        #pass 
        # print("\n[ANALYSIS] Remote analysis selected, receiving data remotely...")
        # #placeholder for Seher
        remote_receiver = RemoteSegmentationReceiver(remote_sender.cache_size())
        remote_receiver.subscribe()
        metadata = remote_receiver.data_w_metadata[0][0] # assuming there is only 1 segmentation done
        data = remote_receiver.data_w_metadata[0][1] # assuming there is only 1 segmentation done

        # print("\n[ANALYSIS] Remote segmentation results received ...")
        # results_dict = {} #remote.recieve results
        # np_array = np.array([]) #remote.recieve results
        # scan_metadata = {} #remote.recieve results
        #fine_scans_tables= analyze_data_remote(np_array, metadata)

    else:
        # For analysis-only mode, return the results
        analysis_results = analyze_data_local(scan_id=scan_id, params=params)
        # Extract fine scans tables if available (created during analysis)
        if analysis_results and 'fine_scans_tables' in analysis_results:
            fine_scans_tables = analysis_results['fine_scans_tables']
            print(f"[WORKFLOW] Captured {len(fine_scans_tables)} fine scans table groups from analysis")

    if not proceed_fine_scans:
        print("\n[INFO] Skipping fine scan queue submission and execution as per flag.")
        return
    
    # For analysis-only mode, return results immediately
    if is_analysis_only:
        print("--- Analysis-only mode complete ---")
        return analysis_results
    
    # C. Queue (Will skip if mode != real)
    submit_fine_scans_to_queue(
        json_path,
        scan_id,
        out_dir,
        params['execution_params'],
        fine_scans_tables=fine_scans_tables
    )
    
    # D. Run (Will skip if mode != real)
    run_fine_scans(is_real)
    
    print("--- Done ---")
    return None  # Explicit return for other modes



def mosaic_overlap_scan_auto(dets = None, ylen = 100, xlen = 100, overlap_per = 5, dwell = 0.01,
                        step_size = 250, plot_elem = ["Cr"],mll = False, 
                        beamline_params=None, initial_scan_path=None, 
                        remote_seg=True, followup_fine_scan=False):
    

    """ Usage <mosaic_overlap_scan_auto(dets=dets_fast, ylen=100, xlen=100, overlap_per=5, dwell=0.01, step_size=250, plot_elem=["Cr"], mll=False, 
    beamline_params=beamline_params, initial_scan_path=initial_scan_path)>"""

    # if dets is None:
    #     dets = dets_fast

    i0_init = sclr2_ch2.get()

    max_travel = 25

    dsx_i = dsx.position
    dsy_i = dsy.position

    smarx_i = smarx.position
    smary_i = smary.position

    scan_dim = max_travel - round(max_travel*overlap_per*0.01)

    x_tile = round(xlen/scan_dim)
    y_tile = round(ylen/scan_dim)

    xlen_updated = scan_dim*x_tile
    ylen_updated = scan_dim*y_tile

    #print(f"{xlen_updated = }, {ylen_updated=}")


    X_position = np.linspace(0,xlen_updated-scan_dim,x_tile)
    Y_position = np.linspace(0,ylen_updated-scan_dim,y_tile)

    X_position_abs = smarx.position+(X_position)
    Y_position_abs = smary.position+(Y_position)

    #print(X_position_abs)
    #print(Y_position_abs)


    #print(X_position)
    #print(Y_position)

    print(f"{xlen_updated = }")
    print(f"{ylen_updated = }")
    print(f"# of x grids = {x_tile}")
    print(f"# of y grids = {y_tile}")
    print(f"individual grid size in um = {scan_dim} x {scan_dim}")

    num_steps = round(max_travel*1000/step_size)

    unit = "minutes"
    fly_time = (num_steps**2)*dwell*2
    num_flys= len(X_position)*len(Y_position)
    total_time = (fly_time*num_flys)/60


    if total_time>60:
        total_time/=60
        unit = "hours"

    ask = input(f"Optimized scan x and y range = {xlen_updated} by {ylen_updated};\n total time = {total_time} {unit}\n Do you wish to continue? (y/n) ")

    if ask == 'y':

        time.sleep(2)
        first_sid = db[-1].start["scan_id"]+1

        if sclr2_ch2.get() < i0_init*0.9:
            RM.item_add(BPlan("peak_the_flux"))
            

        if mll:

            RM.item_add(BPlan("bps.movr", "dsy", ylen_updated/-2))
            RM.item_add(BPlan("bps.movr", "dsx", xlen_updated/-2))
            
            X_position_abs = dsx.position+(X_position)
            Y_position_abs = dsy.position+(Y_position)


        else:
            RM.item_add(BPlan("bps.movr", "smary", ylen_updated/-2))
            RM.item_add(BPlan("bps.movr", "smarx", xlen_updated/-2))
            
            X_position_abs = smarx.position+(X_position)
            Y_position_abs = smary.position+(Y_position)

            print(X_position_abs)
            print(Y_position_abs)


        for i in tqdm.tqdm(Y_position_abs):
                for j in tqdm.tqdm(X_position_abs):
                    print((i,j))
                    #yield from check_for_beam_dump(threshold=5000)
                    RM.item_add(BPlan("bps.sleep", 1)) #cbm catchup time
                    RM.queue_start()

                    fly_dim = scan_dim/2

                    if mll:

                        print(i,j)

                        RM.item_add(BPlan("bps.mov", "dsy", i))
                        RM.item_add(BPlan("bps.mov", "dsx", j))
                        
                        # yield from fly2dpd(dets,dssx,-1*fly_dim,fly_dim,num_steps,dssy,-1*fly_dim,fly_dim,num_steps,dwell)
                        headless_send_queue_coarse_scan(initial_scan_path, 
                                                        remote_seg=remote_seg)

                        RM.item_add(BPlan("bps.sleep", 3))
                        RM.item_add(BPlan("bps.mov", "dssx", 0, "dssy", 0))
                        #insert_xrf_map_to_pdf(-1,plot_elem,'dsx')
                        RM.item_add(BPlan("bps.mov", "dsx", dsx_i))
                        RM.item_add(BPlan("bps.mov", "dsy", dsy_i))
                        

                    else:
                        print(f"{fly_dim = }")
                        RM.item_add(BPlan("bps.mov", "smary", i))
                        RM.item_add(BPlan("bps.mov", "smarx", j))
                        
                        # yield from fly2dpd(dets, zpssx,-1*fly_dim,fly_dim,num_steps,zpssy, -1*fly_dim,fly_dim,num_steps,dwell)
                        headless_send_queue_coarse_scan(initial_scan_path, 
                                                        remote_seg=remote_seg)

                        RM.item_add(BPlan("bps.sleep", 1))
                        RM.item_add(BPlan("bps.mov", "zpssx", 0, "zpssy", 0))
                        

                        #try:
                            #insert_xrf_map_to_pdf(-1,plot_elem[0],'smarx')
                        #except:
                            #plt.close()
                            #pass


                        RM.item_add(BPlan("bps.mov", "smarx", smarx_i))
                        RM.item_add(BPlan("bps.mov", "smary", smary_i))
                    RM.queue_start()
        save_page()

        # plot_mosiac_overlap(grid_shape = (y_tile,x_tile),
        #                     first_scan_num = int(first_sid),
        #                     elem = plot_elem[0],
        #                     show_scan_num = True)

    else:
        return
    

def mosaic_overlap_scan_auto_relative(dets = None, ylen = 100, xlen = 100, overlap_per = 5, dwell = 0.01,
                         step_size = 250, plot_elem = ["Cr"], mll = False, 
                         beamline_params=None, initial_scan_path=None, 
                         remote_seg=True, followup_fine_scan=False):

    # 1. Define the step size for the mosaic grid
    # Since you requested 25 um steps for the grid iteration:
    try:
        if beamline_params:
            with open(beamline_params, 'r') as f:
                beamline_params_dict = json.load(f)
        else:
            beamline_params_dict = {}
    except (FileNotFoundError, json.JSONDecodeError, TypeError) as e:
        print(f"[ERROR] Failed to load beamline_params from {beamline_params}: {e}")
        beamline_params_dict = {}
    
    grid_step = (beamline_params_dict.get("mot1_e")) - (beamline_params_dict.get("mot1_s"))
    grid_step = grid_step*(1-(overlap_per*0.01))

    # 2. Generate the relative step lists
    # This creates a list of positions starting at 0 up to the length
    x_steps_raw = np.arange(grid_step//2, xlen , grid_step)
    y_steps_raw = np.arange(grid_step//2, ylen , grid_step)

    x_steps = x_steps_raw.tolist()
    y_steps = y_steps_raw.tolist()

    print(f"Grid Setup: {len(x_steps)} x {len(y_steps)} tiles.")
    print(f"Total area: {xlen}um x {ylen}um using {grid_step}um steps.")

    # Calculate estimated time (keeping your original logic)
    num_steps_fly = round(25 * 1000 / step_size) # internal fly scan resolution
    fly_time = (num_steps_fly**2) * dwell * 2
    total_time = (fly_time * len(x_steps) * len(y_steps)) / 60
    

    # Select motors based on MLL flag
    mot_x = "dsx" if mll else "smarx"
    mot_y = "dsy" if mll else "smary"
    fine_x = "dssx" if mll else "zpssx"
    fine_y = "dssy" if mll else "zpssy"

    # 3. Iterate over the relative steps
    for y_rel in tqdm.tqdm(y_steps, desc="Y-axis"):
        for x_rel in tqdm.tqdm(x_steps, desc="X-axis"):
            
            # Move motors relatively (movr) from the CURRENT position to the next step
            # Note: We use absolute moves to specific offsets for better trajectory control
            # but we define those offsets relative to where the script STARTED.
            
            print(f"Moving to relative position: X={x_rel}, Y={y_rel}")
            
            # Using bps.movr to move relative to the STARTING point of the whole scan
            # We calculate the move needed to get to the next grid point
            RM.item_add(BPlan("move_relative", mot_x, x_rel))
            RM.item_add(BPlan("move_relative", mot_y, y_rel))
            

            # Execute the fly scan
            headless_send_queue_coarse_scan(
                initial_scan_path, 
                remote_seg=remote_seg
            )

            # Reset internal fine stages to zero before next move
            RM.item_add(BPlan("mov", fine_x, 0, fine_y, 0))
            
            # Return to the local "origin" so the next loop's movr is accurate
            RM.item_add(BPlan("move_relative", mot_x, -x_rel))
            RM.item_add(BPlan("move_relative", mot_y, -y_rel))
            RM.queue_start()
            wait_for_queue_done()

    save_page()
