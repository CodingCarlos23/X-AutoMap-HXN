import json
import tifffile as tiff
import numpy as np
import cv2
from collections import Counter
from pathlib import Path
from hxntools.CompositeBroker import db
import os

from .utils import resize_if_needed, merge_overlapping_boxes_dict, make_json_serializable
from .remote_segmentation import RemoteSegmentationSender

# Create a global instance of the remote sender
remote_sender = RemoteSegmentationSender() 

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
                                                access_tags=["tst_sandbox"])    # access_tags=["synaps_project"])
    
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
            result = scan_container.write_array(stacked_array, key=compound_key, access_tags=["tst_sandbox"])
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
