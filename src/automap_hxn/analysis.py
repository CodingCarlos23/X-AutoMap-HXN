import os
import json
import traceback
from pathlib import Path
import tifffile as tiff
import numpy as np
import time
import cv2
import pandas as pd
from .blobs.detection import detect_blobs
from .blobs.processing import find_union_blobs
from .plotting import plot_analysis_results
from .utils import make_json_serializable, wait_for_element_tiffs, formatted_unions_to_table, normalize_and_dilate, merge_overlapping_boxes_dict
from .export import create_rgb_tiff, create_all_elements_tiff, save_each_blob_as_individual_scan


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
    print(f"[ANALYSIS] Starting analysis for Scan {scan_id} in {out_dir}")
    
    # Skip analysis if remote_seg is True (data sent to remote port, no TIFFs)
    remote_seg = params.get('remote_seg') or params.get('segmentation_params', {}).get('remote_seg', False)
    if remote_seg:
        print("[ANALYSIS] remote_seg=True, skipping local analysis (handled remotely)...")
        return {'error': 'Remote segmentation requested - no local results available'}
    
    # --- 1. Read Scan Parameters ---
    params_json_path = os.path.join(out_dir, f"scan_{scan_id}_params.json")
    
    # Calculate step_size and start positions from scan metadata for accuracy
    step_size = 1.0
    x_start = 0.0
    y_start = 0.0

    if os.path.exists(params_json_path):
        with open(params_json_path, 'r') as f:
            params_data = json.load(f)
            scan_input = params_data.get('start_doc', {}).get('scan', {}).get('scan_input', [])
            if len(scan_input) >= 6:
                # Calculate from actual scan dimensions
                fast_start, fast_end, fast_N = scan_input[0], scan_input[1], scan_input[2]
                step_size = abs(fast_end - fast_start) / fast_N
                x_start = scan_input[0]
                y_start = scan_input[3]
                print(f"[ANALYSIS] Calculated from scan metadata: x_start={x_start}, y_start={y_start}, step_size={step_size}")
            else:
                # Fallback to saved or provided values
                step_size = params_data.get('step_size') or params.get('scan_params', {}).get('step_size') or params.get('step_size', 1.0)
                print(f"[ANALYSIS] Using fallback step_size={step_size}")

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
        tiff_img = tiff.imread(str(tiff_path)).astype(np.float32)

        print(f"{tiff_img.shape=}, dtype={tiff_img.dtype}, min={tiff_img.min()}, max={tiff_img.max()}")
        print(f"{min_thresh=}, {min_area=}, detection_method={detection_method}, method_params={method_params}")
        
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
                      and segmentation/detection parameters.
                      
                      Spatial calibration (for real-world coordinates):
                      - 'step_size': Scan step size in microns (default: 1.0)
                      - 'start_doc.scan.scan_input': [x_start, x_end, x_n, y_start, y_end, y_n]
                        If present, x_start and y_start will be extracted automatically.
                      - Alternatively provide 'x_start' and 'y_start' directly (default: 0.0)
    
    Returns:
        dict: Fine scans tables by group name, each containing:
              - DataFrame with columns: label, cx, cy, num_x, num_y (in microns)
              - Coordinates are in real-world units if spatial parameters provided
              
    Example:
    --------
    >>> # 3D array: (n_elements, height, width)
    >>> element_arrays = np.random.rand(3, 100, 100).astype(np.float32) * 1000
    >>> params = {
    ...     'scan_id': 12345,
    ...     'elem_list': ['Fe', 'Cu', 'Ni'],
    ...     'step_size': 0.5,  # microns per pixel
    ...     'start_doc': {'scan': {'scan_input': [-10, 10, 100, -10, 10, 100]}},
    ...     'segmentation_params': {...}
    ... }
    >>> results = analyze_data_from_arrays(element_arrays, params)
    """
    # Extract parameters
    scan_id = params.get('scan_params', {}).get('scan_id') or params.get('scan_id')
    
    print(f"[ANALYSIS] Analyzing 3D array for Scan {scan_id}")
    print(f"[ANALYSIS] Array shape: {element_arrays.shape}")
    
    # Check input array
    if element_arrays is None or element_arrays.size == 0:
        print("[ANALYSIS] Error: element_arrays is empty")
        return {'error': 'No arrays provided'}
    
    if len(element_arrays.shape) != 3:
        print(f"[ANALYSIS] Error: expected 3D array, got shape {element_arrays.shape}")
        return {'error': 'Array must be 3D (n_elements, height, width)'}
    
    # Get element order from params
    elem_list_of_lists = params.get("export_params", {}).get("elem_list", []) or params.get("elem_list", [])
    if not elem_list_of_lists:
        print("[ANALYSIS] Error: elem_list not found in params")
        return {'error': 'Element list not provided'}
    
    if isinstance(elem_list_of_lists[0], str):
        elem_list_of_lists = [elem_list_of_lists]
    
    all_elements = sorted(list(set(elem for sublist in elem_list_of_lists for elem in sublist)))
    print(f"[ANALYSIS] Element order from params: {all_elements}")
    
    # Verify array has correct number of elements
    if element_arrays.shape[0] != len(all_elements):
        print(f"[ANALYSIS] Warning: array has {element_arrays.shape[0]} elements but {len(all_elements)} expected")
        return {'error': f'Array dimension mismatch: {element_arrays.shape[0]} != {len(all_elements)}'}
    
    # Create dict mapping element names to 2D arrays
    element_array_dict = {elem: element_arrays[i].astype(np.float32) 
                         for i, elem in enumerate(all_elements)}
    
    # --- 1. Extract Analysis Parameters ---
    segmentation = params.get("segmentation_params", {})
    min_thresh = segmentation.get("min_threshold_intensity") or params.get("min_threshold_intensity")
    min_area = segmentation.get("min_threshold_area") or params.get("min_threshold_area")
    detection_method = segmentation.get("blob_detection_method") or params.get("blob_detection_method")
    
    # Calculate spatial parameters from scan metadata for accuracy
    step_size = 1.0
    x_start = 0.0
    y_start = 0.0
    
    # Calculate from start_doc scan_input if available
    scan_input = params.get('start_doc', {}).get('scan', {}).get('scan_input', [])
    if len(scan_input) >= 6:
        # Calculate step_size from actual scan dimensions
        fast_start, fast_end, fast_N = scan_input[0], scan_input[1], scan_input[2]
        step_size = abs(fast_end - fast_start) / fast_N
        x_start = scan_input[0]  # Fast axis start
        y_start = scan_input[3]  # Slow axis start
        print(f"[ANALYSIS] Calculated from scan metadata: x_start={x_start}, y_start={y_start}, step_size={step_size}")
    else:
        # Fallback: extract from scan_params (mot1_s, mot2_s, step_size)
        scan_params = params.get('scan_params', {})
        if scan_params.get('mot1_s') is not None:
            x_start = scan_params['mot1_s']
        if scan_params.get('mot2_s') is not None:
            y_start = scan_params['mot2_s']
        step_size = scan_params.get('step_size') or params.get('step_size', 1.0)
        print(f"[ANALYSIS] Using fallback from scan_params: x_start={x_start}, y_start={y_start}, step_size={step_size}")
    
    print(f"[ANALYSIS] Detection method: {detection_method}, threshold: {min_thresh}, area: {min_area}")
    
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
        
        print(f"[ANALYSIS] Processing {element} ({color})")
        try:
            img = element_array_dict[element]

            print(f"{img.shape=}, dtype={img.dtype}, min={img.min()}, max={img.max()}")
            print(f"{min_thresh=}, {min_area=}, detection_method={detection_method}, method_params={method_params}")

            # Use configurable normalization and dilation parameters
            morphology = params.get('morphology_params', {})
            kernel_size = tuple(morphology.get('normalize_kernel_size') or params.get('normalize_kernel_size', [3, 3]))
            iterations = morphology.get('dilate_iterations') or params.get('dilate_iterations', 2)
            img_norm, img_dilated = normalize_and_dilate(img, kernel_size=kernel_size, iterations=iterations)
            
            # Detect blobs
            b = detect_blobs(img_norm, 
                             img_dilated, 
                             min_thresh,
                             min_area, 
                             color, 
                             f"{element}_array", 
                             method=detection_method,
                             **method_params)
            print(f"{b=}")
            precomputed_blobs[color][(min_thresh, min_area)] = b
            print(f"[ANALYSIS] Found {len(b)} blobs for {element}")
        except Exception as e:
            print(f"[ANALYSIS] ❌ Error processing {element}: {e}")
            traceback.print_exc()
    
    # --- 3. Union & Table Generation Loop ---
    fine_scans_tables = {}
    
    for elem_list in elem_list_of_lists:
        group_name = "".join(elem_list)
        print(f"[ANALYSIS] Processing group: {group_name}")
        
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
            print(f"[ANALYSIS] Single element mode for {group_name}")
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
            print(f"[ANALYSIS] Union mode for {group_name}")
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
            print(f"[ANALYSIS] Skipping {group_name} - no blobs found")
            continue
        
        # Create fine scans table for this group
        if formatted_unions:
            table = formatted_unions_to_table(formatted_unions, save_to=None)
            if not table.empty:
                fine_scans_tables[group_name] = table
                print(f"[ANALYSIS] Created fine scans table with {len(table)} rows")
    
    print("[ANALYSIS] Complete.")
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
    print(f"[ANALYSIS-TABLE] Starting analysis for Scan {scan_id} in {out_dir}")
    
    # Skip analysis if remote_seg is True
    remote_seg = params.get('remote_seg') or params.get('segmentation_params', {}).get('remote_seg', False)
    if remote_seg:
        print("[ANALYSIS-TABLE] remote_seg=True, skipping (no TIFFs available)...")
        return {}
    
    # --- 1. Read Scan Parameters ---
    
    # Calculate step_size and start positions from scan metadata for accuracy
    params_json_path = os.path.join(out_dir, f"scan_{scan_id}_params.json")
    step_size = 1.0
    x_start = 0.0
    y_start = 0.0

    if os.path.exists(params_json_path):
        with open(params_json_path, 'r') as f:
            params_data = json.load(f)
            step_size = params_data.get('step_size', 1.0)
            scan_input = params_data.get('start_doc', {}).get('scan', {}).get('scan_input', [])
            if len(scan_input) >= 6:
                # Calculate from actual scan dimensions
                fast_start, fast_end, fast_N = scan_input[0], scan_input[1], scan_input[2]
                step_size = abs(fast_end - fast_start) / fast_N
                x_start = scan_input[0]
                y_start = scan_input[3]
                print(f"[ANALYSIS] Calculated from scan metadata: x_start={x_start}, y_start={y_start}, step_size={step_size}")
            else:
                # Fallback to saved or provided values
                step_size = params_data.get('step_size') or params.get('scan_params', {}).get('step_size') or params.get('step_size', 1.0)
                print(f"[ANALYSIS] Using fallback step_size={step_size}")

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
