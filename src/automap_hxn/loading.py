import json
import os
from .queue import submit_and_export, submit_fine_scans_to_queue, run_fine_scans
from .analysis import analyze_data_local, analyze_data_remote
from .export import export_xrf_tiled
from automap_hxn.remote_segmentation import RemoteSegmentationReceiver

import warnings
import pandas as pd

from tiled.client import from_uri
c = from_uri('https://tiled.nsls2.bnl.gov')
c_reconstructions = c["tst/sandbox/eugene/synaps/reconstructions"]
c_segmentations = c["tst/sandbox/eugene/synaps/segmentations"]

from .remote_segmentation import RemoteSegmentationSender

# Create a global instance of the remote sender
remote_sender = RemoteSegmentationSender(c_reconstructions)

# import matplotlib
# # This is the equivalent of %matplotlib qt
# matplotlib.use('Qt5Agg')
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# plt.ion()

# Suppress DataFrame fragmentation warnings from databroker
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning, message='.*DataFrame is highly fragmented.*')


def load_params_from_json(json_path, target_id=None):
    """Load parameters from JSON file and perform necessary preprocessing."""

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

    # 3) Get mode from JSON config; set to 'simulation' if not specified
    params['execution_params'] = params.get('execution_params', {})
    mode = params['execution_params']['mode'] = str(params['execution_params'].get('mode', 'simulation')).lower()
    
    # Map mode to legacy real_test for backward compatibility with other functions
    mode_map = {'simulation': 0, 'real': 1, 'offline': 2, 'analysis-only': 3}
    params['real_test'] = mode_map.get(mode, 0)
    
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
    
    # IMPORTANT: If offline, scan_id is mandatory.
    params['scan_id'] = params['scan_params'].get("scan_id", target_id)
    if (mode == 'offline') and (not params['scan_id']):
        print(f"[WARNING] Running in '{mode}' mode but no scan_id provided.")
        # You might want to raise an error or rely on it being in the JSON

    return params


def load_and_queue(json_path, target_id=None, remote_seg=False, proceed_fine_scans=True):
    """
    Main workflow function supporting multiple modes.
    Mode is specified in JSON config file as 'mode' key:
    - 'simulation': Simulation mode
    - 'real': Real scanning mode 
    - 'offline': Offline mode (use existing scan)
    """

    # 1) Load segmentation parameters from JSON
    params = load_params_from_json(json_path, target_id)
    mode = params['execution_params']['mode']
    params['remote_seg'] = remote_seg
    
    # 4) EXECUTE
    print(f"--- Workflow: {os.path.basename(json_path)} (Mode: {mode.capitalize()}) ---")

    # A. Submit / Export
    print(f"\n[STEP A] Submit and/or Export Coarse Scan")
    # This returns scan_id and out_dir which are needed for subsequent steps
    # TODO: Check the `submit_and_export` function; 'segmentation_params/remote_seg' is allways treated as False? (look up `is_remote`)
    scan_id, out_dir = submit_and_export(
        params['execution_params'],
        params['scan_params'],
        params['export_params'],
        params.get('segmentation_params')
    )
    print(f"[{mode.upper()}] Scan ID: {scan_id}, Output Directory: {out_dir}")
    
    # Update params with scan_id and out_dir
    params['scan_id'] = scan_id
    params['out_dir'] = out_dir

    # B. Segmentation
    print(f"\n[STEP B] Segment the Data {'Remotely' if remote_seg else 'Locally'}")
    if remote_seg:
        remote_sender.clear_cache()
        elem_list=params['export_params']['elem_list']
        blocking_receiver = RemoteSegmentationReceiver(c_segmentations, num_tables=len(elem_list))
        blocking_receiver.subscribe()

        export_xrf_tiled(tiled_client=c_reconstructions,
                        scan_id=scan_id, 
                        norm=params['export_params']['export_norm'],
                        elem_list=elem_list, 
                        append_meta_with=params)

        print(f"[DATA], Exported ROI data for remote analysis {scan_id = }.")

        print("[ANALYSIS] Remote analysis selected, receiving data remotely...")
        results, data_w_metadata = blocking_receiver.wait_for_results()

        # metadata = blocking_receiver.data_w_metadata[0][1] # assuming there is only 1 segmentation done
        # data = blocking_receiver.data_w_metadata[0][0] # assuming there is only 1 segmentation done

        # print("\n[ANALYSIS] Remote segmentation results received ...")
        # results_dict = {} #remote.recieve results
        # np_array = np.array([]) #remote.recieve results
        # scan_metadata = {} #remote.recieve results
        #fine_scans_tables= analyze_data_remote(np_array, metadata)

    else:
        segmentation_results = analyze_data_local(scan_id=scan_id, params=params)
        # Extract fine scans tables if available (created during analysis)
        if segmentation_results and 'fine_scans_tables' in segmentation_results:
            fine_scans_tables = segmentation_results['fine_scans_tables']
            print(f"[WORKFLOW] Captured {len(fine_scans_tables)} fine scans table groups from analysis")

    if not proceed_fine_scans:
        print("\n[INFO] Skipping fine scan queue submission and execution as per flag.")
        return
    
    # C. Queue (Will skip if mode != real)
    print(f"\n[STEP C] Queue Fine Scans for Execution")
    submit_fine_scans_to_queue(
        json_path,
        scan_id,
        out_dir,
        params['execution_params'],
        fine_scans_tables=fine_scans_tables
    )
    
    # D. Run (Will skip if mode != real)
    if mode == 'real':
        print(f"\n[STEP D] Run Fine Scans")
        run_fine_scans(True)
    
    print("--- Done ---")
    return None  # Explicit return for other modes

