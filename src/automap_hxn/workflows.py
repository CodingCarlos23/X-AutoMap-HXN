import numpy as np
import tqdm
import json
import time
import os
from .queue import submit_and_export, submit_fine_scans_to_queue, run_fine_scans, wait_for_queue_done, build_coarse_scan_requests
from .loading import load_and_queue, load_params_from_json
from .utils import RM

import pandas as pd
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

from bluesky_queueserver_api import BPlan

def headless_send_queue_coarse_scan(params_path, remote_seg=True, tiled_client = None):
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
                   proceed_fine_scans=proceed_with_fine_scan,
                   tiled_client=tiled_client)


def mosaic_overlap_scan_auto_relative(dets = None, ylen = 100, xlen = 100, overlap_per = 5, dwell = 0.01,
                         step_size = 250, plot_elem = ["Cr"], mll = False, 
                         beamline_params=None, initial_scan_path=None, 
                         remote_seg=True, followup_fine_scan=False,tiled_client=None,
                         ref_scan_id = None):
    '''
    # 1. Define the step size for the mosaic grid
    # Since you requested 25 um steps for the grid iteration:

    #"configs/initial_scan_sim.json"


    mosaic_overlap_scan_auto_relative(dets = None, ylen = 100, xlen = 100, overlap_per = 5, dwell = 0.01,
                         step_size = 250, plot_elem = ["Ni"], mll = False, 
                         beamline_params="configs/initial_scan_sim.json", 
                         initial_scan_path="configs/initial_scan_sim.json", 
                         remote_seg=False, followup_fine_scan=True,
                         tiled_client = c)
    
    '''

    # Determine mode early so we can skip ZMQ entirely in simulation
    _early_params = {}
    if initial_scan_path:
        try:
            with open(initial_scan_path, 'r') as _f:
                _early_params = json.load(_f)
        except Exception:
            pass
    _mode = str(_early_params.get('execution_params', {}).get('mode', 'simulation')).lower()
    _is_real_or_offline = _mode in ('real', 'offline')

    if _is_real_or_offline:
        status = RM.status()
        if not status.get("worker_environment_exists", False):
            response = RM.environment_open()
            if not response.get("success", False):
                raise RuntimeError(f"Could not open QueueServer worker: {response.get('msg', '')}")
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                status = RM.status()
                if status.get("worker_environment_state") == "idle" and status.get("re_state") == "idle":
                    break
                time.sleep(0.25)
            else:
                raise RuntimeError("QueueServer worker did not become ready within 15 seconds.")
    else:
        print(f"[SIM] Skipping QueueServer environment check (mode={_mode})")

    if ref_scan_id and _is_real_or_offline:
            RM.item_execute(BPlan("recover_zp_csan_pos",
                            ref_scan_id,
                            zp_move_flag = 0,
                            smar_move_flag = 1,
                            move_base = 1))

    try:
        if beamline_params:
            with open(beamline_params, 'r') as f:
                beamline_params_dict = json.load(f)
        else:
            beamline_params_dict = {}
    except (FileNotFoundError, json.JSONDecodeError, TypeError) as e:
        print(f"[ERROR] Failed to load beamline_params from {beamline_params}: {e}")
        beamline_params_dict = {}

    # Load full params (with defaults) once for per-tile analysis
    tile_params = {}
    if initial_scan_path:
        try:
            tile_params = load_params_from_json(initial_scan_path)
        except Exception as e:
            print(f"[MOSAIC] Could not load tile params for analysis: {e}")

    proceed_with_fine_scan = tile_params.get('execution_params', {}).get('proceed_with_fine_scan', False)
    mode = str(tile_params.get('execution_params', {}).get('mode', 'simulation')).lower()
    is_real = (mode == 'real')
    is_offline = (mode == 'offline')
    data_wd = tile_params.get('export_params', {}).get('data_wd', '.')

    grid_step = (beamline_params_dict['scan_params'].get("mot1_e")) - (beamline_params_dict['scan_params'].get("mot1_s"))
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
            
            tile_num = y_steps.index(y_rel) * len(x_steps) + x_steps.index(x_rel) + 1
            total_tiles = len(x_steps) * len(y_steps)
            print(f"\n[MOSAIC] Tile {tile_num}/{total_tiles}  →  smarx={x_rel:.1f}µm  smary={y_rel:.1f}µm")

            # Queue all 7 items for this tile before firing the queue once.
            # Previously headless_send_queue_coarse_scan called queue_start()
            # internally, draining the queue before the return moves were added.
            _, coarse_requests = build_coarse_scan_requests(initial_scan_path)
            if is_real or is_offline:
                RM.item_add(BPlan("move_relative", mot_x, x_rel))
                RM.item_add(BPlan("move_relative", mot_y, y_rel))
                for req in coarse_requests:
                    RM.item_add(BPlan(req["plan_name"], *req["plan_args"]))
                RM.item_add(BPlan("mov", fine_x, 0, fine_y, 0))
                RM.item_add(BPlan("move_relative", mot_x, -x_rel))
                RM.item_add(BPlan("move_relative", mot_y, -y_rel))
                RM.queue_start()
                wait_for_queue_done()
            else:
                print(f"[SIM] Would queue: move_relative {mot_x} {x_rel}, move_relative {mot_y} {y_rel}")
                for req in coarse_requests:
                    print(f"[SIM] Would queue: {req['plan_name']} {req['plan_args']}")
                print(f"[SIM] Would queue: mov {fine_x} 0 {fine_y} 0, return moves, queue_start")

            if proceed_with_fine_scan:
                scan_id = None
                if is_real:
                    print("[MOSAIC] Fetching scan_id from databroker...", flush=True)
                    try:
                        from hxntools.CompositeBroker import db
                        scan_id = db[-1].start['scan_id']
                        print(f"[MOSAIC] scan_id={scan_id}", flush=True)
                    except Exception as e:
                        print(f"[MOSAIC] Could not get scan_id from db: {e}")
                if scan_id is None:
                    scan_id = tile_params.get('scan_id') or 111111

                out_dir = os.path.join(data_wd, f"automap_{scan_id}")
                tile_params['out_dir'] = out_dir

                print(f"[MOSAIC] Analyzing tile (scan_id={scan_id}, out_dir={out_dir})...")
                try:
                    from .analysis import analyze_data_local
                    result = analyze_data_local(scan_id=scan_id, params=tile_params)
                    if result and 'fine_scans_tables' in result:
                        # Only fine scan on union blobs (detected across all elements in group).
                        # Individual blobs (single-element groups) are skipped.
                        union_tables = {
                            group: df[df['label'].str.startswith('Union Box')]
                            for group, df in result['fine_scans_tables'].items()
                        }
                        union_tables = {g: df for g, df in union_tables.items() if not df.empty}

                        if union_tables:
                            n_unions = sum(len(df) for df in union_tables.values())
                            print(f"[MOSAIC] {n_unions} union(s) found — queuing fine scans...")
                            submit_fine_scans_to_queue(
                                initial_scan_path, scan_id, out_dir,
                                tile_params['execution_params'],
                                fine_scans_tables=union_tables,
                            )
                            run_fine_scans(is_real or is_offline)
                            wait_for_queue_done()
                        else:
                            print("[MOSAIC] No union blobs detected in this tile — moving to next tile.")
                    else:
                        print("[MOSAIC] No particles detected in this tile — moving to next tile.")
                except Exception as e:
                    print(f"[MOSAIC] Tile analysis failed: {e}")