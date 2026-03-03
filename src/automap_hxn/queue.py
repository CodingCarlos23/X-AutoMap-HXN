from .export import export_xrf_roi_data, export_scan_params
import os
import time
from pathlib import Path
import json
from .utils import RM

from bluesky_queueserver_api import BPlan

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
    ic1_count = scan_params.get('ic1_count', 6000)
    
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
        from hxntools.CompositeBroker import db

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




