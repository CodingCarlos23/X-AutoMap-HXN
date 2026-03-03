import json
import os
import pathlib
import traceback as trackback

import numpy as np
import tifffile as tiff

from app_state import AppState

# Mode flag: 0 = test (coarse scan only), 1 = real (grid scan only)
real_or_test = 1

# if real_or_test == 0:
#     from utils import (
#         detect_blobs,
#         find_union_blobs,
#         headless_send_queue_coarse_scan,
#         headless_send_queue_fine_scan,
#         normalize_and_dilate,
#         save_each_blob_as_individual_scan,
#         wait_for_element_tiffs,
#     )


def load_json_file(path):
    """Loads a single JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.name}")
    with open(path, "r") as f:
        return json.load(f)


def load_parameters(watch_dir):
    """Loads all required JSON parameter files."""
    print("Looking for initial_scan.json")
    try:
        analysis_params = load_json_file(watch_dir / "initial_scan.json")
        print("Loaded initial_scan.json:", analysis_params)
        return analysis_params
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading JSON files: {e}")
        exit(1)


def run_headless_processing():
    """Main function to run the headless processing workflow."""
    state = AppState()

    notebook_dir = pathlib.Path().resolve()
    watch_dir = notebook_dir / "data" / "input"
    watch_dir.mkdir(exist_ok=True)

    analysis_params = load_parameters(watch_dir)
    initial_scan_path = watch_dir / "initial_scan.json"

    # Test mode: run coarse scan only
    if real_or_test == 1:
        print("\nRunning coarse scan (test mode)...")
        #yield from headless_send_queue_coarse_scan(analysis_params, initial_scan_path, 1)

    # Real mode: run grid scan only
    if real_or_test == 1:
        print("\nGrid scan starts here (real mode)")
        yield from mosaic_overlap_scan_auto(
            dets=None,
            ylen=400,
            xlen=100,
            overlap_per=0,
            dwell=0.01,
            step_size=500,
            plot_elem=["None"],
            mll=False,
            beamline_params=analysis_params,
            initial_scan_path=initial_scan_path,
        )

    print("Scans Done")


# Global runner
#run_headless_processing()