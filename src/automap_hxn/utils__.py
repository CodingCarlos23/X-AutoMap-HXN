### THIS IS AN OLD utils.py FILE - some functions have been moved to other submodules, and some have been deleted.
### Please refer to the current version of utils.py for the latest code.

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
import matplotlib
# This is the equivalent of %matplotlib qt
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.ion()

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
container = c["tst/sandbox/eugene/synaps/reconstructions"]

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
