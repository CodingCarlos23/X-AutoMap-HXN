from automap_hxn.workflows import mosaic_overlap_scan_auto_relative, load_and_queue
from tiled.client import from_uri

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
plt.ion()

c = from_uri("https://tiled.nsls2.bnl.gov")


#393976 ref loacation
#for local

if __name__ == "__main__":
    mosaic_overlap_scan_auto_relative(dets = None, ylen = 100, xlen = 100, overlap_per = 5, dwell = 0.01,
                            step_size = 250, plot_elem = ["Ni"], mll = False,
                            beamline_params="configs/initial_scan_sim.json",
                            initial_scan_path="configs/initial_scan_sim.json",
                            remote_seg=True, followup_fine_scan=True,
                            tiled_client = c)