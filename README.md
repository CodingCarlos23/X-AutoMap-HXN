# X-AutoMap-HXN

Automated particle detection and scan planning for the HXN (Hard X-ray Nanoprobe) beamline at NSLS-II.

## What This Does

At HXN, samples are scanned with X-rays to produce element maps (XRF images) showing where different elements like Ni, Fe, Cu are concentrated. Scientists often want to do a quick coarse scan first, then automatically identify interesting particles and queue up detailed high-resolution scans of just those regions.

This tool:
1. Loads XRF element maps (TIFF files) from a coarse scan
2. Overlays up to 3 elements as RGB channels for visualization
3. Detects particles using blob detection or deep learning (cellpose)
4. Finds "union" regions where multiple elements overlap (e.g., particles containing both Ni and Fe)
5. Exports scan coordinates that can be sent to the beamline queue server for automated fine scanning

## Requirements

- Python 3.11 or 3.12
- Platforms: Linux (x86_64), macOS (Intel or Apple Silicon)
- [pixi](https://pixi.sh) for dependency management

## Quick Start

```bash
pixi install
pixi run automap
```

Pixi installs `automap_hxn` from this checkout as an editable package. After
the environment has been installed once, launch the standalone GUI with:

```bash
pixi run automap
```

The package can also be imported by another Qt application:

```python
from automap_hxn.gui import create_automap_widget
```

## Headless Workflow

Run the mosaic pipeline from the terminal without the GUI:

```bash
pixi run automap-headless "configs/live scan demo/live_scan_demo.json"
```

Any config in `configs/` works — pass the path as the only argument.

Behavior is controlled by `execution_params.mode` in the JSON config:

- `offline` — analyzes existing TIFFs from a local folder; no beamline connection needed. Good for local testing.
- `real` — submits coarse and fine scan plans to QueueServer. Beamline use only.

## QueueServer Workflow (Beamline)

For authorized beamline sessions that also need mosaic/piezo orchestration,
`scripts/remote.py` wraps the headless workflow with those extra steps:

```bash
pixi run python scripts/remote.py
```

> **Warning:** `scripts/remote.py` is intended for an authorized beamline
> session. It connects to Tiled and runs the mosaic/headless workflow. Its
> current configuration, `configs/initial_scan_sim.json`, is named like a
> simulation file but presently sets `execution_params.mode` to `real`; it can
> submit scan plans to QueueServer. Do not run it on a development machine or
> against a production QueueServer without reviewing the configuration.

## Troubleshooting

**Segmentation fault / core dump on startup**

If the app crashes immediately with `Segmentation fault (core dumped)`, try these in order:

```bash
# 1. Reinstall the pixi environment (picks up any dependency changes)
pixi install

# 2. Rebuild the fontconfig cache (fixes crashes caused by a stale font cache
#    after pixi modifies the environment)
pixi run fc-cache -fv
```

Then run `pixi run automap` again. The font cache can go stale whenever `pixi install` swaps libraries in the environment, and a bad cache causes Qt to crash when rendering certain Unicode characters.

**"No space left on device" / inotify limit**

If you see `inotify_add_watch failed: No space left on device` before the crash, your system has hit the inotify watch limit (commonly caused by VS Code watching large directories). Raise it:

```bash
sudo sysctl fs.inotify.max_user_watches=524288
```

To persist across reboots: `echo "fs.inotify.max_user_watches=524288" | sudo tee /etc/sysctl.d/60-inotify.conf`

---

## GUI Workflow

1. **Load images**: Select a directory with XRF element TIFFs, pick 3 elements (mapped to RGB)
2. **Configure**: Set microns-per-pixel scale and stage origin coordinates
3. **Detect**: Adjust intensity/area thresholds to identify particles
4. **Find unions**: Locate regions where multiple elements overlap
5. **Queue scans**: Send selected regions to the beamline queue server

## Detection Methods

- `simple` - OpenCV SimpleBlobDetector (default)
- `contours`, `hough`, `watershed` - Traditional CV methods
- `cellpose` - Deep learning segmentation for complex shapes (see `docs/CELLPOSE_INTEGRATION_GUIDE.md`)

## SVG Export

Export XRF intensity arrays as publication-ready SVG figures with contour lines. See `examples/svg_export.py` for a working example (uses `xrf_to_svg` from `automap_hxn.export`):

```bash
pixi run python examples/svg_export.py
```

## Output Files

- `precomputed_blobs.pkl` - Cached detection results
- `union_blobs.json` - Union boxes with real-world coordinates
- `scans/*.json` - Individual scan parameters for queue server

## Key Dependencies

- **tiled** - Remote data access to NSLS-II data
- **bluesky-queueserver-api** - Beamline queue server integration
- **cellpose** - Deep learning segmentation
- **hxntools** - HXN beamline utilities
- **opencv**, **scikit-image** - Image processing
- **QtPy / PySide6** - GUI framework
