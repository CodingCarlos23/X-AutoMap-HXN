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
pixi run python -m automap_hxn.main
```

Pixi installs `automap_hxn` from this checkout as an editable package. After
the environment has been installed once, launch the standalone GUI with:

```bash
pixi run python -m automap_hxn.main
```

The package can also be imported by another Qt application:

```python
from automap_hxn.gui import create_automap_widget
```

## Headless QueueServer Workflow

The standalone GUI performs local TIFF analysis only. The existing headless
workflow is the path that can submit scans to the beamline QueueServer.

```bash
pixi run python scripts/remote.py
```

> **Warning:** `scripts/remote.py` is intended for an authorized beamline
> session. It connects to Tiled and runs the mosaic/headless workflow. Its
> current configuration, `configs/initial_scan_sim.json`, is named like a
> simulation file but presently sets `execution_params.mode` to `real`; it can
> submit scan plans to QueueServer. Do not run it on a development machine or
> against a production QueueServer without reviewing the configuration.

For the core `load_and_queue` workflow, behavior is controlled by
`execution_params.mode` in the JSON configuration:

- `simulation` — prepares the workflow and waits for manually supplied TIFFs;
  it does not submit the coarse or fine scans.
- `offline` — analyzes and exports data from an existing scan ID; it does not
  submit scans.
- `real` — submits coarse and fine scan plans to QueueServer.

`scripts/remote.py` adds mosaic and piezo operations around that core workflow,
so changing its JSON mode alone does not make it a safe local test command.

`src/automap_hxn/main_headless.py` is currently experimental reference code,
not a command-line launcher: its execution call is commented out. Use
`scripts/remote.py` only when the reviewed configuration and beamline context
are appropriate. A dedicated, safe simulation command is planned but is not
yet provided.

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
