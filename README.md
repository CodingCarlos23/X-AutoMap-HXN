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
pixi run python src/automap_hxn/main.py
```

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
- **PyQt5** - GUI framework
