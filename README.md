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

---

## Feature Locator — VLM-Assisted Particle Detection

The **Feature Locator** tab uses a Vision Language Model (VLM) to detect particles in XRF images. It supports two modes: single-element analysis and three-element co-location analysis.

---

### Single Image Mode — Colored Circles Pipeline

When the **"Colored circles on white"** checkbox is enabled, each XRF TIFF is transformed before being sent to the VLM. Every step is fully deterministic (no AI, no randomness) until the VLM call itself.

**Image preprocessing (per TIFF):**

| Step | What happens |
|---|---|
| Load | Raw TIFF read at full bit depth (16-bit / 32-bit preserved) |
| Normalize | Stretch pixel values linearly: dimmest pixel → 0, brightest → 255 |
| Upscale | If shortest dimension < 1092px, scale up bicubically (e.g. 150px → 1092px) |
| Gaussian blur | 3×3 kernel softens single-pixel noise spikes |
| Otsu threshold | Auto-computes the best intensity cutoff from the histogram; pixels above → white, below → black |
| Find contours | OpenCV traces the boundary of every connected white blob |
| Min enclosing circle | For each blob contour, compute the smallest enclosing circle (center + radius) |
| Pad radius | `r = max(r, 3) * 1.3` — ensures even 1-pixel blobs produce a visible circle |
| Draw on white canvas | Solid filled circles on a white RGB background (red in single mode) |
| PNG → base64 | Final image encoded and sent to the VLM |

**VLM prompt used (Sparse XRF):**

```
You are analyzing a sparse XRF (X-ray fluorescence) scan image.
XRF images are characteristically sparse: most of the image is near-zero dark background,
with a small number of isolated bright clusters or particles scattered across the field.
Pixel intensity directly encodes elemental signal strength — brighter means stronger signal.
CRITICAL: Many particles will be very small — just a few pixels across.
Do not overlook tiny bright specks; treat every localized bright point as a candidate particle.
Your task is to identify every bright isolated particle, cluster, or region of elevated signal
against the dark background. Ignore only pure noise (single-pixel random specks at noise floor).
Focus on localized bright spots that stand clearly above the background level, regardless of size.
Each feature may be a single tiny particle, an aggregate, or a compact cluster.
There is no limit on how many features you can return — report ALL of them.
Respond ONLY with valid JSON using this exact schema — no extra text outside the JSON object:
{"features": [{"id": 1, "label": "XRF particle", "x1": 0.10, "y1": 0.20, "x2": 0.45, "y2": 0.60, "description": "bright isolated particle cluster"}], "total_count": 1}
IMPORTANT: All coordinates (x1, y1, x2, y2) must be FRACTIONAL values between 0.0 and 1.0,
where (0,0) is top-left and (1,1) is bottom-right. x1 < x2, y1 < y2.
Draw boxes tightly around the bright core of each feature — exclude surrounding dark background.
For very small particles, the box may only span 0.01–0.05 in each dimension — that is fine.
```

This same prompt is used for all three element calls in XRF ROI mode. In circles mode the VLM is effectively being asked to find colored circles, which is a much simpler visual task than finding faint XRF blobs — the prompt language still fits because each circle represents a particle.

**Output:** `<stem>_features.json` — list of detected feature bounding boxes (fractional 0–1 coordinates) with label and description per feature.

---

### XRF ROI Mode — Three-Element Co-location Analysis

This mode runs three independent VLM calls (one per element), then optionally runs an LLM co-location analysis pass on the results.

#### Step 1 — Triple Detection

Three element TIFFs are selected (e.g. Ca, Fe, S). Element names are auto-extracted from the filenames (`scan_0_Ca.tiff` → `Ca`).

Each TIFF goes through the same preprocessing pipeline as single mode, but with element-specific circle colors:
- **Element 1 → Red circles**
- **Element 2 → Green circles**
- **Element 3 → Blue circles**

Three separate VLM calls are made sequentially. Each receives only its own element's image with the Sparse XRF prompt. Results are combined and saved as:

**`<stem>_xrf_triple_features.json`**
```json
{
  "elements": { "r": "Ca (Red)", "g": "Fe (Green)", "b": "S (Blue)" },
  "r_features": [ { "id": 1, "x1": 0.12, "y1": 0.34, "x2": 0.18, "y2": 0.41, "label": "XRF particle", "description": "..." } ],
  "g_features": [ ... ],
  "b_features": [ ... ],
  "total_count": 92
}
```

The composite image displayed combines the three grayscale channels as R/G/B (false-color), or overlays the three colored-circle images on a white canvas when circles mode is on.

#### Step 2 — Co-location Analysis (LLM)

After triple detection completes, **"Analyze Co-locations"** becomes available. This pass:

1. Reads `_xrf_triple_features.json`
2. Computes all pairwise and triple bounding-box intersections (any overlap, IoU > 0):
   - R+G overlaps where Ca and Fe co-exist
   - R+B overlaps where Ca and S co-exist
   - G+B overlaps where Fe and S co-exist
   - R+G+B overlaps where all three elements co-exist
3. For each overlap zone, records the intersection bounding box
4. Sends a structured prompt to the LLM listing all zones and element identities. The exact prompt sent (with real element names and zone positions filled in) looks like:

```
You are an XRF (X-ray fluorescence) materials scientist.
The scan has 3 elements: Element 1 = Ca (Red), Element 2 = Fe (Green), Element 3 = S (Blue).

The following co-location zones were detected where multiple element signals overlap:
  Zone 1: Ca (Red) + Fe (Green) + S (Blue) co-located at (0.45,0.22)→(0.52,0.30)
  Zone 2: Ca (Red) + S (Blue) co-located at (0.12,0.34)→(0.18,0.41)
  Zone 3: Fe (Green) + S (Blue) co-located at (0.71,0.55)→(0.78,0.62)

For each zone, identify the most likely material, compound, alloy, or phase that explains
why these elements co-exist in the same region. Be concise but specific.

Respond ONLY with valid JSON:
{"zones": [{"id": 1, "suggestion": "stainless steel", "reasoning": "Fe+Cr+Ni is a classic stainless steel signature"}]}
```

The LLM is asked to reason about each zone and return `suggestion` + `reasoning` per zone.
5. Merges LLM suggestions back into the zone records

**`<stem>_xrf_colocate.json`**
```json
{
  "elements": { "r": "Ca (Red)", "g": "Fe (Green)", "b": "S (Blue)" },
  "model": "claude-sonnet-4-6",
  "overlap_zones": [
    {
      "id": 1,
      "element_codes": ["R", "G", "B"],
      "elements": ["Ca (Red)", "Fe (Green)", "S (Blue)"],
      "x1": 0.45, "y1": 0.22, "x2": 0.52, "y2": 0.30,
      "llm_suggestion": "Iron-calcium sulfate (e.g. anhydrite or gypsum with Fe impurities)",
      "reasoning": "Ca + S co-location is a strong indicator of calcium sulfate phases; Fe presence suggests substitution or impurity"
    }
  ],
  "total_zones": 3
}
```

**Black outline boxes** are drawn on the composite image for each zone in `_xrf_colocate.json`. Hovering over a black box shows a tooltip with the zone elements, LLM suggestion, and reasoning.

#### Overlay Controls

After analysis, individual toggle buttons appear on the composite image (top-right corner):

| Button | Controls |
|---|---|
| **Hide All / Show All** | Master toggle for all boxes |
| **● Ca** (red) | Show/hide Element 1 detection boxes |
| **● Fe** (green) | Show/hide Element 2 detection boxes |
| **● S** (blue) | Show/hide Element 3 detection boxes |
| **◆ Zones** (light) | Show/hide black co-location zone boxes |

Tooltips are suppressed for any hidden layer — hovering over a hidden element's region shows nothing.

## Key Dependencies

- **tiled** - Remote data access to NSLS-II data
- **bluesky-queueserver-api** - Beamline queue server integration
- **cellpose** - Deep learning segmentation
- **hxntools** - HXN beamline utilities
- **opencv**, **scikit-image** - Image processing
- **QtPy / PySide6** - GUI framework
