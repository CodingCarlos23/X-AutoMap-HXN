"""
Export XRF intensity data from Tiled as a publication-ready SVG.

Usage:
    pixi run python examples/svg_export.py
"""

from utils.svg_exporter import xrf_to_svg
from tiled.client import from_uri

# Connect to Tiled and fetch an element array
client = from_uri("https://tiled.nsls2.bnl.gov")
arr = client["/tst/sandbox/synaps/reconstructions/automap_393748_1772242656/Ni"][:]

# Export as SVG with contour lines
svg_bytes = xrf_to_svg(arr, metadata={"element": "Ni", "scan_id": "393748"})

with open("Ni_intensity.svg", "wb") as f:
    f.write(svg_bytes)

print("Wrote Ni_intensity.svg")
