# Cellpose Integration for Blob Detection

## Overview
Cellpose has been successfully integrated as a new detection method in the blob detection framework. Cellpose is a deep learning-based segmentation tool that excels at detecting cells and complex biological objects.

## Installation

### Required Packages

#### Using Pixi (Recommended)
```bash
# Basic installation (CPU-only, works on any machine)
pixi install

# Or manually add if not present
pixi add cellpose

# For GPU acceleration (OPTIONAL - only if you have CUDA installed)
# Skip this if you don't have NVIDIA GPU or CUDA toolkit
pixi add pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

> **Note**: The GPU line is completely optional. Cellpose works perfectly on CPU-only systems.

#### Using Pip
```bash
# Basic Cellpose installation
pip install cellpose

# Full installation with GUI (recommended)
pip install cellpose[gui]

# For GPU acceleration (optional, requires CUDA)
pip install cellpose[gui] torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Using Conda
```bash
# Install from conda-forge
conda install -c conda-forge cellpose

# Or create new environment
conda create -n cellpose -c conda-forge cellpose
conda activate cellpose
```

## Usage Examples

### Basic Cellpose Detection
```python
from utils import detect_blobs, normalize_and_dilate
import tifffile as tiff

# Load your XRF image
img = tiff.imread('scan_12345_Ni.tiff').astype(np.float32)
img_norm, img_dilated = normalize_and_dilate(img)

# Detect blobs using Cellpose
blobs = detect_blobs(
    img_dilated, img_norm, 
    min_thresh=50, min_area=100, 
    color='red', file_name='Ni_scan',
    method='cellpose',
    diameter=60,           # Expected particle size in pixels
    model_type='cyto3',    # Cellpose model type
    gpu=False              # Set True for GPU acceleration
)

print(f"Found {len(blobs)} particles")
```

### Advanced Cellpose Parameters
```python
# Fine-tune detection with multiple parameters
blobs = detect_blobs(
    img_dilated, img_norm, 50, 100, 'red', 'test.tiff',
    method='cellpose',
    diameter=60,                # Expected diameter in pixels
    model_type='cyto3',         # 'cyto3', 'cyto2', 'nuclei', 'cyto'
    flow_threshold=0.4,         # Lower = more permissive (0.0-1.0)
    cellprob_threshold=0.0,     # Cell probability threshold
    channels=[0, 0],            # [cytoplasm, nucleus] channels
    min_diameter=30,            # Filter by size range
    max_diameter=100,
    gpu=False                   # Use GPU if available
)
```

### Compare with Other Methods
```python
# Compare Cellpose with traditional methods
from utils import detect_blobs_multi_method

results = detect_blobs_multi_method(
    img_dilated, img_norm, 50, 100, 'red', 'test.tiff',
    methods=['simple', 'contours', 'hough', 'cellpose'],
    combine_results=False
)

for method, blobs in results.items():
    print(f"{method}: {len(blobs)} detections")
```

## Cellpose Model Types

| Model | Best For | Description |
|-------|----------|-------------|
| `cyto3` | General particles, cells | Latest general-purpose model |
| `cyto2` | Cells, spheroids | Previous general model |
| `cyto` | Original cytoplasm model | First Cellpose model |
| `nuclei` | Dense, round objects | Optimized for nuclei detection |

## Parameter Guidelines

### For XRF Particle Detection:
- **diameter**: Set to expected particle size (30-100 px typical)
- **model_type**: `'cyto3'` works well for most particles
- **flow_threshold**: Start with 0.4, lower for more detections
- **cellprob_threshold**: Usually keep at 0.0
- **min/max_diameter**: Filter results by size range

### Performance Tips:
- **CPU vs GPU**: Works fine on CPU - GPU just makes it faster (2-5x speedup)
- **GPU setup**: Only set `gpu=True` if you have CUDA installed
- **No CUDA needed**: Default `gpu=False` works on any machine
- Larger `diameter` values are more robust but slower
- Lower `flow_threshold` detects more objects but may include artifacts
- Use size filtering (`min_diameter`, `max_diameter`) to clean results

## Integration Details

### New Methods Available:
1. `detect_blobs(..., method='cellpose')` - Main function
2. `detect_blobs_cellpose(...)` - Convenience wrapper
3. `get_available_detection_methods()` - Now includes 'cellpose' if installed

### Backward Compatibility:
- All existing code continues to work unchanged
- Cellpose is only added if successfully installed
- Graceful fallback if Cellpose not available

### Output Format:
Cellpose detections include standard blob keys plus:
- `area`: Actual mask area in pixels
- `equiv_diameter`: Equivalent circle diameter
- `bbox`: Bounding box (x1, y1, x2, y2)

## Testing

Run the test suite to verify Cellpose integration:
```bash
# Using pixi
pixi run python test_blob_detection.py

# Or directly if environment is activated
python test_blob_detection.py
```

Try the XRF-specific example:
```bash
# Using pixi  
pixi run python example_cellpose_xrf.py

# Or directly if environment is activated
python example_cellpose_xrf.py
```

## Troubleshooting

### Common Issues:

1. **Import Error**: 
   ```
   ImportError: Cellpose not available
   ```
   **Solution**: 
   - Pixi: `pixi install` (already in pixi.toml) or `pixi add cellpose`
   - Pip: `pip install cellpose[gui]`

2. **GPU/CUDA Issues**:
   ```
   CUDA out of memory
   RuntimeError: No CUDA GPUs available
   ```
   **Solutions**: 
   - **No CUDA installed**: Set `gpu=False` (default) - works fine on CPU
   - **CUDA installed but errors**: Set `gpu=False` or reduce image size
   - **Want GPU speedup**: Install CUDA toolkit first, then add GPU packages

3. **Do I need CUDA?**:
   - **NO** - Cellpose works perfectly on CPU-only systems
   - **YES** - Only if you want GPU acceleration for faster processing
   - CUDA is completely optional for functionality

3. **No Detections**:
   - Try adjusting `diameter` parameter
   - Lower `flow_threshold` (e.g., 0.2)
   - Check if image needs preprocessing

4. **Too Many False Positives**:
   - Increase `flow_threshold` (e.g., 0.6)
   - Adjust `min_diameter`/`max_diameter` filters
   - Try different `model_type`

## When to Use Cellpose vs Other Methods

**Use Cellpose when:**
- ✅ Detecting biological cells or cell-like particles
- ✅ Objects have complex, irregular shapes
- ✅ Traditional methods give poor results
- ✅ You need high-quality segmentation masks
- ✅ Processing time is not critical

**Use traditional methods when:**
- ✅ Objects are simple shapes (circles, ellipses)
- ✅ Fast processing is required
- ✅ Working with very large images
- ✅ Simple threshold-based detection works well
- ✅ Cellpose not available/installable

## Examples Directory

Check out these example files:
- `test_blob_detection.py` - Complete test suite
- `example_cellpose_xrf.py` - XRF-specific Cellpose example