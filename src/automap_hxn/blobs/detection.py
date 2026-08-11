import json
from pathlib import Path

import numpy as np
import cv2
import tifffile
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.segmentation import clear_border
from ..utils import normalize_and_dilate

# Cellpose imports (optional - gracefully handle missing or incompatible ML dependencies)
try:
    from cellpose import models
    from PIL import Image
    CELLPOSE_AVAILABLE = True
    CELLPOSE_IMPORT_ERROR = None
except Exception as error:
    CELLPOSE_AVAILABLE = False
    models = None
    Image = None
    CELLPOSE_IMPORT_ERROR = f"{type(error).__name__}: {error}"

# Cache for Cellpose models to avoid reloading on every detection call
# Key: (model_type, gpu), Value: CellposeModel instance
_CELLPOSE_MODEL_CACHE = {}

# Ultralytics is optional so the rest of the application can run without the
# YOLO runtime and its model weights.
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    YOLO_IMPORT_ERROR = None
except Exception as error:
    YOLO_AVAILABLE = False
    YOLO_IMPORT_ERROR = f"{type(error).__name__}: {error}"
    YOLO = None

# Key: model filename/path, Value: Ultralytics YOLO instance.
_YOLO_MODEL_CACHE = {}
_YOLO_EXPORT_DIR = Path(__file__).resolve().parents[3] / "yolo_exports"


def detect_blobs(img_norm, img_orig, min_thresh, min_area, color, 
                 file_name, method='simple', 
                 include_method_info=False, **kwargs):
    """
    General blob detection function that supports multiple detection methods.
    
    Parameters:
    -----------
    img_norm : np.ndarray
        Normalized image for detection
    img_orig : np.ndarray  
        Original image for intensity calculations
    min_thresh : float
        Minimum threshold for detection
    min_area : float
        Minimum area for blob filtering
    color : str
        Color label for the blobs
    file_name : str
        Name of the file being processed
    method : str
        Detection method to use. Options:
        - 'simple': OpenCV SimpleBlobDetector (default) - Good for general circular/elliptical blobs
        - 'contours': Contour-based detection - Good for irregular shapes
        - 'hough': Hough circle detection - Best for perfect circles
        - 'connected_components': Connected components labeling - Fast, good for well-separated objects
        - 'watershed': Watershed segmentation - Good for touching/overlapping objects
        - 'cellpose': Cellpose deep learning segmentation - Best for cells and complex biological objects
        - 'yolo': Ultralytics YOLO instance segmentation - Returns masks and boxes
    include_method_info : bool
        If True, includes 'method' key in output for compatibility (default: False)
    **kwargs : dict
        Additional method-specific parameters:
        
        For 'simple' method:
            max_threshold=255, max_area=1600, threshold_step=2,
            filter_by_color=False, filter_by_circularity=False, etc.
            
        For 'hough' method:
            max_radius=40, dp=1, min_dist=20, param1=50, param2=30
            
        For 'watershed' method:
            min_distance=10, threshold_abs=0.3
            
        For 'cellpose' method:
            diameter=60, model_type='cyto3', gpu=False, flow_threshold=0.4,
            cellprob_threshold=0.0, channels=[0,0], min_diameter=0, max_diameter=inf
        
    Returns:
    --------
    list : List of detected blob dictionaries with keys:
        'Box', 'center', 'radius', 'color', 'file', 
        'max_intensity', 'mean_intensity', 'mean_dilation',
        'box_x', 'box_y', 'box_size'
        (plus 'method' key if include_method_info=True)
        
    Examples:
    ---------
    # Basic usage (default simple method) - SAME OUTPUT FORMAT AS BEFORE
    blobs = detect_blobs(img_norm, img_orig, 50, 100, 'red', 'test.tiff')
    
    # Use contour detection for irregular shapes
    blobs = detect_blobs(img_norm, img_orig, 50, 100, 'red', 'test.tiff', method='contours')
    
    # Use Hough circles with custom parameters 
    blobs = detect_blobs(img_norm, img_orig, 50, 100, 'red', 'test.tiff', 
                        method='hough', max_radius=50, min_dist=30)
                        
    # Use Cellpose for biological samples
    blobs = detect_blobs(img_norm, img_orig, 50, 100, 'red', 'test.tiff',
                        method='cellpose', diameter=60, model_type='cyto3')
                        method='contours', include_method_info=True)
                        
    # Compare multiple methods (automatically includes method info)
    results = detect_blobs_multi_method(img_norm, img_orig, 50, 100, 'red', 'test.tiff',
                                       methods=['simple', 'contours', 'hough'])
    """
    
    # Method dispatch
    method_map = {
        'simple': _detect_blobs_simple,
        'contours': _detect_blobs_contours, 
        'hough': _detect_blobs_hough_circles,
        'connected_components': _detect_blobs_connected_components,
        'watershed': _detect_blobs_watershed,
        'cellpose': _detect_blobs_cellpose,
        'yolo': _detect_blobs_yolo,
    }
    
    if method not in method_map:
        raise ValueError(f"Unknown detection method: {method}. Available: {list(method_map.keys())}")
    
    # Special check for Cellpose availability
    if method == 'cellpose' and not CELLPOSE_AVAILABLE:
        raise ImportError(f"Cellpose not available. Install with: pip install cellpose[gui]")
    if method == 'yolo' and not YOLO_AVAILABLE:
        raise ImportError("Ultralytics YOLO is not available. Install with: pip install ultralytics")
    
    # Apply morphological preprocessing (normalize and dilate)
    # EXCEPTION: Skip for cellpose - deep learning models need raw/original images
    # Morphological dilation can destroy fine details that cellpose was trained to recognize
    if method in {'cellpose', 'yolo'}: #TODO not clean fix later
        # Use original images for cellpose (no morphological preprocessing)
        processed_norm = img_norm
        processed_dilated = img_orig
    else:
        # Apply morphological preprocessing for all other methods
        processed_norm, processed_dilated = normalize_and_dilate(img_orig, 
                                                                 kernel_size=3, 
                                                                 iterations=1)
    
    # Detect blobs using the selected method
    if method == 'yolo':
        kwargs = {**kwargs, 'output_name': file_name}
    detections = method_map[method](processed_dilated, processed_norm, min_thresh, min_area, **kwargs)
    
    # Convert detections to standard format
    blobs = []
    for idx, detection in enumerate(detections, start=1):
        x, y = detection['center']
        radius = detection['radius']
        box_size = 2 * radius
        box_x, box_y = x - radius, y - radius

        x1, y1 = max(0, box_x), max(0, box_y)
        x2, y2 = min(processed_norm.shape[1], x + radius), min(processed_norm.shape[0], y + radius)
        roi_orig = processed_norm[y1:y2, x1:x2]
        roi_dilated = processed_dilated[y1:y2, x1:x2]

        if roi_orig.size > 0:
            blob_dict = {
                'Box': f"{file_name} Box #{idx}",
                'center': (x, y),
                'radius': radius,
                'color': color,
                'file': file_name,
                'max_intensity': roi_orig.max(),
                'mean_intensity': roi_orig.mean(),
                'mean_dilation': float(roi_dilated.mean()),
                'box_x': box_x,
                'box_y': box_y,
                'box_size': box_size
            }
            
            # Only add method info if requested for backward compatibility
            if include_method_info:
                blob_dict['method'] = method
                
            blobs.append(blob_dict)
    
    return blobs


def _detect_blobs_yolo(img_norm, img_orig, min_thresh, min_area, **kwargs):
    """Run YOLO instance segmentation and return square scan boxes for its masks."""
    if not YOLO_AVAILABLE:
        raise ImportError("Ultralytics YOLO is not available. Install with: pip install ultralytics")

    model_name = kwargs.get('model', 'yolo26s-seg.pt')
    if model_name not in _YOLO_MODEL_CACHE:
        print(f"Loading YOLO segmentation model: {model_name}...")
        _YOLO_MODEL_CACHE[model_name] = YOLO(model_name)
    model = _YOLO_MODEL_CACHE[model_name]

    image = np.asarray(img_norm)
    if image.ndim == 2:
        image = np.repeat(image[..., np.newaxis], 3, axis=2)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    elif image.ndim == 3 and image.shape[2] >= 3:
        image = image[..., :3]
    else:
        raise ValueError(f"YOLO expects a 2D or RGB image, received shape {image.shape}")

    image = image.astype(np.float32, copy=False)
    image_min, image_max = float(np.nanmin(image)), float(np.nanmax(image))
    if image_max <= image_min:
        return []
    image = ((image - image_min) / (image_max - image_min) * 255).clip(0, 255).astype(np.uint8)

    predict_kwargs = {'verbose': False}
    if 'conf' in kwargs:
        predict_kwargs['conf'] = float(kwargs['conf'])
    if 'imgsz' in kwargs:
        predict_kwargs['imgsz'] = kwargs['imgsz']
    tile_size = int(kwargs.get('tile_size', max(image.shape[:2])))
    tile_overlap = int(kwargs.get('tile_overlap', 0))
    max_box_fraction = float(kwargs.get('max_box_fraction', 0.25))
    if tile_size <= 0:
        tile_size = max(image.shape[:2])
    if not 0 <= tile_overlap < tile_size:
        raise ValueError("YOLO tile_overlap must be non-negative and smaller than tile_size")
    if not 0 < max_box_fraction <= 1:
        raise ValueError("YOLO max_box_fraction must be greater than 0 and no larger than 1")
    max_box_size = min(image.shape[:2]) * max_box_fraction

    x_offsets = _yolo_tile_offsets(image.shape[1], tile_size, tile_overlap)
    y_offsets = _yolo_tile_offsets(image.shape[0], tile_size, tile_overlap)
    candidates = []
    for y_offset in y_offsets:
        for x_offset in x_offsets:
            tile = image[y_offset:y_offset + tile_size, x_offset:x_offset + tile_size]
            try:
                results = model(tile, **predict_kwargs)
            except Exception as error:
                print(f"YOLO detection failed: {error}")
                return []
            for result in results:
                candidates.extend(_yolo_result_records(
                    result, x_offset, y_offset, min_area,
                    max_box_size,
                ))

    export_records = []
    for record in sorted(candidates, key=lambda item: item['confidence'] or 0, reverse=True):
        if not any(_yolo_boxes_overlap(record['box'], existing['box']) for existing in export_records):
            export_records.append(record)

    detections = [
        {
            'center': (record['box']['x'] + record['box']['size'] // 2,
                       record['box']['y'] + record['box']['size'] // 2),
            'radius': record['box']['size'] // 2,
        }
        for record in export_records
    ]

    if kwargs.get('export_masks', True):
        _export_yolo_results(image, kwargs.get('output_name', 'yolo_image'), export_records)
    return detections


def _yolo_tile_offsets(length, tile_size, overlap):
    """Return tile starts that cover an image edge-to-edge."""
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    offsets = list(range(0, length - tile_size + 1, stride))
    final_offset = length - tile_size
    if offsets[-1] != final_offset:
        offsets.append(final_offset)
    return offsets


def _yolo_result_records(
    result, x_offset, y_offset, min_area, max_box_size,
):
    """Convert one tiled YOLO result into global-coordinate mask records."""
    if result.masks is None or result.masks.xy is None:
        return []
    confidences = result.boxes.conf.cpu().numpy() if result.boxes is not None else []
    class_ids = result.boxes.cls.cpu().numpy().astype(int) if result.boxes is not None else []
    records = []
    for index, polygon in enumerate(result.masks.xy):
        polygon = np.asarray(polygon)
        if polygon.size == 0 or cv2.contourArea(polygon.astype(np.float32)) < min_area:
            continue
        polygon = polygon.astype(np.int32)
        polygon[:, 0] += x_offset
        polygon[:, 1] += y_offset
        x1, y1 = (int(value) for value in np.floor(polygon.min(axis=0)))
        x2, y2 = (int(value) for value in np.ceil(polygon.max(axis=0)))
        width, height = x2 - x1, y2 - y1
        if width <= 0 or height <= 0:
            continue
        radius = int(np.ceil(max(width, height) / 2))
        if radius * 2 > max_box_size:
            continue
        class_id = int(class_ids[index]) if index < len(class_ids) else None
        class_name = result.names.get(class_id, str(class_id)) if class_id is not None else None
        records.append({
            'polygon': polygon,
            'confidence': float(confidences[index]) if index < len(confidences) else None,
            'class_id': class_id,
            'class_name': class_name,
            # The raw mask bounds are useful for inspecting YOLO's result.
            'yolo_box': {
                'x': x1, 'y': y1,
                'top_left': {'x': x1, 'y': y1},
                'width': width, 'height': height,
            },
            # This square box is what the GUI displays and the fine-scan queue uses.
            'box': {
                'x': x1, 'y': y1,
                'top_left': {'x': x1, 'y': y1},
                'size': radius * 2,
            },
        })
    return records


def _yolo_boxes_overlap(first, second, threshold=0.5):
    """Identify duplicate tile-edge predictions using square-box IoU."""
    left = max(first['x'], second['x'])
    top = max(first['y'], second['y'])
    right = min(first['x'] + first['size'], second['x'] + second['size'])
    bottom = min(first['y'] + first['size'], second['y'] + second['size'])
    intersection = max(0, right - left) * max(0, bottom - top)
    if not intersection:
        return False
    first_area, second_area = first['size'] ** 2, second['size'] ** 2
    return intersection / (first_area + second_area - intersection) >= threshold


def _export_yolo_results(image, output_name, records):
    """Save YOLO input, masks, overlays, and metadata outside version control."""
    _YOLO_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(output_name).stem
    mask_image = np.zeros(image.shape[:2], dtype=np.uint16)
    overlay = image.copy()
    metadata = []

    for label, record in enumerate(records, start=1):
        polygon = record.pop('polygon')
        cv2.fillPoly(mask_image, [polygon], label)
        color = ((37 * label) % 256, (97 * label) % 256, (173 * label) % 256)
        cv2.fillPoly(overlay, [polygon], color)
        cv2.polylines(overlay, [polygon], isClosed=True, color=(255, 255, 255), thickness=1)
        x, y, size = record['box']['x'], record['box']['y'], record['box']['size']
        cv2.rectangle(overlay, (x, y), (x + size, y + size), color=(255, 255, 255), thickness=1)
        if record['confidence'] is not None:
            cv2.putText(overlay, f"{record['confidence']:.2f}", (x, max(0, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        metadata.append(record)

    # A binary 0/255 image is easily visible in ordinary TIFF viewers.  The
    # labeled image is retained separately when per-instance IDs are needed.
    mask_preview = (mask_image > 0).astype(np.uint8) * 255
    tifffile.imwrite(_YOLO_EXPORT_DIR / f"{stem}_yolo_input.tiff", image)
    tifffile.imwrite(_YOLO_EXPORT_DIR / f"{stem}_yolo_masks.tiff", mask_preview)
    tifffile.imwrite(_YOLO_EXPORT_DIR / f"{stem}_yolo_instance_labels.tiff", mask_image)
    cv2.imwrite(
        str(_YOLO_EXPORT_DIR / f"{stem}_yolo_overlay.png"),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
    )
    with (_YOLO_EXPORT_DIR / f"{stem}_yolo_detections.json").open('w') as stream:
        json.dump(metadata, stream, indent=2)


def _detect_blobs_cellpose(img_norm, img_orig, min_thresh, min_area, **kwargs):
    """Cellpose-based blob detection for cell/particle segmentation"""
    if not CELLPOSE_AVAILABLE:
        raise ImportError("Cellpose not available. Install with: pip install cellpose")
    
    # Use img_orig (normalized but NOT dilated) because Cellpose is a deep learning model
    # trained on raw images. Morphological dilation can destroy fine details.
    # img_norm = dilated image (used for simple/contour methods)
    # img_orig = normalized but not dilated (better for deep learning models)
    cellpose_input = img_orig
    
    # Convert to format expected by Cellpose
    if len(cellpose_input.shape) == 2:
        # Convert grayscale to RGB format for Cellpose
        img_rgb = np.stack([cellpose_input, cellpose_input, cellpose_input], axis=2)
    else:
        img_rgb = cellpose_input.copy()
    
    # Normalize to [0,1] range
    img_min, img_max = float(img_rgb.min()), float(img_rgb.max())
    if img_max > img_min:
        img_rgb = (img_rgb - img_min) / (img_max - img_min)
    else:
        # Handle constant image
        return []
    
    # Cellpose parameters
    diameter_guess = kwargs.get('diameter', 30)
    model_type = kwargs.get('model_type', 'cyto3')
    gpu = kwargs.get('gpu', False)
    flow_threshold = kwargs.get('flow_threshold', 0.4)
    cellprob_threshold = kwargs.get('cellprob_threshold', 0.0)
    channels = kwargs.get('channels', [0, 0])  # [cytoplasm, nucleus] channels
    print(f"Running Cellpose with  '{model_type = }' "
          f"and {diameter_guess = }..., "
          f"{gpu = }")
    
    # Initialize model (with caching to avoid reloading)
    cache_key = (model_type, gpu)
    if cache_key not in _CELLPOSE_MODEL_CACHE:
        print(f"Loading Cellpose model: {model_type} (GPU={gpu})...")
        _CELLPOSE_MODEL_CACHE[cache_key] = models.CellposeModel(pretrained_model=model_type, gpu=gpu)
        print(f"Cellpose model loaded and cached.")
    else:
        print(f"Using cached Cellpose model: {model_type} (GPU={gpu})")
    
    model = _CELLPOSE_MODEL_CACHE[cache_key]
    
    # Run detection
    try:
        # Use min_size from kwargs if provided, otherwise fall back to min_area
        cellpose_min_size = kwargs.get('min_size', min_area)
        
        res = model.eval(
            img_rgb,
            channels=channels,
            diameter=diameter_guess,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            min_size=cellpose_min_size,
            batch_size=kwargs.get('batch_size', 8),
            resample=kwargs.get('resample', True),
            tile_overlap=kwargs.get('tile_overlap', 0.1),
            bsize=kwargs.get('bsize', 256),
            augment=kwargs.get('augment', False),
        )

        

        # res = model.eval(
        #     img_rgb,
        #     channels=[0,0],
        #     diameter=30,
        #     flow_threshold=0.4,
        #     cellprob_threshold=0)
        
        print(len(res))
        # Handle different return formats
        if len(res) == 4:
            masks, flows, styles, diams = res
        else:
            masks, flows, styles = res
        # import matplotlib.pyplot as plt
        # plt.imshow(masks)
        # plt.show()
            
    except Exception as e:
        print(f"Cellpose detection failed: {e}")
        return []
    masks = clear_border(masks) #to clear edge boxes
    # Convert masks to boxes and areas
    boxes, areas = _masks_to_boxes_and_areas(masks)
    
    # Filter by diameter range if specified
    min_diameter = kwargs.get('min_diameter', 0)
    max_diameter = kwargs.get('max_diameter', float('inf'))
    
    detections = []
    for box, area in zip(boxes, areas):
        # Check area threshold
        if area < min_area:
            continue
            
        # Check diameter threshold
        equiv_diameter = _area_to_equiv_diameter(area)
        if not (min_diameter <= equiv_diameter <= max_diameter):
            continue
        
        # Calculate center and radius from bounding box
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Use equivalent radius from area for consistency
        radius = equiv_diameter / 2
        
        detections.append({
            'center': (int(center_x), int(center_y)),
            'radius': int(radius),
            'area': area,
            'equiv_diameter': equiv_diameter,
            'bbox': box
        })
    
    return detections


def _detect_blobs_watershed(img_norm, img_orig, min_thresh, min_area, **kwargs):
    """Watershed segmentation for blob detection"""
    from scipy import ndimage
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max
    
    # Apply threshold
    _, binary = cv2.threshold(img_norm, min_thresh, 255, cv2.THRESH_BINARY)
    
    # Distance transform
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    
    # Find local maxima as markers
    local_max_coords = peak_local_max(
        dist_transform, 
        min_distance=kwargs.get('min_distance', 10),
        threshold_abs=kwargs.get('threshold_abs', 0.3 * dist_transform.max())
    )
    
    # Create markers
    markers = np.zeros_like(binary, dtype=np.int32)
    for i, (y, x) in enumerate(local_max_coords):
        markers[y, x] = i + 1
    
    # Apply watershed
    labels = watershed(-dist_transform, markers, mask=binary)
    
    detections = []
    for label_id in np.unique(labels):
        if label_id == 0:  # Skip background
            continue
        
        mask = labels == label_id
        area = np.sum(mask)
        
        if area >= min_area:
            # Calculate centroid
            y_coords, x_coords = np.where(mask)
            x = int(np.mean(x_coords))
            y = int(np.mean(y_coords))
            radius = int(np.sqrt(area / np.pi))
            detections.append({'center': (x, y), 'radius': radius})
    
    return detections


# Helper functions for convenient method-specific detection

def _masks_to_boxes_and_areas(masks):
    """
    Convert Cellpose masks to bounding boxes and areas.
    
    Returns:
        boxes: list of (x1, y1, x2, y2)
        areas: list of mask pixel areas (same order as boxes)
    """
    boxes, areas = [], []
    ids = np.unique(masks)
    ids = ids[ids != 0]  # Skip background
    
    for i in ids:
        ys, xs = np.where(masks == i)
        if xs.size == 0:
            continue
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        boxes.append((x1, y1, x2, y2))
        areas.append(int(xs.size))
        
    return boxes, areas

def detect_blobs_simple(img_norm, img_orig, min_thresh, min_area, color, file_name, include_method_info=False, **kwargs):
    """Convenient wrapper for simple blob detection"""
    return detect_blobs(img_norm, img_orig, min_thresh, min_area, color, file_name, 
                       method='simple', include_method_info=include_method_info, **kwargs)

def detect_blobs_contours(img_norm, img_orig, min_thresh, min_area, color, file_name, include_method_info=False, **kwargs):
    """Convenient wrapper for contour-based blob detection"""
    return detect_blobs(img_norm, img_orig, min_thresh, min_area, color, file_name,
                       method='contours', include_method_info=include_method_info, **kwargs)

def detect_blobs_hough(img_norm, img_orig, min_thresh, min_area, color, file_name, include_method_info=False, **kwargs):  
    """Convenient wrapper for Hough circle detection"""
    return detect_blobs(img_norm, img_orig, min_thresh, min_area, color, file_name,
                       method='hough', include_method_info=include_method_info, **kwargs)

def detect_blobs_connected_components(img_norm, img_orig, min_thresh, min_area, color, file_name, include_method_info=False, **kwargs):
    """Convenient wrapper for connected components detection"""
    return detect_blobs(img_norm, img_orig, min_thresh, min_area, color, file_name,
                       method='connected_components', include_method_info=include_method_info, **kwargs)

def detect_blobs_watershed(img_norm, img_orig, min_thresh, min_area, color, file_name, include_method_info=False, **kwargs):
    """Convenient wrapper for watershed segmentation detection"""
    return detect_blobs(img_norm, img_orig, min_thresh, min_area, color, file_name,
                       method='watershed', include_method_info=include_method_info, **kwargs)

def detect_blobs_cellpose(img_norm, img_orig, min_thresh, min_area, color, file_name, include_method_info=False, **kwargs):
    """Convenient wrapper for Cellpose deep learning segmentation"""
    return detect_blobs(img_norm, img_orig, min_thresh, min_area, color, file_name,
                       method='cellpose', include_method_info=include_method_info, **kwargs)


def detect_blobs_yolo(img_norm, img_orig, min_thresh, min_area, color, file_name, include_method_info=False, **kwargs):
    """Convenient wrapper for Ultralytics YOLO instance segmentation."""
    return detect_blobs(img_norm, img_orig, min_thresh, min_area, color, file_name,
                       method='yolo', include_method_info=include_method_info, **kwargs)


def get_available_detection_methods():
    """Returns list of available detection methods"""
    methods = ['simple', 'contours', 'hough', 'connected_components', 'watershed']
    if CELLPOSE_AVAILABLE:
        methods.append('cellpose')
    if YOLO_AVAILABLE:
        methods.append('yolo')
    return methods


def detect_blobs_multi_method(img_norm, img_orig, min_thresh, min_area, color, file_name, 
                             methods=['simple'], combine_results=True, **kwargs):
    """
    Apply multiple detection methods and optionally combine results.
    
    Parameters:
    -----------
    methods : list
        List of detection methods to apply
    combine_results : bool  
        If True, combine all results into single list. If False, return dict by method.
    **kwargs : dict
        Additional parameters for detection methods
        
    Returns:
    --------
    list or dict : Combined results or dict of results by method
    """
    all_results = {}
    
    for method in methods:
        try:
            blobs = detect_blobs(img_norm, img_orig, min_thresh, min_area, color, file_name,
                               method=method, include_method_info=True, **kwargs)
            all_results[method] = blobs
            print(f"Method '{method}': Found {len(blobs)} blobs")
        except Exception as e:
            print(f"Error with method '{method}': {e}")
            all_results[method] = []
    
    if combine_results:
        # Combine all results (method info already included via include_method_info=True)
        combined_blobs = []
        for method, blobs in all_results.items():
            combined_blobs.extend(blobs)
        return combined_blobs
    
    return all_results


# ---------------- File wrapper ----------------
def _detect_blobs_simple(img_norm, img_orig, min_thresh, min_area, **kwargs):
    """Simple blob detector method (OpenCV SimpleBlobDetector)"""
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = min_thresh
    params.maxThreshold = kwargs.get('max_threshold', 255)
    params.filterByArea = True
    params.minArea = min_area
    params.maxArea = kwargs.get('max_area', 1600)
    params.thresholdStep = kwargs.get('threshold_step', 2)

    params.filterByColor = kwargs.get('filter_by_color', False)
    params.filterByCircularity = kwargs.get('filter_by_circularity', False)
    params.filterByInertia = kwargs.get('filter_by_inertia', False)
    params.filterByConvexity = kwargs.get('filter_by_convexity', False)
    params.minRepeatability = kwargs.get('min_repeatability', 1)
    
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img_norm)
    
    detections = []
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        radius = int(kp.size / 2)
        detections.append({'center': (x, y), 'radius': radius})
    
    return detections

def _detect_blobs_contours(img_norm, img_orig, min_thresh, min_area, **kwargs):
    """Contour-based blob detection"""
    # Apply threshold
    _, binary = cv2.threshold(img_norm, min_thresh, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            # Get bounding circle
            (x, y), radius = cv2.minEnclosingCircle(contour)
            detections.append({'center': (int(x), int(y)), 'radius': int(radius)})
    
    return detections

def _detect_blobs_hough_circles(img_norm, img_orig, min_thresh, min_area, **kwargs):
    """Hough circle detection for circular blobs"""
    # Convert min_area to min_radius (assuming circular blobs)
    min_radius = int(np.sqrt(min_area / np.pi))
    max_radius = kwargs.get('max_radius', 40)
    
    circles = cv2.HoughCircles(
        img_norm,
        cv2.HOUGH_GRADIENT,
        dp=kwargs.get('dp', 1),
        minDist=kwargs.get('min_dist', min_radius * 2),
        param1=kwargs.get('param1', 50),
        param2=kwargs.get('param2', 30),
        minRadius=min_radius,
        maxRadius=max_radius
    )
    
    detections = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            detections.append({'center': (x, y), 'radius': r})
    
    return detections

def _detect_blobs_connected_components(img_norm, img_orig, min_thresh, min_area, **kwargs):
    """Connected components labeling for blob detection"""
    # Apply threshold
    _, binary = cv2.threshold(img_norm, min_thresh, 255, cv2.THRESH_BINARY)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    detections = []
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            x, y = int(centroids[i][0]), int(centroids[i][1])
            # Estimate radius from area
            radius = int(np.sqrt(area / np.pi))
            detections.append({'center': (x, y), 'radius': radius})
    
    return detections

def _area_to_equiv_diameter(area_px):
    """Convert area to equivalent circle diameter: A = π (d/2)^2  -> d = 2*sqrt(A/π)"""
    return 2.0 * np.sqrt(area_px / np.pi)
