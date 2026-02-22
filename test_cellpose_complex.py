#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from utils import normalize_and_dilate, detect_blobs, CELLPOSE_AVAILABLE
from test_blob_detection import create_test_image_complex_shapes
import numpy as np

print('🧪 TESTING CELLPOSE ON COMPLEX SHAPES')
print('=' * 50)
print(f'Cellpose Available: {CELLPOSE_AVAILABLE}')

if CELLPOSE_AVAILABLE:
    np.random.seed(42)
    img, shape_info = create_test_image_complex_shapes()
    img_norm, img_dilated = normalize_and_dilate(img)
    
    try:
        blobs = detect_blobs(img_dilated, img_norm, 50, 100, 'green', 'complex_test.tiff',
                           method='cellpose', include_method_info=True)
        print(f'✅ Cellpose detections on complex shapes: {len(blobs)}')
        if blobs:
            blob = blobs[0]
            print(f'   First detection: Box({blob["box_x"]}, {blob["box_y"]}) size={blob["box_size"]}')
            print(f'   Shape detected: {shape_info[0]["type"] if shape_info else "unknown"}')
    except Exception as e:
        print(f'❌ Cellpose on complex shapes: {type(e).__name__} - {str(e)[:80]}...')
        
    print('\n🔬 ANALYSIS:')
    print('   Cellpose is a deep learning model trained on biological specimens.')
    print('   It works best on cells, nuclei, and organic shapes.')
    print('   Synthetic geometric shapes (L-shapes, crosses, stars) are outside')
    print('   its training domain and may produce unreliable results.')
else:
    print('❌ Cellpose not available')

print('\n🎯 RECOMMENDATION:')
print('   For geometric/synthetic shapes: Use contours or watershed methods')
print('   For biological specimens: Cellpose is excellent')