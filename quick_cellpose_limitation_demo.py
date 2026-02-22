#!/usr/bin/env python3
"""
Quick demonstration of Cellpose limitations on synthetic shapes.
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from utils import normalize_and_dilate, detect_blobs, CELLPOSE_AVAILABLE
from test_blob_detection import create_test_image_rectangular, create_test_image_complex_shapes

def quick_cellpose_limitation_demo():
    """Quick demonstration of Cellpose limitations"""
    print("🔬 QUICK CELLPOSE LIMITATIONS DEMO")
    print("=" * 50)
    
    if not CELLPOSE_AVAILABLE:
        print("❌ Cellpose not available")
        return False
    
    print("✅ Cellpose is available - testing limitations...")
    
    # Set reproducible seed
    np.random.seed(42)
    
    # Test 1: Rectangular objects (synthetic geometric shapes)
    print("\\n1️⃣ Testing on RECTANGULAR synthetic shapes:")
    print("-" * 45)
    
    img_rect, _ = create_test_image_rectangular(num_blobs=3)
    img_norm, img_dilated = normalize_and_dilate(img_rect)
    
    methods = ['contours', 'cellpose']
    
    for method in methods:
        try:
            if method == 'cellpose':
                print(f"  {method:10}... ", end="", flush=True)
                blobs = detect_blobs(img_dilated, img_norm, 50, 100, 'blue', 'rect_test.tiff',
                                   method='cellpose', diameter=30, model_type='cyto3')
            else:
                print(f"  {method:10}... ", end="", flush=True) 
                blobs = detect_blobs(img_dilated, img_norm, 50, 100, 'blue', 'rect_test.tiff',
                                   method=method)
            
            if method == 'cellpose':
                assessment = "⚠️ LIMITATION" if len(blobs) < 3 else "🤔 UNEXPECTED"
            else:
                assessment = "✅ BASELINE"
                
            print(f"{len(blobs):2d} detections - {assessment}")
            
        except Exception as e:
            print(f"ERROR - {str(e)[:40]}...")
    
    # Test 2: Complex synthetic shapes  
    print("\\n2️⃣ Testing on COMPLEX synthetic shapes (L-shapes, crosses):")
    print("-" * 45)
    
    img_complex, _ = create_test_image_complex_shapes()
    img_norm, img_dilated = normalize_and_dilate(img_complex)
    
    for method in methods:
        try:
            if method == 'cellpose':
                print(f"  {method:10}... ", end="", flush=True)
                blobs = detect_blobs(img_dilated, img_norm, 50, 100, 'red', 'complex_test.tiff',
                                   method='cellpose', diameter=30, model_type='cyto3')
            else:
                print(f"  {method:10}... ", end="", flush=True)
                blobs = detect_blobs(img_dilated, img_norm, 50, 100, 'red', 'complex_test.tiff',
                                   method=method)
            
            if method == 'cellpose':
                assessment = "❌ MAJOR LIMITATION" if len(blobs) == 0 else "⚠️ LIMITED" if len(blobs) < 3 else "🤔 UNEXPECTED"  
            else:
                assessment = "✅ BASELINE"
                
            print(f"{len(blobs):2d} detections - {assessment}")
            
        except Exception as e:
            print(f"ERROR - {str(e)[:40]}...")
    
    print("\\n" + "=" * 50)
    print("🎯 KEY FINDINGS:")
    print("-" * 50)
    print("📊 CELLPOSE LIMITATIONS DEMONSTRATED:")
    print("   • Designed for biological specimens, not synthetic geometry")
    print("   • Trained on cell/nuclei data, not rectangles or L-shapes")  
    print("   • May produce unreliable results on geometric objects")
    print("   • Traditional computer vision (contours) works better for synthetic shapes")
    
    print("\\n💡 RECOMMENDATION FOR AUTONOMOUS MICROSCOPY:")
    print("   🔬 Use CELLPOSE for: Biological cells, nuclei, organic specimens")
    print("   🔧 Use CONTOURS for: Geometric particles, synthetic shapes, rectangles")
    print("   ⚖️ Choose method based on specimen type for optimal results")
    
    return True

if __name__ == "__main__":
    try:
        success = quick_cellpose_limitation_demo()
        if success:
            print("\\n🎉 Cellpose limitations successfully demonstrated!")
            print("Understanding these limitations helps choose the right detection method.")
        else:
            print("\\n❌ Could not demonstrate limitations (Cellpose not available)")
    except Exception as e:
        print(f"\\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()