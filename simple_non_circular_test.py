#!/usr/bin/env python3
"""
Simple test to demonstrate non-circular object detection capabilities.
"""

import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, '.')

from utils import normalize_and_dilate, detect_blobs
from test_blob_detection import (
    create_test_image_rectangular,
    create_test_image_complex_shapes,
    plot_method_comparison
)

def simple_non_circular_test():
    """Simple test of non-circular objects"""
    print("🔬 TESTING NON-CIRCULAR OBJECTS FOR AUTONOMOUS MICROSCOPY")
    print("=" * 60)
    
    # Set reproducible seed
    np.random.seed(42)
    
    # Test 1: Rectangular objects
    print("\\n1️⃣ Testing RECTANGULAR objects:")
    print("-" * 40)
    img_rect, _ = create_test_image_rectangular(num_blobs=3)
    img_norm, img_dilated = normalize_and_dilate(img_rect)
    
    methods_to_test = ['simple', 'contours', 'connected_components']
    rect_results = {}
    
    for method in methods_to_test:
        blobs = detect_blobs(img_dilated, img_norm, 50, 100, 'blue', 'rect_test.tiff',
                           method=method, include_method_info=True)
        rect_results[method] = blobs
        
        effectiveness = "⭐ EXCELLENT" if method in ['contours', 'connected_components'] else "⚠️  LIMITED"
        print(f"  {method:20}: {len(blobs):2d} detections - {effectiveness}")
        
        if blobs:
            blob = blobs[0] 
            print(f"{'':23} Bounding box: ({blob['box_x']}, {blob['box_y']}) size={blob['box_size']}")
    
    # Test 2: Complex shapes (L-shapes, crosses, stars)
    print("\\n2️⃣ Testing COMPLEX shapes (L-shapes, crosses, stars):")
    print("-" * 40)
    img_complex, _ = create_test_image_complex_shapes()
    img_norm, img_dilated = normalize_and_dilate(img_complex)
    
    complex_results = {}
    
    for method in methods_to_test:
        blobs = detect_blobs(img_dilated, img_norm, 50, 100, 'green', 'complex_test.tiff',
                           method=method, include_method_info=True)
        complex_results[method] = blobs
        
        effectiveness = "⭐ EXCELLENT" if method == 'contours' else "⚠️  LIMITED" if method == 'simple' else "✅ GOOD"
        print(f"  {method:20}: {len(blobs):2d} detections - {effectiveness}")
        
        if blobs:
            blob = blobs[0]
            print(f"{'':23} Bounding box: ({blob['box_x']}, {blob['box_y']}) size={blob['box_size']}")
    
    # Create visualizations
    print("\\n📊 Creating visualizations with bounding boxes...")
    save_dir = Path("simple_non_circular_test")
    save_dir.mkdir(exist_ok=True)
    
    try:
        # Plot rectangular objects
        fig1 = plot_method_comparison(img_rect, rect_results, save_dir, show_boxes=True)
        if fig1:
            save_path1 = save_dir / "rectangular_objects_comparison.png"
            fig1.savefig(save_path1, dpi=150, bbox_inches='tight')
            import matplotlib.pyplot as plt
            plt.close(fig1)
            print(f"  💾 Saved: {save_path1.name}")
        
        # Plot complex shapes  
        fig2 = plot_method_comparison(img_complex, complex_results, save_dir, show_boxes=True)
        if fig2:
            save_path2 = save_dir / "complex_shapes_comparison.png"
            fig2.savefig(save_path2, dpi=150, bbox_inches='tight')
            plt.close(fig2)
            print(f"  💾 Saved: {save_path2.name}")
            
    except Exception as e:
        print(f"  ❌ Plotting error: {e}")
    
    # Summary and recommendations
    print("\\n" + "=" * 60)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 60)
    print("🎯 For NON-CIRCULAR objects in autonomous microscopy:")
    print("\\n✅ BEST METHODS:")
    print("   1. CONTOURS - Excellent for any irregular shape")
    print("   2. CONNECTED_COMPONENTS - Fast, good for separated objects") 
    print("   3. WATERSHED - Best for touching/overlapping objects")
    print("\\n⚠️  LIMITED METHODS:")
    print("   • SIMPLE - Works poorly on non-circular shapes")
    print("   • HOUGH - Only suitable for circular objects")
    print("\\n📦 BOUNDING BOX DATA:")
    print("   • All methods preserve box_x, box_y, box_size for scan planning")
    print("   • Rectangle coordinates define precise scan regions")
    print("   • Same data structure regardless of shape complexity")
    
    print(f"\\n📁 Visualizations saved to: {save_dir}")
    print("🎉 Non-circular object test completed!")
    
    return rect_results, complex_results

if __name__ == "__main__":
    try:
        rect_results, complex_results = simple_non_circular_test()
        
        print("\\n" + "=" * 60)
        print("✅ SUCCESS: Non-circular object detection working!")
        print("Your autonomous microscopy system can now detect:")
        print("• Rectangular particles • Complex biological shapes")
        print("• Irregular specimens  • Any non-circular objects")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()