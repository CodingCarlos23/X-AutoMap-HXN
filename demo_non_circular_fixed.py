#!/usr/bin/env python3
"""
Demo script specifically for testing blob detection on non-circular objects.
Shows how different detection methods handle various shapes for autonomous microscopy.
"""

import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, '.')

from utils import normalize_and_dilate, detect_blobs, get_available_detection_methods
from test_blob_detection import (
    create_test_image_irregular,
    create_test_image_rectangular, 
    create_test_image_polygonal,
    create_test_image_complex_shapes,
    plot_method_comparison
)

def demo_non_circular_detection():
    """Comprehensive demo of non-circular object detection"""
    print("🔍 NON-CIRCULAR OBJECT DETECTION DEMO")
    print("=" * 60)
    print("Testing blob detection methods on various non-circular shapes")
    print("for autonomous microscopy applications.")
    print("=" * 60)
    
    # Set reproducible seed
    np.random.seed(42)
    
    # Get available methods (exclude Cellpose for synthetic shapes)
    methods = ['simple', 'contours', 'hough', 'connected_components', 'watershed']
    print(f"Testing methods: {', '.join(methods)}")
    
    # Create output directory
    save_dir = Path("non_circular_tests")
    save_dir.mkdir(exist_ok=True)
    print(f"📁 Output directory: {save_dir}")
    
    # Test cases with different shape types
    test_cases = [
        {
            'name': 'Irregular Ellipses',
            'description': 'Elliptical shapes with random orientations',
            'create_func': lambda: create_test_image_irregular(num_blobs=4),
            'best_methods': ['contours', 'connected_components']
        },
        {
            'name': 'Rectangular Objects', 
            'description': 'Various sized rectangles',
            'create_func': lambda: create_test_image_rectangular(num_blobs=4),
            'best_methods': ['contours', 'connected_components', 'watershed']
        },
        {
            'name': 'Polygonal Shapes',
            'description': 'Triangles, pentagons, hexagons with irregular edges',
            'create_func': lambda: create_test_image_polygonal(num_blobs=3),
            'best_methods': ['contours', 'watershed']
        },
        {
            'name': 'Complex Shapes',
            'description': 'L-shapes, crosses, stars, kidney shapes',
            'create_func': lambda: create_test_image_complex_shapes(),
            'best_methods': ['contours', 'watershed']
        }
    ]
    
    overall_results = {}
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\\n{i}. Testing {test_case['name']}:")
        print(f"   {test_case['description']}")
        print("-" * 50)
        
        # Create test image
        img, shape_info = test_case['create_func']()
        img_norm, img_dilated = normalize_and_dilate(img)
        
        case_results = {}
        
        # Test each method
        for method in methods:
            try:
                blobs = detect_blobs(img_dilated, img_norm, 50, 100, 'blue', 
                                   f"{test_case['name'].lower()}_test.tiff",
                                   method=method, include_method_info=True)
                case_results[method] = blobs
                
                # Check if this is a recommended method for this shape type
                is_recommended = method in test_case['best_methods']
                status = "⭐ (recommended)" if is_recommended else ""
                
                print(f"  {method:20}: {len(blobs):2d} detections {status}")
                
                # Show bounding box info for first detection
                if blobs:
                    blob = blobs[0]
                    box_info = f"Box: ({blob['box_x']}, {blob['box_y']}) {blob['box_size']}×{blob['box_size']}"
                    print(f"{'':22}   First detection - {box_info}")
                
            except Exception as e:
                print(f"  {method:20}: ❌ Error - {str(e)[:40]}...")
                case_results[method] = []
        
        # Create visualization
        print(f"  📊 Creating visualization...")
        plot_results = {k: v for k, v in case_results.items() if v}
        if plot_results:
            try:
                fig = plot_method_comparison(img, plot_results, save_dir=save_dir, show_boxes=True)
                if fig:
                    # Save with descriptive filename
                    filename = f"{i:02d}_{test_case['name'].lower().replace(' ', '_')}_comparison.png"
                    save_path = save_dir / filename
                    fig.savefig(save_path, dpi=150, bbox_inches='tight')
                    import matplotlib.pyplot as plt
                    plt.close(fig)
                    print(f"    💾 Saved: {filename}")
            except Exception as e:
                print(f"    ❌ Plotting error: {e}")
        
        overall_results[test_case['name']] = case_results
    
    # Performance analysis
    print("\\n" + "=" * 60)
    print("PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    method_scores = {method: [] for method in methods}
    
    # Calculate detection success rate for each method
    for case_name, case_results in overall_results.items():
        for method in methods:
            detection_count = len(case_results.get(method, []))
            method_scores[method].append(detection_count)
    
    print("\\nAverage detections per test case:")
    for method in methods:
        if method_scores[method]:
            avg_score = np.mean(method_scores[method])
            total_detections = sum(method_scores[method])
            print(f"  {method:20}: {avg_score:4.1f} avg ({total_detections:2d} total)")
    
    # Method recommendations
    print("\\n🎯 METHOD RECOMMENDATIONS FOR NON-CIRCULAR OBJECTS:")
    print("-" * 60)
    
    recommendations = {
        'contours': {
            'description': 'Best overall for irregular shapes',
            'strengths': ['Handles complex boundaries', 'Good for any shape', 'Reliable'],
            'use_cases': ['Irregular biological specimens', 'Complex particles', 'Any non-circular object']
        },
        'watershed': {
            'description': 'Excellent for touching/complex objects', 
            'strengths': ['Separates touching objects', 'Good segmentation', 'Handles concave shapes'],
            'use_cases': ['Overlapping particles', 'Complex cellular structures', 'Clustered objects']
        },
        'connected_components': {
            'description': 'Fast and reliable for separated objects',
            'strengths': ['Very fast', 'Simple and robust', 'Good for well-separated objects'],
            'use_cases': ['Isolated particles', 'Non-overlapping specimens', 'Simple shapes']
        },
        'simple': {
            'description': 'Limited effectiveness on non-circular shapes',
            'strengths': ['Fast', 'Good parameter control'],
            'use_cases': ['Approximately circular objects only']
        },
        'hough': {
            'description': 'Not suitable for non-circular objects',
            'strengths': ['Perfect for circles'],
            'use_cases': ['Circular objects only']
        }
    }
    
    for method, info in recommendations.items():
        print(f"\\n{method.upper()}:")
        print(f"  📝 {info['description']}")
        print(f"  ✅ Strengths: {', '.join(info['strengths'])}")
        print(f"  🎯 Use cases: {', '.join(info['use_cases'])}")
    
    # Final summary
    print("\\n" + "=" * 60)
    print("SUMMARY FOR AUTONOMOUS MICROSCOPY")
    print("=" * 60)
    print("🔬 For non-circular biological specimens and particles:")
    print("   1️⃣  PRIMARY: Use 'contours' method for most reliable detection")
    print("   2️⃣  BACKUP: Use 'watershed' for touching/overlapping objects")
    print("   3️⃣  FAST: Use 'connected_components' for well-separated simple shapes")
    print("\\n📊 All bounding box coordinates (box_x, box_y, box_size) are preserved")
    print("   for scan planning regardless of detection method used.")
    
    print(f"\\n📁 All visualizations saved to: {save_dir}")
    
    # List generated files
    plot_files = list(save_dir.glob("*.png"))
    if plot_files:
        print(f"\\n📋 Generated {len(plot_files)} visualization files:")
        for i, file_path in enumerate(sorted(plot_files), 1):
            print(f"   {i}. {file_path.name}")
    
    print("\\n🎉 Non-circular object detection demo completed!")
    
    return overall_results


if __name__ == "__main__":
    try:
        results = demo_non_circular_detection()
        print("\\n" + "=" * 60)
        print("Demo completed successfully! Check the generated plots to see")
        print("how each method handles different non-circular object types.")
        print("=" * 60)
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()