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
                is_recommended = method in test_case['best_methods']_case['best_methods']\n                status = \"⭐ (recommended)\" if is_recommended else \"\"\n                \n                print(f\"  {method:20}: {len(blobs):2d} detections {status}\")\n                \n                # Show bounding box info for first detection\n                if blobs:\n                    blob = blobs[0]\n                    box_info = f\"Box: ({blob['box_x']}, {blob['box_y']}) {blob['box_size']}×{blob['box_size']}\"\n                    print(f\"{'':22}   First detection - {box_info}\")\n                \n            except Exception as e:\n                print(f\"  {method:20}: ❌ Error - {str(e)[:40]}...\")\n                case_results[method] = []\n        \n        # Create visualization\n        print(f\"  📊 Creating visualization...\")\n        plot_results = {k: v for k, v in case_results.items() if v}\n        if plot_results:\n            try:\n                fig = plot_method_comparison(img, plot_results, save_dir=save_dir, show_boxes=True)\n                if fig:\n                    # Save with descriptive filename\n                    filename = f\"{i:02d}_{test_case['name'].lower().replace(' ', '_')}_comparison.png\"\n                    save_path = save_dir / filename\n                    fig.savefig(save_path, dpi=150, bbox_inches='tight')\n                    import matplotlib.pyplot as plt\n                    plt.close(fig)\n                    print(f\"    💾 Saved: {filename}\")\n            except Exception as e:\n                print(f\"    ❌ Plotting error: {e}\")\n        \n        overall_results[test_case['name']] = case_results\n    \n    # Performance analysis\n    print(\"\\n\" + \"=\" * 60)\n    print(\"PERFORMANCE ANALYSIS\")\n    print(\"=\" * 60)\n    \n    method_scores = {method: [] for method in methods}\n    \n    # Calculate detection success rate for each method\n    for case_name, case_results in overall_results.items():\n        for method in methods:\n            detection_count = len(case_results.get(method, []))\n            method_scores[method].append(detection_count)\n    \n    print(\"\\nAverage detections per test case:\")\n    for method in methods:\n        if method_scores[method]:\n            avg_score = np.mean(method_scores[method])\n            total_detections = sum(method_scores[method])\n            print(f\"  {method:20}: {avg_score:4.1f} avg ({total_detections:2d} total)\")\n    \n    # Method recommendations\n    print(\"\\n🎯 METHOD RECOMMENDATIONS FOR NON-CIRCULAR OBJECTS:\")\n    print(\"-\" * 60)\n    \n    recommendations = {\n        'contours': {\n            'description': 'Best overall for irregular shapes',\n            'strengths': ['Handles complex boundaries', 'Good for any shape', 'Reliable'],\n            'use_cases': ['Irregular biological specimens', 'Complex particles', 'Any non-circular object']\n        },\n        'watershed': {\n            'description': 'Excellent for touching/complex objects', \n            'strengths': ['Separates touching objects', 'Good segmentation', 'Handles concave shapes'],\n            'use_cases': ['Overlapping particles', 'Complex cellular structures', 'Clustered objects']\n        },\n        'connected_components': {\n            'description': 'Fast and reliable for separated objects',\n            'strengths': ['Very fast', 'Simple and robust', 'Good for well-separated objects'],\n            'use_cases': ['Isolated particles', 'Non-overlapping specimens', 'Simple shapes']\n        },\n        'simple': {\n            'description': 'Limited effectiveness on non-circular shapes',\n            'strengths': ['Fast', 'Good parameter control'],\n            'use_cases': ['Approximately circular objects only']\n        },\n        'hough': {\n            'description': 'Not suitable for non-circular objects',\n            'strengths': ['Perfect for circles'],\n            'use_cases': ['Circular objects only']\n        }\n    }\n    \n    for method, info in recommendations.items():\n        print(f\"\\n{method.upper()}:\")\n        print(f\"  📝 {info['description']}\")\n        print(f\"  ✅ Strengths: {', '.join(info['strengths'])}\")\n        print(f\"  🎯 Use cases: {', '.join(info['use_cases'])}\")\n    \n    # Final summary\n    print(\"\\n\" + \"=\" * 60)\n    print(\"SUMMARY FOR AUTONOMOUS MICROSCOPY\")\n    print(\"=\" * 60)\n    print(\"🔬 For non-circular biological specimens and particles:\")\n    print(\"   1️⃣  PRIMARY: Use 'contours' method for most reliable detection\")\n    print(\"   2️⃣  BACKUP: Use 'watershed' for touching/overlapping objects\")\n    print(\"   3️⃣  FAST: Use 'connected_components' for well-separated simple shapes\")\n    print(\"\\n📊 All bounding box coordinates (box_x, box_y, box_size) are preserved\")\n    print(\"   for scan planning regardless of detection method used.\")\n    \n    print(f\"\\n📁 All visualizations saved to: {save_dir}\")\n    \n    # List generated files\n    plot_files = list(save_dir.glob(\"*.png\"))\n    if plot_files:\n        print(f\"\\n📋 Generated {len(plot_files)} visualization files:\")\n        for i, file_path in enumerate(sorted(plot_files), 1):\n            print(f\"   {i}. {file_path.name}\")\n    \n    print(\"\\n🎉 Non-circular object detection demo completed!\")\n    \n    return overall_results\n\n\nif __name__ == \"__main__\":\n    try:\n        results = demo_non_circular_detection()\n        print(\"\\n\" + \"=\" * 60)\n        print(\"Demo completed successfully! Check the generated plots to see\")\n        print(\"how each method handles different non-circular object types.\")\n        print(\"=\" * 60)\n    except Exception as e:\n        print(f\"❌ Demo failed: {e}\")\n        import traceback\n        traceback.print_exc()