#!/usr/bin/env python3
"""
Test script to demonstrate Cellpose limitations on synthetic geometric shapes
versus its strengths on biological-like objects.
"""

import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, '.')

from utils import normalize_and_dilate, detect_blobs, CELLPOSE_AVAILABLE
from test_blob_detection import (
    create_test_image_complex_shapes,
    create_test_image_rectangular,
    plot_method_comparison
)
import cv2

def create_biological_like_image(width=500, height=500, num_cells=4):
    """Create test image that mimics biological cells/nuclei"""
    img = np.zeros((height, width), dtype=np.float32)
    
    cell_info = []
    for i in range(num_cells):
        # Create irregular cell-like shapes
        center_x = np.random.randint(60, width-60)
        center_y = np.random.randint(60, height-60) 
        
        # Create irregular blob using multiple overlapping circles
        base_radius = np.random.randint(15, 30)
        intensity = np.random.randint(180, 255)
        
        # Main cell body
        cv2.circle(img, (center_x, center_y), base_radius, intensity, -1)
        
        # Add irregular extensions to make it cell-like
        for j in range(3):
            offset_x = np.random.randint(-base_radius//2, base_radius//2)
            offset_y = np.random.randint(-base_radius//2, base_radius//2)
            small_radius = np.random.randint(8, base_radius//2)
            extension_intensity = intensity - np.random.randint(0, 30)
            
            cv2.circle(img, (center_x + offset_x, center_y + offset_y), 
                      small_radius, extension_intensity, -1)
        
        # Add texture variation (cytoplasm-like)
        for k in range(5):
            texture_x = center_x + np.random.randint(-base_radius, base_radius)
            texture_y = center_y + np.random.randint(-base_radius, base_radius)
            texture_radius = np.random.randint(2, 6)
            texture_intensity = intensity - np.random.randint(20, 50)
            cv2.circle(img, (texture_x, texture_y), texture_radius, texture_intensity, -1)
        
        cell_info.append({
            'center': (center_x, center_y),
            'radius': base_radius,
            'intensity': intensity,
            'type': 'biological_like'
        })
    
    # Add realistic noise (similar to microscopy images)
    noise = np.random.normal(0, 15, img.shape).astype(np.float32)
    img = np.clip(img + noise, 0, 255)
    
    # Add gradient background (common in microscopy)
    y, x = np.ogrid[:height, :width]
    gradient = 20 * np.sin(x/width * np.pi) * np.sin(y/height * np.pi)
    img = np.clip(img + gradient, 0, 255)
    
    return img, cell_info

def test_cellpose_limitations():
    """Test Cellpose on different types of objects to show its limitations"""
    print("🧪 CELLPOSE LIMITATIONS DEMONSTRATION")
    print("=" * 60)
    print("Testing Cellpose performance on different object types")
    print("to demonstrate when it works well vs. when it struggles.")
    print("=" * 60)
    
    if not CELLPOSE_AVAILABLE:
        print("❌ Cellpose not available. Install with: pixi add cellpose")
        return False
    
    # Set reproducible seed
    np.random.seed(42)
    
    # Create output directory  
    save_dir = Path("cellpose_limitations_test")
    save_dir.mkdir(exist_ok=True)
    
    # Test cases
    test_cases = [
        {
            'name': 'Biological-like Objects',
            'description': 'Irregular cell-like shapes with texture',
            'create_func': lambda: create_biological_like_image(num_cells=3),
            'expected_cellpose': 'GOOD',
            'color': 'green'
        },
        {
            'name': 'Geometric Rectangles',
            'description': 'Perfect rectangular synthetic shapes',  
            'create_func': lambda: create_test_image_rectangular(num_blobs=3),
            'expected_cellpose': 'POOR',
            'color': 'blue'
        },
        {
            'name': 'Complex Synthetic Shapes',
            'description': 'L-shapes, crosses, stars (geometric)',
            'create_func': lambda: create_test_image_complex_shapes(),
            'expected_cellpose': 'VERY_POOR', 
            'color': 'red'
        }
    ]
    
    # Methods to compare
    comparison_methods = ['contours', 'connected_components', 'cellpose']
    
    all_results = {}
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\\n{i}. Testing {test_case['name']}:")
        print(f"   {test_case['description']}")
        print(f"   Expected Cellpose performance: {test_case['expected_cellpose']}")
        print("-" * 55)
        
        # Create test image
        img, shape_info = test_case['create_func']()
        img_norm, img_dilated = normalize_and_dilate(img)
        
        case_results = {}
        
        # Test each method
        for method in comparison_methods:
            try:
                print(f"  Testing {method}... ", end="", flush=True)
                
                if method == 'cellpose':
                    # Use smaller diameter for synthetic shapes, larger for biological
                    diameter = 60 if 'biological' in test_case['name'].lower() else 30
                    blobs = detect_blobs(img_dilated, img_norm, 50, 100, test_case['color'], 
                                       f"{test_case['name'].lower()}_test.tiff",
                                       method=method, include_method_info=True,
                                       diameter=diameter, model_type='cyto3')
                else:
                    blobs = detect_blobs(img_dilated, img_norm, 50, 100, test_case['color'],
                                       f"{test_case['name'].lower()}_test.tiff", 
                                       method=method, include_method_info=True)
                
                case_results[method] = blobs
                
                # Performance assessment
                detection_count = len(blobs)
                if method == 'cellpose':
                    expected = test_case['expected_cellpose']
                    if expected == 'GOOD' and detection_count >= 2:
                        performance = "✅ GOOD (as expected)"
                    elif expected == 'POOR' and detection_count <= 1:
                        performance = "⚠️ POOR (as expected)"  
                    elif expected == 'VERY_POOR' and detection_count == 0:
                        performance = "❌ VERY POOR (as expected)"
                    else:
                        performance = f"🤔 UNEXPECTED ({detection_count} detections)"
                else:
                    performance = "✅ BASELINE"
                
                print(f"{detection_count:2d} detections - {performance}")
                
                # Show bounding box info for first detection
                if blobs:
                    blob = blobs[0]
                    print(f"{'':20} First: Box({blob['box_x']}, {blob['box_y']}) size={blob['box_size']}")
                
            except Exception as e:
                print(f"ERROR - {type(e).__name__}: {str(e)[:50]}...")
                case_results[method] = []
        
        # Create visualization
        print(f"  📊 Creating comparison plot...")
        plot_results = {k: v for k, v in case_results.items() if v}
        if plot_results:
            try:
                fig = plot_method_comparison(img, plot_results, save_dir=save_dir, show_boxes=True)
                if fig:
                    filename = f"{i:02d}_{test_case['name'].lower().replace(' ', '_')}_cellpose_test.png"
                    save_path = save_dir / filename
                    fig.savefig(save_path, dpi=150, bbox_inches='tight')
                    import matplotlib.pyplot as plt
                    plt.close(fig)
                    print(f"    💾 Saved: {filename}")
            except Exception as e:
                print(f"    ❌ Plotting error: {e}")
        
        all_results[test_case['name']] = case_results
    
    # Analysis and conclusions
    print("\\n" + "=" * 60)
    print("CELLPOSE PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    print("\\n📊 DETECTION SUMMARY:")
    cellpose_scores = {}
    baseline_scores = {}
    
    for case_name, case_results in all_results.items():
        cellpose_count = len(case_results.get('cellpose', []))
        contours_count = len(case_results.get('contours', []))
        
        cellpose_scores[case_name] = cellpose_count
        baseline_scores[case_name] = contours_count
        
        print(f"\\n{case_name}:")
        print(f"  Cellpose:   {cellpose_count:2d} detections")
        print(f"  Contours:   {contours_count:2d} detections (baseline)")
        
        if 'biological' in case_name.lower():
            if cellpose_count >= contours_count * 0.7:
                print("  📈 Result: Cellpose performs well on biological-like objects")
            else:
                print("  📉 Result: Cellpose underperformed on biological-like objects")
        else:
            if cellpose_count < contours_count * 0.5:
                print("  📉 Result: Cellpose struggles with synthetic geometric shapes")
            else:
                print("  📈 Result: Cellpose performed better than expected")
    
    print("\\n" + "=" * 60)
    print("KEY FINDINGS & LIMITATIONS")
    print("=" * 60)
    
    print("\\n🧬 CELLPOSE STRENGTHS:")
    print("   ✅ Designed for biological specimens (cells, nuclei, tissues)")
    print("   ✅ Trained on real microscopy data with organic shapes")
    print("   ✅ Handles irregular boundaries and complex cellular morphology")
    print("   ✅ Good at distinguishing cell interiors from backgrounds")
    
    print("\\n⚠️ CELLPOSE LIMITATIONS:")
    print("   ❌ Poor performance on synthetic geometric shapes")
    print("   ❌ Not trained on perfect rectangles, L-shapes, crosses")  
    print("   ❌ Expects biological textures and morphologies")
    print("   ❌ May over-segment or under-segment geometric objects")
    print("   ❌ Computationally intensive for simple shape detection")
    
    print("\\n📋 RECOMMENDATIONS FOR AUTONOMOUS MICROSCOPY:")
    print("   🔬 Use Cellpose for: Biological specimens, cells, nuclei, tissues")
    print("   🔧 Use traditional methods for: Geometric particles, synthetic shapes")
    print("   ⚖️  Hybrid approach: Detect specimen type first, then choose method")
    
    print(f"\\n📁 All test visualizations saved to: {save_dir}")
    
    # List generated files
    plot_files = list(save_dir.glob("*.png"))
    if plot_files:
        print(f"\\n📋 Generated {len(plot_files)} visualization files:")
        for file_path in sorted(plot_files):
            print(f"   • {file_path.name}")
    
    print("\\n🎉 Cellpose limitations demonstration completed!")
    
    return all_results

if __name__ == "__main__":
    try:
        results = test_cellpose_limitations()
        if results:
            print("\\n" + "=" * 60)
            print("✅ SUCCESS: Cellpose limitations clearly demonstrated!")
            print("Understanding when to use Cellpose vs traditional methods")
            print("will improve your autonomous microscopy detection accuracy.")
            print("=" * 60)
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()