#!/usr/bin/env python3
"""
Quick demo of the plotting functionality for blob detection
"""

import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, '.')

# Demo the plotting capabilities
from utils import normalize_and_dilate, detect_blobs
from test_minimal_blob_detection import create_test_image, plot_minimal_results
from test_blob_detection import plot_detection_results, plot_method_comparison

def run_plotting_demo():
    """Run a quick demonstration of all plotting features"""
    print("🎨 BLOB DETECTION PLOTTING DEMO")
    print("=" * 50)
    
    # Create test image
    print("📸 Creating test image...")
    img = create_test_image()
    img_norm, img_dilated = normalize_and_dilate(img)
    
    # Test different methods
    print("🔍 Testing detection methods...")
    methods = ['simple', 'contours', 'hough', 'connected_components']
    
    method_results = {}
    
    for method in methods:
        try:
            blobs = detect_blobs(img_dilated, img_norm, 50, 100, 'red', 'test.tiff',
                               method=method, include_method_info=True)
            method_results[method] = blobs
            print(f"  {method}: {len(blobs)} detections")
        except Exception as e:
            print(f"  {method}: Error - {e}")
            method_results[method] = []
    
    # Create output directory
    save_dir = Path("demo_plots")
    save_dir.mkdir(exist_ok=True)
    print(f"📁 Output directory: {save_dir}")
    
    # Demo 1: Individual method plots
    print("\n📊 Creating individual method plots...")
    for method, blobs in method_results.items():
        if blobs:
            save_path = save_dir / f"demo_{method}.png"
            fig = plot_detection_results(
                img, blobs, 
                title=f"Demo: {method.upper()} Detection", 
                method=method,
                save_path=save_path,
                show_boxes=True  # Show bounding boxes for autonomous microscopy
            )
            if fig:
                import matplotlib.pyplot as plt
                plt.close(fig)
    
    # Demo 2: Method comparison plot
    print("\n🆚 Creating method comparison plot...")
    if method_results:
        # Filter out empty results
        comparison_results = {k: v for k, v in method_results.items() if v}
        if comparison_results:
            fig = plot_method_comparison(img, comparison_results, save_dir, show_boxes=True)
            if fig:
                import matplotlib.pyplot as plt
                plt.close(fig)
    
    # Demo 3: Minimal plotting function
    print("\n📋 Testing minimal plotting function...")
    if method_results:
        save_path = save_dir / "demo_minimal_results.png"
        fig = plot_minimal_results(img, method_results, save_path, show_boxes=True)
        if fig:
            import matplotlib.pyplot as plt
            plt.close(fig)
    
    print(f"\n✅ Demo completed successfully!")
    print(f"📂 All plots saved to: {save_dir}")
    print("\n🎯 Summary of generated plots:")
    
    # List generated files
    plot_files = list(save_dir.glob("*.png"))
    for i, plot_file in enumerate(plot_files, 1):
        print(f"  {i}. {plot_file.name}")
    
    print(f"\n🎉 Generated {len(plot_files)} visualization plots!")

if __name__ == "__main__":
    # Set random seed for reproducible results
    np.random.seed(42)
    
    try:
        run_plotting_demo()
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()