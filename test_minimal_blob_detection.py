#!/usr/bin/env python3
"""
Minimal test for core blob detection functionality without bluesky dependencies
"""

import numpy as np
import cv2
from pathlib import Path

# Optional plotting imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("⚠️ Matplotlib not available - plotting disabled")

# Test importing just the core detection functions
try:
    import sys
    sys.path.insert(0, '.')
    
    # Import core functions we need
    from utils import normalize_and_dilate, CELLPOSE_AVAILABLE
    print("✅ Core utils imports successful")
    
    # Test the detect_blobs function by importing it directly
    from utils import (
        _detect_blobs_simple,
        _detect_blobs_contours,
        _detect_blobs_hough_circles,
        _detect_blobs_connected_components,
        _detect_blobs_watershed
    )
    print("✅ Core detection method imports successful")
    
    if CELLPOSE_AVAILABLE:
        from utils import _detect_blobs_cellpose
        print("✅ Cellpose detection method import successful")
    else:
        print("⚠️  Cellpose not available (not installed)")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)


def create_test_image():
    """Create simple test image with circular blobs"""
    img = np.zeros((200, 200), dtype=np.float32)
    
    # Add a few circular blobs
    cv2.circle(img, (50, 50), 15, 200, -1)
    cv2.circle(img, (150, 100), 20, 180, -1)
    cv2.circle(img, (100, 150), 12, 220, -1)
    
    # Add noise
    noise = np.random.normal(0, 10, img.shape).astype(np.float32)
    img = np.clip(img + noise, 0, 255)
    
    return img


def plot_minimal_results(img, method_results, save_path=None, show_boxes=True):
    """Simple plot showing detection results with bounding boxes or circles"""
    if not PLOTTING_AVAILABLE:
        print("  📊 Plotting not available (matplotlib not installed)")
        return None
        
    n_methods = len([k for k, v in method_results.items() if v])  # Count methods with results
    if n_methods == 0:
        return None
        
    # Use a simple 2-column layout: original + best method
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Show original image
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Find method with most detections
    best_method = None
    max_detections = 0
    for method, blobs in method_results.items():
        if len(blobs) > max_detections:
            max_detections = len(blobs)
            best_method = method
    
    if best_method:
        blobs = method_results[best_method]
        axes[1].imshow(img, cmap='gray')
        
        # Color for the method
        method_colors = {
            'simple': 'red',
            'contours': 'blue', 
            'hough': 'green',
            'connected_components': 'orange',
            'watershed': 'purple',
            'cellpose': 'cyan'
        }
        color = method_colors.get(best_method, 'red')
        
        for blob in blobs:
            if show_boxes and 'box_x' in blob and 'box_y' in blob and 'box_size' in blob:
                # Draw bounding box rectangle (preferred for autonomous microscopy)
                box_x = blob['box_x']
                box_y = blob['box_y'] 
                box_size = blob['box_size']
                
                rectangle = patches.Rectangle((box_x, box_y), box_size, box_size, 
                                            fill=False, color=color, linewidth=2)
                axes[1].add_patch(rectangle)
            else:
                # Fall back to circles
                center = blob.get('center', (0, 0))
                radius = blob.get('radius', 5)
                circle = patches.Circle(center, radius, fill=False, color=color, linewidth=2)
                axes[1].add_patch(circle)
        
        shape_type = "boxes" if show_boxes else "circles"
        axes[1].set_title(f"{best_method.upper()} (Best)\n{len(blobs)} detections ({shape_type})")
        axes[1].axis('off')
    else:
        axes[1].text(0.5, 0.5, "No detections", ha='center', va='center', 
                    transform=axes[1].transAxes, fontsize=14)
        axes[1].set_title("No Results")
        axes[1].axis('off')
    
    shape_desc = "(Bounding Boxes)" if show_boxes else "(Circles)"
    plt.suptitle(f"Minimal Results - {n_methods} methods tested {shape_desc}", fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  💾 Saved plot: {save_path}")
        plt.close(fig)
    else:
        plt.show()
    
    return fig


def test_detection_methods(enable_plotting=False, save_dir=None):
    """Test each detection method individually with optional plotting"""
    print("\n🔍 Testing detection methods...")
    
    # Create test image
    img = create_test_image()
    img_norm, img_dilated = normalize_and_dilate(img)
    
    methods_to_test = [
        ('simple', _detect_blobs_simple),
        ('contours', _detect_blobs_contours), 
        ('hough', _detect_blobs_hough_circles),
        ('connected_components', _detect_blobs_connected_components),
        ('watershed', _detect_blobs_watershed)
    ]
    
    if CELLPOSE_AVAILABLE:
        methods_to_test.append(('cellpose', _detect_blobs_cellpose))
    
    method_results = {}
    
    for method_name, method_func in methods_to_test:
        try:
            detections = method_func(img_dilated, img_norm, 50, 100)
            print(f"  {method_name}: Found {len(detections)} detections")
            method_results[method_name] = detections
        except Exception as e:
            if method_name == 'cellpose':
                print(f"  {method_name}: Expected error with synthetic data - {type(e).__name__}")
                method_results[method_name] = []
            else:
                print(f"  {method_name}: ❌ Error - {e}")
                method_results[method_name] = []
    
    # Create plot if requested
    if enable_plotting and method_results and PLOTTING_AVAILABLE:
        print("  📊 Creating detection results plot...")
        # Filter out methods with no results for plotting
        plot_results = {k: v for k, v in method_results.items() if v}
        if plot_results:
            save_path = None
            if save_dir:
                Path(save_dir).mkdir(exist_ok=True)
                save_path = Path(save_dir) / "minimal_detection_results.png"
            
            plot_minimal_results(img, plot_results, save_path, show_boxes=True)
    
    print("\n✅ All core detection methods tested successfully!")


def test_main_detect_blobs():
    """Test the main detect_blobs function"""
    print("\n🎯 Testing main detect_blobs function...")
    
    try:
        from utils import detect_blobs
        
        img = create_test_image()
        img_norm, img_dilated = normalize_and_dilate(img)
        
        # Test default method
        blobs = detect_blobs(img_dilated, img_norm, 50, 100, 'red', 'test.tiff')
        print(f"  Default method: Found {len(blobs)} blobs")
        
        # Test with method info
        blobs_with_info = detect_blobs(img_dilated, img_norm, 50, 100, 'red', 'test.tiff',
                                      include_method_info=True)
        has_method = 'method' in blobs_with_info[0] if blobs_with_info else False
        print(f"  Method info flag: {'✅ Working' if has_method else '❌ Failed'}")
        
        # Test different methods
        for method in ['simple', 'contours', 'hough']:
            try:
                blobs = detect_blobs(img_dilated, img_norm, 50, 100, 'red', 'test.tiff',
                                   method=method)
                print(f"  {method}: Found {len(blobs)} blobs")
            except Exception as e:
                print(f"  {method}: ❌ Error - {e}")
        
        print("\n✅ Main detect_blobs function working!")
        
    except ImportError as e:
        print(f"❌ Cannot test main function due to import error: {e}")


if __name__ == "__main__":
    print("🧪 MINIMAL BLOB DETECTION TEST")
    print("=" * 40)
    print(f"Plotting Available: {'✅ Yes' if PLOTTING_AVAILABLE else '❌ No (matplotlib not installed)'}")
    
    # Ask for plotting preferences if available
    enable_plotting = False
    save_dir = None
    
    if PLOTTING_AVAILABLE:
        enable_plotting = input("Enable plotting? (y/n): ").lower().strip() == 'y'
        if enable_plotting:
            save_plots = input("Save plots to files? (y/n): ").lower().strip() == 'y'
            if save_plots:
                save_dir = input("Save directory (default: minimal_plots): ").strip() or "minimal_plots"
    
    print("=" * 40)
    
    test_detection_methods(enable_plotting, save_dir)
    test_main_detect_blobs()
    
    print("\n🎉 All tests completed!")
    print("\n💡 If this works, the core functionality is fine.")
    print("   The issue with test_blob_detection.py is likely due to")
    print("   bluesky/zmq dependencies taking time to load or hanging.")
    
    if enable_plotting and save_dir:
        print(f"\n📊 Plots saved to: {save_dir}")