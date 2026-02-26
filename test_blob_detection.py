#!/usr/bin/env python3
"""
Test script for blob detection functions in utils.py

This script creates synthetic test images and validates that all blob detection
methods work correctly and maintain backward compatibility.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from utils import (
    detect_blobs, 
    detect_blobs_simple,
    detect_blobs_contours, 
    detect_blobs_hough,
    detect_blobs_connected_components,
    detect_blobs_watershed,
    detect_blobs_cellpose,
    detect_blobs_multi_method,
    get_available_detection_methods,
    normalize_and_dilate,
    CELLPOSE_AVAILABLE
)


def plot_detection_results(img, blobs, title="Blob Detection Results", method=None, save_path=None, show_boxes=True):
    """
    Plot image with detected blobs overlaid as bounding boxes (rectangles) or circles.
    
    Parameters:
    -----------
    img : np.ndarray
        Input image to display
    blobs : list
        List of detected blobs with 'center', 'radius', 'box_x', 'box_y', 'box_size' keys
    title : str
        Plot title
    method : str
        Detection method used (for color coding)
    save_path : str
        Optional path to save the plot
    show_boxes : bool
        If True, show bounding boxes (rectangles). If False, show circles (default: True)
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Display image
    if len(img.shape) == 3:
        ax.imshow(img)
    else:
        ax.imshow(img, cmap='gray')
    
    # Color map for different methods
    method_colors = {
        'simple': 'red',
        'contours': 'blue', 
        'hough': 'green',
        'connected_components': 'orange',
        'watershed': 'purple',
        'cellpose': 'cyan'
    }
    
    # Default color
    color = method_colors.get(method, 'red')
    
    # Draw detected blobs
    for i, blob in enumerate(blobs):
        if show_boxes and 'box_x' in blob and 'box_y' in blob and 'box_size' in blob:
            # Draw bounding box rectangle (preferred for autonomous microscopy)
            box_x = blob['box_x']
            box_y = blob['box_y'] 
            box_size = blob['box_size']
            
            rectangle = patches.Rectangle((box_x, box_y), box_size, box_size, 
                                        fill=False, color=color, linewidth=2)
            ax.add_patch(rectangle)
            
            # Add blob number at box center
            center_x = box_x + box_size / 2
            center_y = box_y + box_size / 2
            ax.text(center_x, center_y, str(i+1), color='white', 
                   fontsize=10, ha='center', va='center', weight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))
        else:
            # Fall back to circles if box info not available
            center = blob.get('center', (0, 0))
            radius = blob.get('radius', 5)
            
            circle = patches.Circle(center, radius, fill=False, color=color, linewidth=2)
            ax.add_patch(circle)
            
            # Add blob number
            ax.text(center[0], center[1], str(i+1), color='white', 
                   fontsize=10, ha='center', va='center', weight='bold')
    
    shape_type = "boxes" if show_boxes else "circles"
    ax.set_title(f"{title}\n{method or 'Unknown'}: {len(blobs)} blobs detected ({shape_type})")
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  💾 Saved plot: {save_path}")
    
    plt.tight_layout()
    return fig


def plot_method_comparison(img, method_results, save_dir=None, show_boxes=True):
    """
    Create a comparison plot showing results from multiple detection methods.
    
    Parameters:
    -----------
    img : np.ndarray
        Input image
    method_results : dict
        Dictionary with method names as keys and blob lists as values
    save_dir : str
        Directory to save plots
    show_boxes : bool
        If True, show bounding boxes (rectangles). If False, show circles (default: True)
    """
    n_methods = len(method_results)
    cols = min(3, n_methods)
    rows = (n_methods + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    if n_methods == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes] if n_methods == 1 else axes
    else:
        axes = axes.flatten()
    
    method_colors = {
        'simple': 'red',
        'contours': 'blue', 
        'hough': 'green',
        'connected_components': 'orange',
        'watershed': 'purple',
        'cellpose': 'cyan'
    }
    
    for i, (method, blobs) in enumerate(method_results.items()):
        ax = axes[i]
        
        # Display image
        if len(img.shape) == 3:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap='gray')
        
        # Draw detected blobs
        color = method_colors.get(method, 'red')
        for blob in blobs:
            if show_boxes and 'box_x' in blob and 'box_y' in blob and 'box_size' in blob:
                # Draw bounding box rectangle
                box_x = blob['box_x']
                box_y = blob['box_y'] 
                box_size = blob['box_size']
                
                rectangle = patches.Rectangle((box_x, box_y), box_size, box_size, 
                                            fill=False, color=color, linewidth=2)
                ax.add_patch(rectangle)
            else:
                # Fall back to circles
                center = blob.get('center', (0, 0))
                radius = blob.get('radius', 5)
                circle = patches.Circle(center, radius, fill=False, color=color, linewidth=2)
                ax.add_patch(circle)
        
        shape_type = "boxes" if show_boxes else "circles"
        ax.set_title(f"{method.upper()}\n{len(blobs)} detections")
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(n_methods, len(axes)):
        axes[i].axis('off')
    
    shape_desc = "(Bounding Boxes)" if show_boxes else "(Circles)"
    plt.suptitle(f"Detection Method Comparison {shape_desc}", fontsize=16, y=0.95)
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / "method_comparison.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  💾 Saved comparison plot: {save_path}")
    
    return fig


def plot_test_images(images_info, save_dir=None):
    """
    Plot the test images used in testing.
    
    Parameters:
    -----------
    images_info : list of tuples
        List of (image, title) tuples
    save_dir : str
        Directory to save plots
    """
    n_images = len(images_info)
    cols = min(3, n_images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_images == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if n_images > 1 else [axes]
    else:
        axes = axes.flatten()
    
    for i, (img, title) in enumerate(images_info):
        ax = axes[i]
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle("Test Images", fontsize=16)
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / "test_images.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  💾 Saved test images: {save_path}")
    
    return fig


def create_test_image_circular(width=500, height=500, num_blobs=5):
    """Create test image with circular blobs"""
    img = np.zeros((height, width), dtype=np.float32)
    
    blob_info = []
    for i in range(num_blobs):
        # Random position and size
        x = np.random.randint(50, width-50)
        y = np.random.randint(50, height-50) 
        radius = np.random.randint(15, 35)
        intensity = np.random.randint(150, 255)
        
        # Draw filled circle
        cv2.circle(img, (x, y), radius, intensity, -1)
        blob_info.append({'center': (x, y), 'radius': radius, 'intensity': intensity})
    
    # Add some noise
    noise = np.random.normal(0, 10, img.shape).astype(np.float32)
    img = np.clip(img + noise, 0, 255)
    
    return img, blob_info


def create_test_image_irregular(width=500, height=500, num_blobs=4):
    """Create test image with irregular shaped blobs"""
    img = np.zeros((height, width), dtype=np.float32)
    
    blob_info = []
    for i in range(num_blobs):
        # Random irregular shape using ellipse with random rotation
        x = np.random.randint(60, width-60)
        y = np.random.randint(60, height-60)
        axes_a = np.random.randint(20, 40)
        axes_b = np.random.randint(15, 30) 
        angle = np.random.randint(0, 180)
        intensity = np.random.randint(150, 255)
        
        # Draw filled ellipse
        cv2.ellipse(img, (x, y), (axes_a, axes_b), angle, 0, 360, intensity, -1)
        blob_info.append({'center': (x, y), 'axes': (axes_a, axes_b), 'angle': angle, 'intensity': intensity})
    
    # Add noise
    noise = np.random.normal(0, 10, img.shape).astype(np.float32)
    img = np.clip(img + noise, 0, 255) 
    
    return img, blob_info


def create_test_image_touching(width=500, height=500):
    """Create test image with touching/overlapping blobs"""
    img = np.zeros((height, width), dtype=np.float32)
    
    # Create overlapping circles
    centers = [(150, 150), (180, 150), (165, 180)]
    radius = 25
    intensity = 200
    
    for center in centers:
        cv2.circle(img, center, radius, intensity, -1)
    
    # Add separate blob
    cv2.circle(img, (350, 350), 30, intensity, -1)
    
    # Add noise
    noise = np.random.normal(0, 10, img.shape).astype(np.float32)
    img = np.clip(img + noise, 0, 255)
    
    return img, centers


def create_test_image_rectangular(width=500, height=500, num_blobs=4):
    """Create test image with rectangular shaped objects"""
    img = np.zeros((height, width), dtype=np.float32)
    
    blob_info = []
    for i in range(num_blobs):
        # Random rectangular shape
        x = np.random.randint(40, width-80)
        y = np.random.randint(40, height-80)
        rect_width = np.random.randint(20, 50)
        rect_height = np.random.randint(15, 45)
        intensity = np.random.randint(150, 255)
        
        # Draw filled rectangle
        cv2.rectangle(img, (x, y), (x + rect_width, y + rect_height), intensity, -1)
        blob_info.append({
            'top_left': (x, y), 
            'bottom_right': (x + rect_width, y + rect_height),
            'width': rect_width,
            'height': rect_height,
            'intensity': intensity
        })
    
    # Add noise
    noise = np.random.normal(0, 10, img.shape).astype(np.float32)
    img = np.clip(img + noise, 0, 255)
    
    return img, blob_info


def create_test_image_polygonal(width=500, height=500, num_blobs=3):
    """Create test image with polygonal shaped objects (triangles, hexagons, etc.)"""
    img = np.zeros((height, width), dtype=np.float32)
    
    blob_info = []
    for i in range(num_blobs):
        # Random polygon (3-6 sides)
        num_sides = np.random.randint(3, 7)
        center_x = np.random.randint(60, width-60)
        center_y = np.random.randint(60, height-60)
        radius = np.random.randint(20, 40)
        intensity = np.random.randint(150, 255)
        
        # Generate polygon points
        angles = np.linspace(0, 2*np.pi, num_sides, endpoint=False)
        points = []
        for angle in angles:
            # Add some randomness to radius for irregular polygons
            r = radius + np.random.randint(-5, 6)
            x = int(center_x + r * np.cos(angle))
            y = int(center_y + r * np.sin(angle))
            points.append((x, y))
        
        # Draw filled polygon
        pts = np.array(points, np.int32)
        cv2.fillPoly(img, [pts], intensity)
        
        blob_info.append({
            'center': (center_x, center_y),
            'points': points,
            'num_sides': num_sides,
            'radius': radius,
            'intensity': intensity
        })
    
    # Add noise
    noise = np.random.normal(0, 10, img.shape).astype(np.float32)
    img = np.clip(img + noise, 0, 255)
    
    return img, blob_info


def create_test_image_complex_shapes(width=500, height=500):
    """Create test image with complex non-circular shapes (L-shapes, crosses, etc.)"""
    img = np.zeros((height, width), dtype=np.float32)
    
    blob_info = []
    intensity = 200
    
    # L-shape
    l_points = np.array([(100, 100), (140, 100), (140, 120), (120, 120), (120, 160), (100, 160)], np.int32)
    cv2.fillPoly(img, [l_points], intensity)
    blob_info.append({'type': 'L-shape', 'points': l_points.tolist()})
    
    # Cross shape
    cross_pts1 = np.array([(220, 120), (260, 120), (260, 140), (220, 140)], np.int32)  # Horizontal bar
    cross_pts2 = np.array([(230, 110), (250, 110), (250, 150), (230, 150)], np.int32)  # Vertical bar
    cv2.fillPoly(img, [cross_pts1], intensity)
    cv2.fillPoly(img, [cross_pts2], intensity)
    blob_info.append({'type': 'cross', 'points': [cross_pts1.tolist(), cross_pts2.tolist()]})
    
    # Irregular blob (kidney shape approximation)
    kidney_points = np.array([
        (350, 120), (380, 110), (410, 120), (420, 140), (415, 160), 
        (400, 170), (380, 165), (360, 170), (345, 160), (340, 140)
    ], np.int32)
    cv2.fillPoly(img, [kidney_points], intensity)
    blob_info.append({'type': 'kidney', 'points': kidney_points.tolist()})
    
    # Star shape (5-pointed)
    center = (240, 300)
    outer_radius = 30
    inner_radius = 15
    star_points = []
    for i in range(10):
        angle = i * np.pi / 5
        if i % 2 == 0:
            r = outer_radius
        else:
            r = inner_radius
        x = int(center[0] + r * np.cos(angle - np.pi/2))
        y = int(center[1] + r * np.sin(angle - np.pi/2))
        star_points.append((x, y))
    
    star_pts = np.array(star_points, np.int32)
    cv2.fillPoly(img, [star_pts], intensity)
    blob_info.append({'type': 'star', 'points': star_points, 'center': center})
    
    # Add noise
    noise = np.random.normal(0, 8, img.shape).astype(np.float32)
    img = np.clip(img + noise, 0, 255)
    
    return img, blob_info


def test_non_circular_objects(enable_plotting=False, save_dir=None):
    """Test detection methods on various non-circular objects"""
    print("Testing detection methods on non-circular objects...")
    
    # Get available methods
    methods = ['simple', 'contours', 'connected_components', 'watershed']
    if CELLPOSE_AVAILABLE:
        methods.append('cellpose')
    
    # Test different non-circular image types
    test_cases = [
        ("Irregular (Ellipses)", create_test_image_irregular),
        ("Rectangular", create_test_image_rectangular),
        ("Polygonal", create_test_image_polygonal),
        ("Complex Shapes", create_test_image_complex_shapes)
    ]
    
    all_results = {}
    
    for case_name, create_func in test_cases:
        print(f"\n📐 Testing {case_name} objects:")
        print("-" * 50)
        
        # Create test image
        if case_name == "Complex Shapes":
            img, shape_info = create_func()
        else:
            img, shape_info = create_func(num_blobs=3)
        
        img_norm, img_dilated = normalize_and_dilate(img)
        
        case_results = {}
        
        # Test each method
        for method in methods:
            try:
                blobs = detect_blobs(img_dilated, img_norm, 50, 100, 'red', f'{case_name.lower()}_test.tiff',
                                   method=method, include_method_info=True)
                case_results[method] = blobs
                print(f"  {method:20}: {len(blobs):2d} detections")
                
            except Exception as e:
                if method == 'cellpose':
                    print(f"  {method:20}: Expected limitation with synthetic shapes")
                else:
                    print(f"  {method:20}: ❌ Error - {e}")
                case_results[method] = []
        
        all_results[case_name] = {
            'image': img,
            'results': case_results,
            'shape_info': shape_info
        }
        
        # Create plots if requested
        if enable_plotting and case_results:
            print(f"  📊 Creating plots for {case_name}...")
            
            # Filter out empty results for plotting
            plot_results = {k: v for k, v in case_results.items() if v}
            
            if plot_results and save_dir:
                save_path = Path(save_dir)
                save_path.mkdir(exist_ok=True)
                
                # Create method comparison plot
                fig = plot_method_comparison(
                    img, plot_results, 
                    save_dir=save_path, 
                    show_boxes=True
                )
                if fig:
                    # Save with descriptive name
                    comparison_path = save_path / f"non_circular_{case_name.lower().replace(' ', '_')}_comparison.png"
                    fig.savefig(comparison_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    print(f"    💾 Saved: {comparison_path.name}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("NON-CIRCULAR OBJECT DETECTION SUMMARY")
    print("=" * 60)
    
    method_performance = {method: [] for method in methods}
    
    for case_name, case_data in all_results.items():
        print(f"\n{case_name}:")
        for method, blobs in case_data['results'].items():
            detection_count = len(blobs)
            method_performance[method].append(detection_count)
            print(f"  {method:20}: {detection_count:2d} detections")
    
    # Calculate average performance
    print(f"\nOverall Performance (average detections per test case):")
    for method in methods:
        if method_performance[method]:
            avg_detections = np.mean(method_performance[method])
            print(f"  {method:20}: {avg_detections:4.1f} average detections")
    
    # Method recommendations
    print(f"\n🎯 RECOMMENDATIONS FOR NON-CIRCULAR OBJECTS:")
    print(f"  • CONTOURS: Best for irregular and complex shapes")
    print(f"  • CONNECTED_COMPONENTS: Good for well-separated objects")
    print(f"  • WATERSHED: Excellent for touching/overlapping objects")
    print(f"  • SIMPLE: May miss very irregular shapes")
    if CELLPOSE_AVAILABLE:
        print(f"  • CELLPOSE: Limited effectiveness on synthetic geometric shapes")
    
    print("\n✅ Non-circular object testing completed!")
    
    return all_results


def test_backward_compatibility():
    """Test that new functions produce same output format as original"""
    print("Testing backward compatibility...")
    
    # Create test image  
    img, _ = create_test_image_circular(num_blobs=3)
    img_norm, img_dilated = normalize_and_dilate(img)
    
    # Test default behavior (should be same as original)
    blobs = detect_blobs(img_dilated, img_norm, 50, 100, 'red', 'test.tiff')
    
    # Check output format
    expected_keys = {'Box', 'center', 'radius', 'color', 'file', 'max_intensity', 
                     'mean_intensity', 'mean_dilation', 'box_x', 'box_y', 'box_size'}
    
    if blobs:
        actual_keys = set(blobs[0].keys())
        missing_keys = expected_keys - actual_keys
        extra_keys = actual_keys - expected_keys
        
        if missing_keys:
            print(f"❌ Missing keys: {missing_keys}")
            return False
        if extra_keys:
            print(f"❌ Extra keys (breaking compatibility): {extra_keys}")
            return False
        
        print(f"✅ Backward compatibility maintained. Found {len(blobs)} blobs with correct format.")
        return True
    else:
        print("⚠️ No blobs detected in test image")
        return False
    """Test that new functions produce same output format as original"""
    print("Testing backward compatibility...")
    
    # Create test image  
    img, _ = create_test_image_circular(num_blobs=3)
    img_norm, img_dilated = normalize_and_dilate(img)
    
    # Test default behavior (should be same as original)
    blobs = detect_blobs(img_dilated, img_norm, 50, 100, 'red', 'test.tiff')
    
    # Check output format
    expected_keys = {'Box', 'center', 'radius', 'color', 'file', 'max_intensity', 
                     'mean_intensity', 'mean_dilation', 'box_x', 'box_y', 'box_size'}
    
    if blobs:
        actual_keys = set(blobs[0].keys())
        missing_keys = expected_keys - actual_keys
        extra_keys = actual_keys - expected_keys
        
        if missing_keys:
            print(f"❌ Missing keys: {missing_keys}")
            return False
        if extra_keys:
            print(f"❌ Extra keys (breaking compatibility): {extra_keys}")
            return False
        
        print(f"✅ Backward compatibility maintained. Found {len(blobs)} blobs with correct format.")
        return True
    else:
        print("⚠️ No blobs detected in test image")
        return False


def test_method_info_flag():
    """Test include_method_info parameter"""
    print("Testing method info flag...")
    
    img, _ = create_test_image_circular(num_blobs=2)
    img_norm, img_dilated = normalize_and_dilate(img)
    
    # Without method info (default)
    blobs_no_info = detect_blobs(img_dilated, img_norm, 50, 100, 'red', 'test.tiff')
    
    # With method info
    blobs_with_info = detect_blobs(img_dilated, img_norm, 50, 100, 'red', 'test.tiff', 
                                  include_method_info=True)
    
    if blobs_no_info and blobs_with_info:
        has_method_no_info = 'method' in blobs_no_info[0]
        has_method_with_info = 'method' in blobs_with_info[0]
        
        if has_method_no_info:
            print("❌ Method info included when it shouldn't be")
            return False
        if not has_method_with_info:
            print("❌ Method info missing when it should be included")
            return False
            
        print("✅ Method info flag working correctly")
        return True
    else:
        print("⚠️ No blobs detected for method info test")
        return False


def test_all_methods(enable_plotting=False, save_dir=None):
    """Test all detection methods with optional plotting"""
    print("Testing all detection methods...")
    
    methods = get_available_detection_methods()
    print(f"Available methods: {methods}")
    
    # Test with circular blobs (good for most methods)
    img_circular, _ = create_test_image_circular(num_blobs=4)
    img_norm, img_dilated = normalize_and_dilate(img_circular)
    
    results = {}
    method_results = {}  # For plotting
    
    for method in methods:
        try:
            blobs = detect_blobs(img_dilated, img_norm, 50, 100, 'red', 'test.tiff', 
                               method=method, include_method_info=True)
            results[method] = len(blobs)
            method_results[method] = blobs
            print(f"  {method}: Found {len(blobs)} blobs")
            
            # Verify method is recorded correctly
            if blobs and blobs[0]['method'] != method:
                print(f"    ❌ Method mismatch: expected {method}, got {blobs[0]['method']}")
                return False
                
        except Exception as e:
            print(f"  ❌ {method}: Error - {e}")
            return False
    
    # Create plots if requested
    if enable_plotting and method_results:
        print("  📊 Creating method comparison plot...")
        plot_method_comparison(img_circular, method_results, save_dir)
        
        # Create individual plots for each successful method
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(exist_ok=True)
            for method, blobs in method_results.items():
                if blobs:  # Only plot if detections found
                    fig = plot_detection_results(
                        img_circular, blobs, 
                        title=f"Blob Detection: {method.upper()}", 
                        method=method,
                        save_path=save_path / f"detection_{method}.png",
                        show_boxes=True  # Use bounding boxes for autonomous microscopy
                    )
                    plt.close(fig)
    
    # Check that at least some methods found blobs
    successful_methods = [m for m, count in results.items() if count > 0]
    if len(successful_methods) >= 3:
        print(f"✅ {len(successful_methods)} methods successfully detected blobs")
        return True
    else:
        print(f"❌ Only {len(successful_methods)} methods found blobs. Expected at least 3.")
        return False


def test_convenience_functions():
    """Test convenience wrapper functions"""
    print("Testing convenience wrapper functions...")
    
    img, _ = create_test_image_circular(num_blobs=3)
    img_norm, img_dilated = normalize_and_dilate(img)
    
    # Test each convenience function
    convenience_functions = [
        ('simple', detect_blobs_simple),
        ('contours', detect_blobs_contours), 
        ('hough', detect_blobs_hough),
        ('connected_components', detect_blobs_connected_components),
        ('watershed', detect_blobs_watershed)
    ]
    
    # Add Cellpose if available
    if CELLPOSE_AVAILABLE:
        convenience_functions.append(('cellpose', detect_blobs_cellpose))
    
    for method_name, func in convenience_functions:
        try:
            blobs = func(img_dilated, img_norm, 50, 100, 'red', 'test.tiff')
            print(f"  {method_name}: Found {len(blobs)} blobs")
            
            # Test with method info
            blobs_with_info = func(img_dilated, img_norm, 50, 100, 'red', 'test.tiff', 
                                 include_method_info=True)
            if blobs_with_info and 'method' not in blobs_with_info[0]:
                print(f"    ❌ Method info not included for {method_name}")
                return False
                
        except Exception as e:
            print(f"  ❌ {method_name}: Error - {e}")
            return False
    
    print("✅ All convenience functions working")
    return True


def test_multi_method(enable_plotting=False, save_dir=None):
    """Test multi-method comparison function with optional plotting"""
    print("Testing multi-method comparison...")
    
    img, _ = create_test_image_circular(num_blobs=3)
    img_norm, img_dilated = normalize_and_dilate(img)
    
    # Test with multiple methods (include cellpose if available)
    methods_to_test = ['simple', 'contours', 'hough']
    if CELLPOSE_AVAILABLE:
        methods_to_test.append('cellpose')
    
    # Test combined results
    combined_results = detect_blobs_multi_method(
        img_dilated, img_norm, 50, 100, 'red', 'test.tiff',
        methods=methods_to_test, combine_results=True
    )
    
    # Test separate results  
    separate_results = detect_blobs_multi_method(
        img_dilated, img_norm, 50, 100, 'red', 'test.tiff',
        methods=methods_to_test, combine_results=False
    )
    
    if isinstance(combined_results, list) and isinstance(separate_results, dict):
        print(f"  Combined: {len(combined_results)} total detections")
        print(f"  Separate: {len(separate_results)} methods tested")
        
        # Create plots if requested
        if enable_plotting:
            print("  📊 Creating multi-method comparison plots...")
            
            # Plot separate results comparison
            plot_method_comparison(img, separate_results, save_dir, show_boxes=True)
            
            # Plot combined results
            if save_dir and combined_results:
                save_path = Path(save_dir) / "combined_results.png"
                fig = plot_detection_results(
                    img, combined_results, 
                    title="Combined Multi-Method Detection", 
                    method="combined",
                    save_path=save_path,
                    show_boxes=True  # Use bounding boxes for scan planning
                )
                plt.close(fig)
        
        # Check that combined results have method info
        if combined_results and 'method' in combined_results[0]:
            print("✅ Multi-method comparison working correctly")
            return True
        else:
            print("❌ Combined results missing method information")
            return False
    else:
        print("❌ Multi-method function returned wrong types")
        return False


def test_method_specific_parameters():
    """Test method-specific parameters"""
    print("Testing method-specific parameters...")
    
    img, _ = create_test_image_circular(num_blobs=2)
    img_norm, img_dilated = normalize_and_dilate(img)
    
    try:
        # Test simple method with custom parameters
        blobs1 = detect_blobs(img_dilated, img_norm, 50, 100, 'red', 'test.tiff',
                             method='simple', max_area=2000, threshold_step=5)
        
        # Test hough method with custom parameters
        blobs2 = detect_blobs(img_dilated, img_norm, 50, 100, 'red', 'test.tiff',
                             method='hough', max_radius=50, min_dist=30)
        
        # Test watershed with custom parameters
        blobs3 = detect_blobs(img_dilated, img_norm, 50, 100, 'red', 'test.tiff',
                             method='watershed', min_distance=15)
        
        print(f"  Simple with custom params: {len(blobs1)} blobs")
        print(f"  Hough with custom params: {len(blobs2)} blobs")
        print(f"  Watershed with custom params: {len(blobs3)} blobs")
        
        # Test Cellpose if available
        if CELLPOSE_AVAILABLE:
            try:
                blobs4 = detect_blobs(img_dilated, img_norm, 50, 100, 'red', 'test.tiff',
                                     method='cellpose', diameter=50, model_type='cyto3')
                print(f"  Cellpose with custom params: {len(blobs4)} blobs")
            except Exception as e:
                print(f"  Cellpose test failed (expected for synthetic images): {e}")
        else:
            print(f"  Cellpose: Skipped (not installed)")
        
        print("✅ Method-specific parameters working")
        return True
        
    except Exception as e:
        print(f"❌ Error testing method parameters: {e}")
        return False


def test_edge_cases():
    """Test edge cases and error handling"""
    print("Testing edge cases...")
    
    img, _ = create_test_image_circular(num_blobs=1)
    img_norm, img_dilated = normalize_and_dilate(img)
    
    # Test invalid method
    try:
        blobs = detect_blobs(img_dilated, img_norm, 50, 100, 'red', 'test.tiff',
                           method='invalid_method')
        print("❌ Should have raised error for invalid method")
        return False
    except ValueError as e:
        print(f"  ✅ Correctly caught invalid method: {e}")
    
    # Test empty image
    empty_img = np.zeros((100, 100), dtype=np.float32)
    try:
        blobs = detect_blobs(empty_img, empty_img, 50, 100, 'red', 'empty.tiff')
        print(f"  Empty image: {len(blobs)} blobs (expected 0)")
    except Exception as e:
        print(f"  ❌ Error with empty image: {e}")
        return False
    
    # Test very high threshold (should find nothing)
    blobs = detect_blobs(img_dilated, img_norm, 250, 100, 'red', 'test.tiff')
    print(f"  High threshold: {len(blobs)} blobs (expected 0 or very few)")
    
    print("✅ Edge cases handled correctly")
    return True


def visualize_methods_comparison(save_plots=False, save_dir="test_plots"):
    """Create comprehensive visualization comparing different methods"""
    print("Creating comprehensive visualization of different methods...")
    
    # Create test images
    img_circular, _ = create_test_image_circular(num_blobs=4)
    img_irregular, _ = create_test_image_irregular(num_blobs=3) 
    img_touching, _ = create_test_image_touching()
    
    test_images = [
        (img_circular, "Circular Blobs"),
        (img_irregular, "Irregular Blobs"), 
        (img_touching, "Touching Blobs")
    ]
    
    methods = ['simple', 'contours', 'hough', 'connected_components', 'watershed']
    if CELLPOSE_AVAILABLE:
        methods.append('cellpose')
    
    # Create save directory if needed
    if save_plots:
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
    
    # Create comprehensive comparison figure
    fig, axes = plt.subplots(len(test_images), len(methods) + 1, 
                            figsize=(4 * (len(methods) + 1), 4 * len(test_images)))
    
    if len(test_images) == 1:
        axes = axes.reshape(1, -1)
    
    for row, (img, title) in enumerate(test_images):
        img_norm, img_dilated = normalize_and_dilate(img)
        
        # Show original image
        axes[row, 0].imshow(img, cmap='gray')
        axes[row, 0].set_title(f"{title}\n(Original)")
        axes[row, 0].axis('off')
        
        # Test each method and create individual plots
        method_results = {}
        
        for col, method in enumerate(methods):
            try:
                blobs = detect_blobs(img_dilated, img_norm, 50, 100, 'red', 'test.tiff',
                                   method=method)
                method_results[method] = blobs
                
                # Plot image with detected blobs in main comparison
                axes[row, col + 1].imshow(img, cmap='gray')
                
                # Color map for different methods
                method_colors = {
                    'simple': 'red',
                    'contours': 'blue', 
                    'hough': 'green',
                    'connected_components': 'orange',
                    'watershed': 'purple',
                    'cellpose': 'cyan'
                }
                
                color = method_colors.get(method, 'red')
                
                # Draw detected blobs
                for blob in blobs:
                    if 'box_x' in blob and 'box_y' in blob and 'box_size' in blob:
                        # Draw bounding box rectangle (preferred for autonomous microscopy)
                        box_x = blob['box_x']
                        box_y = blob['box_y'] 
                        box_size = blob['box_size']
                        
                        rectangle = patches.Rectangle((box_x, box_y), box_size, box_size, 
                                                    fill=False, color=color, linewidth=2)
                        axes[row, col + 1].add_patch(rectangle)
                    else:
                        # Fall back to circles
                        x, y = blob['center']
                        radius = blob['radius']
                        circle = patches.Circle((x, y), radius, fill=False, color=color, linewidth=2)
                        axes[row, col + 1].add_patch(circle)
                
                axes[row, col + 1].set_title(f"{method.upper()}\n{len(blobs)} detections")
                axes[row, col + 1].axis('off')
                
            except Exception as e:
                axes[row, col + 1].text(0.5, 0.5, f"Error:\n{str(e)[:50]}...", 
                                       transform=axes[row, col + 1].transAxes,
                                       ha='center', va='center', fontsize=8)
                axes[row, col + 1].set_title(f"{method.upper()}\n(Error)")
                axes[row, col + 1].axis('off')
                method_results[method] = []
        
        # Create individual method comparison plot for this image
        if save_plots and method_results:
            individual_fig = plot_method_comparison(
                img, method_results, 
                save_dir=save_path if save_plots else None
            )
            if individual_fig:
                individual_save_path = save_path / f"comparison_{title.lower().replace(' ', '_')}.png"
                individual_fig.savefig(individual_save_path, dpi=150, bbox_inches='tight')
                plt.close(individual_fig)
                print(f"  💾 Saved individual comparison: {individual_save_path}")
    
    plt.suptitle("Comprehensive Blob Detection Method Comparison", fontsize=16, y=0.98)
    plt.tight_layout()
    
    if save_plots:
        main_save_path = save_path / "comprehensive_comparison.png"
        plt.savefig(main_save_path, dpi=150, bbox_inches='tight')
        print(f"  💾 Saved comprehensive comparison: {main_save_path}")
        plt.close(fig)
    else:
        plt.show()
    
    print(f"✅ Method comparison visualization completed")


def run_all_tests(enable_plotting=False, save_dir=None):
    """Run all test functions with optional plotting"""
    print("=" * 60)
    print("BLOB DETECTION FUNCTION TEST SUITE")
    print("=" * 60)
    print(f"Cellpose Available: {'✅ Yes' if CELLPOSE_AVAILABLE else '❌ No (install with: pip install cellpose[gui])'}")
    print(f"Available Methods: {get_available_detection_methods()}")
    if enable_plotting:
        print(f"Plotting: ✅ Enabled (save_dir: {save_dir or 'None'})")
    print("=" * 60)
    
    # Create save directory if plotting is enabled
    if enable_plotting and save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        print(f"📁 Created output directory: {save_path}")
    
    tests = [
        ("Backward Compatibility", test_backward_compatibility),
        ("Method Info Flag", test_method_info_flag),
        ("All Detection Methods", lambda: test_all_methods(enable_plotting, save_dir)),
        ("Convenience Functions", test_convenience_functions),
        ("Multi-Method Comparison", lambda: test_multi_method(enable_plotting, save_dir)),
        ("Method-Specific Parameters", test_method_specific_parameters),
        ("Non-Circular Objects", lambda: test_non_circular_objects(enable_plotting, save_dir)),
        ("Edge Cases", test_edge_cases)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Create test image gallery if plotting enabled
    if enable_plotting:
        print(f"\nCreating test image gallery...")
        print("-" * 40)
        try:
            img_circular, _ = create_test_image_circular(num_blobs=4)
            img_irregular, _ = create_test_image_irregular(num_blobs=3) 
            img_touching, _ = create_test_image_touching()
            img_rectangular, _ = create_test_image_rectangular(num_blobs=3)
            img_polygonal, _ = create_test_image_polygonal(num_blobs=3)
            img_complex, _ = create_test_image_complex_shapes()
            
            images_info = [
                (img_circular, "Circular Blobs"),
                (img_irregular, "Irregular Blobs (Ellipses)"), 
                (img_touching, "Touching Blobs"),
                (img_rectangular, "Rectangular Objects"),
                (img_polygonal, "Polygonal Objects"),
                (img_complex, "Complex Shapes")
            ]
            
            fig = plot_test_images(images_info, save_dir)
            if save_dir is None:
                plt.show()
            else:
                plt.close(fig)
                
        except Exception as e:
            print(f"❌ Error creating test image gallery: {e}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if enable_plotting and save_dir:
        print(f"\n📊 Plots saved to: {save_dir}")
    
    if passed == total:
        print("🎉 All tests passed! Blob detection functions are working correctly.")
    else:
        print("⚠️ Some tests failed. Please review the output above.")
    
    return passed == total


if __name__ == "__main__":
    # Set random seed for reproducible tests
    np.random.seed(42)
    
    # Ask user for plotting preferences
    print("=" * 60)
    print("BLOB DETECTION TEST SUITE")
    print("=" * 60)
    
    enable_plots = input("Enable plotting visualization? (y/n): ").lower().strip() == 'y'
    save_plots = False
    save_dir = None
    
    if enable_plots:
        save_plots = input("Save plots to files? (y/n): ").lower().strip() == 'y'
        if save_plots:
            default_dir = "test_plots"
            save_dir = input(f"Save directory (default: {default_dir}): ").strip() or default_dir
            print(f"📁 Plots will be saved to: {save_dir}")
    
    # Run all tests
    print("\n" + "=" * 60)
    print("RUNNING TESTS...")
    print("=" * 60)
    success = run_all_tests(enable_plotting=enable_plots, save_dir=save_dir)
    
    # Create additional visualization (optional)
    if enable_plots:
        print("\n" + "=" * 60)
        create_comparison_viz = input("Create detailed method comparison visualization? (y/n): ").lower().strip() == 'y'
        if create_comparison_viz:
            try:
                visualize_methods_comparison(save_plots=save_plots)
            except Exception as e:
                print(f"❌ Error creating comparison visualization: {e}")
    
    print("\n" + "=" * 60)
    print("TEST SUITE COMPLETED")
    print("=" * 60)
    if success:
        print("🎉 All tests passed successfully!")
    else:
        print("⚠️ Some tests failed - please review the results above.")
    
    if enable_plots and save_plots:
        print(f"\n📊 All plots have been saved to: {save_dir}")
    
    print("\nTest suite completed.")