#!/usr/bin/env python3
"""
Test script for blob detection functions in utils.py

This script creates synthetic test images and validates that all blob detection
methods work correctly and maintain backward compatibility.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
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


def test_all_methods():
    """Test all detection methods"""
    print("Testing all detection methods...")
    
    methods = get_available_detection_methods()
    print(f"Available methods: {methods}")
    
    # Test with circular blobs (good for most methods)
    img_circular, _ = create_test_image_circular(num_blobs=4)
    img_norm, img_dilated = normalize_and_dilate(img_circular)
    
    results = {}
    
    for method in methods:
        try:
            blobs = detect_blobs(img_dilated, img_norm, 50, 100, 'red', 'test.tiff', 
                               method=method, include_method_info=True)
            results[method] = len(blobs)
            print(f"  {method}: Found {len(blobs)} blobs")
            
            # Verify method is recorded correctly
            if blobs and blobs[0]['method'] != method:
                print(f"    ❌ Method mismatch: expected {method}, got {blobs[0]['method']}")
                return False
                
        except Exception as e:
            print(f"  ❌ {method}: Error - {e}")
            return False
    
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


def test_multi_method():
    """Test multi-method comparison function"""
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


def visualize_methods_comparison(save_plots=False):
    """Create visualization comparing different methods"""
    print("Creating visualization of different methods...")
    
    # Create test images
    img_circular, _ = create_test_image_circular(num_blobs=4)
    img_irregular, _ = create_test_image_irregular(num_blobs=3) 
    img_touching, _ = create_test_image_touching()
    
    test_images = [
        (img_circular, "Circular Blobs"),
        (img_irregular, "Irregular Blobs"), 
        (img_touching, "Touching Blobs")
    ]
    
    methods = ['simple', 'contours', 'hough', 'connected_components']
    if CELLPOSE_AVAILABLE:
        methods.append('cellpose')
    
    fig, axes = plt.subplots(len(test_images), len(methods) + 1, 
                            figsize=(20, 12))
    
    for row, (img, title) in enumerate(test_images):
        img_norm, img_dilated = normalize_and_dilate(img)
        
        # Show original image
        axes[row, 0].imshow(img, cmap='gray')
        axes[row, 0].set_title(f"{title}\n(Original)")
        axes[row, 0].axis('off')
        
        # Test each method
        for col, method in enumerate(methods):
            try:
                blobs = detect_blobs(img_dilated, img_norm, 50, 100, 'red', 'test.tiff',
                                   method=method)
                
                # Plot image with detected blobs
                axes[row, col + 1].imshow(img, cmap='gray')
                
                # Draw detected blobs
                for blob in blobs:
                    x, y = blob['center']
                    radius = blob['radius']
                    circle = plt.Circle((x, y), radius, fill=False, color='red', linewidth=2)
                    axes[row, col + 1].add_patch(circle)
                
                axes[row, col + 1].set_title(f"{method}\n({len(blobs)} blobs)")
                axes[row, col + 1].axis('off')
                
            except Exception as e:
                axes[row, col + 1].text(0.5, 0.5, f"Error:\n{str(e)}", 
                                       transform=axes[row, col + 1].transAxes,
                                       ha='center', va='center')
                axes[row, col + 1].set_title(f"{method}\n(Error)")
                axes[row, col + 1].axis('off')
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('blob_detection_comparison.png', dpi=150, bbox_inches='tight')
        print("  Saved comparison plot as 'blob_detection_comparison.png'")
    
    plt.show()


def run_all_tests():
    """Run all test functions"""
    print("=" * 60)
    print("BLOB DETECTION FUNCTION TEST SUITE")
    print("=" * 60)
    print(f"Cellpose Available: {'✅ Yes' if CELLPOSE_AVAILABLE else '❌ No (install with: pip install cellpose[gui])'}")
    print(f"Available Methods: {get_available_detection_methods()}")
    print("=" * 60)
    
    tests = [
        ("Backward Compatibility", test_backward_compatibility),
        ("Method Info Flag", test_method_info_flag),
        ("All Detection Methods", test_all_methods),
        ("Convenience Functions", test_convenience_functions),
        ("Multi-Method Comparison", test_multi_method),
        ("Method-Specific Parameters", test_method_specific_parameters),
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
    
    if passed == total:
        print("🎉 All tests passed! Blob detection functions are working correctly.")
    else:
        print("⚠️ Some tests failed. Please review the output above.")
    
    return passed == total


if __name__ == "__main__":
    # Set random seed for reproducible tests
    np.random.seed(42)
    
    # Run all tests
    success = run_all_tests()
    
    # Create visualization (optional)
    print("\n" + "=" * 60)
    create_viz = input("Create visualization comparison? (y/n): ").lower().strip() == 'y'
    if create_viz:
        visualize_methods_comparison(save_plots=True)
    
    print("\nTest suite completed.")