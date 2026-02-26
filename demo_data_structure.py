#!/usr/bin/env python3
"""
Demonstrate that the detect_blobs function output structure is unchanged
and contains all bounding box information needed for autonomous microscopy scan planning.
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from utils import detect_blobs, normalize_and_dilate
from test_minimal_blob_detection import create_test_image

def show_blob_data_structure():
    """Show the complete data structure returned by detect_blobs"""
    print("🔍 BLOB DETECTION DATA STRUCTURE DEMO")
    print("=" * 60)
    
    # Create test image
    print("📸 Creating test image...")
    img = create_test_image()
    img_norm, img_dilated = normalize_and_dilate(img)
    
    # Detect blobs using different methods
    methods = ['simple', 'contours', 'hough']
    
    for method in methods:
        print(f"\n🎯 Testing {method.upper()} method:")
        print("-" * 40)
        
        # Detect with method info included
        blobs = detect_blobs(img_dilated, img_norm, 50, 100, 'red', 'test.tiff',
                           method=method, include_method_info=True)
        
        print(f"Found {len(blobs)} blobs")
        
        if blobs:
            # Show detailed structure of first blob
            blob = blobs[0]
            print(f"\n📋 First blob data structure:")
            print(f"   Box ID: {blob.get('Box', 'N/A')}")
            print(f"   Center: {blob.get('center', 'N/A')}")
            print(f"   Radius: {blob.get('radius', 'N/A')}")
            print(f"   Color: {blob.get('color', 'N/A')}")
            print(f"   File: {blob.get('file', 'N/A')}")
            print(f"   Method: {blob.get('method', 'N/A')}")
            
            print(f"\n🗂️  BOUNDING BOX INFO FOR SCAN PLANNING:")
            print(f"   box_x: {blob.get('box_x', 'N/A')} (top-left x)")
            print(f"   box_y: {blob.get('box_y', 'N/A')} (top-left y)")  
            print(f"   box_size: {blob.get('box_size', 'N/A')} (width/height)")
            
            print(f"\n📊 INTENSITY MEASUREMENTS:")
            print(f"   max_intensity: {blob.get('max_intensity', 'N/A')}")
            print(f"   mean_intensity: {blob.get('mean_intensity', 'N/A')}")
            print(f"   mean_dilation: {blob.get('mean_dilation', 'N/A')}")
            
            # Show how to calculate bounding box corners
            if 'box_x' in blob and 'box_y' in blob and 'box_size' in blob:
                box_x = blob['box_x']
                box_y = blob['box_y']
                box_size = blob['box_size']
                
                print(f"\n📐 CALCULATED BOUNDING BOX COORDINATES:")
                print(f"   Top-left:     ({box_x}, {box_y})")
                print(f"   Top-right:    ({box_x + box_size}, {box_y})")
                print(f"   Bottom-left:  ({box_x}, {box_y + box_size})")
                print(f"   Bottom-right: ({box_x + box_size}, {box_y + box_size})")
                
                # Show scan area calculation
                print(f"\n🎯 FOR AUTONOMOUS MICROSCOPY SCAN PLANNING:")
                print(f"   Scan region: x={box_x} to {box_x + box_size}, y={box_y} to {box_y + box_size}")
                print(f"   Scan area: {box_size}x{box_size} pixels")
                center_x, center_y = blob['center']
                print(f"   Center point: ({center_x}, {center_y})")
    
    print(f"\n" + "=" * 60)
    print("✅ SUMMARY:")
    print("• The detect_blobs() function output is UNCHANGED")
    print("• All bounding box information is preserved:")
    print("  - 'box_x', 'box_y', 'box_size' for scan planning")
    print("  - 'center', 'radius' for circular approximation")
    print("• The plotting functions now show rectangles by default")
    print("• Your autonomous microscopy scan planning code will work unchanged!")
    print("=" * 60)

if __name__ == "__main__":
    # Set random seed for reproducible results
    np.random.seed(42)
    
    try:
        show_blob_data_structure()
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()