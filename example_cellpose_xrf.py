#!/usr/bin/env python3
"""
Example script demonstrating Cellpose blob detection for XRF images

This script shows how to use the new Cellpose method in the blob detection framework
for analyzing X-ray fluorescence microscopy data.
"""

import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from pathlib import Path

# Check if we have the required modules
try:
    from utils import detect_blobs, normalize_and_dilate, CELLPOSE_AVAILABLE
    if not CELLPOSE_AVAILABLE:
        print("❌ Cellpose not available. Install with: pip install cellpose[gui]")
        exit(1)
except ImportError as e:
    print(f"❌ Error importing utils: {e}")
    exit(1)


def analyze_xrf_with_cellpose(image_path, output_dir="cellpose_results"):
    """
    Analyze XRF image using Cellpose for particle detection.
    
    Parameters:
    -----------
    image_path : str or Path
        Path to the XRF TIFF image
    output_dir : str  
        Directory to save results
    """
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"Analyzing XRF image: {image_path}")
    
    # Load image
    try:
        img = tiff.imread(str(image_path)).astype(np.float32)
        print(f"Image shape: {img.shape}")
        print(f"Image range: {img.min():.2f} - {img.max():.2f}")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return
    
    # Normalize and dilate for preprocessing
    img_norm, img_dilated = normalize_and_dilate(img)
    
    # Detect blobs using different methods for comparison
    methods_to_test = {
        'simple': {'max_area': 2000},
        'contours': {},
        'hough': {'max_radius': 50, 'min_dist': 20}, 
        'cellpose': {
            'diameter': 60,           # Expected particle diameter in pixels
            'model_type': 'cyto3',    # Cellpose model (cyto3 good for particles)
            'flow_threshold': 0.4,    # Lower = more permissive
            'cellprob_threshold': 0.0, # Cell probability threshold  
            'min_diameter': 30,       # Filter particles by size
            'max_diameter': 100,
            'gpu': False              # Set to True if you have GPU
        }
    }
    
    # Parameters for blob detection
    min_thresh = 50     # Minimum image threshold
    min_area = 100      # Minimum area in pixels
    element_name = Path(image_path).stem  # Use filename as element name
    
    results = {}
    
    # Test each method
    for method, params in methods_to_test.items():
        print(f"\nTesting {method} method...")
        try:
            blobs = detect_blobs(
                img_dilated, img_norm, 
                min_thresh, min_area, 
                'red', element_name,
                method=method, 
                include_method_info=True,
                **params
            )
            results[method] = blobs
            print(f"  Found {len(blobs)} particles")
            
            # Show some details for Cellpose
            if method == 'cellpose' and blobs:
                print(f"  Sample detection details:")
                for i, blob in enumerate(blobs[:3]):  # Show first 3
                    if 'equiv_diameter' in blob:
                        print(f"    Particle {i+1}: diameter={blob['equiv_diameter']:.1f}px, area={blob.get('area', 'N/A')}")
                    
        except Exception as e:
            print(f"  ❌ Error with {method}: {e}")
            results[method] = []
    
    # Create comparison visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Show original image
    axes[0].imshow(img, cmap='viridis')
    axes[0].set_title(f'Original XRF Image\n{element_name}')
    axes[0].axis('off')
    
    # Show processed image
    axes[1].imshow(img_dilated, cmap='gray')
    axes[1].set_title('Processed Image\n(Normalized & Dilated)')
    axes[1].axis('off')
    
    # Show results from each method
    method_names = list(methods_to_test.keys())
    colors = ['red', 'blue', 'green', 'orange']
    
    for i, method in enumerate(method_names):
        ax_idx = i + 2
        if ax_idx < len(axes):
            axes[ax_idx].imshow(img, cmap='viridis', alpha=0.7)
            
            blobs = results.get(method, [])
            for blob in blobs:
                x, y = blob['center'] 
                radius = blob['radius']
                circle = plt.Circle((x, y), radius, fill=False, 
                                  color=colors[i % len(colors)], linewidth=2)
                axes[ax_idx].add_patch(circle)
            
            axes[ax_idx].set_title(f'{method.upper()}\n{len(blobs)} particles')
            axes[ax_idx].axis('off')
    
    # Hide unused subplots
    for i in range(len(method_names) + 2, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Save comparison plot
    output_file = output_path / f"{element_name}_cellpose_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ Comparison plot saved: {output_file}")
    
    # Save detailed results
    summary_file = output_path / f"{element_name}_results_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"XRF Blob Detection Results: {element_name}\\n")
        f.write("=" * 50 + "\\n\\n")
        f.write(f"Image: {image_path}\\n")
        f.write(f"Image shape: {img.shape}\\n")
        f.write(f"Parameters: min_thresh={min_thresh}, min_area={min_area}\\n\\n")
        
        for method, blobs in results.items():
            f.write(f"{method.upper()} Method: {len(blobs)} particles\\n")
            if method == 'cellpose' and blobs:
                f.write("  Cellpose-specific details:\\n")
                for i, blob in enumerate(blobs):
                    f.write(f"    Particle {i+1}: center={blob['center']}, ")
                    if 'equiv_diameter' in blob:
                        f.write(f"diameter={blob['equiv_diameter']:.1f}px, ")
                    if 'area' in blob:
                        f.write(f"area={blob['area']}px²")
                    f.write("\\n")
            f.write("\\n")
    
    print(f"✅ Results summary saved: {summary_file}")
    
    # Show plot
    plt.show()
    
    return results


def create_synthetic_xrf_example():
    """Create a synthetic XRF-like image for testing"""
    print("Creating synthetic XRF example...")
    
    # Create synthetic data resembling XRF map
    img = np.zeros((256, 256), dtype=np.float32)
    
    # Add particles with varying intensities and sizes
    particles = [
        (50, 50, 20, 1000),    # (x, y, radius, intensity)
        (120, 80, 15, 800),
        (200, 150, 25, 1200),
        (80, 180, 18, 900),
        (180, 60, 22, 1100),
        (60, 120, 12, 600),
        (150, 200, 16, 750)
    ]
    
    for x, y, radius, intensity in particles:
        # Create circular particle with some noise
        yy, xx = np.ogrid[:256, :256]
        mask = (xx - x)**2 + (yy - y)**2 <= radius**2
        img[mask] += intensity
    
    # Add background noise and texture
    noise = np.random.normal(0, 50, img.shape)
    img = np.maximum(img + noise, 0)  # Keep non-negative
    
    # Add some texture/grain structure  
    grain = np.random.random(img.shape) * 100
    img += grain
    
    # Save synthetic image
    output_file = "synthetic_xrf_example.tiff"
    tiff.imwrite(output_file, img.astype(np.float32))
    print(f"✅ Synthetic XRF image saved: {output_file}")
    
    return output_file


def main():
    """Main function to run Cellpose XRF analysis examples"""
    
    print("🔬 Cellpose XRF Blob Detection Example")
    print("=" * 50)
    
    # Check if user has a real XRF image to analyze
    print("\\nDo you have an XRF TIFF image to analyze?")
    print("1. Yes - I'll provide the path")
    print("2. No - Create a synthetic example")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        image_path = input("Enter path to XRF TIFF image: ").strip()
        if not Path(image_path).exists():
            print(f"❌ File not found: {image_path}")
            print("Creating synthetic example instead...")
            image_path = create_synthetic_xrf_example()
    else:
        image_path = create_synthetic_xrf_example()
    
    # Analyze the image
    print(f"\\n🔍 Starting analysis...")
    results = analyze_xrf_with_cellpose(image_path)
    
    # Print summary
    print("\\n📊 ANALYSIS COMPLETE")
    print("=" * 30)
    for method, blobs in results.items():
        print(f"{method}: {len(blobs)} particles detected")
    
    print("\\n💡 Tips for using Cellpose with XRF data:")
    print("• Adjust 'diameter' parameter based on expected particle size")
    print("• Lower 'flow_threshold' for more permissive detection") 
    print("• Use 'cyto3' model for general particles, 'nuclei' for dense objects")
    print("• Set GPU=True for faster processing if you have CUDA GPU")
    print("• Filter results using min_diameter/max_diameter for size selection")


if __name__ == "__main__":
    main()