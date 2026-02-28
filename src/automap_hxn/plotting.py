import matplotlib.patches as patches
import matplotlib.pyplot as plt

def plot_image_with_boxes(image, formatted_unions, title="Analysis Results", save_path=None):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(image)
    
    for name, info in formatted_unions.items():
        # 1. Extract Center
        if 'image_center' in info:
            cx, cy = info['image_center']
        else:
            continue
            
        # 2. Extract Size/Radius
        if 'image_radius' in info:
            radius = info['image_radius']
            size = radius * 2
            # Offset center to find bottom-left corner
            x = cx - radius
            y = cy - radius
        else:
            # Fallback if size is provided directly
            size = info.get('image_length', 10) 
            x = cx - size / 2
            y = cy - size / 2

        # 3. Draw the Rectangle
        # We use size for both width and height to make it a square
        rect = patches.Rectangle((x, y), size, size, linewidth=2, 
                                 edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        
        # 4. Add Label
        ax.text(x, y - 2, name, fontsize=9, color='red', weight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Plot saved to: {save_path}")
    plt.tight_layout()
    ax.set_title(title)
    plt.show()
    plt.pause(0.1)


def plot_analysis_results(tiff_paths, elem_list, formatted_unions_dict, out_dir, group_name=None):
    """
    Plot analysis results with bounding boxes for each element.
    
    Args:
        tiff_paths: dict of element -> TIFF path
        elem_list: list of elements
        formatted_unions_dict: dict of group_name -> formatted_unions
        out_dir: output directory for saving plots
        group_name: specific group to plot (if None, plots all groups)
    """
    
    if group_name:
        groups_to_plot = {group_name: formatted_unions_dict.get(group_name, {})}
    else:
        groups_to_plot = formatted_unions_dict
    
    for gname, formatted_unions in groups_to_plot.items():
        if not formatted_unions:
            print(f"⏭️ Skipping {gname}: no unions/blobs found")
            continue
        
        # Get the first element's image for visualization
        first_element = None
        for elem in elem_list:
            if elem in tiff_paths:
                first_element = elem
                break
        
        if not first_element:
            print(f"❌ No TIFF found for visualization in {gname}")
            continue
        
        
        tiff_path = tiff_paths[first_element]
        image = tiff.imread(str(tiff_path)).astype(np.float32)
        
        # Normalize for display
        image_norm = cv2.normalize(np.nan_to_num(image), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Create plot
        title = f"Analysis Results - Group {gname} (Element: {first_element})"
        save_path = Path(out_dir) / f"analysis_plot_{gname}.png"
        
        plot_image_with_boxes(image_norm, formatted_unions, title=title, save_path=str(save_path))
