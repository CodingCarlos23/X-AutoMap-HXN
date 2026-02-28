import io

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt


def xrf_intensity_to_svg(array, metadata=None):
    """
    Convert XRF intensity array to SVG with contour visualization.

    Creates publication-ready SVG with contour lines, colorbar,
    axes labels, and title.

    Parameters
    ----------
    array : numpy.ndarray
        2D array of XRF intensity values. 3D arrays with shape (1, H, W)
        will be squeezed to 2D.
    metadata : dict, optional
        Metadata dictionary. Recognized keys:
        - 'element': Element name (e.g., 'Ni', 'Fe')
        - 'scan_id': Scan identifier

    Returns
    -------
    bytes
        SVG content as bytes.
    """
    # Handle 3D arrays (1, H, W) -> squeeze to 2D
    if array.ndim == 3 and array.shape[0] == 1:
        array = array.squeeze(0)

    if array.ndim != 2:
        raise ValueError(f"Expected 2D array, got {array.ndim}D")

    # Extract metadata
    element = metadata.get('element', 'unknown') if metadata else 'unknown'
    scan_id = metadata.get('scan_id', '') if metadata else ''

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot filled contours with colorbar
    levels = 10
    contour = ax.contourf(array, levels=levels, cmap='viridis')
    ax.contour(array, levels=levels, colors='black', linewidths=0.5, alpha=0.5)

    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Intensity')

    # Labels and title
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_title(f'XRF Intensity: {element}' + (f' (Scan {scan_id})' if scan_id else ''))
    ax.set_aspect('equal')

    # Save to SVG
    buffer = io.BytesIO()
    fig.savefig(buffer, format='svg', bbox_inches='tight')
    plt.close(fig)

    return buffer.getvalue()
