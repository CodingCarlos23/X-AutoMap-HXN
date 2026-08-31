
def boxes_intersect(b1, b2):
    x1_min, y1_min = b1['box_x'], b1['box_y']
    x1_max, y1_max = x1_min + b1['box_size'], y1_min + b1['box_size']

    x2_min, y2_min = b2['box_x'], b2['box_y']
    x2_max, y2_max = x2_min + b2['box_size'], y2_min + b2['box_size']

    return not (x1_max < x2_min or x1_min > x2_max or y1_max < y2_min or y1_min > y2_max)


def union_box_dimensions(b1, b2, b3):
    """
    Computes the union box of three blobs using their box_x, box_y, and box_size.
    The union box is defined by the min bottom-left and max top-right corners.
    Returns:
        center (tuple): (x, y) of union box center
        length (float): side length of union box
        area (float): area of union box
    """
    # bottom-left corners
    bl_x = [b1['box_x'], b2['box_x'], b3['box_x']]
    bl_y = [b1['box_y'], b2['box_y'], b3['box_y']]
   
    # top-right corners
    tr_x = [b1['box_x'] + b1['box_size'], b2['box_x'] + b2['box_size'], b3['box_x'] + b3['box_size']]
    tr_y = [b1['box_y'] + b1['box_size'], b2['box_y'] + b2['box_size'], b3['box_y'] + b3['box_size']]
   
    # union box bounds
    min_x = min(bl_x)
    min_y = min(bl_y)
    max_x = max(tr_x)
    max_y = max(tr_y)
   
    # center of union box
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
   
    # side length and area
    width = max_x - min_x
    height = max_y - min_y
    length = max(width, height)  # make it square
    area = length * length
   
    return (center_x, center_y), float(length), float(area)


def union_center(b1, b2, b3):
    """
    Computes the center of the union box of three blobs.
    Uses the union_box_dimensions function to avoid repeating logic.
    """
    center, _, _ = union_box_dimensions(b1, b2, b3)
    return center


def _union_overlap(u1, u2):
    """Returns (iou, intersection/u1_area, intersection/u2_area) in pixel space."""
    cx1, cy1 = u1['center']
    l1 = u1['length'] / 2
    cx2, cy2 = u2['center']
    l2 = u2['length'] / 2
    ix = max(0, min(cx1 + l1, cx2 + l2) - max(cx1 - l1, cx2 - l2))
    iy = max(0, min(cy1 + l1, cy2 + l2) - max(cy1 - l1, cy2 - l2))
    intersection = ix * iy
    union_area = u1['area'] + u2['area'] - intersection
    iou = intersection / union_area if union_area > 0 else 0.0
    frac1 = intersection / u1['area'] if u1['area'] > 0 else 0.0
    frac2 = intersection / u2['area'] if u2['area'] > 0 else 0.0
    return iou, frac1, frac2


def _dedup_unions(union_objects, overlap_thresh):
    """Remove redundant unions — keeps larger box when:
    - IoU > overlap_thresh, OR
    - the smaller union is fully contained inside the larger one.
    """
    if not union_objects:
        return union_objects
    sorted_keys = sorted(union_objects, key=lambda k: union_objects[k]['area'], reverse=True)
    kept = []
    for k in sorted_keys:
        u = union_objects[k]
        discard = False
        for j in kept:
            v = union_objects[j]
            iou, frac_u, frac_v = _union_overlap(u, v)
            # u is the smaller one (sorted desc); frac_u = how much of u is inside v
            if frac_u >= 0.9 or iou > overlap_thresh:
                discard = True
                break
        if not discard:
            kept.append(k)
    return {new_idx + 1: union_objects[k] for new_idx, k in enumerate(kept)}


def find_union_blobs(blobs, microns_per_pixel_x, microns_per_pixel_y, true_origin_x, true_origin_y, overlap_thresh=0.5):
    blobs_by_color = {color: [] for color in blobs}

    for color, blob_dict in blobs.items():
        for coord_key, blob_list in blob_dict.items():
            blobs_by_color[color].extend(blob_list)

    union_objects = {}
    union_index = 1
    reds = blobs_by_color.get('red', [])
    greens = blobs_by_color.get('green', [])
    blues = blobs_by_color.get('blue', [])

    for r in reds:
        for g in greens:
            if not boxes_intersect(r, g):
                continue
            for b in blues:
                if boxes_intersect(r, b) and boxes_intersect(g, b):
                    cx, cy = union_center(r, g, b)
                    _, length, area = union_box_dimensions(r, g, b)
                    top_left_x = cx - length // 2
                    top_left_y = cy - length // 2
                    bottom_right_x = top_left_x + length
                    bottom_right_y = top_left_y + length

                    real_cx = (cx * microns_per_pixel_x) + true_origin_x
                    real_cy = (cy * microns_per_pixel_y) + true_origin_y
                    real_length_x = length * microns_per_pixel_x
                    real_length_y = length * microns_per_pixel_y
                    real_area = real_length_x * real_length_y

                    real_top_left = (
                        (top_left_x * microns_per_pixel_x) + true_origin_x,
                        (top_left_y * microns_per_pixel_y) + true_origin_y
                    )
                    real_bottom_right = (
                        (bottom_right_x * microns_per_pixel_x) + true_origin_x,
                        (bottom_right_y * microns_per_pixel_y) + true_origin_y
                    )

                    union_obj = {
                        # Original fields (used by formatter)
                        'center': [cx, cy],
                        'length': length,
                        'area': area,

                        # Alias for compatibility with merge logic
                        'image_center': [cx, cy],
                        'image_length': length,
                        'image_area_px²': area,

                        # Real-world
                        'real_center_um': [real_cx, real_cy],
                        'real_size_um': [real_length_x, real_length_y],
                        'real_area_um\u00b2': real_area,
                        'real_top_left_um': list(real_top_left),
                        'real_bottom_right_um': list(real_bottom_right),
                    }

                    union_objects[union_index] = union_obj
                    union_index += 1

    before = len(union_objects)
    union_objects = _dedup_unions(union_objects, overlap_thresh)
    after = len(union_objects)
    if before != after:
        print(f"[UNION] Dedup removed {before - after} redundant unions ({before} → {after}, IoU thresh={overlap_thresh}, containment always on)")

    return union_objects

