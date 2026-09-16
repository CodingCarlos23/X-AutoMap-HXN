
def boxes_intersect(b1, b2):
    x1_min, y1_min = b1['box_x'], b1['box_y']
    x1_max, y1_max = x1_min + b1['box_size'], y1_min + b1['box_size']

    x2_min, y2_min = b2['box_x'], b2['box_y']
    x2_max, y2_max = x2_min + b2['box_size'], y2_min + b2['box_size']

    return not (x1_max < x2_min or x1_min > x2_max or y1_max < y2_min or y1_min > y2_max)


def union_box_dimensions(*blobs):
    """
    Computes the bounding box that covers all given blobs (2 or 3).
    Returns:
        center (tuple): (x, y) of union box center
        length (float): side length (square) of union box
        area (float): area of union box
    """
    min_x = min(b['box_x'] for b in blobs)
    min_y = min(b['box_y'] for b in blobs)
    max_x = max(b['box_x'] + b['box_size'] for b in blobs)
    max_y = max(b['box_y'] + b['box_size'] for b in blobs)

    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    length = float(max(max_x - min_x, max_y - min_y))
    return (center_x, center_y), length, length * length


def union_center(*blobs):
    center, _, _ = union_box_dimensions(*blobs)
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
    print(f"[UNION] blobs — red={len(reds)}, green={len(greens)}, blue={len(blues)}")

    def _make_union(*group):
        (cx, cy), length, area = union_box_dimensions(*group)
        tl_x, tl_y = cx - length / 2, cy - length / 2
        br_x, br_y = tl_x + length, tl_y + length
        real_cx = cx * microns_per_pixel_x + true_origin_x
        real_cy = cy * microns_per_pixel_y + true_origin_y
        real_lx = length * microns_per_pixel_x
        real_ly = length * microns_per_pixel_y
        return {
            'center': [cx, cy], 'length': length, 'area': area,
            'image_center': [cx, cy], 'image_length': length, 'image_area_px²': area,
            'real_center_um': [real_cx, real_cy],
            'real_size_um': [real_lx, real_ly],
            'real_area_um²': real_lx * real_ly,
            'real_top_left_um': [tl_x * microns_per_pixel_x + true_origin_x,
                                  tl_y * microns_per_pixel_y + true_origin_y],
            'real_bottom_right_um': [br_x * microns_per_pixel_x + true_origin_x,
                                      br_y * microns_per_pixel_y + true_origin_y],
        }

    if blues:
        # 3-element: all three must intersect pairwise
        for r in reds:
            for g in greens:
                if not boxes_intersect(r, g):
                    continue
                for b in blues:
                    if boxes_intersect(r, b) and boxes_intersect(g, b):
                        union_objects[union_index] = _make_union(r, g, b)
                        union_index += 1
    else:
        # 2-element: red ∩ green
        for r in reds:
            for g in greens:
                if boxes_intersect(r, g):
                    union_objects[union_index] = _make_union(r, g)
                    union_index += 1
        print(f"[UNION] 2-element pairs checked: {len(reds) * len(greens)}, raw unions before dedup: {len(union_objects)}")

    before = len(union_objects)
    union_objects = _dedup_unions(union_objects, overlap_thresh)
    after = len(union_objects)
    if before != after:
        print(f"[UNION] Dedup removed {before - after} redundant unions ({before} → {after}, IoU thresh={overlap_thresh}, containment always on)")

    return union_objects

