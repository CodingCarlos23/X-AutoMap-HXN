
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


def _merge_two_unions(u1, u2):
    """Return a new union box that is the bounding box of u1 and u2 in both pixel and real space."""
    cx1, cy1 = u1['center']
    l1 = u1['length'] / 2
    cx2, cy2 = u2['center']
    l2 = u2['length'] / 2

    px_x1 = min(cx1 - l1, cx2 - l2)
    px_x2 = max(cx1 + l1, cx2 + l2)
    px_y1 = min(cy1 - l1, cy2 - l2)
    px_y2 = max(cy1 + l1, cy2 + l2)
    new_length = float(max(px_x2 - px_x1, px_y2 - px_y1))
    new_cx = (px_x1 + px_x2) / 2
    new_cy = (px_y1 + px_y2) / 2
    new_area = new_length * new_length

    tl1 = u1['real_top_left_um']
    br1 = u1['real_bottom_right_um']
    tl2 = u2['real_top_left_um']
    br2 = u2['real_bottom_right_um']
    rx1 = min(tl1[0], tl2[0])
    rx2 = max(br1[0], br2[0])
    ry1 = min(tl1[1], tl2[1])
    ry2 = max(br1[1], br2[1])
    real_cx = (rx1 + rx2) / 2
    real_cy = (ry1 + ry2) / 2
    real_lx = rx2 - rx1
    real_ly = ry2 - ry1

    return {
        'center': [new_cx, new_cy], 'length': new_length, 'area': new_area,
        'image_center': [new_cx, new_cy], 'image_length': new_length, 'image_area_px²': new_area,
        'real_center_um': [real_cx, real_cy],
        'real_size_um': [real_lx, real_ly],
        'real_area_um²': real_lx * real_ly,
        'real_top_left_um': [rx1, ry1],
        'real_bottom_right_um': [rx2, ry2],
    }


def _merge_unions(union_objects, overlap_thresh):
    """Iteratively merge union boxes whose IoU exceeds overlap_thresh into one encompassing box.

    Repeats until every remaining pair is under the threshold, so chains of
    3+ overlapping boxes collapse fully in successive passes.
    """
    if not union_objects:
        return union_objects

    boxes = list(union_objects.values())
    pass_num = 0
    total_merges = 0

    changed = True
    while changed:
        changed = False
        pass_num += 1
        new_boxes = []
        used = set()

        for i in range(len(boxes)):
            if i in used:
                continue
            current = boxes[i]
            for j in range(i + 1, len(boxes)):
                if j in used:
                    continue
                iou, _, _ = _union_overlap(current, boxes[j])
                if iou > overlap_thresh:
                    current = _merge_two_unions(current, boxes[j])
                    used.add(j)
                    changed = True
                    total_merges += 1
            used.add(i)
            new_boxes.append(current)
        boxes = new_boxes

    if total_merges:
        print(f"[UNION] Merged {total_merges} overlapping pair(s) over {pass_num} pass(es) "
              f"(IoU thresh={overlap_thresh}) → {len(boxes)} box(es) remain")

    return {idx + 1: box for idx, box in enumerate(boxes)}


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
    union_objects = _merge_unions(union_objects, overlap_thresh)
    after = len(union_objects)
    if before != after:
        print(f"[UNION] {before} raw union(s) → {after} after merging (IoU thresh={overlap_thresh})")

    return union_objects

