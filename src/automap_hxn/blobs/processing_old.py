
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

    # Merge source blobs from both unions
    merged_source_blobs = list(u1.get('source_blobs', []))
    for blob in u2.get('source_blobs', []):
        if blob not in merged_source_blobs:
            merged_source_blobs.append(blob)

    # Merge colors from both unions
    merged_colors = list(set(u1.get('colors', []) + u2.get('colors', [])))

    return {
        'center': [new_cx, new_cy], 'length': new_length, 'area': new_area,
        'image_center': [new_cx, new_cy], 'image_length': new_length, 'image_area_px²': new_area,
        'real_center_um': [real_cx, real_cy],
        'real_size_um': [real_lx, real_ly],
        'real_area_um²': real_lx * real_ly,
        'real_top_left_um': [rx1, ry1],
        'real_bottom_right_um': [rx2, ry2],
        'source_blobs': merged_source_blobs,
        'colors': merged_colors,
        'element_count': len(merged_colors),
    }


def _merge_unions(union_objects, overlap_thresh):
    """Iteratively merge union boxes whose IoU exceeds overlap_thresh, or where
    the smaller box is fully contained within the larger one.

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
                iou, frac_i, frac_j = _union_overlap(current, boxes[j])
                if iou > overlap_thresh or max(frac_i, frac_j) >= 0.8:
                    current = _merge_two_unions(current, boxes[j])
                    used.add(j)
                    changed = True
                    total_merges += 1
            used.add(i)
            new_boxes.append(current)
        boxes = new_boxes

    if total_merges:
        print(f"[UNION-OVERLAP] Merged {total_merges} overlapping pair(s) over {pass_num} pass(es) "
              f"(IoU thresh={overlap_thresh}) → {len(boxes)} box(es) remain")

    return {idx + 1: box for idx, box in enumerate(boxes)}


def boxes_overlap_iou(b1, b2):
    """
    Calculate IoU (Intersection over Union) between two blob boxes.
    Returns: (iou, intersection/b1_area, intersection/b2_area)
    """
    x1_min, y1_min = b1['box_x'], b1['box_y']
    x1_max, y1_max = x1_min + b1['box_size'], y1_min + b1['box_size']

    x2_min, y2_min = b2['box_x'], b2['box_y']
    x2_max, y2_max = x2_min + b2['box_size'], y2_min + b2['box_size']

    # Calculate intersection
    ix = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    iy = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    intersection = ix * iy

    # Calculate areas
    b1_area = b1['box_size'] * b1['box_size']
    b2_area = b2['box_size'] * b2['box_size']
    union_area = b1_area + b2_area - intersection

    # Calculate metrics
    iou = intersection / union_area if union_area > 0 else 0.0
    frac1 = intersection / b1_area if b1_area > 0 else 0.0
    frac2 = intersection / b2_area if b2_area > 0 else 0.0

    return iou, frac1, frac2


def boxes_have_sufficient_overlap(b1, b2, overlap_thresh):
    """
    Check if two boxes have sufficient overlap based on IoU threshold.
    Returns True if IoU > overlap_thresh OR if either box is 80%+ contained in the other.
    """
    iou, frac1, frac2 = boxes_overlap_iou(b1, b2)
    return iou > overlap_thresh or max(frac1, frac2) >= 0.8


def find_union_blobs_by_category(blobs, microns_per_pixel_x, microns_per_pixel_y, true_origin_x, true_origin_y, overlap_thresh=0.5, categories=None):
    """
    Find union boxes categorized by element count (3-element, 2-element, 1-element).

    Detects all possible overlapping combinations and solo blobs, organizing them
    into categories for filtering.

    Parameters:
    -----------
    blobs : dict
        Dictionary of {color: {coord_key: [blob_list]}}
    microns_per_pixel_x, microns_per_pixel_y : float
        Pixel to micron conversion factors
    true_origin_x, true_origin_y : float
        Real-world origin coordinates
    overlap_thresh : float
        IoU threshold for considering boxes as overlapping (default 0.5)
    categories : list or None
        List of categories to include. Options: '3-element', '2-element', '1-element'.
        If None, returns all categories.

    Returns:
    --------
    dict : {
        '3-element': {index: union_box_dict},  # red+green+blue
        '2-element': {index: union_box_dict},  # any two colors
        '1-element': {index: union_box_dict},  # solo blobs
    }
    """
    if categories is None:
        categories = ['3-element', '2-element', '1-element']

    blobs_by_color = {color: [] for color in blobs}
    for color, blob_dict in blobs.items():
        for _coord_key, blob_list in blob_dict.items():
            blobs_by_color[color].extend(blob_list)

    reds = blobs_by_color.get('red', [])
    greens = blobs_by_color.get('green', [])
    blues = blobs_by_color.get('blue', [])
    print(f"[UNION-CATEGORY] blobs — red={len(reds)}, green={len(greens)}, blue={len(blues)} (IoU thresh={overlap_thresh})")

    def _make_union(*group, colors_included=None, source_blobs=None):
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
            'colors': colors_included or [],
            'element_count': len(group),
            'source_blobs': source_blobs or list(group),  # Store references to source blobs
        }

    result = {
        '3-element': {},
        '2-element': {},
        '1-element': {},
    }

    # Track which blobs have been used in unions
    used_reds = set()
    used_greens = set()
    used_blues = set()

    union_index = {'3-element': 1, '2-element': 1, '1-element': 1}

    # 3-element unions: red+green+blue
    if '3-element' in categories and blues:
        for i, r in enumerate(reds):
            for j, g in enumerate(greens):
                if not boxes_have_sufficient_overlap(r, g, overlap_thresh):
                    continue
                for k, b in enumerate(blues):
                    if (boxes_have_sufficient_overlap(r, b, overlap_thresh) and
                        boxes_have_sufficient_overlap(g, b, overlap_thresh)):
                        result['3-element'][union_index['3-element']] = _make_union(
                            r, g, b, colors_included=['red', 'green', 'blue']
                        )
                        union_index['3-element'] += 1
                        used_reds.add(i)
                        used_greens.add(j)
                        used_blues.add(k)

        # Merge overlapping 3-element unions
        before = len(result['3-element'])
        result['3-element'] = _merge_unions(result['3-element'], overlap_thresh)
        after = len(result['3-element'])
        if before != after:
            print(f"[UNION-CATEGORY] 3-element: {before} raw → {after} after merging")

    # 2-element unions
    if '2-element' in categories:
        # red+green
        for i, r in enumerate(reds):
            for j, g in enumerate(greens):
                if boxes_have_sufficient_overlap(r, g, overlap_thresh):
                    # Skip if both are already in a 3-element union
                    if i in used_reds and j in used_greens and blues:
                        continue
                    result['2-element'][union_index['2-element']] = _make_union(
                        r, g, colors_included=['red', 'green']
                    )
                    union_index['2-element'] += 1
                    used_reds.add(i)
                    used_greens.add(j)

        # red+blue
        if blues:
            for i, r in enumerate(reds):
                for k, b in enumerate(blues):
                    if boxes_have_sufficient_overlap(r, b, overlap_thresh):
                        if i in used_reds and k in used_blues:
                            continue
                        result['2-element'][union_index['2-element']] = _make_union(
                            r, b, colors_included=['red', 'blue']
                        )
                        union_index['2-element'] += 1
                        used_reds.add(i)
                        used_blues.add(k)

        # green+blue
        if blues:
            for j, g in enumerate(greens):
                for k, b in enumerate(blues):
                    if boxes_have_sufficient_overlap(g, b, overlap_thresh):
                        if j in used_greens and k in used_blues:
                            continue
                        result['2-element'][union_index['2-element']] = _make_union(
                            g, b, colors_included=['green', 'blue']
                        )
                        union_index['2-element'] += 1
                        used_greens.add(j)
                        used_blues.add(k)

        # Merge overlapping 2-element unions
        before = len(result['2-element'])
        result['2-element'] = _merge_unions(result['2-element'], overlap_thresh)
        after = len(result['2-element'])
        if before != after:
            print(f"[UNION-CATEGORY] 2-element: {before} raw → {after} after merging")

    # 1-element boxes: solo blobs that don't overlap with anything
    if '1-element' in categories:
        for i, r in enumerate(reds):
            if i not in used_reds:
                result['1-element'][union_index['1-element']] = _make_union(
                    r, colors_included=['red']
                )
                union_index['1-element'] += 1

        for j, g in enumerate(greens):
            if j not in used_greens:
                result['1-element'][union_index['1-element']] = _make_union(
                    g, colors_included=['green']
                )
                union_index['1-element'] += 1

        for k, b in enumerate(blues):
            if k not in used_blues:
                result['1-element'][union_index['1-element']] = _make_union(
                    b, colors_included=['blue']
                )
                union_index['1-element'] += 1

        print(f"[UNION-CATEGORY] 1-element (solo): {len(result['1-element'])} boxes")

    print(f"[UNION-CATEGORY] Results: 3-elem={len(result['3-element'])}, "
          f"2-elem={len(result['2-element'])}, 1-elem={len(result['1-element'])}")

    return result


def find_union_blobs(blobs, microns_per_pixel_x, microns_per_pixel_y, true_origin_x, true_origin_y, overlap_thresh=0.5):
    """
    NEW ALGORITHM: Find union boxes from overlapping blobs using overlap_thresh for initial pairing.

    Unlike the original algorithm which uses simple intersection (any touch counts),
    this version requires boxes to have IoU > overlap_thresh (or 80%+ containment)
    to be paired together.

    This function returns only the highest-level unions (3-element if blues exist, otherwise 2-element).
    For granular control over which union types to get, use find_union_blobs_by_category.

    Parameters:
    -----------
    blobs : dict
        Dictionary of {color: {coord_key: [blob_list]}}
    microns_per_pixel_x, microns_per_pixel_y : float
        Pixel to micron conversion factors
    true_origin_x, true_origin_y : float
        Real-world origin coordinates
    overlap_thresh : float
        IoU threshold for considering boxes as overlapping (default 0.5)

    Returns:
    --------
    dict : {index: union_box_dict}
    """
    blobs_by_color = {color: [] for color in blobs}

    for color, blob_dict in blobs.items():
        for _coord_key, blob_list in blob_dict.items():
            blobs_by_color[color].extend(blob_list)

    union_objects = {}
    union_index = 1
    reds = blobs_by_color.get('red', [])
    greens = blobs_by_color.get('green', [])
    blues = blobs_by_color.get('blue', [])
    print(f"[UNION-OVERLAP] blobs — red={len(reds)}, green={len(greens)}, blue={len(blues)} (IoU thresh={overlap_thresh})")

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
        # 3-element: all three must have sufficient pairwise overlap
        for r in reds:
            for g in greens:
                if not boxes_have_sufficient_overlap(r, g, overlap_thresh):
                    continue
                for b in blues:
                    if (boxes_have_sufficient_overlap(r, b, overlap_thresh) and
                        boxes_have_sufficient_overlap(g, b, overlap_thresh)):
                        union_objects[union_index] = _make_union(r, g, b)
                        union_index += 1
    else:
        # 2-element: red and green must have sufficient overlap
        for r in reds:
            for g in greens:
                if boxes_have_sufficient_overlap(r, g, overlap_thresh):
                    union_objects[union_index] = _make_union(r, g)
                    union_index += 1
        print(f"[UNION-OVERLAP] 2-element pairs checked: {len(reds) * len(greens)}, raw unions before dedup: {len(union_objects)}")

    before = len(union_objects)
    union_objects = _merge_unions(union_objects, overlap_thresh)
    after = len(union_objects)
    if before != after:
        print(f"[UNION-OVERLAP] {before} raw union(s) → {after} after merging (IoU thresh={overlap_thresh})")

    return union_objects
