
"""
helper methods for tightening the changed area
"""

"""
Parameters
- raw - image
- x,y,w,h - bbox
- max_strip_px_h, max_strip_px_w - max strip height or width
- min_h, min_w - 
- strip_thickness -
- max_iter_per_side -
- thr_frac -
- inner_mean - 
"""

def peel_top(raw, x,y,w,h, inner_mean, max_strip_px_h, min_h=12, strip_thickness=4, max_iter_per_side=100, thr_frac=0.35):
    removed = 0
    iter_count = 0
    while removed < max_strip_px_h and (h - removed) > min_h and iter_count < max_iter_per_side:
        th = min(strip_thickness, h - removed - min_h)
        if th <= 0:
            break
        row_start = y + removed
        row_end = row_start + th
        strip = raw[row_start:row_end, x:x+w]
        if strip.size == 0:
            break
        strip_mean = float(strip.mean())
        if strip_mean < thr_frac * inner_mean:
            removed += th
        else:
            break
        iter_count += 1
    return removed

def peel_bottom(raw, x,y,w,h, inner_mean, max_strip_px_h, min_h=12, strip_thickness=4, max_iter_per_side=100, thr_frac=0.35):
    removed = 0
    iter_count = 0
    while removed < max_strip_px_h and (h - removed) > min_h and iter_count < max_iter_per_side:
        th = min(strip_thickness, h - removed - min_h)
        if th <= 0:
            break
        row_end = y + h - removed
        row_start = row_end - th
        strip = raw[row_start:row_end, x:x+w]
        if strip.size == 0:
            break
        strip_mean = float(strip.mean())
        if strip_mean < thr_frac * inner_mean:
            removed += th
        else:
            break
        iter_count += 1
    return removed

def peel_left(raw, x,y,w,h, inner_mean, max_strip_px_w, min_w=12, strip_thickness=4, max_iter_per_side=100, thr_frac=0.35):
    removed = 0
    iter_count = 0
    while removed < max_strip_px_w and (w - removed) > min_w and iter_count < max_iter_per_side:
        th = min(strip_thickness, w - removed - min_w)
        if th <= 0:
            break
        col_start = x + removed
        col_end = col_start + th
        strip = raw[y:y+h, col_start:col_end]
        if strip.size == 0:
            break
        strip_mean = float(strip.mean())
        if strip_mean < thr_frac * inner_mean:
            removed += th
        else:
            break
        iter_count += 1
    return removed

def peel_right(raw, x,y,w,h, inner_mean, max_strip_px_w, min_w=12, strip_thickness=4, max_iter_per_side=100, thr_frac=0.35):
    removed = 0
    iter_count = 0
    while removed < max_strip_px_w and (w - removed) > min_w and iter_count < max_iter_per_side:
        th = min(strip_thickness, w - removed - min_w)
        if th <= 0:
            break
        col_end = x + w - removed
        col_start = col_end - th
        strip = raw[y:y+h, col_start:col_end]
        if strip.size == 0:
            break
        strip_mean = float(strip.mean())
        if strip_mean < thr_frac * inner_mean:
            removed += th
        else:
            break
        iter_count += 1
    return removed
