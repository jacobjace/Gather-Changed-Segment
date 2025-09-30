"""
imports 
"""
import numpy as np
import cv2
from peel_methods import peel_top, peel_bottom, peel_right, peel_left

"""
helper methods
"""
def _safe_gray(img):
    if img is None:
        return None
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.copy()

def _norm01(a, eps=1e-9):
    m = a.min()
    a2 = a.astype(np.float32) - m
    mx = a2.max()
    if mx > eps:
        a2 /= (mx + eps)
    return a2

def tighten_by_strips(before, after, bbox,
                      strip_thickness=4,
                      thr_frac=0.35,
                      max_total_strip=0.5,
                      max_iter_per_side=100,
                      min_w=12, min_h=12):
    
    H, W = before.shape[:2]
    x, y, w, h = map(int, bbox)
    x = max(0, min(x, W-1)); y = max(0, min(y, H-1))
    w = max(1, min(w, W - x)); h = max(1, min(h, H - y))

    gray_b = _safe_gray(before).astype(np.float32)
    gray_a = _safe_gray(after).astype(np.float32)
    raw = np.abs(gray_a - gray_b)

    # initial mean
    inner = raw[y:y+h, x:x+w]
    if inner.size == 0:
        return (x,y,w,h)
    inner_mean = float(inner.mean())
    if inner_mean < 1e-6:
        return (x,y,w,h)

    max_strip_px_w = int(max_total_strip * w)
    max_strip_px_h = int(max_total_strip * h)

    # peel inwards from each side 
    passes = 2
    for _ in range(passes):
        top_r = peel_top(raw, x,y,w,h, inner_mean, max_strip_px_h)
        y += top_r; h -= top_r
        bottom_r = peel_bottom(raw, x,y,w,h, inner_mean, max_strip_px_h)
        h -= bottom_r
        left_r = peel_left(raw, x,y,w,h, inner_mean, max_strip_px_w)
        x += left_r; w -= left_r
        right_r = peel_right(raw, x,y,w,h, inner_mean, max_strip_px_w)
        w -= right_r

        # recompute inner_mean for the smaller interior
        if w > 0 and h > 0:
            inner2 = raw[y:y+h, x:x+w]
            if inner2.size > 0:
                new_inner_mean = float(inner2.mean())
                # update inner_mean 
                inner_mean = min(inner_mean, max(1e-6, new_inner_mean))
        else:
            break

    x = max(0, x); y = max(0, y)
    w = max(1, min(W - x, w)); h = max(1, min(H - y, h))

    return (x, y, w, h)
