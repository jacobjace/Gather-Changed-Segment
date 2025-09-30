import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from collections import deque
from tighten_by_strips import tighten_by_strips, _safe_gray, _norm01
def detect_popup_adaptive(before, after, save_paths,
                          block_size=32,
                          weight_normdiff=1,
                          weight_edge=0.4,
                          weight_ssim=0.1,
                          min_component_blocks=4,
                          min_blocks_abs=4,
                          max_blocks_frac=0.6,
                          target_energy_frac=0.50,
                          mean_std_k=0.6,
                          percentile_min=98,
                          relax_if_too_small=True,
                          relax_steps=2,
                          relax_factor=0.5,
                          expand_px=0,
                          max_area_frac=0.9,
                          min_area_frac=0.0000,
                          ):
    """
    Returns (bbox, cropped, vis) where bbox = (x,y,w,h) or (None, None, vis)
    """
    if isinstance(before, str):
        before = cv2.imread(before)
    if isinstance(after, str):
        after = cv2.imread(after)
    if before is None or after is None:
        raise ValueError("Images couldn't be loaded")

    h, w = before.shape[:2]
    if after.shape[:2] != (h, w):
        after = cv2.resize(after, (w, h), interpolation=cv2.INTER_LINEAR)

    gray_b = _safe_gray(before).astype(np.float32)
    gray_a = _safe_gray(after).astype(np.float32)

    sob_bx = cv2.Sobel(gray_b, cv2.CV_32F, 1, 0, ksize=3)
    sob_by = cv2.Sobel(gray_b, cv2.CV_32F, 0, 1, ksize=3)
    sob_ax = cv2.Sobel(gray_a, cv2.CV_32F, 1, 0, ksize=3)
    sob_ay = cv2.Sobel(gray_a, cv2.CV_32F, 0, 1, ksize=3)
    mag_b = np.sqrt(sob_bx*sob_bx + sob_by*sob_by)
    mag_a = np.sqrt(sob_ax*sob_ax + sob_ay*sob_ay)

    bs = block_size
    nx = (w + bs - 1) // bs
    ny = (h + bs - 1) // bs

    normdiff = np.zeros((ny, nx), dtype=np.float32)
    edgediff = np.zeros((ny, nx), dtype=np.float32)
    ssimmap = np.zeros((ny, nx), dtype=np.float32) if ssim is not None else None

    eps = 1e-6
    for by in range(ny):
        for bx in range(nx):
            x0 = bx * bs; y0 = by * bs
            x1 = min(w, x0 + bs); y1 = min(h, y0 + bs)
            b_block = gray_b[y0:y1, x0:x1]
            a_block = gray_a[y0:y1, x0:x1]

            # local normalized diff: (robust to brightness)
            mb = float(b_block.mean()); ma = float(a_block.mean())
            sb = float(b_block.std()); sa = float(a_block.std())
            sb = max(sb, eps); sa = max(sa, eps)
            nb = (b_block - mb) / sb
            na = (a_block - ma) / sa
            ndiff = float(np.mean(np.abs(nb - na)))
            normdiff[by, bx] = ndiff

            # edge magnitude diff (mean magnitude difference)
            mag_b_mean = float(mag_b[y0:y1, x0:x1].mean())
            mag_a_mean = float(mag_a[y0:y1, x0:x1].mean())
            edgediff[by, bx] = abs(mag_a_mean - mag_b_mean)

            # per-block ssim 
            if ssim is not None:
                try:
                    b_u8 = np.clip(b_block, 0, 255).astype(np.uint8)
                    a_u8 = np.clip(a_block, 0, 255).astype(np.uint8)
                    s = ssim(b_u8, a_u8, data_range=255)
                    ssimmap[by, bx] = 1.0 - s
                except Exception:
                    ssimmap[by, bx] = 0.0

    # normalize 
    n_norm = _norm01(normdiff)
    n_edge = _norm01(edgediff)
    n_ssim = _norm01(ssimmap) if ssimmap is not None else np.zeros_like(n_norm)

    # combined score
    total_w = weight_normdiff + weight_edge + weight_ssim
    if total_w <= 0: total_w = 1.0
    score = (weight_normdiff * n_norm + weight_edge * n_edge + weight_ssim * n_ssim) / total_w
    # smooth 
    score = cv2.GaussianBlur(score, (3,3), 0)

    flat = score.flatten()
    sorted_idx = np.argsort(flat)[::-1]  
    sorted_vals = flat[sorted_idx]
    total_energy = sorted_vals.sum() + eps

    def select_by_mean_std(k):
        thr = flat.mean() + k * flat.std()
        sel = (score >= thr).astype(np.uint8)
        return sel

    def select_by_cumulative_energy(target_frac):
        # take top blocks until cumulative sum reaches target_frac of total energy
        cum = np.cumsum(sorted_vals)
        cutoff_idx = np.searchsorted(cum, target_frac * total_energy)
        cutoff_idx = max(1, cutoff_idx)
        if cutoff_idx >= len(sorted_vals):
            cutoff_idx = len(sorted_vals) - 1
        val_thr = sorted_vals[cutoff_idx]
        sel = (score >= val_thr).astype(np.uint8)
        return sel

    def select_by_percentile(pct):
        thr = np.percentile(flat, pct)
        sel = (score >= thr).astype(np.uint8)
        return sel

    def sel_to_bbox(sel_grid):
        visited = np.zeros_like(sel_grid, dtype=np.uint8)
        comps = []
        dirs = [(1,0),(-1,0),(0,1),(0,-1)]
        for y0 in range(ny):
            for x0 in range(nx):
                if sel_grid[y0, x0] and not visited[y0, x0]:
                    q = deque()
                    q.append((y0, x0))
                    visited[y0, x0] = 1
                    comp_blocks = []
                    comp_score = 0.0
                    while q:
                        yy, xx = q.popleft()
                        comp_blocks.append((yy, xx))
                        comp_score += float(score[yy, xx])
                        for dy, dx in dirs:
                            nyb, nxb = yy + dy, xx + dx
                            if 0 <= nyb < ny and 0 <= nxb < nx and sel_grid[nyb, nxb] and not visited[nyb, nxb]:
                                visited[nyb, nxb] = 1
                                q.append((nyb, nxb))
                    comps.append({
                        'blocks': comp_blocks,
                        'score_sum': comp_score,
                        'num_blocks': len(comp_blocks)
                    })
        if not comps:
            return None, []

        img_blocks = nx * ny
        best_comp = None
        best_val = -1e12
        for c in comps:
            frac = c['num_blocks'] / float(img_blocks)
            if c['num_blocks'] < min_component_blocks:
                continue
            if frac > max_blocks_frac:
                continue
            val = c['score_sum'] * (c['num_blocks'] ** 0.5)
            if val > best_val:
                best_val = val
                best_comp = c
        # fallback if we filtered everything
        if best_comp is None:
            best_comp = max(comps, key=lambda x: x['score_sum'])
        # convert block list to bbox
        bys = [b for (b,a) in best_comp['blocks']]
        bxs = [a for (b,a) in best_comp['blocks']]
        min_bx, max_bx = min(bxs), max(bxs)
        min_by, max_by = min(bys), max(bys)
        x0 = min_bx * bs; y0 = min_by * bs
        x1 = min(w, (max_bx + 1) * bs); y1 = min(h, (max_by + 1) * bs)
        x0 = max(0, x0 - expand_px); y0 = max(0, y0 - expand_px)
        x1 = min(w, x1 + expand_px); y1 = min(h, y1 + expand_px)
        bbox = (x0, y0, x1 - x0, y1 - y0)
        return bbox, comps

    strategies = [
        ('mean_std', lambda: select_by_mean_std(mean_std_k)),
        ('cum_energy', lambda: select_by_cumulative_energy(target_energy_frac)),
        ('percentile', lambda: select_by_percentile(percentile_min))
    ]

    chosen_bbox = None
    chosen_comps = []
    chosen_sel = None

    for name, fn in strategies:
        sel = fn()
        num_selected = sel.sum()
        if num_selected < min_blocks_abs:
            chosen_sel = sel  # keep for debugging if needed
            continue
        if num_selected > int(max_blocks_frac * nx * ny) and name != 'cum_energy':
            chosen_sel = sel
            continue
        # convert to bbox and ensure it meets area constraints
        bbox, comps = sel_to_bbox(sel)
        if bbox is None:
            chosen_sel = sel
            continue
        bw = bbox[2]; bh = bbox[3]
        area_frac = (bw * bh) / float(w * h)
        if area_frac < min_area_frac:
            # too small
            chosen_sel = sel
            continue
        if area_frac > max_area_frac:
            # too large
            chosen_sel = sel
            continue
        chosen_bbox = bbox
        chosen_comps = comps
        chosen_sel = sel
        break

    # If none chosen and relax allowed, run relax steps lowering strictness
    if chosen_bbox is None and relax_if_too_small:
        cur_target = target_energy_frac
        cur_mean_k = mean_std_k
        cur_pct = percentile_min
        for r in range(relax_steps):
            cur_target = min(0.99, cur_target + (1.0 - cur_target) * relax_factor)
            cur_mean_k = max(0.0, cur_mean_k * (1.0 - relax_factor))
            cur_pct = max(50, int(cur_pct * (1.0 - relax_factor)))
            sel = select_by_cumulative_energy(cur_target)
            bbox, comps = sel_to_bbox(sel)
            if bbox is not None:
                bw = bbox[2]; bh = bbox[3]
                area_frac = (bw * bh) / float(w * h)
                if area_frac >= min_area_frac and area_frac <= max_area_frac:
                    chosen_bbox = bbox
                    chosen_comps = comps
                    chosen_sel = sel
                    break

    # image copies
    vis_heat_map = after.copy()
    vis_cropped = after.copy()

    if chosen_bbox is None:
        # no confident detection. produce heat overlay for debugging
        score_px = cv2.resize((score * 255.0).astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR)
        heat = cv2.applyColorMap(score_px, cv2.COLORMAP_JET)
        vis_heat_map = cv2.addWeighted(after, 0.75, heat, 0.25, 0)
        return None, None, vis_heat_map
    
    new_choosen_bbox = tighten_by_strips(before, after,chosen_bbox)
    x,y,ww,hh = new_choosen_bbox
    

    # visualize heat map of the changed areas
    score_px = cv2.resize((score * 255.0).astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR)
    heat = cv2.applyColorMap(score_px, cv2.COLORMAP_JET)
    vis_heat_map = cv2.addWeighted(after, 0.75, heat, 0.25, 0)
    cv2.rectangle(vis_heat_map, (x, y), (x+ww, y+hh), (0,255,0), 2)

    # cropping changed portion 
    cropped_image = vis_cropped[y:y+hh, x:x+ww]

    # saving 
    if len(save_paths) > 0:
        cv2.imwrite(save_paths[0], vis_heat_map)
    if len(save_paths) > 1:
        cv2.imwrite(save_paths[1], cropped_image)
    
    return new_choosen_bbox, cropped_image, vis_heat_map