import os
import math
import cv2
import numpy as np
from typing import List, Tuple, Dict

# -----------------------------
# Utils: IO + silhouette
# -----------------------------

def read_image(path: str) -> np.ndarray:
    """Read image as BGR uint8."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    return img

def binarize_silhouette(img_bgr: np.ndarray, bg_is_white: bool = True, thr: int = 245) -> np.ndarray:
    """
    Convert to binary silhouette (uint8 0/1).
    Assumes background is near-white if bg_is_white=True.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if bg_is_white:
        # foreground are darker pixels
        mask = (gray < thr).astype(np.uint8)
    else:
        # foreground are brighter pixels
        mask = (gray > thr).astype(np.uint8)

    return mask

def largest_connected_component(bin01: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component in a binary (0/1) image."""
    bin255 = (bin01 * 255).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin255, connectivity=8)
    if num <= 1:
        return bin01.copy()

    # stats: [label, x, y, w, h, area] ; label 0 is background
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = 1 + int(np.argmax(areas))
    out = (labels == largest_idx).astype(np.uint8)
    return out

def fill_holes(bin01: np.ndarray) -> np.ndarray:
    """Fill holes in a binary silhouette."""
    bin255 = (bin01 * 255).astype(np.uint8)
    h, w = bin255.shape
    flood = bin255.copy()
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(flood, mask, seedPoint=(0, 0), newVal=255)
    inv = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(bin255, inv)
    return (filled > 0).astype(np.uint8)

def remove_ground_plane(bin01: np.ndarray, density_thr: float = 0.80) -> np.ndarray:
    """
    Remove ground-plane rows from a binary silhouette.
    Scans from the bottom upward; rows whose foreground density exceeds
    density_thr are treated as ground and zeroed out.  Stops as soon as
    a row drops below the threshold (i.e. only removes a contiguous
    bottom band, not isolated dense rows higher up).
    """
    out = bin01.copy()
    W = bin01.shape[1]
    for row in range(bin01.shape[0] - 1, -1, -1):
        if out[row].sum() / max(1, W) >= density_thr:
            out[row] = 0
        else:
            break
    return out

def preprocess_silhouette(img_bgr: np.ndarray, bg_is_white: bool = True,
                          thr: int = 245, remove_ground: bool = False) -> np.ndarray:
    """Binarize + keep largest CC + fill holes + optional ground removal."""
    s = binarize_silhouette(img_bgr, bg_is_white=bg_is_white, thr=thr)
    s = largest_connected_component(s)
    s = fill_holes(s)
    if remove_ground:
        s = remove_ground_plane(s)
    return s

def area(bin01: np.ndarray) -> int:
    return int(bin01.sum())

# -----------------------------
# Core centroid + alignment
# -----------------------------

def erode_binary(bin01: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return bin01.copy()
    k = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    out = cv2.erode((bin01*255).astype(np.uint8), kernel, iterations=1)
    return (out > 0).astype(np.uint8)

def centroid_xy(bin01: np.ndarray) -> Tuple[float, float]:
    """
    Centroid in (x, y). If empty, return center (0,0) as sentinel.
    """
    ys, xs = np.nonzero(bin01)
    if len(xs) == 0:
        return (0.0, 0.0)
    return (float(xs.mean()), float(ys.mean()))

def scale_binary(bin01: np.ndarray, scale: float) -> np.ndarray:
    """Scale binary mask with nearest-neighbor."""
    h, w = bin01.shape
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize((bin01*255).astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return (resized > 0).astype(np.uint8)

def place_centered(bin01: np.ndarray, out_hw: Tuple[int, int], anchor_xy: Tuple[float, float]) -> np.ndarray:
    """
    Place bin01 onto an output canvas so that anchor_xy (in source coords)
    lands at canvas center.
    """
    out_h, out_w = out_hw
    src_h, src_w = bin01.shape
    canvas = np.zeros((out_h, out_w), dtype=np.uint8)

    cx = out_w / 2.0
    cy = out_h / 2.0
    ax, ay = anchor_xy

    # top-left position in canvas for src (float), then round to int
    x0 = int(round(cx - ax))
    y0 = int(round(cy - ay))

    # compute paste bounds
    x1 = max(0, x0)
    y1 = max(0, y0)
    x2 = min(out_w, x0 + src_w)
    y2 = min(out_h, y0 + src_h)

    if x2 <= x1 or y2 <= y1:
        return canvas

    src_x1 = x1 - x0
    src_y1 = y1 - y0
    src_x2 = src_x1 + (x2 - x1)
    src_y2 = src_y1 + (y2 - y1)

    canvas[y1:y2, x1:x2] = bin01[src_y1:src_y2, src_x1:src_x2]
    return canvas

def center_align_set(
    gen_sils: List[np.ndarray],
    out_hw: Tuple[int, int] = (512, 512),
    core_radius_frac: float = 0.02,
) -> List[np.ndarray]:
    """
    Center-align each silhouette onto an out_hw canvas using core centroid.
    No scaling applied — silhouettes are placed at original size.
    """
    assert len(gen_sils) == 4

    # core erosion radius (in pixels) based on output height
    r = int(round(out_hw[0] * core_radius_frac))

    aligned = []
    for s in gen_sils:
        core = erode_binary(s, r)
        ax, ay = centroid_xy(core)
        if ax == 0.0 and ay == 0.0:
            ax, ay = centroid_xy(s)

        aligned.append(place_centered(s, out_hw, (ax, ay)))

    return aligned

# -----------------------------
# Mirror consistency metrics
# -----------------------------

def mirror_h(bin01: np.ndarray) -> np.ndarray:
    return np.fliplr(bin01)

def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a > 0, b > 0).sum()
    union = np.logical_or(a > 0, b > 0).sum()
    return float(inter) / float(max(1, union))

def scanline_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Mean absolute difference between row-wise foreground counts, normalized by width.
    Lower is better.
    """
    a_proj = a.sum(axis=1).astype(np.float32)
    b_proj = b.sum(axis=1).astype(np.float32)
    mad = np.mean(np.abs(a_proj - b_proj))
    w = a.shape[1]
    return float(mad / max(1.0, w))

def evaluate_pairs(normed: List[np.ndarray]) -> Dict[str, Dict[str, float]]:
    """
    normed order: [front, right, back, left]
    Returns metrics for LR and FB mirror checks.
    """
    F, R, B, L = normed
    metrics = {}

    # Right vs Left: flip right, compare with left
    Rm = mirror_h(R)
    metrics["LR"] = {
        "iou_mirror": iou(Rm, L),
        "scan_mirror": scanline_distance(Rm, L),
    }

    # Front vs Back: flip front, compare with back
    Fm = mirror_h(F)
    metrics["FB"] = {
        "iou_mirror": iou(Fm, B),
        "scan_mirror": scanline_distance(Fm, B),
    }

    return metrics

def pass_fail(metrics: Dict[str, Dict[str, float]],
              iou_thr_lr: float = 0.80,
              scan_thr_lr: float = 0.06,
              iou_thr_fb: float = 0.80,
              scan_thr_fb: float = 0.07) -> Dict[str, bool]:
    """
    Hard gate: require both metrics to be good enough (you可以改成更保守的逻辑)
    """
    lr_ok = (metrics["LR"]["iou_mirror"] >= iou_thr_lr) and (metrics["LR"]["scan_mirror"] <= scan_thr_lr)
    fb_ok = (metrics["FB"]["iou_mirror"] >= iou_thr_fb) and (metrics["FB"]["scan_mirror"] <= scan_thr_fb)
    return {"LR_ok": lr_ok, "FB_ok": fb_ok, "all_ok": lr_ok and fb_ok}

# -----------------------------
# Example main
# -----------------------------

def run_pipeline(
    gen_paths: List[str],
    out_hw=(512, 512),
    thr_gen=245,
    core_radius_frac=0.02,
    remove_ground: bool = False,
):
    """
    gen_paths must be a list of 4 paths in order: [front, right, back, left]
    Set remove_ground=True for models that have a visible ground plane.
    """
    gen_imgs = [read_image(p) for p in gen_paths]

    gen_sils = [preprocess_silhouette(im, bg_is_white=True, thr=thr_gen,
                                      remove_ground=remove_ground) for im in gen_imgs]

    normed = center_align_set(gen_sils, out_hw=out_hw, core_radius_frac=core_radius_frac)

    metrics = evaluate_pairs(normed)
    decision = pass_fail(metrics)

    return {
        "metrics": metrics,
        "decision": decision,
        "normed_silhouettes": normed,
    }

# -----------------------------
# Composite splitting
# -----------------------------

def split_composite_to_tmp(composite_path: str, tmp_dir: str, prefix: str,
                           target_hw: Tuple[int, int] = (512, 512)) -> List[str]:
    """
    Split a 2×2 composite image into 4 temporary files.
    Layout (matches Unity 2flipped output):
        TL = front,  TR = right
        BL = back,   BR = left
    Each tile is resized to target_hw so that gen and mask tiles share the
    same resolution before area-based scale computation.
    Returns paths in order [front, right, back, left].
    """
    img = cv2.imread(composite_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read composite: {composite_path}")
    H, W = img.shape[:2]
    h, w = H // 2, W // 2

    tiles = {
        "front": img[:h, :w],
        "right": img[:h, w:],
        "back":  img[h:, :w],
        "left":  img[h:, w:],
    }

    th, tw = target_hw
    os.makedirs(tmp_dir, exist_ok=True)
    paths = []
    for view in ["front", "right", "back", "left"]:
        tile = cv2.resize(tiles[view], (tw, th), interpolation=cv2.INTER_AREA)
        out_path = os.path.join(tmp_dir, f"{prefix}_{view}.png")
        cv2.imwrite(out_path, tile)
        paths.append(out_path)
    return paths


# -----------------------------
# Batch main
# -----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_MAP = {
    "bladeLego":    "BladeLego",
    "cactus":       "cactus",
    "church":       "church",
    "defenseTower": "tower",
    "iceCreamShop": "icecreamShop",
    "standingApple":"StandingApple",
}

# Models whose renders include a visible ground plane that should be removed
# before silhouette comparison.
HAS_GROUND = {"church", "defenseTower"}

if __name__ == "__main__":
    import shutil

    tmp_root = os.path.join(BASE_DIR, "_tmp_tiles")

    print(f"{'Model':<18} {'V':>2}  "
          f"{'LR_iou':>7} {'LR_scan':>8}  "
          f"{'FB_iou':>7} {'FB_scan':>8}  {'pass':>6}")
    print("-" * 68)

    for model_key, gt_name in MODEL_MAP.items():
        gen_dir   = os.path.join(BASE_DIR, model_key, "unityresults", "2flipped")
        has_ground = model_key in HAS_GROUND

        for n in range(1, 6):
            gen_composite = os.path.join(gen_dir, f"{n}.png")
            if not os.path.isfile(gen_composite):
                continue

            gen_paths = split_composite_to_tmp(gen_composite,
                                               os.path.join(tmp_root, model_key),
                                               f"gen{n}")

            try:
                out = run_pipeline(gen_paths, remove_ground=has_ground)
                lr  = out["metrics"]["LR"]
                fb  = out["metrics"]["FB"]
                dec = out["decision"]
                status = ("PASS"     if dec["all_ok"]
                          else "ALL-fail" if not dec["LR_ok"] and not dec["FB_ok"]
                          else "LR-fail"  if not dec["LR_ok"]
                          else "FB-fail")
                print(f"{model_key:<18} {n:>2}  "
                      f"{lr['iou_mirror']:>7.3f} {lr['scan_mirror']:>8.4f}  "
                      f"{fb['iou_mirror']:>7.3f} {fb['scan_mirror']:>8.4f}  {status:>6}")
            except Exception as e:
                print(f"{model_key:<18} {n:>2}  ERROR: {e}")

    # Clean up temp tiles
    if os.path.isdir(tmp_root):
        shutil.rmtree(tmp_root, ignore_errors=True)
