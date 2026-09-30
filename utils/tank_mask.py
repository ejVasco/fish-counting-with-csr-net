# utils/tank_mask.py
# detect circular tank rim and builds mask to restrict density predictions and GT to tank interior


import json
import os

import cv2
import numpy as np

# manually marked masks per dataset (written by mark_mask.py):
#   "tank": traced outline of the water [[x/W, y/H], ...], overrides hough detection
#   "exclude": circles (e.g. center pipe) to cut out of the tank mask
MANUAL_MASKS_PATH = "tank_masks.json"


def detect_tank_circle(img_bgr, min_frac=0.25, max_frac=0.55):
    """
    find tank outer rim via hough circle transform

    args
        img_bgr: image loaded by cv2.imread
        min_frac/max_frac: expected tank radious as fraction of image min(image width, image height) may have to tune later
    """

    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 9)

    min_r = int(min_frac * min(H, W))
    max_r = int(max_frac * min(H, W))

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=max(H, W),
        param1=80,
        param2=60,
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is None:
        return None

    x, y, r = circles[0, 0]
    return float(x), float(y), float(r)


def circle_to_fractional(circle, shape):
    """
    convert a pixel space circle to be resolution independent
    """
    H, W = shape
    cx, cy, cr = circle
    return cx / W, cy / H, cr / min(H, W)


def fractional_to_mask(frac_circle, shape, shrink=0.97):
    """
    build binary float mask (1 inside tank, 0 outside)
    shrink arg pulls the mask in to avoid including the rim
    """
    H, W = shape
    if frac_circle is None:
        return np.zeros((H, W), dtype=np.float32)

    mask = np.zeros((H, W), dtype=np.float32)
    cxf, cyf, rf = frac_circle
    cx, cy, r = cxf * W, cyf * H, rf * min(H, W)
    cv2.circle(mask, (int(cx), int(cy)), int(r * shrink), 1.0, -1)
    return mask


def load_manual_masks(path=MANUAL_MASKS_PATH):
    """
    returns {dataset_name: {"tank": [(xf, yf), ...] or None, "exclude": [frac_circle, ...]}}
    from the manual mask file, or {} if the file doesn't exist yet
    """
    if not os.path.isfile(path):
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    manual = {}
    for name, entry in data.items():
        tank = entry.get("tank")
        manual[name] = {
            "tank": [tuple(p) for p in tank] if tank else None,
            "exclude": [tuple(c) for c in entry.get("exclude", [])],
        }
    return manual


def polygon_to_mask(frac_points, shape, upsample=16):
    """
    rasterize traced tank outline (fractions of W, H) into a float mask
    edge cells get the fraction of their area inside the outline (drawn at {upsample}x, then area-averaged)
    """
    H, W = shape
    hi = np.zeros((H * upsample, W * upsample), dtype=np.float32)
    pts = np.array(
        [[xf * W * upsample, yf * H * upsample] for xf, yf in frac_points],
        dtype=np.int32,
    )
    cv2.fillPoly(hi, [pts], 1.0)
    return cv2.resize(hi, (W, H), interpolation=cv2.INTER_AREA)


def build_mask(frac_circle, shape, manual=None):
    """
    final mask for one image: traced outline if the dataset has one, else the hough circle,
    minus any excluded circles. manual is one entry from load_manual_masks (or None)
    """
    manual = manual or {}
    if manual.get("tank"):
        mask = polygon_to_mask(manual["tank"], shape)
    else:
        mask = fractional_to_mask(frac_circle, shape)
    return apply_exclusions(mask, manual.get("exclude"))


def apply_exclusions(mask, exclusions, upsample=16):
    """
    cut each fractional circle (cx/W, cy/H, r/min(H,W)) out of the mask
    at 1/8 res the pipe is only ~1-4 cells wide, so whole-cell rasterizing badly oversizes it;
    instead each cell is scaled by the fraction of it outside the circles (drawn at {upsample}x, then area-averaged)
    """
    if not exclusions:
        return mask
    H, W = mask.shape
    hi = np.zeros((H * upsample, W * upsample), dtype=np.float32)
    for cxf, cyf, rf in exclusions:
        cx, cy, r = cxf * W, cyf * H, rf * min(H, W)
        cv2.circle(
            hi,
            (int(round(cx * upsample)), int(round(cy * upsample))),
            int(round(r * upsample)),
            1.0,
            -1,
        )
    covered = cv2.resize(hi, (W, H), interpolation=cv2.INTER_AREA)
    return mask * (1.0 - covered)


def compute_dataset_circle(images_dir, filenames, sample_n=5):
    """
    averages detected circle as fractions over sample frames
    CALLED once per dataset since frames are a fixed angle
    """
    fracs = []
    for f in filenames[:sample_n]:
        img = cv2.imread(os.path.join(images_dir, f))
        if img is None:
            continue
        c = detect_tank_circle(img)
        if c is not None:
            fracs.append(circle_to_fractional(c, img.shape[:2]))

    if not fracs:
        return None
    return tuple(np.mean(fracs, axis=0))
