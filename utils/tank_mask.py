# utils/tank_mask.py
# detect circular tank rim and builds mask to restrict density predictions and GT to tank interior


import os

import cv2
import numpy as np
from sympy.printing.codeprinter import cxxcode

from utils.test_density import img


def detect_tank_circle(img_bgr, min_frac=0.25, max_frac=0.55):
    """
    find tank outer rim via hough circle transform

    args
        img_bgr: image loaded by cv2.imread
        min_frac/max_frac: expected tank radious as fraction of image min(image width, image height) may have to tune later
    """

    H, W, _ = img_bgr.shape[:2]
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
