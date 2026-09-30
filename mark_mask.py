# mark_mask.py
#
# manually mark a dataset's mask; camera is fixed per dataset so this only needs doing once per dataset
#
# usage:
# python mark_mask.py <dataset_name> [--tank] [--frame N]
#   default (pipe mode): left click center, then a point on the edge (one pair per circle, as many circles as needed)
#   --tank: left click points around the edge of the water (not the rim), in order; outline closes itself
#   backspace / right click: undo last click     enter: finish
# saves to tank_masks.json as fractions of the image size, only replacing the part being marked
#   pipe circles: (cx/W, cy/H, r/min(H,W)), same convention as the hough tank circle
#   tank outline: [[x/W, y/H], ...], replaces hough detection for that dataset

import argparse
import json
import math
import os

import cv2
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Polygon

from utils.tank_mask import MANUAL_MASKS_PATH, circle_to_fractional

DATA_ROOT = "datasets"


def draw_saved(ax, entry, H, W):
    """show what's already saved for this dataset in yellow"""
    for cxf, cyf, rf in entry.get("exclude", []):
        ax.add_patch(
            Circle((cxf * W, cyf * H), rf * min(H, W), fill=False, color="yellow", lw=1)
        )
    if entry.get("tank"):
        pts = [(xf * W, yf * H) for xf, yf in entry["tank"]]
        ax.add_patch(Polygon(pts, closed=True, fill=False, color="yellow", lw=1))


def clicks_to_circles(clicks, H, W, ax):
    if len(clicks) % 2:
        print("odd number of clicks, dropping the last one")
        clicks = clicks[:-1]
    circles = []
    for (cx, cy), (ex, ey) in zip(clicks[0::2], clicks[1::2]):
        r = math.hypot(ex - cx, ey - cy)
        circles.append(circle_to_fractional((cx, cy, r), (H, W)))
        ax.add_patch(Circle((cx, cy), r, fill=False, color="red", lw=2))
    for c in circles:
        print("  circle (cx/W, cy/H, r/min):", tuple(round(float(v), 4) for v in c))
    return [[round(float(v), 5) for v in c] for c in circles]


def clicks_to_outline(clicks, H, W, ax):
    ax.add_patch(Polygon(clicks, closed=True, fill=False, color="red", lw=2))
    print(f"  outline with {len(clicks)} points")
    return [[round(x / W, 5), round(y / H, 5)] for x, y in clicks]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--tank", action="store_true", help="trace the tank outline")
    ap.add_argument("--frame", type=int, default=0, help="index into sorted frames")
    args = ap.parse_args()

    images_dir = os.path.join(DATA_ROOT, args.dataset, "images")
    frames = sorted(f for f in os.listdir(images_dir) if f.endswith(".jpg"))
    img_path = os.path.join(images_dir, frames[args.frame])
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]

    data = {}
    if os.path.isfile(MANUAL_MASKS_PATH):
        with open(MANUAL_MASKS_PATH, "r") as f:
            data = json.load(f)
    entry = data.get(args.dataset, {})

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.imshow(img)
    draw_saved(ax, entry, H, W)
    if args.tank:
        how = "click points around the water's edge"
    else:
        how = "click center then edge per circle"
    ax.set_title(
        f"{args.dataset} / {frames[args.frame]}\n"
        f"{how}, enter to finish (yellow = currently saved)"
    )

    clicks = plt.ginput(n=-1, timeout=0, mouse_add=1, mouse_pop=3)
    if len(clicks) < (3 if args.tank else 2):
        print("not enough clicks, nothing saved")
        return

    if args.tank:
        key, value = "tank", clicks_to_outline(clicks, H, W, ax)
    else:
        key, value = "exclude", clicks_to_circles(clicks, H, W, ax)
    fig.canvas.draw()
    plt.pause(0.1)

    if input(f"replace {key} for {args.dataset}? [y/N] ").lower() != "y":
        print("not saved")
        return

    entry[key] = value
    entry[f"{key}_marked_on"] = frames[args.frame]
    entry.pop("marked_on", None)  # older pipe-only format
    data[args.dataset] = entry
    with open(MANUAL_MASKS_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"saved {key} for {args.dataset} -> {MANUAL_MASKS_PATH}")


if __name__ == "__main__":
    main()
