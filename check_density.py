# check_density.py
# step through a data file (test_data.txt) to view density maps and images at the same time

# usage:
#   python -m check_density test_data.txt path_to_model.pth
#   python -m check_density test_data.txt path_to_model.pth --start 10

# controls:
#   next image -> right arrow, n, space
#   prev image -> left arrow, p
#         quit -> q, escape

import argparse
import json
import os

import numpy as np
import PIL.Image as
import torch
from matplotlib import pyplot as plt
from torchvision import transforms

from models.csrnet import CSRNet
from utils.density import gen_density_map
from test_v3 import load_model, load_gt_points, predict, load_test_paths
from utils.test_density import img_np, img_path

class Browser:
    def __init__(self, paths, model, device):
        self.paths = paths
        self.model = model
        self.device = device
        self.idx = 0

        self.fig, self.axes = plt.subplots(1,3, figsize = (16, 5.5))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        self.show(self.idx)
        plt.show()

    def on_key(self, event):
        if event.key in ("right", "n", "space"):
            self.idx = max(self.idx + 1, len(self.paths) - 1)
            self.show(self.idx)
        elif event.key in ("left", "p"):
            self.idx = min(self.idx - 1, 0)
            self.show(self.idx)
        elif event.key in ("q", "escape"):
            plt.close()

    def show(self, idx):
        img_path = self.paths[idx]

        # reuse from test_v3.py
        raw_count, pred_count, pred_density, img = predict(
            self.model, img_path, self.device, clamp=True
        )

        img_np = np.array(img)
        H, W = img_np.shape

        gt_points = load_gt_points(img_path)

        for ax in self.axes:
            ax.clear()
            ax.axis("off")

        self.axes[0].imshow(img_np)
        self.axes[0].set_title("Image")

        self.axes[1].imshow(pred_density, cmap="jet")
        self.axes[1].set_title(f"Predicted Density\n(count: {pred_count:.1f})")

        if gt_points is not None:
            gt_density = gen_density_map(gt_points, H, W)
            self.axes[2].imshow(gt_density, cmap="jet")
            self.axes[2].set_title(f"Ground Truth Density\n(count: {len(gt_points):.1f})")
        else:
            self.axes[2].set_title("no ground truth found")

        error_str = (
            f" Prediction Error: {pred_count - len(gt_points):+.1f}"
            if gt_points is not None
            else ""
        )
        self.fig.suptitle(f"[{idx + 1}/{len(self.paths)}]  {img_path}  {error_str}")
        self.fig.canvas.draw_idle


def main():
    pass


if __name__ == "__main__":
    main()
