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
import os

import numpy as np
import torch
from matplotlib import pyplot as plt

from test_v3 import load_gt_points, load_model, load_test_paths, predict
from utils.density import gen_density_map


class Browser:
    def __init__(self, paths, model, device):
        self.paths = paths
        self.model = model
        self.device = device
        self.idx = 0

        self.fig, self.axes = plt.subplots(1, 3, figsize=(16, 5.5))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

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
        _, pred_count, pred_density, img = predict(
            self.model, img_path, self.device, clamp=True
        )  # _ is raw count but i don't use it

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
            self.axes[2].set_title(
                f"Ground Truth Density\n(count: {len(gt_points):.1f})"
            )
        else:
            self.axes[2].set_title("no ground truth found")

        error_str = (
            f" Prediction Error: {pred_count - len(gt_points):+.1f}"
            if gt_points is not None
            else ""
        )
        self.fig.suptitle(f"[{idx + 1}/{len(self.paths)}]  {img_path}  {error_str}")
        self.fig.canvas.draw_idle()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("data_path", help="path to .txt file of image paths")
    parser.add_argument("model_path", help="path to trained .pth model checkpoint file")
    parser.add_argument(
        "--start", type=int, default=0, help="index of image to start browsing from"
    )

    args = parser.parse_args()

    if not os.path.isfile(args.data_path):
        parser.error(f" data file not found: {args.data_path}")
    if not os.path.isfile(args.model_path):
        parser.error(f"model file not found: {args.model_path}")

    # reuse test_v3.py file load
    paths = load_test_paths(args.data_path)
    if not paths:
        parser.error(f"no images found in {args.data_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    print(f"loaded {len(paths)} images from {args.data_path}")
    print("controls: right/n/space = next, left/p = prev, q/esc = quit")

    model = load_model(args.model_path, device)
    start = max(0, min(args.start, len(paths) - 1))
    browser = Browser(paths, model, device)
    browser.idx = start
    browser.show(start)


if __name__ == "__main__":
    main()
