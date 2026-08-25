# utils/test_density.py
# script to test density.py

import json
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from utils.density import gen_density_map

# paths
IMG_DIR = "datasets/yellow_tank_1/images"
ANN_PATH = "datasets/yellow_tank_1/annotations.json"

# selected image, all we need is the dimensions for testing
IMG_NAME = "image_201.jpg"
img_path = os.path.join(IMG_DIR, IMG_NAME)
img = Image.open(img_path).convert("RGB")
img_np = np.array(img)  # convert PIL img -> np array
H, W, _ = img_np.shape

with open(ANN_PATH, "r") as f:
    annotations = json.load(f)  # opens json as pyton dict.
points = annotations[IMG_NAME]

density = gen_density_map((H, W), points, sigma=4)

print(f"number of points: {len(points)}")
print(f"density sum: {density.sum():.4f}")

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.title("og img")
plt.imshow(img_np)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("density map")
plt.imshow(density, cmap="jet")
plt.colorbar()
plt.axis("off")

plt.show()
