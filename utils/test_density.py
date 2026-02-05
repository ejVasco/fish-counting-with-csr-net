# script to test density.py

import json 
import os
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image 

from utils.density import gen_density_map

# paths
IMG_DIR =  "datasets/GX011278-fish-1-10/images"
ANN_PATH = "datasets/GX011278-fish-1-10/annotations.json"

# selected image 
IMG_NAME="image_0.jpg"

img_path = os.path.join(IMG_DIR, IMG_NAME)
img = Image.open(img_path).convert("RGB")
img_np = np.array(img)

H,W, _ = img_np.shape 

with open(ANN_PATH, "r") as f:
    annotations = json.load(f) 
    
points = annotations[IMG_NAME]

density = gen_density_map((H,W), points, sigma=4)

print(f"number of points: {len(points)}")
print(f"density sum: {density.sum():.4f}")

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.title("og img")
plt.imshow(density, cmap="jet")
plt.colorbar()
plt.axis("off")

plt.show()
