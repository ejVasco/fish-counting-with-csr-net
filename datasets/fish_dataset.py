import os 
import json
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

from utils.density import gen_density_map  

class FishDataset(Dataset): #custom dataset inherited from base "Dataset" from torch.utils
    def __init__(self, root_dir, dataset_names):
        """
        Args:
            root_dir: root folder
            dataset_names: list of dataset folder identifiers
        """
        self.samples = [] # list for storing dictionaries of image paths and points

        for dataset_name in dataset_names:
            # create paths
            dataset_path = os.path.join(root_dir, dataset_name)
            images_dir = os.path.join(dataset_path, "images")
            ann_path = os.path.join(dataset_path, "annotations.json")

            with open(ann_path, "r") as f:
                annotations = json.load(f) # open annotations as python dict.

            for image_name, points in annotations.items():
                # ignore non image files
                if not image_name.endswith(".jpg"):
                    continue 

                # add each image and points to samples dict.
                image_path = os.path.join(images_dir, image_name) 
                self.samples.append({
                    "image_path": image_path,
                    "points": points 
                })

    def __len__(self):
        return len(self.samples)
    
    # returns 1 sample image and its density map
    def __getitem__(self, index):
        sample = self.samples[index]

        # loading image
        image = cv2.imread(sample["image_path"])
        if image is None:
            raise RuntimeError(f"failed to load image: {sample["image_path"]}") 

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)/255.0 # converts from ints to floats (0-255)->(0.0-1.0)

        H, W, _ = image.shape

        # generate density map
        points = sample["points"]
        density = gen_density_map((H, W), points)

        # csrnet downsamples by 8 
        density = cv2.resize(
            density,
            (W // 8, H // 8),
            interpolation=cv2.INTER_CUBIC
        )

        # adjust so density sum reains the same after downsamples
        density *= (H*W) / ((H // 8) * (W // 8))

        image = torch.from_numpy(image).permute(2,0,1) # convert to pytorch format (H,W,C)->(C,H,W)
        density = torch.from_numpy(density).unsqueeze(0) # add a channel dimension (reuired by cnn)

        return image, density

