import os 
import json
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

from utils.density import gen_density_map  

class FishDataset(Dataset):
    def __init__(self, root_dir, dataset_names):
        """_summary_

        Args:
            root_dir (_type_): _description_
            dataset_names (_type_): _description_
        """
        self.samples = []

        for dataset_name in dataset_names:
            # create paths based on root path
            dataset_path = os.path.join(root_dir, dataset_name)
            images_dir = os.path.join(dataset_path, "images")
            ann_path = os.path.join(dataset_path, "annotations.json")

            with open(ann_path, "r") as f:
                annotations = json.load(f)

            for image_name, points in annotations.items():
                
                # ignore non image files
                if not image_name.endswith(".jpg"):
                    continue 

                image_path = os.path.join(images_dir, image_name)

                self.samples.append({
                    "image_path": image_path,
                    "points": points 
                })

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample = self.samples[index]

        # loading image
        image = cv2.imread(sample["image_path"])
        if image is None:
            raise RuntimeError(f"faild load image") # TODO: make this output the specific image

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)/255.0

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

        density *= (H*W) / ((H // 8) * (W // 8))

        image = torch.from_numpy(image).permute(2,0,1)
        density = torch.from_numpy(density).unsqueeze(0)

        return image, density

