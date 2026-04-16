import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.density import gen_density_map


class FishDataset(Dataset):
    def __init__(
        self, datasets_dir=None, dataset_names=None, max_size=512, samples=None
    ):
        """
        can be constructed with exculsively 2 options (not both)
        1. folder names
        2. samples list
        Args:
            datasets_dir: folder of datasets relative to project root
            dataset_names: names of dataset folders in root_dir
            max_size: used to resize images so longest side is this, default=1024
            samples: list of {"image_path relative to project root":str, "points":list} to use instead of dataset names
        """  # TODO:
        self.max_size = max_size

        if samples is not None:
            # construction method 2: uses list of samples
            self.samples = samples
        else:
            # construction method 1: list of folder names used to create list of sample, points list
            if datasets_dir is None or dataset_names is None:
                raise ValueError(
                    "'samples' or 'datasets_dir' & 'dataset_names' not provided"
                )
                self.samples = []
                for dataset_name in dataset_names:
                    dataset_path = os.path.join(datasets_dir, dataset_name)
                    images_dir = os.path.join(dataset_path, "images")
                    json_path = os.path.join(dataset_path, "annotations.json")

                    with open(json_path, "r") as j:
                        annotations = json.load(j)

                    for image_name, points in annotations.items():
                        if not image_name.endswith(
                            ".jpg"
                        ):  # all images in this are jpg
                            continue
                        image_path = os.path.join(images_dir, image_name)
                        self.samples.append(
                            {"image_path": image_path, "points": points}
                        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        image = cv2.imread(sample["image_path"])  # load image with opencv
        if image is None:
            raise RuntimeError(f"failed to load image {sample['image_path']}")

        # convert loaded image for compatability
        image = cv2.cvtColor(
            image, cv2.COLOR_BGR2RGB
        )  # convert from opencv bgr to standard rgb
        image = (
            image.astype(np.float32) / 255.0
        )  # converts image values from int to float
        # important for numpy and pytorch (and therefore my csrnet)

        # resize image if longest size has more pixels than {max_size}
        H, W, _ = image.shape
        scale = 1.0
        if self.max_size is not None:
            scale = min(self.max_size / W, self.max_size / H, 1.0)
        new_W = int(W * scale)
        new_H = int(H * scale)
        if scale != 1.0:
            image = cv2.resize(image, (new_W, new_H))

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std

        # load and scale points to resized image
        points = sample["points"]
        scaled_points = []
        for x, y in points:
            scaled_points.append((x * scale, y * scale))

        density = gen_density_map((new_H, new_W), scaled_points)
        # downsample by 8 for csrnet
        density = cv2.resize(
            density, (new_W // 8, new_H // 8), interpolation=cv2.INTER_AREA
        )
        density *= 64.0

        # convert to pythorch format
        image = torch.from_numpy(image).permute(2, 0, 1)
        # added dimension channel (required by cnn)
        density = torch.from_numpy(density).unsqueeze(0)
        return image, density
