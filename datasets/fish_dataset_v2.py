# datasets/fish_dataset_v2.py
import json
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.density import gen_density_map


def _add_shadow(image):
    H, W, _ = image.shape

    # pick random x positions for shadow "stripe"
    x1 = random.randint(0, W)
    x2 = random.randint(0, W)

    # build polygon cut diagonally across img
    pts = np.array(
        [
            [min(x1, x2), 0],
            [max(x1, x2), 0],
            [max(x1, x2) + random.randint(-W // 3, W // 3), H],
            [max(x1, x2) + random.randint(-W // 3, W // 3), H],
        ],
        dtype=np.int32,
    )

    mask = np.zeros((H, W), dtype=np.float32)
    cv2.fillPoly(mask, [pts], 1.0)

    shadow_strength = random.uniform(0.3, 0.7)  # darkness value
    image = image * (1 - mask[:, :, np.newaxis] * shadow_strength)
    return np.clip(image, 0, 1).astype(np.float32)


def _augment(image):
    """
    augmentations to use on float32 rgb image in [0,1] range
    density map isn't affgected, only pixel values on the image
    """
    # -- brightness / contrast
    if random.random() > 0.5:
        brightness = random.uniform(-0.3, 0.3)
        image = np.clip(image + brightness, 0, 1)

    if random.random() > 0.5:
        contrast = random.uniform(0.6, 1.4)
        mean = image.mean()
        image = np.clip((image - mean) * contrast + mean, 0, 1)

    # -- gamma
    if random.random() > 0.5:
        gamma = random.uniform(0.6, 1.4)
        image = np.power(image, gamma).astype(np.float32)

    # -- synth shadows
    # random dark polygons attempt to "teach " model that dark spot != fish
    if random.random() > 0.4:
        image = _add_shadow(image)

    return image


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

        # augment to attempt compensation for shadows (via lighting changes and shadows changes)
        image = _augment(image)

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std

        # load and scale points to resized image
        points = sample["points"]
        scaled_points = []
        for x, y in points:
            scaled_points.append((x * scale, y * scale))

        density = gen_density_map((new_H, new_W), scaled_points)

        # downsample gt to match csrnet output
        density = cv2.resize(
            density, (new_W // 8, new_H // 8), interpolation=cv2.INTER_AREA
        )
        density = density * (8 * 8)

        # random flipping horizontally
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
            density = np.fliplr(density).copy()
        # if random.random() > 0.5:
        #     image = np.flipud(image).copy()
        # density = np.flipud(density).copy()
        # leave as copies to avoid numpy "stride issues" ?

        # convert to torch tensor
        density = torch.from_numpy(density).unsqueeze(0).float()

        # convert image to torch format
        image = torch.from_numpy(image).permute(2, 0, 1)

        return image, density
