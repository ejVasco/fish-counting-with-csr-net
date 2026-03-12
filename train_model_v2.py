import json
import os
import random

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from models.csrnet import CSRNet

DATA_ROOT = "datasets"  # path relative to project home dir
ALL_DATASETS = [
    "fake_test_dataset",
    "10_fish",
    "blue_tank_2",
    "GX011278-fish-1-10",
    "GX011279-fish-11-20",
    "GX011284-fish-41-50",
    "yellow_tank_1",
    "yellow_tank_2",
]

# how the dataset is split, adds up to 1.0
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# training config
# RNG_SEED = 42 # use to reproduce training with same seed
BATCH_SIZE = 4
NUM_EPOCHS = 50
LEARNING_RATE = 1e-5
CHECKPOINT_DIR = "checkpoints"  # path relatie to project home dir
TEST_FILES_OUT = "test_data.txt"  # outputs files to input into test_model.py


def gather_samples(
    dataset_root: str, dataset_names: list[str]
) -> dict[str, list[dict]]:
    """
    Reads annotations.json for each dataset and returns a dictionary of:
       dataset_name -> list of dictionaries
    which mimics the logic in Fish Dataset
    """
    gathered: dict[str, list[dict]] = {}
    for name in dataset_names:
        dataset_path = os.path.join(dataset_root, name)
        images_dir = os.path.join(dataset_path, "images")
        json_path = os.path.join(dataset_path, "annotations.json")

        print(f"  gathering samples from dataset: {dataset_path}\n")

        if not os.path.exists(json_path):
            print(
                f"    skipping dataset: {dataset_path} . due to missing path: {json_path}\n"
            )
            continue
        with open(json_path, "r") as f:
            annotations = json.load(f)

        samples = []
        for img_name, points in annotations.items():
            if img_name.endswith(".jpg"):
                samples.append(
                    {"image_path": os.path.join(images_dir, img_name), "points": points}
                )
        if not samples:
            print(f"    skipping dataset: {dataset_path}, due to no .jpg entries found\n")
            continue

        gathered[name] = samples
    return gathered


def split_samples(gathered_samples: dict[str,list[dict]],train_ratio:float, val_ratio: float):



def save_test_paths(
    test_samples: list[dict], out_path: str
):  # test_samples is a list of dictionaries because its a taken from database jsons

    # convert to list of paths
    paths = []
    for s in test_samples:
        paths.append(s["image_path"])

    with open(out_path, "w") as f:
        f.write("\n".join(paths) + "\n")

    print(f" list of test files saved to {out_path} with {len(paths)} files\n")


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    if TRAIN_RATIO + VAL_RATIO + TEST_RATIO != 1.0:
        print(f"invalid split ratios. train: {TRAIN_RATIO}, val: {VAL_RATIO}, test: {TEST_RATIO}")
        return

    # gather samples
    print("gathering samples from datasets\n")
    gathered_samples = gather_samples(DATA_ROOT, ALL_DATASETS)
    total = sum(len(v) for v in gathered_samples.values())
    print(f" total gathered samples: {total}\n")

    # split samples
    print("splitting gathered samples\n")


if __name__ == "__main__":
    main()
