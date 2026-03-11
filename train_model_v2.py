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


def main():
    pass


if __name__ == "__main__":
    main()
