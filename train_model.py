import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np


# training
def train_model(dataset_path, 
                epochs=10, 
                batch_size=4, 
                lr=1e-5, 
                device="cuda" if torch.cuda.is_available() else "cpu"):
    pass

if __name__ == "__main__":
    dataset_root = os.path.join("frames")
    train_model(dataset_root)