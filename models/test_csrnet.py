import torch

from datasets.fish_dataset import FishDataset
from models.csrnet import CSRNet

# load datasets
TRAIN_DATASETS = ["GX011278-fish-1-10"]
dataset = FishDataset("datasets", TRAIN_DATASETS)

# use single sample
img, gt_density = dataset[0]
img = img.unsqueeze(0)

# init model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CSRNet(load_pretrained=False).to(device)
img = img.to(device)
gt_density = gt_density.to(device)

# forward
with torch.no_grad():
    pred_density = model(img)

print(f"input img shape: {img.shape}")
print(f"gt density shape: {gt_density.shape}")
print(f"predicted density shape:{pred_density.shape}")
