from datasets.fish_dataset import FishDataset

# list of folders to turn into datasets
TRAIN_DATASETS = [
    "GX011278-fish-1-10",
]

# initialize FishDataset class
dataset = FishDataset("datasets", TRAIN_DATASETS)

# fetch first dataset item
img, den = dataset[0]

# prints image tensor shape
print(img.shape)
# prints density map tensor shape and sum
print(den.shape)
print(den.sum())
