from datasets.fish_dataset import FishDataset

TRAIN_DATASETS = [
    "GX011278-fish-1-10",
]

dataset = FishDataset("datasets", TRAIN_DATASETS)

img, den = dataset[0]

print(img.shape)
print(den.shape)
print(den.sum())
