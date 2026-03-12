import json
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

# from PIL import Image
from torch.utils.data import DataLoader  # , Dataset

# from torchvision import transforms
from datasets.fish_dataset_v2 import FishDataset
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
# TEST_RATIO = 1.0 - TRAIN_RATIO - VAL_RATIO during code execution

# training config
RNG_SEED = None  # use to reproduce splits for training
BATCH_SIZE = 4
NUM_EPOCHS = 50
LEARNING_RATE = 1e-5
CHECKPOINT_DIR = "checkpoints"  # path relatie to project home dir
TEST_FILES_OUT = "test_data.txt"  # outputs files to input into test_model.py


def pad_collate(batch):
    """
    make imgs in a batch uniform dimensions via padding as required by torch nn
    (and density data)
    """
    imgs, densities = zip(*batch)

    # get max H and max W in this batch
    max_H = max(img.shape[1] for img in imgs)
    max_W = max(img.shape[2] for img in imgs)

    padded_imgs = []
    padded_densities = []
    for img, density in zip(imgs, densities):
        # image has [channels, height, width]
        pad_H = max_H - img.shape[1]
        pad_W = max_W - img.shape[2]
        padded_imgs.append(F.pad(img, (0, pad_W, 0, pad_H)))

        # density has [1, height//8, width//8]
        den_pad_H = (max_H // 8) - density.shape[1]
        den_pad_W = (max_W // 8) - density.shape[2]
        padded_densities.append(F.pad(density, (0, den_pad_W, 0, den_pad_H)))

    return torch.stack(padded_imgs), torch.stack(padded_densities)


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

        # print(f"  gathering samples from dataset: {dataset_path}\n")

        if not os.path.exists(json_path):
            print(
                f"    skipping dataset: {dataset_path} . due to missing path: {json_path}"
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
            print(f"    skipping dataset: {dataset_path}, due to no .jpg entries found")
            continue

        gathered[name] = samples
    return gathered


def split_samples(
    gathered_samples: dict[str, list[dict]],
    train_ratio: float,
    val_ratio: float,
    seed=None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Each dataset gets split and included in training, validation, and testing.
    Optional seed to use consistent splits
    """
    train_samples, val_samples, test_samples = [], [], []
    rng = random.Random(seed)

    test_ratio = 1.0 - train_ratio - val_ratio
    if train_ratio <= 0.0 or val_ratio <= 0.0 or test_ratio <= 0.0:
        # print(f"invalid train/val/ratio split: {train_ratio}:{val_ratio}:{test_ratio}")
        raise ValueError(
            f"invalid train/val/test split: {train_ratio}/{val_ratio}/{test_ratio}"
        )

    for name, samples in gathered_samples.items():
        # print(f"  splitting {name}\n")
        shuffled = samples[:]  # copy samples instead of shufflying samples just in case
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        # ensure every split has at least 1 sample
        n_train = min(n_train, n - 2)
        n_val = min(n_val, n - 1)
        n_test = n - n_train - n_val

        train_samples.extend(shuffled[:n_train])
        val_samples.extend(shuffled[n_train : n_train + n_val])
        test_samples.extend(shuffled[n_train + n_val :])

        print(
            f"    {name} has {n} images -> training:{n_train}, val:{n_val}, test:{n_test}"
        )

    return train_samples, val_samples, test_samples


def save_test_paths(
    test_samples: list[dict], out_path: str
):  # test_samples is a list of dictionaries because its a taken from database jsons

    # convert to list of paths
    paths = []
    for s in test_samples:
        paths.append(s["image_path"])

    with open(out_path, "w") as f:
        f.write("\n".join(paths) + "\n")

    print(f" list of test files saved to {out_path} with {len(paths)} files")


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # gather samples
    print("gathering samples from datasets")
    gathered_samples = gather_samples(DATA_ROOT, ALL_DATASETS)
    total = sum(len(v) for v in gathered_samples.values())
    print(f" total gathered samples: {total}")

    # split samples
    print("splitting gathered samples")
    train_samples, val_samples, test_samples = split_samples(
        gathered_samples, TRAIN_RATIO, VAL_RATIO, seed=RNG_SEED
    )

    # save test samples to txt to use for testing later:
    save_test_paths(test_samples, TEST_FILES_OUT)

    # from samples -> datasets
    train_dataset = FishDataset(samples=train_samples)
    val_dataset = FishDataset(samples=val_samples)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=pad_collate,
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = CSRNet(load_pretrained=True).to(device)

    criterion = nn.MSELoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        # train
        model.train()
        train_loss = 0.0

        for imgs, gt_density in train_loader:
            imgs = imgs.to(device)
            gt_density = gt_density.to(device)

            pred_density = model(imgs)
            loss = criterion(pred_density, gt_density)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        train_loss /= len(train_loader)
        print(f"[Epoch {epoch + 1}/{NUM_EPOCHS}] train loss: {train_loss:.4f}")

        # validate
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for imgs, gt_density in val_loader:
                imgs = imgs.to(device)
                gt_density = gt_density.to(device)
                val_loss += criterion(model(imgs), gt_density).item()

        val_loss /= len(val_loader)
        print(f"  val loss {val_loss:.4f}")

        # checkpoint every epoch
        chkpnt_path = os.path.join(CHECKPOINT_DIR, f"csrnet_epoch{epoch + 1}.pth")
        torch.save(model.state_dict(), chkpnt_path)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(
                f"    - new best model at epoch {epoch + 1} with val loss {best_val_loss:.4f}"
            )


if __name__ == "__main__":
    main()
