# train_v3.py
import json
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.fish_dataset_v2 import FishDataset
from models.csrnet import CSRNet

DATA_ROOT = "datasets"

ALL_DATASETS = [
    "fake_test_dataset",
    "10_fish",
    "blue_tank_2",
    "GX011278-fish-1-10",
    "GX011279-fish-11-20",
    "GX011284-fish-41-50",
    "yellow_tank_1",
    "yellow_tank_2",
    "calibration_0fish",
]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

RNG_SEED = 71
BATCH_SIZE = 4
NUM_EPOCHS = 25
CHECKPOINT_DIR = "checkpoints"
TEST_FILES_OUT = "test_data.txt"


def pad_collate(batch):
    imgs, densities = zip(*batch)

    max_H = max(img.shape[1] for img in imgs)
    max_W = max(img.shape[2] for img in imgs)

    padded_imgs = []
    padded_densities = []

    for img, density in zip(imgs, densities):
        pad_H = max_H - img.shape[1]
        pad_W = max_W - img.shape[2]
        padded_imgs.append(F.pad(img, (0, pad_W, 0, pad_H)))

        den_pad_H = (max_H // 8) - density.shape[1]
        den_pad_W = (max_W // 8) - density.shape[2]
        padded_densities.append(F.pad(density, (0, den_pad_W, 0, den_pad_H)))

    return torch.stack(padded_imgs), torch.stack(padded_densities)


def gather_samples(dataset_root: str, dataset_names: list[str]):
    gathered = {}

    for name in dataset_names:
        dataset_path = os.path.join(dataset_root, name)
        images_dir = os.path.join(dataset_path, "images")
        json_path = os.path.join(dataset_path, "annotations.json")

        if not os.path.exists(json_path):
            print(f"skipping dataset {name}: missing annotations")
            continue

        with open(json_path, "r") as f:
            annotations = json.load(f)

        samples = []
        for img_name, points in annotations.items():
            if img_name.endswith(".jpg"):
                samples.append(
                    {"image_path": os.path.join(images_dir, img_name), "points": points}
                )

        if samples:
            gathered[name] = samples

    return gathered


def split_samples(gathered_samples, train_ratio, val_ratio, seed=None):
    rng = random.Random(seed)

    train, val, test = [], [], []

    for name, samples in gathered_samples.items():
        shuffled = samples[:]
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train += shuffled[:n_train]
        val += shuffled[n_train : n_train + n_val]
        test += shuffled[n_train + n_val :]

        print(
            f"{name}: train={n_train}, val={n_val}, test={len(shuffled) - n_train - n_val}"
        )

    return train, val, test


def save_paths(samples, path):
    with open(path, "w") as f:
        f.write("\n".join(s["image_path"] for s in samples))


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("loading data")
    gathered = gather_samples(DATA_ROOT, ALL_DATASETS)
    print("total:", sum(len(v) for v in gathered.values()))

    train_s, val_s, test_s = split_samples(
        gathered, TRAIN_RATIO, VAL_RATIO, seed=RNG_SEED
    )

    save_paths(test_s, TEST_FILES_OUT)
    save_paths(train_s, "train_data.txt")
    save_paths(val_s, "val_data.txt")

    train_ds = FishDataset(samples=train_s)
    val_ds = FishDataset(samples=val_s)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=pad_collate,
    )

    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model = CSRNet(load_pretrained=True).to(device)

    mse_loss = nn.MSELoss()

    # better learning balance compared to previous scripts
    optimizer = torch.optim.Adam(
        [
            {"params": model.frontend.parameters(), "lr": 5e-6},
            {"params": model.backend.parameters(), "lr": 2e-5},
        ]
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

    best_mae = float("inf")

    for epoch in range(NUM_EPOCHS):
        # ------------ train- ---------------
        model.train()
        total_loss = 0.0

        for imgs, gt in train_loader:
            imgs, gt = imgs.to(device), gt.to(device)

            pred = model(imgs)

            # combine density + count supervision
            density_loss = mse_loss(pred, gt)
            count_loss = F.l1_loss(pred.sum(dim=[1, 2, 3]), gt.sum(dim=[1, 2, 3]))

            loss = 0.8 * density_loss + 0.2 * count_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        print(f"[epoch {epoch + 1}] train loss: {total_loss / len(train_loader):.6f}")

        # ---------- validation ---------
        model.eval()

        total_abs_error = 0.0
        count = 0

        with torch.no_grad():
            for imgs, gt_density in val_loader:
                imgs = imgs.to(device)
                gt_density = gt_density.to(device)

                pred = model(imgs)

                pred_count = pred.sum().item()
                gt_count = gt_density.sum().item()

                total_abs_error += abs(pred_count - gt_count)
                count += 1

        mae = total_abs_error / count

        print(f"mean abs error: {mae:.4f}")

        # ---------- chcekpoint -----------
        if mae < best_mae:
            best_mae = mae
            torch.save(model.state_dict(), "best_model.pth")
            print("saved best model")


if __name__ == "__main__":
    main()
