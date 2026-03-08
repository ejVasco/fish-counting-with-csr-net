import os
import torch
import torch.nn as nn 
from torch.utils.data import DataLoader

from datasets.fish_dataset import FishDataset 
from models.csrnet import CSRNet

#-----------------------------
# config 
DATA_ROOT = "datasets"
TRAIN_DATASETS = [
    "yellow_tank_2",
    "GX011279-fish-11-20",
]
VAL_DATASETS = [
    "yellow_tank_1",
    "blue_tank_2",
]

BATCH_SIZE = 4 
NUM_EPOCHS = 50 
LEARNING_RATE = 1e-5 
CHECKPOINT_DIR = "checkpoints"


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True) # make/check checkpoint dir exist
#----------------------------- 
# devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
#-------------------------------- 
# datasets & loaders 
    train_dataset = FishDataset(DATA_ROOT, TRAIN_DATASETS)
    val_dataset = FishDataset(DATA_ROOT, VAL_DATASETS)

    train_loader = DataLoader(
        train_dataset, 
        batch_size = BATCH_SIZE, 
        shuffle = True,
        num_workers = 4,
        pin_memory = True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size = 1,
        shuffle = False
    )

    print(f"Num of train samples {len(train_dataset)}")
    print(f"Num of val samples {len(val_dataset)}")
#------------------------------- 
# model 
    model = CSRNet(load_pretrained=True).to(device)
#------------------------------------ 
# loss & optimizer 
    criterion = nn.MSELoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# ----------------------------------- 
# training loop

    best_epoch = -1 # will store best epoch model 
    best_val_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
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
        print(f"[Epoch: {epoch+1}/{NUM_EPOCHS}] train loss: {train_loss:.4f}")
#------------------------------ 
# validation (still in loop)
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for imgs, gt_density in val_loader:
                imgs = imgs.to(device)
                gt_density = gt_density.to(device)

                pred_density = model(imgs)
                loss = criterion(pred_density, gt_density)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"    val loss: {val_loss:.4f}")
# ---------------------------- 
# save checkpoint 
        ckpt_path = os.path.join(
            CHECKPOINT_DIR, f"csrnet_epoch_{epoch+1}.pth"
        )
        torch.save(model.state_dict(), ckpt_path)
#-------------------------------- 
# save best epoch / checkpoint 
        if val_loss < best_val_loss:
            best_val_loss = val_loss 
            best_epoch = epoch+1 
            torch.save(model.state_dict(), "best_model.pth")
            print(f"new best model at epoch {epoch+1}")


if __name__ == "__main__":
    main()
