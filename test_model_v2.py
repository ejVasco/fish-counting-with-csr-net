# ./test_model_v2.py
import os
import sys

import numpy as np
import PIL.Image as Image
import torch
from matplotlib import cm
from matplotlib import pyplot as plt
from torchvision import transforms

from models.csrnet import CSRNet


def main():
    usage = "Usage:\ntemp"

    # ---- arguments ------------
    # todo: add these args and defaults to usage later
    args = sys.argv[1:]
    if any(h in args for h in ["--h", "--help"]):
        print(usage)
        return
    clamp = "--no-clamp" not in args
    save = "--save" in args
    headless = "--headless" in args

    positional = [a for a in args if not a.startswith("--")]
    if not positional:
        print(usage)
        return
    model_path = positional[1] if len(positional) > 1 else "test_data.txt"
    data_path = positional[0] if len(positional) > 0 else "best_model.pth"

    # ---------validate arguments ----------
    if os.path.isfile(data_path) or not data_path.endswith(".txt"):
        print(f"Invalid datapath: {data_path}, ensure file exists and is .txt")
        print(usage)
        return
    if os.path.isfile(model_path) or not model_path.endswith(".pth"):
        print(f"Invalid modelpath: {model_path}, ensure file exists and is .pth")
        print(usage)
        return

    print(f"PATHS:\n   data_path = {data_path}\n  model_path = {model_path}")

    with open(data_path) as f:
        image_paths = [line.strip() for line in f if line.strip()]


if __name__ == "__main__":
    main()
