# ./test_model_v2.py
import json
import os
import sys

import numpy as np
import PIL.Image as Image
import torch
from matplotlib import cm
from matplotlib import pyplot as plt
from torchvision import transforms

from models.csrnet import CSRNet


def load_gt_count(img_path):
    """
    looks up ground truth fish count for image by reading corresponding annotations.json

    expects paths in form: ./datasets/<dataset_name>/images/<img>.jpg
    returns the point count as an int or None if the annotation can't be found
    """
    # walk up 2 levels:
    # Walk up two levels:  images/ -> dataset_dir -> find annotations.json
    images_dir = os.path.dirname(img_path)
    dataset_dir = os.path.dirname(images_dir)
    json_path = os.path.join(dataset_dir, "annotations.json")

    if not os.path.isfile(json_path):
        return None

    img_name = os.path.basename(img_path)

    try:
        with open(json_path, "r") as f:
            annotations = json.load(f)
        points = annotations.get(img_name)
        if points is None:
            return None
        return len(points)
    except Exception:
        return None


def load_model(model_path, device):
    model = CSRNet()
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # handles both checkpoints (raw state dict and { "model_state_dict " : ...})
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def predict(model, img_path, device, clamp=True):
    img = Image.open(img_path).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)

    raw_count = float(output.sum().cpu().item())

    if clamp:
        output = output.clamp(min=0)

    density = output.squeeze().cpu().numpy()
    final_count = float(density.sum())

    return raw_count, final_count, density, img


def visualize(
    img, density, raw_count, final_count, img_path, show=True, save_path=None
):
    pass


def print_summary(results, clamp):
    counts = [r["final"] for r in results]
    raw_counts = [r["raw"] for r in results]
    n_negative = sum(1 for r in raw_counts if r < 0)

    print(
        "----------------------------------------------\n"
        f" RESULTS SUMMARY ({len(results)} images)\n"
        "------------------------------------------------\n"
        f"  {'Image':<45} {'Raw':>8} {'Clamped':>8} {'Rounded':>7}\n"
        "---------------------------------------------------\n"
    )

    # TODO - finish print summary


def main():
    usage = "Usage:\ntemp"

    # ---- arguments ------------
    # "--save" , "--headless" , "[modelpath].pth" , "[testdata].txt"
    # todo: add these args and defaults to usage later
    args = sys.argv[1:]
    if any(h in args for h in ["--h", "--help"]):
        print(usage)
        return
    save = "--save" in args
    headless = "--headless" in args
    clamp = False  # not "--no-clamp" in args  # basically whether to throw out negative results

    positional = [a for a in args if not a.startswith("--")]
    if not positional:
        print(usage)
        return
    model_path = positional[1] if len(positional) > 1 else "test_data.txt"
    data_path = positional[0] if len(positional) > 0 else "best_model.pth"

    # ---------validate arguments ----------
    if not os.path.isfile(data_path) or not data_path.endswith(".txt"):
        print(f"Invalid datapath: {data_path}, ensure file exists and is .txt")
        print(usage)
        return
    if not os.path.isfile(model_path) or not model_path.endswith(".pth"):
        print(f"Invalid modelpath: {model_path}, ensure file exists and is .pth")
        print(usage)
        return
    # print(f"PATHS:\n  data_path = {data_path}\n  model_path = {model_path}")

    # ---------- reading test data ----------_ -
    with open(data_path) as f:
        img_paths = [line.strip() for line in f if line.strip()]

    # validate files exist
    missing_img_files = [
        p for p in img_paths if not os.path.isfile(p)
    ]  # array of missing files

    if missing_img_files:
        print(f"Warning: {len(missing_img_files)} files in {data_path} not found:")
        for m in missing_img_files:
            print(f"    -> {m}")

        img_paths = [
            p for p in img_paths if os.path.isfile(p)
        ]  # only keep existing files

    if not img_paths:
        print(f"error: no paths found in {data_path}]")
        return

    # ------------- device and model ---------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        "------------------------------------------\n"
        "Full summary before running testing\n"
        f"  model    : {model_path}\n"
        f"  data     : {data_path}\n"
        f"  device   : {device}\n"
        f"  clamp    : {clamp}\n"
        f"  save     : {save}\n"
        f"  headless : {headless}\n"
        "-------------------------------------------\n"
    )

    model = load_model(model_path, device)

    # ------------- batch testing --------------
    results = []
    total = len(img_paths)

    for (
        i,
        img_path,
    ) in enumerate(img_paths, 1):
        try:
            raw, final, density, og_img = predict(model, img_path, device, clamp=clamp)
        except Exception as e:
            print(f"    error predicting {img_path} ({e})")
            continue

        gt_count = load_gt_count(img_path)
        gt_str = f"{gt_count}" if gt_count is not None else "N/A"
        error_str = (
            f"{abs(round(final) - gt_count):+d}" if gt_count is not None else "N/A"
        )

        flag = " <- negative" if raw < 0 else ""
        print(
            f"[{i:>2}/{total}] raw: {raw:>+8.2f}\n"
            f"  clamped: {final:>6.1f}  rounded: {round(final):>4}  gt: {gt_str:>4}  error: {error_str}\n"
            f"  {img_path}\n"
        )

        results.append(
            {
                "path": img_path,
                "name": os.path.basename(img_path),
                "raw": raw,
                "final": final,
                "gt": gt_count,
            }
        )

        save_path = None
        if save:
            pass  # TODO: savinhg

        if not headless or save:
            visualize(
                og_img,
                density,
                raw,
                final,
                img_path=img_path,
                show=not headless,
                save_path=save_path,
            )

    gt_results = [r for r in results if r["gt"] is not None]
    if gt_results:
        mae = sum(abs(round(r["final"]) - r["gt"]) for r in gt_results) / len(
            gt_results
        )
        print(f"MAE over {len(gt_results)} images: {mae:.4f}")


if __name__ == "__main__":
    main()
