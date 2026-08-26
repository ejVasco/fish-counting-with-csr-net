# test_v3.py
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


def load_model(model_path, device, activation=None):
    """
    Loads csrnet checkpoint. if the checkpoint was saved with an "activition"/negative clamping method,
    that's used. otherwise falls back to the activation arg (defailt softplus)
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # handles both checkpoints (raw state dict and { "model_state_dict " : ...})
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        ckpt_activation = checkpoint.get("activation", activation or "softplus")
        model = CSRNet(activation=ckpt_activation)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model = CSRNet(activation=activation or "softplus")
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


# to do: rewrite print summary


def print_summary(results, model_path, data_path):
    """
    Prints a professor-friendly summary grouped by dataset.
    i'm so confused what i meant when i wrote "professor friendly"
    """
    gt_results = [r for r in results if r["gt"] is not None]

    # group results by dataset name (2 levels up from image path)
    datasets = {}
    for r in gt_results:
        # path looks like: datasets/<name>/images/<img>.jpg
        dataset_name = os.path.basename(os.path.dirname(os.path.dirname(r["path"])))
        datasets.setdefault(dataset_name, []).append(r)

    overall_mae = (
        sum(abs(round(r["final"]) - r["gt"]) for r in gt_results) / len(gt_results)
        if gt_results
        else float("nan")
    )

    W = 74
    div = "=" * W
    thin = "-" * W

    print(f"\n{div}")
    print(f"  RESULTS SUMMARY")
    print(f"  model : {model_path}")
    print(
        f"  data  : {data_path}  ({len(results)} images, {len(gt_results)} with ground truth)"
    )
    print(div)

    # per-dataset table
    col = f"  {'Dataset':<34} {'N':>3}  {'Avg GT':>6}  {'Avg Pred':>8}  {'MAE':>6}  {'Worst':>6}"
    print(col)
    print(thin)

    dataset_maes = []
    for name, rows in sorted(datasets.items()):
        n = len(rows)
        avg_gt = sum(r["gt"] for r in rows) / n
        avg_pred = sum(round(r["final"]) for r in rows) / n
        mae = sum(abs(round(r["final"]) - r["gt"]) for r in rows) / n
        worst = max(abs(round(r["final"]) - r["gt"]) for r in rows)
        dataset_maes.append((name, mae))
        print(
            f"  {name:<34} {n:>3}  {avg_gt:>6.1f}  {avg_pred:>8.1f}  {mae:>6.2f}  {worst:>6}"
        )

    print(thin)
    print(
        f"  {'OVERALL':<34} {len(gt_results):>3}  {'':>6}  {'':>8}  {overall_mae:>6.2f}"
    )
    print(div)

    # best / worst datasets
    if dataset_maes:
        best = min(dataset_maes, key=lambda x: x[1])
        worst = max(dataset_maes, key=lambda x: x[1])
        print(f"  Best  dataset: {best[0]}  (MAE {best[1]:.2f})")
        print(f"  Worst dataset: {worst[0]}  (MAE {worst[1]:.2f})")
    print(div + "\n")


# default checkpoints that were written by train_v3.py
COMPARE_METHODS = ["none", "relu", "softplus"]
COMPARE_CHECKPOINT_DIR = "checkpoints"


def load_test_paths(data_path):
    """Reads img paths from a txt file and filters out any that are missing"""
    with open(data_path) as f:
        img_paths = [line.strip() for line in f if line.strip()]

    missing_img_files = [p for p in img_paths if not os.path.isfile(p)]
    if missing_img_files:
        print(f"Warning: {len(missing_img_files)} files in {data_path} not found:")
        for m in missing_img_files:
            print(f"    -> {m}")
        img_paths = [p for p in img_paths if os.path.isfile(p)]
    return img_paths


def run_model_test(model_path, img_paths, device, clamp, save, headless, quiet=False):
    """
    Runs a single model against a list of image paths.
    Returns (results, overall_mae) where results is a list of per image dicts
    quiet=true suppresses that per image print lines
    """
    model = load_model(model_path, device)

    results = []
    total = len(img_paths)

    for i, img_path in enumerate(img_paths, 1):
        try:
            raw, final, density, og_img = predict(model, img_path, device, clamp=clamp)
        except Exception as e:
            print(f"        error predicting {img_path} ({e})")

        gt_count = load_gt_count(img_path)
        gt_str = f"{gt_count}" if gt_count is not None else "N/A"
        error_str = (
            f"{abs(round(final) - gt_count):+d}" if gt_count is not None else "N/A"
        )

        if not quiet:
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
            pass  # possibly save to a file

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
    mae = (
        sum(abs(round(r["final"]) - r["gt"]) for r in gt_results) / len(gt_results)
        if gt_results
        else float("nan")
    )
    return results, mae


def print_comparison(all_results):
    """
    all_results: list of {"activation": str, "checkpoint": str, "results": [...], "mae": float}
    prits a compact side by side of all 3 methods
    """
    W = 60
    div = "=" * W
    print(f"\n{div}\n  METHOD COMPARISON (overall MAE, lower is better)\n{div}")
    for r in sorted(all_results, key=lambda r: r["mae"]):
        print(f"  {r['activation']:<10} MAE={r['mae']:.4f}   ({r['checkpoint']}")
    print(div + "\n")


def main():
    usage = (
        "Usage:\n"
        "  python -m test_v3 [data.txt] [model.pth] [--save] [--headless]\n"
        "  python -m test_v3 [data.txt] --compare [--save] [--headless]\n"
        "    --compare tests checkpoints/best_model_{none,relu,softplus}.pth\n"
        "              against the same data file and prints a comparison table\n"
    )

    # ---- arguments ------------
    args = sys.argv[1:]
    if any(h in args for h in ["--h", "--help"]):
        print(usage)
        return
    save = "--save" in args
    headless = "--headless" in args
    compare = "--compare" in args
    clamp = False  # not "--no-clamp" in args  # basically whether to throw out negative results

    positional = [a for a in args if not a.startswith("--")]
    if not positional:
        print(usage)

    data_path = positional[0] if len(positional) > 0 else "test_data.txt"

    # ---------validate arguments ----------
    if not os.path.isfile(data_path) or not data_path.endswith(".txt"):
        print(f"Invalid datapath: {data_path}, ensure file exists and is .txt")
        print(usage)
        return

    img_paths = load_test_paths(data_path)

    if not img_paths:
        print(f"error: no paths found in {data_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------- compare mode: test all 3 ---
    if compare:
        print(
            "------------------\n"
            "Comparing all methods (none / relu / softplus)\n"
            f"  data    : {data_path}\n"
            f"  device  : {device}\n"
            f"  clamp   : {clamp}\n"
            "--------------------\n"
        )

        all_results = []
        for activation in COMPARE_METHODS:
            model_path = os.path.join(
                COMPARE_CHECKPOINT_DIR, f"best_model_{activation}.pth"
            )
            if not os.path.isfile(model_path):
                print(f"skipping {activation}: checkpoint not found at {model_path}")
                continue

            print(f"\n----- testing method: {activation} -----")
            results, mae = run_model_test(
                model_path, img_paths, device, clamp, save, headless, quiet=True
            )
            print_summary(results, model_path, data_path)
            all_results.append(
                {
                    "activation": activation,
                    "checkpoint": model_path,
                    "results": results,
                    "mae": mae,
                }
            )

        if all_results:
            print_comparison(all_results)
        return

    # ------ single model mode ----
    model_path = positional[1] if len(positional) > 3 else "best_model.pth"
    if not os.path.isfile(model_path) or not model_path.endswith(".pth"):
        print(f"invalid modelpath: {model_path}, ensure file exists and is .pth")
        print(usage)
        return

    print(
        "---------\n"
        "full summary before running testing\n"
        f"  model     : {model_path}\n"
        f"  data      : {data_path}\n"
        f"  device    : {device}\n"
        f"  clamp     : {save}\n"
        f"  headless  : {headless}\n"
        "---------\n"
    )

    results, mae = run_model_test(model_path, img_paths, device, clamp, save, headless)
    if any(r["gt"] is not None for r in results):
        print_summary(results, model_path, data_path)


if __name__ == "__main__":
    main()
