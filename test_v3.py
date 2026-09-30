# test_v3.py
import json
import os
import sys

import cv2
import numpy as np
import PIL.Image as Image
import torch
from matplotlib import cm
from matplotlib import pyplot as plt
from torchvision import transforms

from models.csrnet import CSRNet
from utils.tank_mask import (
    build_mask,
    circle_to_fractional,
    compute_dataset_circle,
    detect_tank_circle,
    load_manual_masks,
)

_MASK_CIRCLE_CACHE = {}
_MANUAL_MASKS = load_manual_masks()


def _get_mask_for(img, img_path, shape, use_dataset_calibration):
    """
    build tank mask for prediction: traced outline if the image's dataset has one (no hough needed),
    else hough circle; minus any manual exclusions (pipe)
    optional flag for using dataset calibration, set to true to use dataset calibration and save time for known datasets
    """
    images_dir = os.path.dirname(img_path)
    dataset_dir = os.path.dirname(images_dir)
    manual = _MANUAL_MASKS.get(os.path.basename(dataset_dir))

    if manual and manual["tank"]:
        return build_mask(None, shape, manual)

    if use_dataset_calibration:
        if dataset_dir not in _MASK_CIRCLE_CACHE:
            filenames = sorted(f for f in os.listdir(images_dir) if f.endswith(".jpg"))
            _MASK_CIRCLE_CACHE[dataset_dir] = compute_dataset_circle(
                images_dir, filenames
            )
        return build_mask(_MASK_CIRCLE_CACHE[dataset_dir], shape, manual)

    img_bgr = np.array(img)[:, :, ::-1].copy()
    circle = detect_tank_circle(img_bgr)
    frac_circle = (
        circle_to_fractional(circle, img_bgr.shape[:2]) if circle is not None else None
    )
    return build_mask(frac_circle, shape, manual)


def load_gt_points(img_path):
    """
    looks up ground truth (x,y) points for image by reading corresponding annotations.json

    expects paths in form: ./datasets/<dataset_name>/images/<img>.jpg
    returns a list of (x,y) tuples or None if the annotation can't be found
    """
    # walk up 2 levels to find annotations.json
    images_dir = os.path.dirname(img_path)
    dataset_dir = os.path.dirname(images_dir)
    json_path = os.path.join(dataset_dir, "annotations.json")

    if not os.path.isfile(json_path):
        return None

    img_name = os.path.basename(img_path)

    try:
        with open(json_path, "r") as f:
            annotations = json.load(f)
        return annotations.get(img_name)
    except Exception:
        return None


def load_gt_count(img_path):
    """
    looks up ground truth fish count for image by reading corresponding annotations.json
    returns the count as an int or None if the annotation can't be found
    """
    points = load_gt_points(img_path)
    if points is None:
        return None
    return len(points)


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


MAX_SIZE = 512  # must match FishDataset max_size used in training


def resize_like_training(img):
    """
    Resizes the image to so longest side is MAX_SIZE
    """
    W, H = img.size
    scale = min(MAX_SIZE / W, MAX_SIZE / H, 1.0)
    if scale == 1.0:
        return img
    new_W, new_H = int(W * scale), int(H * scale)
    resized = cv2.resize(np.array(img), (new_W, new_H))
    return Image.fromarray(resized)


TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def predict(
    model, img_path, device, clamp=True, apply_mask=True, use_dataset_calibration=False
):
    img = Image.open(img_path).convert("RGB")
    tensor = TRANSFORM(resize_like_training(img)).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)

    raw_count = float(output.sum().cpu().item())

    if clamp:
        output = output.clamp(min=0)

    density = output.squeeze().cpu().numpy()

    if apply_mask:
        mask = _get_mask_for(img, img_path, density.shape, use_dataset_calibration)
        density *= mask

    final_count = float(density.sum())

    return raw_count, final_count, density, img


def visualize(
    img, density, raw_count, final_count, img_path, show=True, save_path=None
):
    pass


def compute_metrics(rows):
    """
    count metrics over results with ground truth, using rounded predictions
      mae  - mean absolute error, in fish
      rmse - root mean squared error, in fish (penalizes large misses more)
      mape - mean absolute percentage error over images with gt > 0 (percent of the true count)
    """
    rows = [r for r in rows if r["gt"] is not None]
    if not rows:
        nan = float("nan")
        return {"n": 0, "mae": nan, "rmse": nan, "mape": nan, "worst": 0}
    errs = [abs(round(r["final"]) - r["gt"]) for r in rows]
    pct = [e / r["gt"] * 100 for e, r in zip(errs, rows) if r["gt"] > 0]
    return {
        "n": len(rows),
        "mae": sum(errs) / len(errs),
        "rmse": (sum(e * e for e in errs) / len(errs)) ** 0.5,
        "mape": sum(pct) / len(pct) if pct else float("nan"),
        "worst": max(errs),
    }


def print_summary(results, model_path, data_path):
    """
    Prints a summary table grouped by dataset
    """
    gt_results = [r for r in results if r["gt"] is not None]

    # group results by dataset name (2 levels up from image path)
    datasets = {}
    for r in gt_results:
        # path looks like: datasets/<name>/images/<img>.jpg
        dataset_name = os.path.basename(os.path.dirname(os.path.dirname(r["path"])))
        datasets.setdefault(dataset_name, []).append(r)

    W = 84
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
    col = f"  {'Dataset':<30} {'N':>3}  {'Avg GT':>6}  {'Avg Pred':>8}  {'MAE':>6}  {'RMSE':>6}  {'MAPE':>6}  {'Worst':>5}"
    print(col)
    print(thin)

    dataset_metrics = []
    for name, rows in sorted(datasets.items()):
        m = compute_metrics(rows)
        avg_gt = sum(r["gt"] for r in rows) / m["n"]
        avg_pred = sum(round(r["final"]) for r in rows) / m["n"]
        dataset_metrics.append((name, m))
        print(
            f"  {name:<30} {m['n']:>3}  {avg_gt:>6.1f}  {avg_pred:>8.1f}  {m['mae']:>6.2f}  {m['rmse']:>6.2f}  {m['mape']:>5.1f}%  {m['worst']:>5}"
        )

    overall = compute_metrics(gt_results)
    print(thin)
    print(
        f"  {'OVERALL':<30} {overall['n']:>3}  {'':>6}  {'':>8}  {overall['mae']:>6.2f}  {overall['rmse']:>6.2f}  {overall['mape']:>5.1f}%  {overall['worst']:>5}"
    )
    print(div)
    print("  MAE/RMSE in fish; MAPE = mean |error| / true count per image (gt > 0 only)")

    # best / worst datasets, by percent error so large and small tanks compare fairly
    ranked = [(n, m) for n, m in dataset_metrics if m["mape"] == m["mape"]]
    if ranked:
        best = min(ranked, key=lambda x: x[1]["mape"])
        worst = max(ranked, key=lambda x: x[1]["mape"])
        print(f"  Best  dataset: {best[0]}  (MAPE {best[1]['mape']:.1f}%, MAE {best[1]['mae']:.2f})")
        print(f"  Worst dataset: {worst[0]}  (MAPE {worst[1]['mape']:.1f}%, MAE {worst[1]['mae']:.2f})")
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
    Returns (results, metrics) where results is a list of per image dicts and metrics is from compute_metrics
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
            continue

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

    return results, compute_metrics(results)


def print_comparison(all_results):
    """
    all_results: list of {"activation": str, "checkpoint": str, "results": [...], "metrics": dict}
    prits a compact side by side of all 3 methods
    """
    W = 72
    div = "=" * W
    print(f"\n{div}\n  METHOD COMPARISON (overall, lower is better)\n{div}")
    for r in sorted(all_results, key=lambda r: r["metrics"]["mae"]):
        m = r["metrics"]
        print(
            f"  {r['activation']:<10} MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}  MAPE={m['mape']:.1f}%   ({r['checkpoint']})"
        )
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
            results, metrics = run_model_test(
                model_path, img_paths, device, clamp, save, headless, quiet=True
            )
            print_summary(results, model_path, data_path)
            all_results.append(
                {
                    "activation": activation,
                    "checkpoint": model_path,
                    "results": results,
                    "metrics": metrics,
                }
            )

        if all_results:
            print_comparison(all_results)
        return

    # ------ single model mode ----
    model_path = positional[1] if len(positional) > 1 else "best_model.pth"
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
        f"  clamp     : {clamp}\n"
        f"  save      : {save}\n"
        f"  headless  : {headless}\n"
        "---------\n"
    )

    results, _ = run_model_test(model_path, img_paths, device, clamp, save, headless)
    if any(r["gt"] is not None for r in results):
        print_summary(results, model_path, data_path)


if __name__ == "__main__":
    main()
