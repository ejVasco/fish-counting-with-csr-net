# spreadsheet.py
#
# write one frame's density map to a .xlsx to see how the model with no negative clamping behaves
#
# usage:
# python spreadsheet.py <image.jpg> <checkpoint.pth> [--out density_dump.xlsx] [--calibrated]

import argparse
import os
from json import load

import cv2
import numpy as np
import torch
from openpyxl import Workbook
from openpyxl.formatting.rule import CellIsRule, ColorScaleRule
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

from test_v3 import _get_mask_for, load_gt_points, load_model, predict
from utils.density import gen_density_map

FONT = Font(name="Arial", size=9)
BOLD = Font(name="Arial", size=10, bold=True)


def write_grid(ws, grid, number_format="0.000"):
    """write 2d array starting at A1, small fixed-width columns"""
    H, W = grid.shape
    for r in range(H):
        for c in range(W):
            cell = ws.cell(row=r + 1, column=c + 1, value=float(grid[r, c]))
            cell.font = FONT
            cell.number_format = number_format
    for c in range(1, W + 1):
        ws.column_dimensions[get_column_letter(c)].width = 6.5
    ws.freeze_panes = "A1"


def add_diverging_format(ws, grid):
    """blue for negative, white for 0, red for positive + bold red font for negative cell"""
    ref = f"A1:{get_column_letter(grid.shape[1])}{grid.shape[0]}"
    lim = float(max(abs(grid.min()), abs(grid.max()), 1e-6))
    ws.conditional_formatting.add(
        ref,
        ColorScaleRule(
            start_type="num",
            start_value=-lim,
            start_color="4F81BD",
            mid_type="num",
            mid_value=0,
            mid_color="FFFFFF",
            end_type="num",
            end_value=lim,
            end_color="C0504D",
        ),
    )
    ws.conditional_formatting.add(
        ref,
        CellIsRule(
            operator="lessThan",
            formula=["0"],
            font=Font(name="Arial", size=9, bold=True, color="0000FF"),
        ),
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("img_path")
    ap.add_argument("model_path")
    ap.add_argument("--out", default="density.xlsx")
    ap.add_argument(
        "--calibrated",
        action="store_true",
        help="use dataset level tank circle as mask",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_path, device)

    # raw output no clamping no masking
    _, _, raw, img = predict(
        model, args.img_path, device, clamp=False, apply_mask=False
    )
    mask = _get_mask_for(img, args.img_path, raw.shape, args.calibrated)
    masked = raw * mask

    # gt resampled to the output grid
    gt_grid = None
    pts = load_gt_points(args.img_path)
    if pts is not None:
        W, H = img.size
        gt_full = gen_density_map((H, W), pts, sigma=4)
        h, w = raw.shape
        gt_grid = cv2.resize(gt_full, (w, h), interpolation=cv2.INTER_AREA)
        gt_grid = gt_grid * (H * W) / (h * w)

    wb = Workbook()
    ws_sum = wb.active
    ws_sum.title = "Summary"
    ws_raw = wb.create_sheet("Raw")
    ws_masked = wb.create_sheet("Masked")
    ws_mask = wb.create_sheet("Mask")

    write_grid(ws_raw, raw)
    write_grid(ws_masked, masked)
    write_grid(ws_mask, mask)
    add_diverging_format(ws_raw, raw)
    add_diverging_format(ws_masked, masked)

    if gt_grid is not None:
        ws_gt = wb.create_sheet("GT")
        write_grid(ws_gt, gt_grid)

    h, w = raw.shape
    rng = f"A1:{get_column_letter(w)}{h}"

    rows = [
        ("image", args.img_path),
        ("model", args.model_path),
        ("grid (rows x cols)", f"{h} x {w}"),
        (
            "mask source",
            "dataset calibration" if args.calibrated else "per-image Hough",
        ),
        (None, None),
        ("Raw: total (no mask)", f"=SUM(Raw!{rng})"),
        ("Raw: sum of negatives", f'=SUMIF(Raw!{rng},"<0")'),
        ("Raw: sum of positives", f'=SUMIF(Raw!{rng},">0")'),
        ("Raw: min cell", f"=MIN(Raw!{rng})"),
        ("Raw: max cell", f"=MAX(Raw!{rng})"),
        ("Raw: # negative cells", f'=COUNTIF(Raw!{rng},"<0")'),
        (None, None),
        ("Masked: total (= predicted count)", f"=SUM(Masked!{rng})"),
        ("Masked: sum of negatives", f'=SUMIF(Masked!{rng},"<0")'),
        ("Masked: sum of positives", f'=SUMIF(Masked!{rng},">0")'),
        ("Masked: min cell", f"=MIN(Masked!{rng})"),
        ("Masked: # negative cells (inside tank)", f'=COUNTIF(Masked!{rng},"<0")'),
        ("Tank cells in mask", f"=SUM(Mask!{rng})"),
    ]

    if gt_grid is not None:
        rows += [
            (None, None),
            ("GT count (annotated points)", len(pts)),
            ("GT grid total (masked-equivalent check)", f"=SUM(GT!{rng})"),
        ]

    for i, (label, value) in enumerate(rows, start=1):
        if label is None:
            continue
        a = ws_sum.cell(row=i, column=1, value=label)
        b = ws_sum.cell(row=i, column=2, value=value)
        a.font = BOLD
        b.font = FONT
        if isinstance(value, str) and value.startswith("="):
            b.number_format = "0.000"
    ws_sum.column_dimensions["A"].width = 42
    ws_sum.column_dimensions["B"].width = 60

    wb.save(args.out)
    print(f"saved {args.out}  (grid {h} x {w})")
    print("open in Excel/LibreOffice; formulas calculate on open")


if __name__ == "__main__":
    main()
