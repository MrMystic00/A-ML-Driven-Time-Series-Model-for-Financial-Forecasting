"""
Plot the model's DIRECT multi-day forecast (--predict_window P, e.g. 3) from
dates YOU choose, using the checkpoint trained by train.py.

For each --anchor_date (the last real day of the look-back window) it saves one
PDF showing: the look-back, the P predicted days, and the actual prices for
those days (ground truth).

Use the SAME flags as the training run, so the checkpoint path matches:

    python plot_window.py --anchor_date 2026-03-20,2026-04-13 \
        --root_path ./dataset --data_path SP500_stock.csv \
        --behavior_data_path SP500_cfb_exp3.csv --enc_in 1 --enc_in_be 8 \
        --train_window 8 --predict_window 3 --batch_size 1 ... (rest as in training)

Options:
    --anchor_date   one date or a comma-separated list; must lie inside the TEST set
                    and have P real days after it (no forecasting past the data).
    --out_dir       folder for the PDFs (default ./pic), file: window_<date>.pdf
    --day_labels    on (default) / off -- dot and "Day N" label at each forecast day.
"""

import argparse
import os
import sys

import numpy as np
import torch

from train import build_setting, get_args
from exp.exp_craft import Exp_Craft
from utils.tools import visual


def find_anchor_index(dates, anchor_date):
    import pandas as pd

    normalized = [pd.Timestamp(d).normalize() for d in dates]
    target = pd.Timestamp(anchor_date).normalize()
    if target in normalized:
        return normalized.index(target)
    raise ValueError(
        f"anchor_date {anchor_date} is not a date in the test set "
        f"({normalized[0].date()} to {normalized[-1].date()}, {len(dates)} dates). "
        f"Weekends/holidays are not in the data -- pick a trading day."
    )


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--anchor_date", type=str, required=True,
                        help="Last real day of the look-back window; comma-separated for several")
    parser.add_argument("--out_dir", type=str, default="./pic")
    parser.add_argument("--day_labels", type=str, default="on", choices=["on", "off"])
    known, remaining = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + remaining
    args = get_args()
    P, L = args.predict_window, args.train_window

    exp = Exp_Craft(args)
    setting = build_setting(args)
    ckpt = f"{args.checkpoints}/{setting}/checkpoint.pth"
    exp.model.load_state_dict(torch.load(ckpt, map_location=exp.device))
    exp.model.eval()

    test_data, _ = exp._get_data(flag="test")
    dates = test_data.dates
    n = len(test_data.data_x)
    os.makedirs(known.out_dir, exist_ok=True)

    for anchor in [a.strip() for a in known.anchor_date.split(",") if a.strip()]:
        a_idx = find_anchor_index(dates, anchor)
        s_begin = a_idx - L + 1
        s_end = s_begin + L
        r_end = s_end + P
        if s_begin < 0:
            raise ValueError(f"{anchor}: too close to the start of the test set for train_window={L}")
        if r_end > n:
            raise ValueError(f"{anchor}: only {n - s_end} real days follow it, but predict_window={P}")

        dev = exp.device
        batch_x = torch.tensor(test_data.data_x[s_begin:s_end], dtype=torch.float32).unsqueeze(0).to(dev)
        behavior_x = torch.tensor(test_data.data_be[s_begin:s_end], dtype=torch.float32).unsqueeze(0).to(dev)
        behavior_y = torch.tensor(test_data.data_be[s_end:r_end], dtype=torch.float32).unsqueeze(0).to(dev)
        behavior_matrix = torch.tensor(test_data.behavior_matrix_full[s_end:r_end],
                                       dtype=torch.float32).unsqueeze(0).to(dev)

        with torch.no_grad():
            y, _, _ = exp.model(batch_x, behavior_x, behavior_y, behavior_matrix)
        preds = y[0, :, -1].cpu().numpy()                       # P forecasts, target column

        history = test_data.data_x[s_begin:s_end, -1].copy()
        truth = test_data.data_x[s_end:r_end, -1].copy()
        all_dates = list(dates[s_begin:r_end])

        gt = np.concatenate([history, truth])
        pd_values = np.concatenate([history, preds])
        labels = None
        if known.day_labels == "on":
            labels = [(dates[s_end + i], preds[i], f"Day {i + 1}") for i in range(P)]

        out = os.path.join(known.out_dir, f"window_{str(anchor).replace('/', '-')}.pdf")
        visual(gt, pd_values, dates=all_dates, name=out, point_labels=labels)

        err = truth - preds
        print(f"[{anchor}] saved {out}")
        print(f"  dates:     {[str(d)[:10] for d in dates[s_end:r_end]]}")
        print(f"  actual:    {np.round(truth, 2).tolist()}")
        print(f"  predicted: {np.round(preds, 2).tolist()}")
        print(f"  error:     {np.round(err, 2).tolist()}   RMSE={np.sqrt((err ** 2).mean()):.4f}")


if __name__ == "__main__":
    main()