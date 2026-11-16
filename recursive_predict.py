"""
Recursive (iterative) multi-step forecasting using ONE model trained with
--predict_window 1, instead of training a separate model per horizon.

Mechanism:
    1. Load the real train_window-length history.
    2. Predict day 1.
    3. Drop the oldest day from the window, append day 1's prediction as the
       newest day.
    4. Predict day 2 from this updated window. Repeat for as many
       --horizon_steps as requested.

This resolves the inconsistency seen with separately-trained per-horizon
models (a 1-day model and a 4-day model can disagree with each other, since
they're independent networks) -- here every day's forecast comes from the
same single model, applied repeatedly, so the trajectory is self-consistent
by construction.

TRADE-OFF: recursive forecasting compounds error. Day 3's prediction is built
on top of day 1 and day 2's (possibly wrong) predictions, so errors can grow
with each step -- unlike direct multi-step forecasting, which predicts every
horizon independently from only real, unmodified history.

--behavior_rollout_mode controls how the behavior series' history window is
extended at each step:
    'real'      -- append the ACTUAL real future behavior value for that day.
                   Only valid when backtesting against historical data (where
                   the true future already exists on record). Isolates PRICE
                   forecast error from behavior-forecast error.
    'predicted' -- append the model's own single-step behavior forecast
                   (be_y) instead. Genuine no-lookahead forecasting, usable
                   beyond the range of available historical data, but
                   compounds BOTH price and behavior forecast error together.

NOTE: as of this script, behavior_matrix_full in data_loader.py is a
placeholder hardcoded to all zeros, so behavior_y's content (real vs
predicted) currently has NO effect on the model's output either way --
only behavior_x's rolling (controlled by this flag) actually matters. Once
you implement a real CFB availability mask, the mode/matrix content will
start to matter for the "known in advance" portions too.

Usage:
    python train.py --predict_window 1 ...   (train once)
    python recursive_predict.py --anchor_date 2026-04-13 --horizon_steps 3 \
        --behavior_rollout_mode predicted --train_window 9 ...

Add --day_labels off to hide the dots and "Day N" labels (default: on).
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import torch

from train import build_setting, get_args
from exp.exp_craft import Exp_Craft
from utils.tools import visual


def find_anchor_index(dates, anchor_date):
    import pandas as pd

    dates = list(dates)
    try:
        normalized = [pd.Timestamp(d).normalize() for d in dates]
        target = pd.Timestamp(anchor_date).normalize()
    except Exception:
        normalized = dates
        target = anchor_date

    if target in normalized:
        return normalized.index(target)

    str_dates = [str(d) for d in dates]
    if str(anchor_date) in str_dates:
        return str_dates.index(str(anchor_date))

    lo = normalized[0] if normalized else "?"
    hi = normalized[-1] if normalized else "?"
    raise ValueError(
        f"anchor_date {anchor_date} not found in test_data.dates. "
        f"Test set covers {lo} to {hi} ({len(dates)} dates). "
        f"Pick a date within that range."
    )


def run_rollout(exp, test_data, sample_index, train_window, horizon_steps, behavior_mode):
    """
    Runs the recursive rollout starting from a real window, returning the
    list of predicted values (target column, one per day) and the model's
    own single-step behavior forecasts (for reference/debugging).
    """
    device = exp.device
    s_begin = sample_index
    s_end = s_begin + train_window

    batch_x = torch.tensor(
        test_data.data_x[s_begin:s_end], dtype=torch.float32
    ).unsqueeze(0).to(device)
    behavior_x = torch.tensor(
        test_data.data_be[s_begin:s_end], dtype=torch.float32
    ).unsqueeze(0).to(device)

    preds = []
    behavior_preds = []

    with torch.no_grad():
        for step in range(horizon_steps):
            r_begin = s_end + step
            r_end = r_begin + 1  # predict_window=1

            # behavior_y / behavior_matrix for this single step. Since
            # behavior_matrix_full is currently all-zero (see module
            # docstring), the actual content of behavior_y doesn't reach
            # the model's computation -- passing real data here is safe and
            # future-proofs this script once a real mask is implemented.
            if r_end <= len(test_data.data_be):
                behavior_y = torch.tensor(
                    test_data.data_be[r_begin:r_end], dtype=torch.float32
                ).unsqueeze(0).to(device)
                behavior_matrix = torch.tensor(
                    test_data.behavior_matrix_full[r_begin:r_end], dtype=torch.float32
                ).unsqueeze(0).to(device)
            else:
                # Past the end of available data (genuine future beyond
                # your dataset) -- no real behavior_y exists; zeros are safe
                # since they'd be masked out anyway, and there's nothing
                # else we could use.
                behavior_y = torch.zeros(1, 1, behavior_x.shape[-1], device=device)
                behavior_matrix = torch.zeros(1, 1, behavior_x.shape[-1], device=device)

            y, be_y, be_y_known = exp.model(batch_x, behavior_x, behavior_y, behavior_matrix)

            pred_value = y[0, 0, -1].item()  # target column, this step's forecast
            preds.append(pred_value)
            behavior_preds.append(be_y[0, 0].cpu().numpy().copy())

            # --- Roll batch_x forward: drop oldest day, append this prediction ---
            new_x_row = batch_x[:, -1:, :].clone()
            new_x_row[:, :, -1] = y[:, 0, -1]  # only the target column is the
                                                 # model's own forecast; other
                                                 # enc_in columns (if enc_in>1)
                                                 # hold at their last real value
            batch_x = torch.cat([batch_x[:, 1:, :], new_x_row], dim=1)

            # --- Roll behavior_x forward, per --behavior_rollout_mode ---
            if behavior_mode == "real" and r_end <= len(test_data.data_be):
                new_be_row = torch.tensor(
                    test_data.data_be[r_begin:r_end], dtype=torch.float32
                ).unsqueeze(0).to(device)
            else:
                new_be_row = be_y  # model's own single-step behavior forecast
            behavior_x = torch.cat([behavior_x[:, 1:, :], new_be_row], dim=1)

    return preds, behavior_preds


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--anchor_date", type=str, required=True,
                         help="Date of the last real lookback-window point")
    parser.add_argument("--horizon_steps", type=int, default=3,
                         help="How many days to recursively forecast")
    parser.add_argument(
        "--behavior_rollout_mode", type=str, default="predicted",
        choices=["real", "predicted"],
        help=(
            "How to extend the behavior series' history each step. "
            "'real': use actual real future behavior values (backtesting "
            "only -- requires the true future to already exist on record). "
            "'predicted': use the model's own single-step behavior forecast "
            "(genuine no-lookahead forecasting)."
        ),
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help=(
            "Output PDF path. If omitted, a unique name is generated: "
            "./pic/recursive_<anchor>_h<steps>_<mode>_<YYYYmmdd-HHMMSS>.pdf. "
            "An existing file is never overwritten: _2, _3, ... is appended."
        ),
    )
    parser.add_argument(
        "--day_labels", type=str, default="on", choices=["on", "off"],
        help=(
            "on (default): mark every forecast day with a dot and a 'Day N' label. "
            "off: plot only the lines (use for long horizons, where 100+ labels overlap)."
        ),
    )
    known_args, remaining = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + remaining
    args = get_args()

    if args.predict_window != 1:
        raise ValueError(
            f"recursive_predict.py requires a checkpoint trained with "
            f"--predict_window 1 (each recursive step predicts exactly one "
            f"day), but --predict_window {args.predict_window} was given. "
            f"Retrain with --predict_window 1 first."
        )

    exp = Exp_Craft(args)
    setting = build_setting(args)
    ckpt_path = f"{args.checkpoints}/{setting}/checkpoint.pth"
    exp.model.load_state_dict(torch.load(ckpt_path, map_location=exp.device))
    exp.model.eval()

    test_data, _ = exp._get_data(flag="test")
    dates = test_data.dates
    anchor_idx = find_anchor_index(dates, known_args.anchor_date)
    sample_index = anchor_idx - args.train_window + 1
    if sample_index < 0:
        raise ValueError(
            f"anchor_date {known_args.anchor_date} is too close to the "
            f"start of the test set for train_window={args.train_window}"
        )

    preds, behavior_preds = run_rollout(
        exp, test_data, sample_index, args.train_window,
        known_args.horizon_steps, known_args.behavior_rollout_mode,
    )

    s_begin = sample_index
    s_end = s_begin + args.train_window
    history = test_data.data_x[s_begin:s_end, -1].copy()
    dates_history = dates[s_begin:s_end]

    n_true_available = min(known_args.horizon_steps, len(test_data.data_x) - s_end)
    true_future = test_data.data_x[s_end: s_end + n_true_available, -1].copy()
    dates_future = dates[s_end: s_end + known_args.horizon_steps]

    # visual() expects true/preds of the SAME length (it concatenates history
    # + each and plots them together) -- pad true_future with NaN past what's
    # actually on record, so the ground-truth line simply stops there.
    if n_true_available < known_args.horizon_steps:
        pad = np.full(known_args.horizon_steps - n_true_available, np.nan)
        true_future_padded = np.concatenate([true_future, pad])
    else:
        true_future_padded = true_future

    gt = np.concatenate([history, true_future_padded])
    pd_values = np.concatenate([history, np.array(preds)])
    all_dates = list(dates_history) + list(dates_future)

    # Mark each recursively-forecast day with a dot + "Day N" label, since
    # the prediction line alone doesn't show where each individual step's
    # forecast actually landed.
    if known_args.day_labels == "on":
        point_labels = [
            (dates_future[i], preds[i], f'Day {i + 1}')
            for i in range(known_args.horizon_steps)
        ]
    else:
        point_labels = None  # visual() then draws just the two lines

    out_path = known_args.out
    if out_path is None:
        anchor_tag = str(known_args.anchor_date).replace("/", "-").replace(" ", "")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = (f"./pic/recursive_{anchor_tag}_h{known_args.horizon_steps}_"
                    f"{known_args.behavior_rollout_mode}_{stamp}.pdf")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    if os.path.exists(out_path):                      # never overwrite an earlier result
        stem, ext = os.path.splitext(out_path)
        k = 2
        while os.path.exists(f"{stem}_{k}{ext}"):
            k += 1
        out_path = f"{stem}_{k}{ext}"

    visual(gt, pd_values, dates=all_dates, name=out_path, point_labels=point_labels)

    print(f"Saved recursive rollout plot to {out_path}")
    print(f"Predicted values (day 1..{known_args.horizon_steps}): {preds}")
    if n_true_available < known_args.horizon_steps:
        print(
            f"Note: only {n_true_available} of {known_args.horizon_steps} "
            f"future days have real ground truth on record -- the rest are "
            f"genuine forecasts beyond your available data."
        )


if __name__ == "__main__":
    main()