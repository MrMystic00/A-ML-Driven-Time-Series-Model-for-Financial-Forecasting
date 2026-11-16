import argparse
import os
import random

import numpy as np
import torch

from exp.exp_craft import Exp_Craft


def get_args():
    parser = argparse.ArgumentParser(
        description="Train the CRAFT financial forecasting model"
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="CRAFT",
        choices=["CRAFT"],
    )

    # Data
    parser.add_argument(
        "--root_path",
        type=str,
        default="./dataset",
        help="Folder containing the CSV files",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="AAPL_Adj_Close.csv",
        help="Main target-series CSV",
    )
    parser.add_argument(
        "--behavior_data_path",
        type=str,
        default="AAPL_Behavior.csv",
        help="Cross-future-behaviour CSV",
    )

    # Train / validation / test split
    parser.add_argument(
        "--train_size",
        type=float,
        default=0.7,
        help="Fraction (0-1) or absolute row count (>1) used for training",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.1,
        help="Fraction (0-1) or absolute row count (>1) used for validation",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Fraction (0-1) or absolute row count (>1) used for testing",
    )

    # Data augmentation (training set only)
    parser.add_argument(
        "--augmentation_types",
        type=str,
        default="none",
        choices=["none", "jitter"],
        help="Augmentation to apply to the TRAINING set only ('none' to disable, default).",
    )
    parser.add_argument(
        "--aug_jitter_std",
        type=float,
        default=0.01,
        help="Jitter noise std, as a fraction of each series' mean magnitude",
    )
    parser.add_argument(
        "--augmentation_mode",
        type=str,
        default="online",
        choices=["online", "duplicate"],
        help=(
            "online (default): each training index is re-augmented with a "
            "fresh random draw every epoch, dataset length unchanged. "
            "duplicate: the dataset is extended so every epoch contains "
            "BOTH the original clean window AND --n_augmented_copies "
            "freshly-augmented copies of it, all seen together each epoch."
        ),
    )
    parser.add_argument(
        "--n_augmented_copies",
        type=int,
        default=1,
        help="duplicate mode only: how many augmented copies per original training window",
    )
    parser.add_argument(
        "--aug_copy_fraction",
        type=float,
        default=1.0,
        help=(
            "duplicate mode only: fraction (0-1] of training windows that "
            "get an augmented copy. 1.0 (default) = every window. E.g. 0.3 "
            "= only 30%% of windows get copies, evenly spread across the "
            "training timeline; the rest appear only once (clean)."
        ),
    )

    # Window sizes
    parser.add_argument(
        "--train_window",
        type=int,
        default=60,
        help="Number of historical observations",
    )
    parser.add_argument(
        "--predict_window",
        type=int,
        default=5,
        help="Number of future observations to predict",
    )

    # Input dimensions
    parser.add_argument(
        "--enc_in",
        type=int,
        default=1,
        help="Number of columns in the main input series",
    )
    parser.add_argument(
        "--enc_in_be",
        type=int,
        default=1,
        help="Number of behaviour-series columns",
    )

    # Deprecated: the External Trend Guide (hierarchical) module was removed from
    # this framework. Accepted and ignored so older command lines keep working.
    parser.add_argument(
        "--node_number",
        type=int,
        default=1,
        help=argparse.SUPPRESS,
    )

    # Training
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--train_epochs",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--convergence_threshold",
        type=float,
        default=1e-5,
        help=(
            "Minimum improvement in validation loss that counts as progress "
            "(early-stopping min_delta). If an epoch does not beat the best "
            "validation loss by at least this much, the early-stopping "
            "counter is incremented; training stops after --patience such "
            "epochs in a row. Default 1e-5. 0.0 means any improvement, however "
            "small, resets the counter."
        ),
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
    )

    # Loss
    parser.add_argument(
        "--loss",
        type=str,
        default="mse",
        choices=["mse"],
    )
    parser.add_argument(
        "--main_weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--behavior_weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--known_weight",
        type=float,
        default=1.0,
    )

    # Regularization
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.05,
        help="Dropout probability inside each MLP encoder/decoder in CRAFT",
    )
    parser.add_argument(
        "--hidden_layers",
        type=int,
        default=2,
        help=(
            "Number of Linear layers per MLP (input+output count as 2; "
            "e.g. 2 = no hidden layers beyond input/output, 3 = one hidden layer)"
        ),
    )
    parser.add_argument(
        "--ridge_alpha",
        type=float,
        default=0.1,
        help="L2 penalty used in the closed-form Koopman ridge regression solve",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="L2 weight decay passed to the optimizer",
    )
    parser.add_argument(
        "--behavior_norm",
        type=str,
        default="column",
        choices=["column", "global"],
        help=("column (default): each behavior column is normalised on its own scale. "
              "global: legacy -- one mean/std for the whole behavior tensor, which crushes "
              "small-scale columns (VIX, macro, flags) next to price-scale ones."),
    )
    parser.add_argument(
        "--behavior_relu",
        type=str,
        default="known_only",
        choices=["both", "known_only", "none"],
        help=("Which behavior outputs are clamped at 0 by a final ReLU. both = legacy "
              "(a clamped output that goes negative gets no gradient and stays 0). "
              "known_only (default): clamp only the known-behavior head, whose target is 0. "
              "none: no clamp. The price output is always clamped."),
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "nadam"],
        help=(
            "adam: standard Adam. nadam: Adam with Nesterov momentum "
            "(estimates the gradient at the anticipated next position "
            "rather than the current one)"
        ),
    )
    parser.add_argument(
        "--momentum_decay",
        type=float,
        default=4e-3,
        help="NAdam-only: controls how the momentum coefficient itself decays over time",
    )

    # Output-feature selection used by Exp_Craft
    parser.add_argument(
        "--features",
        type=str,
        default="M",
        choices=["M", "S", "MS"],
        help="M: multivariate, S: univariate, MS: multivariate-to-univariate",
    )

    # Learning-rate adjustment
    parser.add_argument(
        "--lradj",
        type=str,
        default="type1",
        choices=["type1", "type2", "none"],
    )

    # Checkpoints
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="./checkpoints",
    )

    # Hardware
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Use CUDA when available",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--use_multi_gpu",
        action="store_true",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="0",
    )

    # Execution
    parser.add_argument(
        "--is_training",
        type=int,
        default=1,
        choices=[0, 1],
        help="1 trains and tests; 0 only tests a saved model",
    )
    parser.add_argument(
        "--do_predict",
        action="store_true",
        help="Generate predictions after training/testing",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()
    if args.node_number != 1:
        print("WARNING: --node_number is deprecated (the External Trend Guide module was "
              "removed) and is ignored.")
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_device_args(args):
    # Do not request a GPU when CUDA is unavailable.
    args.use_gpu = bool(args.use_gpu and torch.cuda.is_available())

    if args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        args.device_ids = [
            int(device_id)
            for device_id in args.devices.split(",")
        ]
        args.gpu = args.device_ids[0]
    else:
        args.device_ids = [args.gpu]


def format_split_size(value):
    """
    Formats a train/val/test size for use in a folder name.
    Fractions (<=1) are shown as whole-number percentages (0.7 -> '70').
    Absolute row counts (>1) are shown as-is (500 -> '500').
    """
    if value <= 1:
        return str(int(round(value * 100)))
    return str(int(value))


def build_setting(args):
    setting = (
        f"CRAFT"
        f"_tw{args.train_window}"
        f"_pw{args.predict_window}"
        f"_ei{args.enc_in}"
        f"_be{args.enc_in_be}"
        f"_bs{args.batch_size}"
        f"_lr{args.learning_rate}"
        f"_mw{args.main_weight}"
        f"_bw{args.behavior_weight}"
        f"_kw{args.known_weight}"
        f"_dp{args.dropout}"
        f"_hl{args.hidden_layers}"
        f"_ra{args.ridge_alpha}"
        f"_wd{args.weight_decay}"
        f"_opt{args.optimizer}"
        f"_split{format_split_size(args.train_size)}"
        f"-{format_split_size(args.val_size)}"
        f"-{format_split_size(args.test_size)}"
    )

    # Behavior-handling suffix, omitted only for the legacy combination so old
    # checkpoint folders keep their names and new runs never reuse them.
    bnorm = getattr(args, "behavior_norm", "global")
    brelu = getattr(args, "behavior_relu", "both")
    if not (bnorm == "global" and brelu == "both"):
        setting += f"_bn-{bnorm}_br-{brelu}"

    # Only append augmentation info when augmentation is actually enabled, so
    # existing checkpoint paths for runs without augmentation stay unchanged.
    aug_types = getattr(args, "augmentation_types", "none")
    if aug_types and aug_types.lower() != "none":
        mode = getattr(args, "augmentation_mode", "online")
        safe_aug = aug_types.replace(";", "__").replace(",", "+")
        setting += f"_aug-{safe_aug}-{mode}"
        if mode == "duplicate" and ";" not in aug_types:
            setting += f"x{getattr(args, 'n_augmented_copies', 1)}"

    return setting


def main():
    args = get_args()
    set_seed(args.seed)
    configure_device_args(args)

    os.makedirs(args.checkpoints, exist_ok=True)

    fractional = [args.train_size <= 1, args.val_size <= 1, args.test_size <= 1]
    if any(fractional) and not all(fractional):
        raise ValueError(
            "train_size/val_size/test_size must be all fractions (<=1) or all "
            "absolute row counts (>1) -- mixing the two is not supported."
        )
    if all(fractional) and abs(args.train_size + args.val_size + args.test_size - 1.0) > 1e-6:
        raise ValueError(
            f"train_size + val_size + test_size must sum to 1.0 when given as "
            f"fractions (got {args.train_size + args.val_size + args.test_size})."
        )

    setting = build_setting(args)

    print("\nExperiment configuration")
    print("-" * 60)
    for name, value in vars(args).items():
        print(f"{name}: {value}")
    print("-" * 60)
    print(f"Setting: {setting}\n")

    experiment = Exp_Craft(args)

    if args.is_training == 1:
        print("Starting training...")
        experiment.train(setting)

        print("\nStarting testing...")
        experiment.test(setting)
    else:
        print("Loading checkpoint and testing...")
        experiment.test(setting, test=1)

    if args.do_predict:
        print("\nGenerating predictions...")
        experiment.predict(setting, load=True)

    print("\nFinished.")


if __name__ == "__main__":
    main()