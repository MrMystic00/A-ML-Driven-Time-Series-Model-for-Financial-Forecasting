"""
Expected shapes per sample (before batching):
    seq_x            : [train_window,   enc_in]
    seq_y            : [predict_window, enc_in]
    behavior_seq_x   : [train_window,   enc_in_be]
    behavior_seq_y   : [predict_window, enc_in_be]
    behavior_matrix  : [predict_window, enc_in_be]  binary, 1 = known at
                                                      inference time

Samples are served in chronological order (the DataLoader does not shuffle).
"""

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


def _parse_dates(col):
    """
    The project's CSVs use two date styles: ISO (2024-01-02, written by the data
    notebook) and day-first (2/1/2024 and 16/1/2024, as in decomposition.csv).
    A bare pd.to_datetime guesses month-first from the first row, so 2/1/2024 is
    read as 1 Feb and the file then fails at the first day above 12. ISO is parsed
    as ISO; anything else is read day-first, the project convention.
    """
    s = col.astype(str).str.strip()
    try:
        if s.str.match(r"^\d{4}-\d{2}-\d{2}").all():
            return pd.to_datetime(s)
        return pd.to_datetime(s, dayfirst=True)
    except ValueError as e:
        raise ValueError(
            "Could not parse the 'Date' column (examples: "
            f"{s.iloc[:3].tolist()}). Use YYYY-MM-DD or D/M/YYYY. Original error: {e}"
        ) from e


class Dataset_Craft(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 data_path='data.csv', behavior_data_path='behavior.csv',
                 scale=True):
        self.args = args
        self.train_window = size[0] if size else args.train_window
        self.predict_window = size[1] if size else args.predict_window

        assert flag in ['train', 'val', 'test', 'pred']
        self.flag = flag
        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path
        self.behavior_data_path = behavior_data_path

        # Data augmentation (train split only). Only 'jitter' is supported,
        # so this is just an on/off flag -- no per-copy recipe system needed
        # (that only made sense when multiple distinct techniques existed).
        aug_types_str = getattr(args, "augmentation_types", "none").strip().lower()
        if aug_types_str not in ("none", "", "jitter"):
            raise ValueError(
                f"Unknown augmentation_types {aug_types_str!r}; only "
                f"'jitter' or 'none' is supported."
            )
        self.aug_enabled = aug_types_str == "jitter"

        self.aug_mode = getattr(args, "augmentation_mode", "online")
        self.aug_jitter_std = getattr(args, "aug_jitter_std", 0.01)
        self.n_aug_copies = max(0, getattr(args, "n_augmented_copies", 1))
        self.aug_copy_fraction = min(1.0, max(0.0, getattr(args, "aug_copy_fraction", 1.0)))

        self.__read_data__()
        self._setup_aug_copy_indices()

    def _setup_aug_copy_indices(self):
        """
        Precomputes WHICH training windows get an augmented copy, evenly
        spaced across the training timeline (not just the first N windows,
        which would bias the augmented subset toward one time period).
        A fraction of 1.0 selects every window (original behavior).
        """
        base_len = self._base_len()
        n_selected = int(round(base_len * self.aug_copy_fraction))
        n_selected = max(0, min(base_len, n_selected))

        if n_selected >= base_len:
            self.aug_copy_indices = list(range(base_len))
        elif n_selected <= 0:
            self.aug_copy_indices = []
        else:
            self.aug_copy_indices = sorted(
                set(np.linspace(0, base_len - 1, n_selected, dtype=int).tolist())
            )

    def __read_data__(self):
        main_path = f"{self.root_path}/{self.data_path}"
        behavior_path = f"{self.root_path}/{self.behavior_data_path}"

        df_raw = pd.read_csv(main_path)
        df_be = pd.read_csv(behavior_path)

        # Save dates separately for plotting only
        if "Date" in df_raw.columns:
            df_raw["Date"] = _parse_dates(df_raw["Date"])
            dates = df_raw["Date"].to_numpy()
        else:
            dates = None

        # Keep model inputs exactly as before
        df_raw_numeric = df_raw.select_dtypes(include=[np.number])
        df_be_numeric = df_be.select_dtypes(include=[np.number])

        if len(df_raw_numeric) != len(df_be_numeric):
            raise ValueError(
                "The main and behavior CSV files must contain the "
                "same number of rows."
            )

        if df_raw_numeric.shape[1] != self.args.enc_in:
            raise ValueError(
                f"enc_in={self.args.enc_in}, but the main CSV has "
                f"{df_raw_numeric.shape[1]} numeric columns."
            )

        if df_be_numeric.shape[1] != self.args.enc_in_be:
            raise ValueError(
                f"enc_in_be={self.args.enc_in_be}, but the behavior CSV has "
                f"{df_be_numeric.shape[1]} numeric columns."
            )

        raw_values = df_raw_numeric.to_numpy(dtype=np.float32)
        behavior_values = df_be_numeric.to_numpy(dtype=np.float32)

        self.behavior_matrix_full = (
            ~np.isnan(behavior_values)
        ).astype(np.float32)

        behavior_values = np.nan_to_num(
            behavior_values,
            nan=0.0,
        )

        total_length = len(raw_values)

        train_size = getattr(self.args, "train_size", 0.7)
        val_size = getattr(self.args, "val_size", 0.1)
        test_size = getattr(self.args, "test_size", 0.2)

        if train_size <= 1 and val_size <= 1 and test_size <= 1:
            # Fractional split (e.g. 0.7 / 0.1 / 0.2)
            train_end = int(total_length * train_size)
            val_end = train_end + int(total_length * val_size)
        else:
            # Absolute row-count split (e.g. 500 / 100 / 150)
            train_end = int(train_size)
            val_end = train_end + int(val_size)

        if val_end > total_length:
            raise ValueError(
                f"train_size + val_size ({train_end} + {val_end - train_end} = "
                f"{val_end}) exceeds the dataset length ({total_length}). "
                f"Reduce train_size/val_size/test_size."
            )

        # Include historical overlap so the first validation/test window has
        # enough preceding observations. Defines with rows are used in training, validation and training phase 
        borders = {
            "train": (
                0,
                train_end,
            ),
            "val": (
                max(0, train_end - self.train_window),
                val_end,
            ),
            "test": (
                max(0, val_end - self.train_window),
                total_length,
            ),
            "pred": (
                max(
                    0,
                    total_length
                    - self.train_window
                    - self.predict_window,
                ),
                total_length,
            ),
        }

        start, end = borders[self.flag]

        self.data_x = raw_values[start:end]
        self.data_be = behavior_values[start:end]

        # Dates are stored only for test plotting
        if dates is not None:
            self.dates = dates[start:end]
        else:
            self.dates = None

        # Temporary mask.
        # Replace this with the actual CFB availability mask.
        self.behavior_matrix_full = np.zeros_like(
            self.data_be,
            dtype=np.float32,
        )

    def _base_len(self):
        return len(self.data_x) - self.train_window - self.predict_window + 1

    def _duplicate_active(self):
        return (
            self.flag == "train"
            and self.aug_mode == "duplicate"
            and self.aug_enabled
            and self.n_aug_copies > 0
            and len(self.aug_copy_indices) > 0
        )

    def __getitem__(self, index):
        base_len = self._base_len()

        if self._duplicate_active():
            # Block layout: [0, base_len) = ALL clean originals.
            # [base_len, base_len + m) = augmented copy #1, but only over
            # the SELECTED subset (m = len(aug_copy_indices), m <= base_len
            # when aug_copy_fraction < 1.0). Each subsequent copy block is
            # also size m. Within a copy block, order follows
            # aug_copy_indices (already sorted -> chronological order
            # preserved, so shuffle=False batching stays group-consistent).
            m = len(self.aug_copy_indices)
            if index < base_len:
                original_index = index
                apply_aug = False
            else:
                offset = index - base_len
                pos_in_block = offset % m
                original_index = self.aug_copy_indices[pos_in_block]
                apply_aug = True
        else:
            original_index = index
            apply_aug = (
                self.flag == "train"
                and self.aug_mode == "online"
                and self.aug_enabled
            )

        s_begin = original_index
        s_end = s_begin + self.train_window
        r_begin = s_end
        r_end = r_begin + self.predict_window

        seq_x = self.data_x[s_begin:s_end].copy()
        seq_y = self.data_x[r_begin:r_end].copy()
        behavior_seq_x = self.data_be[s_begin:s_end].copy()
        behavior_seq_y = self.data_be[r_begin:r_end].copy()
        behavior_matrix = self.behavior_matrix_full[r_begin:r_end]

        if apply_aug:
            # Pure noise -- independent per series is fine, since it's meant
            # to emulate measurement noise, not a market-level shift.
            seq_x = seq_x + np.random.normal(
                0, self.aug_jitter_std * (np.abs(seq_x).mean() + 1e-6), seq_x.shape
            )
            behavior_seq_x = behavior_seq_x + np.random.normal(
                0, self.aug_jitter_std * (np.abs(behavior_seq_x).mean() + 1e-6),
                behavior_seq_x.shape,
            )

            seq_x = seq_x.astype(np.float32)
            seq_y = seq_y.astype(np.float32)
            behavior_seq_x = behavior_seq_x.astype(np.float32)
            behavior_seq_y = behavior_seq_y.astype(np.float32)

        return seq_x, seq_y, behavior_seq_x, behavior_seq_y, behavior_matrix

    def __len__(self):
        base_len = self._base_len()
        if self._duplicate_active():
            return base_len + self.n_aug_copies * len(self.aug_copy_indices)
        return base_len