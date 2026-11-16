import numpy as np
import torch
import matplotlib.pyplot as plt

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    else:
        lr_adjust = {}

    if epoch in lr_adjust:
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {:.5g}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.5f} --> {val_loss:.5f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


def visual(true, preds=None, dates=None, name='./pic/test.pdf', point_labels=None):
    """
    point_labels: optional list of (x, y, text) tuples -- draws a dot and a
    text label at each specified point (e.g. marking each recursively
    forecast day as "Day 1", "Day 2", ... directly on the prediction line,
    useful when overlapping lines would otherwise hide each other).
    Defaults to None, leaving all existing behavior unchanged.
    """
    plt.figure(figsize=(12, 5))

    if dates is None:
        x = np.arange(len(true))
    else:
        x = dates

    plt.plot(
        x,
        true,
        label='GroundTruth',
        linewidth=2,
        zorder=3,
    )

    if preds is not None:
        plt.plot(
            x,
            preds,
            label='Prediction',
            linewidth=2,
            zorder=2,
        )

    if point_labels:
        for px, py, label in point_labels:
            plt.scatter([px], [py], color='#2a2a2a', zorder=5, s=28)
            plt.annotate(
                label,
                (px, py),
                textcoords='offset points',
                xytext=(0, 9),
                ha='center',
                fontsize=8,
                color='#2a2a2a',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.75),
            )

    plt.xlabel("Date")
    plt.ylabel("S&P 500 Adjusted Close")
    plt.legend()

    if dates is not None:
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(name, bbox_inches='tight')
    plt.close()


_ORDINAL_WORDS = [
    None, 'First', 'Second', 'Third', 'Fourth', 'Fifth', 'Sixth', 'Seventh',
    'Eighth', 'Ninth', 'Tenth', 'Eleventh', 'Twelfth', 'Thirteenth',
    'Fourteenth', 'Fifteenth', 'Sixteenth', 'Seventeenth', 'Eighteenth',
    'Nineteenth', 'Twentieth',
]


def _ordinal_word(n):
    """1 -> 'First', 2 -> 'Second', ... falls back to a numeric ordinal
    (e.g. '21st') beyond the spelled-out list above."""
    if 0 < n < len(_ORDINAL_WORDS):
        return _ORDINAL_WORDS[n]
    suffix = 'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f'{n}{suffix}'


# Distinct, high-contrast colors for up to ~8 horizons. Extend if needed.
_HORIZON_COLORS = [
    '#e6550d',  # orange
    '#31a354',  # green
    '#756bb1',  # purple
    '#e7298a',  # pink
    '#1f78b4',  # light blue
    '#b15928',  # brown
    '#666666',  # gray
]


def visual_multi_horizon(
    history,
    true_future,
    pred_dict,
    dates_history=None,
    dates_future=None,
    name='./pic/multi_horizon.pdf',
    ylabel='S&P 500 Adjusted Close',
):
    """
    history:     1D array, ground-truth values for the L-length lookback window.
    true_future: 1D array, ground-truth values for the P-length future window
                 (use the longest horizon you have, e.g. 4 days).
    pred_dict:   dict {horizon_length (int) -> 1D array of predicted values},
                 e.g. {1: [...], 2: [...], 3: [...], 4: [...]}.
                 Each array's length must equal its key.
    dates_history / dates_future: optional x-axis labels (e.g. date strings).
                 If omitted, integer positions are used.
    name:        output file path.
    """
    history = np.asarray(history).reshape(-1)
    true_future = np.asarray(true_future).reshape(-1)

    L = len(history)
    P = len(true_future)

    if dates_history is not None and dates_future is not None:
        x_hist = list(dates_history)
        x_future = list(dates_future)
    else:
        x_hist = list(np.arange(L))
        x_future = list(np.arange(L, L + P))

    plt.figure(figsize=(12, 5))

    # --- Ground truth: one continuous line across lookback + true future ---
    gt_x = x_hist + x_future
    gt_y = np.concatenate([history, true_future])
    plt.plot(gt_x, gt_y, color='navy', linewidth=2.2, label='Ground Truth', zorder=3)

    # --- Each prediction horizon branches off the last historical point ---
    join_x, join_y = x_hist[-1], history[-1]

    for idx, h in enumerate(sorted(pred_dict.keys())):
        pred_vals = np.asarray(pred_dict[h]).reshape(-1)
        if len(pred_vals) != h:
            raise ValueError(
                f"pred_dict[{h}] has length {len(pred_vals)}, expected {h}"
            )
        if h > len(x_future):
            raise ValueError(
                f"horizon {h} exceeds available future x-positions ({len(x_future)}); "
                f"pass a longer dates_future/true_future."
            )
        px = [join_x] + x_future[:h]
        py = np.concatenate([[join_y], pred_vals])
        color = _HORIZON_COLORS[idx % len(_HORIZON_COLORS)]
        plt.plot(px, py, color=color, linewidth=2, label=f'{_ordinal_word(h)} Day Prediction', zorder=2)

    plt.xlabel('Date')
    plt.ylabel(ylabel)
    plt.legend()

    if dates_history is not None:
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(name, bbox_inches='tight')
    plt.close()