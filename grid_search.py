"""
Hyperparameter grid search for CRAFT.

Reuses train.py's get_args()/set_seed()/configure_device_args()/build_setting()
so every combination trains and checkpoints exactly like a normal `python
train.py ...` run would -- grid_search.py just loops over combinations and
compares validation loss at the end.

Usage:
    python grid_search.py \
        --root_path ./dataset --data_path SP500_stock.csv \
        --behavior_data_path SP500_behaviour.csv \
        --train_size 0.7 --val_size 0.1 --test_size 0.2 \
        --enc_in 1 --enc_in_be 5 \
        --learning_rate_grid 1e-3,1e-2 \
        --batch_size_grid 16,32 \
        --dropout_grid 0.0,0.1 \
        --train_epochs 20

Any flag ending in "_grid" takes a comma-separated list and is swept.
Any flag WITHOUT "_grid" (e.g. --train_epochs, --patience, --root_path) is
just a normal train.py flag, held fixed across every combination.
If you omit a *_grid flag entirely, that hyperparameter is held fixed at
whatever value/default the matching non-grid flag has (e.g. --learning_rate).

To sweep exact train/val/test splits (instead of the cross product of the three
*_size_grid flags), use e.g.:
    --split_grid 80/10/10,70/15/15,70/20/10,70/10/20,60/20/20,60/15/25,60/25/15,50/25/25

Results (one row per combination) are saved to a NEW, time-stamped Excel
workbook every run: ./grid_results/<run_name>_YYYYmmdd_HHMMSS.xlsx (sheets:
Runs, SeedAvg when seeds are swept, Setup), plus a CSV with the same name.
Use --run_name to label a run (e.g. --run_name cfb_groups), --results_dir to
change the folder, or --results_csv to force a specific CSV path.
"""

import argparse
import copy
import csv
import datetime
import itertools
import math
import os
import random
import statistics
import sys

from train import build_setting, configure_device_args, get_args, set_seed
from exp.exp_craft import Exp_Craft


GRID_SPEC = [
    # (grid flag name, base args attribute, type)
    ("learning_rate_grid", "learning_rate", float),
    ("batch_size_grid", "batch_size", int),
    ("train_window_grid", "train_window", int),
    ("predict_window_grid", "predict_window", int),
    ("main_weight_grid", "main_weight", float),
    ("behavior_weight_grid", "behavior_weight", float),
    ("known_weight_grid", "known_weight", float),
    ("train_size_grid", "train_size", float),
    ("val_size_grid", "val_size", float),
    ("test_size_grid", "test_size", float),
    ("dropout_grid", "dropout", float),
    ("hidden_layers_grid", "hidden_layers", int),
    ("ridge_alpha_grid", "ridge_alpha", float),
    ("weight_decay_grid", "weight_decay", float),
    ("optimizer_grid", "optimizer", str),
    ("momentum_decay_grid", "momentum_decay", float),
    ("aug_copy_fraction_grid", "aug_copy_fraction", float),
    ("augmentation_types_grid", "augmentation_types", str),
    ("augmentation_mode_grid", "augmentation_mode", str),
    ("aug_jitter_std_grid", "aug_jitter_std", float),
    ("behavior_norm_grid", "behavior_norm", str),
    ("behavior_relu_grid", "behavior_relu", str),
    ("convergence_threshold_grid", "convergence_threshold", float),
    ("seed_grid", "seed", int),
]


def parse_grid_args():
    parser = argparse.ArgumentParser(add_help=False)
    for grid_flag, _, _ in GRID_SPEC:
        parser.add_argument(f"--{grid_flag}", type=str, default=None,
                             help="Comma-separated values to sweep")
    parser.add_argument("--results_csv", type=str, default=None,
                         help="Force a specific CSV path (default: auto-named next to the Excel file)")
    parser.add_argument("--results_dir", type=str, default="./grid_results",
                         help="Folder for the time-stamped Excel/CSV results")
    parser.add_argument("--run_name", type=str, default="grid_search",
                         help="Label used at the start of the result file names")
    parser.add_argument("--split_grid", type=str, default=None,
                         help="Exact train/val/test split triples, comma-separated, e.g. "
                              "80/10/10,70/15/15,60/20/20 (percentages summing to 100, or "
                              "fractions summing to 1). Runs ONLY these triples instead of "
                              "the cross product of --train/val/test_size_grid.")
    parser.add_argument("--random_seeds", type=int, default=None,
                         help="Draw N random seeds instead of listing them (see --seed_grid)")
    # NOTE: deliberately not named --seed_*: argparse prefix-matches "--seed" and
    # a second --seed* flag would make a plain "--seed 42" ambiguous.
    parser.add_argument("--rng_master_seed", type=int, default=None,
                         help="Seed for drawing the random seeds, to reproduce a --random_seeds run")
    known, remaining = parser.parse_known_args()
    return known, remaining


def parse_split_grid(s):
    """
    "80/10/10,70/15/15" -> [(0.8, 0.1, 0.1), (0.7, 0.15, 0.15)].
    Each triple must sum to 100 (percentages) or to 1 (fractions).
    """
    triples = []
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split("/")
        if len(parts) != 3:
            raise SystemExit(f"--split_grid entry {item!r} must look like 70/15/15 (train/val/test).")
        try:
            tr, va, te = (float(p) for p in parts)
        except ValueError:
            raise SystemExit(f"--split_grid entry {item!r} contains a non-numeric value.")
        total = tr + va + te
        if abs(total - 100.0) < 1e-6:
            tr, va, te = tr / 100.0, va / 100.0, te / 100.0
        elif abs(total - 1.0) > 1e-6:
            raise SystemExit(f"--split_grid entry {item!r} sums to {total:g}; it must sum to 100 (or 1).")
        triples.append((tr, va, te))
    if not triples:
        raise SystemExit("--split_grid is empty.")
    return triples


def to_list(s, cast):
    """
    Parses a *_grid value into a list of `cast`-typed values.

    Three formats are accepted:
      - comma-separated: "1e-3,5e-3,1e-2"          -> [0.001, 0.005, 0.01]
      - linear range "start:stop:step" (inclusive of stop):
            "1e-2:1e-3:-1e-3" -> [0.01, 0.009, ..., 0.001]
      - log-scale range "start:stop:logN" (N points, evenly spaced in log
        space -- use this for regularization terms like weight_decay/
        ridge_alpha where a wide linear sweep would badly under-sample the
        small end):
            "1e-6:1e-1:log5" -> [1e-6, ~3.16e-5, ~1e-3, ~3.16e-2, 1e-1]
    """
    if s is None:
        return None

    if ":" in s:
        parts = s.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"Range grid value {s!r} must have exactly 3 parts "
                f"(start:stop:step or start:stop:logN), e.g. "
                f"'1e-2:1e-3:-1e-3' or '1e-6:1e-1:log5'"
            )
        start_str, stop_str, third = parts

        if third.lower().startswith("log"):
            n_str = third[3:]
            if not n_str.isdigit() or int(n_str) < 2:
                raise ValueError(
                    f"Log-scale range {s!r} must end in 'log' followed by an "
                    f"integer point count >= 2, e.g. '1e-6:1e-1:log5'"
                )
            n_points = int(n_str)
            start, stop = float(start_str), float(stop_str)
            if start <= 0 or stop <= 0:
                raise ValueError(
                    f"Log-scale range {s!r} requires positive start and stop "
                    f"(can't take log of 0 or a negative number)"
                )
            ratio = (stop / start) ** (1.0 / (n_points - 1))
            values = [start * (ratio ** i) for i in range(n_points)]
            return [cast(round(v, 15)) for v in values]

        start, stop, step = (cast(p) for p in parts)
        if step == 0:
            raise ValueError(f"Range step cannot be 0 in {s!r}")

        n_steps = int(round((stop - start) / step)) + 1
        if n_steps <= 0:
            raise ValueError(
                f"Range {s!r} produces no values -- check that step's sign "
                f"matches the direction from start to stop"
            )

        values = []
        for i in range(n_steps):
            v = start + i * step
            if (step > 0 and v > stop + abs(step) * 1e-9) or \
               (step < 0 and v < stop - abs(step) * 1e-9):
                break
            values.append(round(v, 10))
        return values

    return [cast(x) for x in s.split(",")]


def aggregate_over_seeds(results, keys):
    """
    Groups result rows that share every hyperparameter except `seed`, and
    returns one row per group with the mean and (sample) std of the metrics.
    """
    group_keys = [k for k in keys if k != "seed"]
    metrics = ["epochs_trained", "vali_loss", "vali_acc", "test_loss", "test_acc"]
    groups = {}
    for r in results:
        groups.setdefault(tuple(r[k] for k in group_keys), []).append(r)

    rows = []
    for gk, rs in groups.items():
        row = dict(zip(group_keys, gk))
        row["n_seeds"] = len(rs)
        for m in metrics:
            vals = [r[m] for r in rs if r[m] is not None]
            if vals:
                row[f"{m}_mean"] = statistics.mean(vals)
                row[f"{m}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
            else:
                row[f"{m}_mean"] = None
                row[f"{m}_std"] = None
        rows.append(row)
    return rows


def unique_path(path):
    """Returns `path`, or path_2, path_3, ... if it already exists (never overwrites)."""
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    i = 2
    while os.path.exists(f"{base}_{i}{ext}"):
        i += 1
    return f"{base}_{i}{ext}"


def save_excel(path, runs, seedavg, setup_items):
    """
    Writes an .xlsx workbook: 'Runs' (one row per run), 'SeedAvg' (if given)
    and 'Setup' (command line, swept and fixed settings). Returns True on success.
    """
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.utils import get_column_letter
    except ImportError:
        print("openpyxl is not installed (pip install openpyxl): skipped the "
              "Excel file; the CSV was still saved.")
        return False

    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill("solid", start_color="4472C4", end_color="4472C4")

    def number_format(col):
        c = col.lower()
        if "acc" in c:
            return "0.0%"
        if "loss" in c:
            return "0.00000"
        return None

    def write_table(ws, rows):
        cols = list(rows[0].keys())
        ws.append(cols)
        for r in rows:
            ws.append([r.get(c) for c in cols])
        for j, c in enumerate(cols, start=1):
            head = ws.cell(row=1, column=j)
            head.font, head.fill = header_font, header_fill
            head.alignment = Alignment(horizontal="center", wrap_text=True)
            widest = max([len(str(c))] + [len(str(r.get(c, ""))) for r in rows])
            ws.column_dimensions[get_column_letter(j)].width = min(widest + 2, 45)
            fmt = number_format(c)
            if fmt:
                for i in range(2, len(rows) + 2):
                    ws.cell(row=i, column=j).number_format = fmt
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions

    wb = Workbook()
    ws_runs = wb.active
    ws_runs.title = "Runs"
    write_table(ws_runs, runs)

    if seedavg:
        write_table(wb.create_sheet("SeedAvg"), seedavg)

    ws_setup = wb.create_sheet("Setup")
    ws_setup.append(["Item", "Value"])
    for k, v in setup_items:
        ws_setup.append([k, str(v)])
    for cell in ws_setup[1]:
        cell.font, cell.fill = header_font, header_fill
    ws_setup.column_dimensions["A"].width = 34
    ws_setup.column_dimensions["B"].width = 110
    ws_setup.freeze_panes = "A2"

    wb.save(path)
    return True


def main():
    original_cmd = " ".join(sys.argv)          # recorded in the workbook's Setup sheet
    started = datetime.datetime.now()          # used to name this run's result files
    # argparse prefix-matches "--seed 42" to "--seed_grid", so a plain --seed
    # must not switch on seed-averaging mode.
    plain_seed_flag = any(a == "--seed" or a.startswith("--seed=") for a in sys.argv[1:])
    grid_args, remaining = parse_grid_args()
    explicit_seed_grid = grid_args.seed_grid is not None and not plain_seed_flag
    if grid_args.random_seeds is not None and explicit_seed_grid:
        raise SystemExit("Use either --seed_grid or --random_seeds, not both.")
    use_seed_grid = explicit_seed_grid or grid_args.random_seeds is not None
    sys.argv = [sys.argv[0]] + remaining
    base_args = get_args()  # all normal train.py flags, parsed as usual

    grid = {}
    for grid_flag, base_attr, cast in GRID_SPEC:
        values = to_list(getattr(grid_args, grid_flag), cast)
        grid[base_attr] = values if values is not None else [getattr(base_args, base_attr)]

    if grid_args.random_seeds is not None:
        if grid_args.random_seeds < 1:
            raise SystemExit("--random_seeds must be >= 1")
        master = (grid_args.rng_master_seed if grid_args.rng_master_seed is not None
                  else random.SystemRandom().randrange(10**9))
        grid["seed"] = random.Random(master).sample(range(1, 100000), grid_args.random_seeds)
        print(f"Random seeds drawn (rng_master_seed={master}): {grid['seed']}")
        print("To reproduce: --rng_master_seed {0}  or  --seed_grid {1}\n".format(
            master, ",".join(str(s) for s in grid["seed"])))

    keys = list(grid.keys())
    split_keys = ("train_size", "val_size", "test_size")
    split_triples = None
    if grid_args.split_grid is not None:
        # Note: argparse prefix-matching may have put a plain --train_size/--val_size/
        # --test_size into the *_size_grid slots; with --split_grid those are ignored.
        if any(len(grid[k]) > 1 for k in split_keys):
            print("WARNING: --split_grid given together with --train/val/test_size_grid; "
                  "the *_size_grid values are ignored.")
        split_triples = parse_split_grid(grid_args.split_grid)
        # Cross product of every OTHER hyperparameter, with each exact split triple.
        other_keys = [k for k in keys if k not in split_keys]
        combos = []
        for triple in split_triples:
            for other in itertools.product(*[grid[k] for k in other_keys]):
                values = dict(zip(other_keys, other))
                values.update(zip(split_keys, triple))
                combos.append(tuple(values[k] for k in keys))
        for i, k in enumerate(split_keys):  # so the Setup sheet lists what was swept
            grid[k] = list(dict.fromkeys(t[i] for t in split_triples))
    else:
        combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f"Grid search: {len(combos)} combination(s) to train\n")

    results = []
    best = None

    for combo_num, combo in enumerate(combos, start=1):
        args = copy.deepcopy(base_args)
        for k, v in zip(keys, combo):
            setattr(args, k, v)

        fractional = [args.train_size <= 1, args.val_size <= 1, args.test_size <= 1]
        if any(fractional) and not all(fractional):
            print(f"[{combo_num}/{len(combos)}] Skipping {dict(zip(keys, combo))}: "
                  f"train_size/val_size/test_size must be all fractions or all "
                  f"absolute row counts, not a mix")
            continue
        if all(fractional) and abs(args.train_size + args.val_size + args.test_size - 1.0) > 1e-6:
            print(f"[{combo_num}/{len(combos)}] Skipping {dict(zip(keys, combo))}: "
                  f"train_size + val_size + test_size = "
                  f"{args.train_size + args.val_size + args.test_size:.3g}, must sum to 1.0")
            continue

        set_seed(args.seed)
        configure_device_args(args)
        os.makedirs(args.checkpoints, exist_ok=True)

        setting = build_setting(args)
        if use_seed_grid:
            # keep one checkpoint folder per seed instead of overwriting
            setting = f"{setting}_seed{args.seed}"
        print(f"[{combo_num}/{len(combos)}] {setting}")
        for k, v in zip(keys, combo):
            print(f"    {k}: {v}")

        try:
            exp = Exp_Craft(args)
            exp.train(setting)  # trains, early-stops, and loads the best checkpoint
            epochs_trained = getattr(exp, "epochs_trained", None)

            vali_data, vali_loader = exp._get_data(flag="val")
            criterion = exp._select_criterion()
            vali_loss, vali_acc = exp.vali(vali_data, vali_loader, criterion)

            test_data, test_loader = exp._get_data(flag="test")
            test_loss, test_acc = exp.vali(test_data, test_loader, criterion)
        except Exception as e:
            print(
                f"    -> Skipping {dict(zip(keys, combo))}: training failed "
                f"with error: {e}\n"
            )
            continue

        row = dict(zip(keys, combo))
        row["epochs_trained"] = epochs_trained
        row["vali_loss"] = vali_loss
        row["vali_acc"] = vali_acc
        row["test_loss"] = test_loss
        row["test_acc"] = test_acc
        row["setting"] = setting
        results.append(row)

        print(f"    -> epochs_trained: {epochs_trained}/{args.train_epochs}")
        print(f"    -> vali_loss: {vali_loss:.5f}, vali_direction_acc: {vali_acc:.2%}")
        print(f"    -> test_loss: {test_loss:.5f}, test_direction_acc: {test_acc:.2%}\n")

        if best is None or vali_loss < best["vali_loss"]:
            best = row

    if not results:
        print("No valid combinations were trained (see the skip/error messages above).")
        return

    # Every run gets its own time-stamped file names, so earlier results are never overwritten.
    os.makedirs(grid_args.results_dir, exist_ok=True)
    stem = f"{grid_args.run_name}_{started:%Y%m%d_%H%M%S}"
    xlsx_path = unique_path(os.path.join(grid_args.results_dir, stem + ".xlsx"))
    csv_path = grid_args.results_csv or unique_path(os.path.join(grid_args.results_dir, stem + ".csv"))

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {len(results)} result(s) to {csv_path}")

    agg = None
    if use_seed_grid:
        agg = aggregate_over_seeds(results, keys)
        base, ext = os.path.splitext(csv_path)
        agg_path = f"{base}_seedavg{ext or '.csv'}"
        with open(agg_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
            writer.writeheader()
            writer.writerows(agg)
        print(f"Saved {len(agg)} seed-averaged row(s) to {agg_path}")

    setup_items = [
        ("Started", started.isoformat(timespec="seconds")),
        ("Command line", original_cmd),
        ("Runs completed / planned", f"{len(results)} / {len(combos)}"),
    ]
    if split_triples:
        setup_items.append(("Swept: split (train/val/test)",
                            ", ".join("/".join(f"{x * 100:g}" for x in t) for t in split_triples)))
    for k, vals in grid.items():
        if len(vals) > 1:
            setup_items.append((f"Swept: {k}", ", ".join(str(v) for v in vals)))
    for k, v in sorted(vars(base_args).items()):
        if k not in grid or len(grid[k]) == 1:
            setup_items.append((f"Fixed: {k}", v))
    if save_excel(xlsx_path, results, agg, setup_items):
        print(f"Saved Excel workbook to {xlsx_path}")

    if use_seed_grid:
        best_agg = min(agg, key=lambda r: r["vali_loss_mean"])
        print("\nBest combination (lowest MEAN vali_loss across seeds):")
        for k, v in best_agg.items():
            print(f"    {k}: {v}")
        return

    print("\nBest combination (lowest vali_loss):")
    for k, v in best.items():
        print(f"    {k}: {v}")


if __name__ == "__main__":
    main()