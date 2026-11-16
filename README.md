# A Machine-Learning-Driven Time Series Model for Financial Forecasting

Final-year project (MH4900), Nanyang Technological University.

This repository adapts **CRAFT**, a Koopman-operator neural forecaster, to
**S&P 500 adjusted-close prediction**. The model is given *cross-future behaviour*
(CFB): information that is already known on the forecast date about future days,
such as scheduled FOMC and CPI releases, consensus forecasts (CPI, Federal Funds
rate, crude oil inventory) and sentiment scores. It is compared with a Naive
(persistence) baseline and an ARMA–GARCH benchmark.

## 1. Project overview

- **Koopman Predictor Module (KPM):** a closed-form ridge-regression operator maps
  the CFB latent state to the price latent state and produces an initial forecast.
- **Internal Trend Mining Module (ITM):** refines that forecast by completing partial
  CFB information with a shared linear operator and a small convolutional adaptation layer.
- **Series decomposition:** a moving average splits the price into trend and seasonal
  parts.
- **External Trend Guide (ETG):** removed from this adaptation 
- **Loss:** a sum of three RMSE terms (price, behaviour, known-future behaviour).
- **Training:** NAdam/Adam, learning rate halved each epoch (`--lradj type1`), early
  stopping on validation loss.

## 2. Repository layout

The imports expect this layout. Adjust it if your folders differ.

```
.
├── train.py                        # train and test one model
├── grid_search.py                  # hyperparameter grid search (Excel/CSV results)
├── recursive_predict.py            # multi-day recursive forecast plot from a chosen date
├── plot_window.py                  # direct multi-day forecast plot from chosen dates
├── sp500_ARMA-GARCH_and_naive      # runs the ARMA-GARCH and Naive forecast
├── requirements.txt
├── dataset/                        # CSV files (see Section 4)
├── checkpoints/                    # saved models (created automatically)
├── exp/                            # exp_basic.py, exp_craft.py
├── data_provider/                  # data_factory.py, data_loader.py
├── models/                         # CRAFT.py
│   └── craft_modules/              # decomposition.py
├── base_model/                     # mlp.py
└── utils/                          # tools.py, metrics.py
    ├── data/                       # scaler.py
    └── Linear_Algebraic_Solution/  # LA_Solution.py (ridge regression)
```

## 3. Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 4. Data

Put the CSV files in `dataset/` (or pass another folder with `--root_path`).

- **Main series** (`--data_path`): a `Date` column plus the price columns. The last
  column is the forecast target (adjusted close). `--enc_in` is the number of price
  columns used.
- **Behaviour series** (`--behavior_data_path`): a `Date` column plus the CFB columns
  (event flags, consensus forecasts, sentiment, and so on). `--enc_in_be` must equal
  the number of behaviour columns.
- Both files must cover the same dates, in the same order.

## Command-Line Arguments

**Data and model**

| Argument | Description | Default |
|---|---|---|
| `root_path` | Folder containing the CSV files | `"./dataset"` |
| `data_path` | Main target-series CSV | `"AAPL_Adj_Close.csv"` |
| `behavior_data_path` | Cross-future-behaviour CSV | `"AAPL_Behavior.csv"` |
| `train_size` | Fraction (0–1) or absolute row count (>1) used for training | `0.7` |
| `val_size` | Fraction (0–1) or absolute row count (>1) used for validation | `0.1` |
| `test_size` | Fraction (0–1) or absolute row count (>1) used for testing | `0.2` |
| `train_window` | Number of historical observations (look-back window) | `60` |
| `predict_window` | Number of future observations to predict | `5` |
| `enc_in` | Number of columns in the main input series | `1` |
| `enc_in_be` | Number of behaviour-series columns | `1` |
| `node_number` | Deprecated (ETG removed); accepted and ignored | `1` |
| `dropout` | Dropout probability inside each MLP encoder/decoder in CRAFT | `0.05` |
| `hidden_layers` | Number of Linear layers per MLP (input and output count as 2) | `2` |
| `ridge_alpha` | L2 penalty used in the closed-form Koopman ridge-regression solve | `0.1` |
| `behavior_norm` | `column`: each behaviour column normalised on its own scale; `global`: one mean/std for the whole behaviour tensor | `"column"` |
| `behavior_relu` | Which behaviour outputs are clamped at 0: `both`, `known_only`, `none` | `"known_only"` |

The three split sizes must all be fractions that sum to 1, or all absolute row counts.

**Training**

| Argument | Description | Default |
|---|---|---|
| `batch_size` | Batch size | `32` |
| `train_epochs` | Maximum number of epochs trained | `20` |
| `learning_rate` | Initial learning rate | `1e-3` |
| `lradj` | Learning-rate schedule: `type1` (halved every epoch), `type2` (fixed steps), `none` (constant) | `"type1"` |
| `patience` | Early stopping: number of epochs without enough improvement before stopping | `5` |
| `convergence_threshold` | Minimum decrease in validation loss that counts as progress; an epoch that improves by less adds one to the early-stopping counter. `0` means any improvement counts | `1e-5` |
| `weight_decay` | L2 weight decay passed to the optimizer during backpropagation | `0.0` |
| `optimizer` | `adam`: standard Adam; `nadam`: Adam with Nesterov momentum | `"adam"` |
| `momentum_decay` | NAdam only: how the momentum coefficient decays over time | `4e-3` |
| `main_weight`, `behavior_weight`, `known_weight` | Weights of the price, behaviour and known-behaviour RMSE terms | `1.0` each |
| `seed` | Random seed | `42` |
| `use_gpu` | Flag: use CUDA when available | off |

**Data augmentation (training set only)**

| Argument | Description | Default |
|---|---|---|
| `augmentation_types` | `none` or `jitter` | `"none"` |
| `aug_jitter_std` | Jitter noise std, as a fraction of each series' mean magnitude | `0.01` |
| `augmentation_mode` | `online` (fresh noise every epoch) or `duplicate` (adds noisy copies) | `"online"` |
| `n_augmented_copies` | Duplicate mode: noisy copies per window | `1` |
| `aug_copy_fraction` | Duplicate mode: fraction of windows that get copies | `1.0` |

**Recursive prediction** (`recursive_predict.py`; also needs the same model arguments as training)

| Argument | Description | Default |
|---|---|---|
| `anchor_date` | Last real day of the look-back window, inside the test set | required |
| `horizon_steps` | Number of days to predict recursively | `3` |
| `behavior_rollout_mode` | `real`: use actual future behaviour values (backtesting only, the true future must already exist); `predicted`: use the model's own single-step behaviour forecast (genuine no-lookahead forecasting) | `"predicted"` |
| `out` | Output PDF path; if omitted, a unique name is generated in `./pic` | auto |
| `day_labels` | `on`/`off`: dot and "Day N" label on each forecast day | `"on"` |

`recursive_predict.py` needs a model trained with `--predict_window 1`.

## Commands Used to Reproduce the Experiments

Run the commands from the repository root. Each Python file takes the hyperparameters
above as command-line arguments.

| Python file | Use | Example |
|---|---|---|
| `train.py` | Train the network and evaluate it | see below |
| `grid_search.py` | Find the best hyperparameters by grid search | see below |
| `recursive_predict.py` | Plot truth versus recursive forecasts from a chosen date | see below |
| `plot_window.py` | Plot a direct multi-day forecast from chosen dates | same flags as `train.py`, plus `--anchor_date` |

**Training**

```bash
python train.py \
  --root_path ./dataset --data_path SP500_stock_test.csv \
  --behavior_data_path SP500_behaviour_sentiment.csv \
  --train_window 8 --predict_window 1 --enc_in 4 --enc_in_be 8 \
  --batch_size 2 --train_epochs 50 --learning_rate 9e-3 \
  --train_size 0.6 --val_size 0.15 --test_size 0.25 \
  --dropout 0 --ridge_alpha 2.73841963426315 --weight_decay 0.000001 \
  --seed 2
```

**Grid search.** Add `_grid` to the name of any hyperparameter you want to sweep. A
sweep takes a comma-separated list (`1e-3,1e-2`), a range `start:stop:step`
(`5:10:1`), or a log-spaced range `start:stop:logN`, which gives `N` values evenly
spaced on a log scale (`1e-6:1e2:log5`). Arguments without `_grid` are held fixed.

```bash
python grid_search.py \
  --root_path ./dataset --data_path SP500_stock_test.csv \
  --behavior_data_path SP500_behaviour_sentiment.csv \
  --train_window_grid 5:10:1 --predict_window 1 --enc_in 4 --enc_in_be 13 \
  --batch_size_grid 1:5:1 --train_epochs 50 --learning_rate 9e-3 \
  --train_size 0.6 --val_size 0.2 --test_size 0.2 \
  --dropout 0 --ridge_alpha 2.73841963426315 --weight_decay 0.000001
```

Each run writes a new time-stamped workbook to `./grid_results/` (sheets `Runs`,
`SeedAvg` when seeds are swept, and `Setup`) and a CSV with the same name. To sweep
exact split triples use `--split_grid 80/10/10,70/15/15,60/20/20`. To average over
seeds use `--seed_grid 1,2,3` or `--random_seeds 5`.

**Recursive prediction.** Use the same model arguments as the training run so the
saved checkpoint is found:

```bash
python recursive_predict.py \
  --anchor_date 2026-02-15 --horizon_steps 3 --behavior_rollout_mode predicted \
  --root_path ./dataset --data_path SP500_stock_test.csv \
  --behavior_data_path SP500_behaviour_sentiment.csv \
  --train_window 8 --predict_window 1 --enc_in 4 --enc_in_be 8 \
  --batch_size 2 --train_epochs 50 --learning_rate 9e-3 \
  --train_size 0.6 --val_size 0.15 --test_size 0.25 \
  --dropout 0 --hidden_layers 2 --ridge_alpha 2.73841963426315 --weight_decay 1e-6
```

### Reproducibility notes

- Checkpoints are saved to `./checkpoints/<setting>/checkpoint.pth`, where the setting
  name is built from the look-back and prediction windows, `enc_in`, `enc_in_be`,
  batch size, learning rate, loss weights, dropout, hidden layers, ridge alpha,
  weight decay, optimizer, the split and the behaviour options. `recursive_predict.py`
  and `plot_window.py` must therefore be called with the same values as training.
- Seeds are set with `--seed`. Report averages over several seeds, because the
  differences between many variants are within seed-to-seed noise.
- Validation loss is the sum of the three RMSE terms (not the price RMSE alone).
  Select hyperparameters on validation loss and read the test set once.

## Acknowledgements

Based on the CRAFT architecture 

Zhang, Y., Bu, K., Zhuang, Z., Xie, T., Yu, Y., Li, D., Guo, Y., & Lv, D. (2025). CRAFT:
Time series forecasting with cross-future behavior awareness. Proceedings of the
Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-
25), 7056–7064. https://doi.org/10.24963/ijcai.2025/785

Supervised by Assoc Prof Xiang Liming.
