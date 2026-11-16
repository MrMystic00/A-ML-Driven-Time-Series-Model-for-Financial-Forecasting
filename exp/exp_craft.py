"""
  - Batch unpacking: DLinear's loader yields
        (batch_x, batch_y, batch_x_mark, batch_y_mark)
    CRAFT's forward needs a second, parallel behavior series plus a
    "known future" mask, so the loader (see data_provider/data_loader_craft.py)
    must yield
        (batch_x, batch_y, behavior_x, behavior_y, behavior_matrix)
    time-mark tensors are dropped since CRAFT's forward() never consumes them.

  - No decoder input (`dec_inp`): that's an Autoformer/Informer encoder-decoder
    concept. CRAFT, like DLinear, is direct multi-step, so it's removed.

  - Loss: DLinear/Autoformer/Informer all optimize a single nn.MSELoss on one
    output tensor. CRAFT's forward returns three tensors (y, be_y, be_y_known),
    so _select_criterion returns a CraftLoss that combines all three terms.
    Weights are exposed as CLI args (main_weight / behavior_weight /
    known_weight) rather than hardcoded, since I don't know your intended
    relative importance between the three signals.

  - args.loss == 'ql': craft.py referenced this flag but the quantile-loss
    implementation wasn't in the file you shared. CraftLoss below raises
    NotImplementedError for 'ql' rather than silently doing plain MSE —
    fill in QuantileLoss once you have the pinball-loss spec.
"""

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from models.CRAFT import Model as CRAFT

import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time
import warnings

warnings.filterwarnings('ignore')


class CraftLoss(nn.Module):
    """
    Composite loss for CRAFT's three outputs.

    total = main_weight     * RMSE(y,          batch_y)
          + behavior_weight * RMSE(be_y,       behavior_y)
          + known_weight    * RMSE(be_y_known, behavior_y * behavior_matrix)

    NOTE: this trains directly on RMSE, not MSE. A small epsilon is added
    inside the sqrt for gradient stability -- as MSE approaches 0,
    d(sqrt(MSE))/d(pred) blows up towards infinity without it, since it's
    proportional to 1/sqrt(MSE). Switching from MSE to RMSE changes the
    actual optimization target (RMSE's gradient is naturally rescaled by
    1/(2*RMSE) relative to MSE's), so loss values and grid search results
    from before this change are NOT directly comparable to results after it.
    """
    def __init__(self, main_weight=1.0, behavior_weight=1.0, known_weight=1.0, loss_type='mse', rmse_eps=1e-8):
        super(CraftLoss, self).__init__()
        if loss_type == 'ql':
            # TODO: plug in quantile / pinball loss here once the quantile
            # levels and formula are specified — not present in craft.py.
            raise NotImplementedError(
                "args.loss == 'ql' was referenced in craft.py but no quantile "
                "loss implementation was provided. Add one here."
            )
        self.mse = nn.MSELoss()
        self.rmse_eps = rmse_eps
        self.main_weight = main_weight
        self.behavior_weight = behavior_weight
        self.known_weight = known_weight

    def _rmse(self, pred, true):
        return torch.sqrt(self.mse(pred, true) + self.rmse_eps)

    def forward(self, y, be_y, be_y_known, batch_y, behavior_y, behavior_matrix):
        # Calculate all the losses
        main_loss = self._rmse(y, batch_y)
        behavior_loss = self._rmse(be_y, behavior_y)
        known_loss = self._rmse(be_y_known, behavior_y * behavior_matrix)
        total = (self.main_weight * main_loss
                 + self.behavior_weight * behavior_loss
                 + self.known_weight * known_loss)
        return total, main_loss, behavior_loss, known_loss


class Exp_Craft(Exp_Basic):
    def __init__(self, args):
        super(Exp_Craft, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'CRAFT': CRAFT,
        }
        model = model_dict[self.args.model](self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        # Requires a Dataset registered in data_provider/data_factory.py that
        # yields (batch_x, batch_y, behavior_x, behavior_y, behavior_matrix) —
        # see data_provider/data_loader_craft.py for the expected shapes.
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        optimizer_name = getattr(self.args, "optimizer", "adam")
        weight_decay = getattr(self.args, "weight_decay", 0.0)

        if optimizer_name == "nadam":
            model_optim = optim.NAdam(
                self.model.parameters(),
                lr=self.args.learning_rate,
                weight_decay=weight_decay,
                momentum_decay=getattr(self.args, "momentum_decay", 4e-3),
            )
        elif optimizer_name == "adam":
            model_optim = optim.Adam(
                self.model.parameters(),
                lr=self.args.learning_rate,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer {optimizer_name!r}; choose 'adam' or 'nadam'")

        return model_optim

    def _select_criterion(self):
        # Getting the loss of prediction data, behavior data and known weight data
        criterion = CraftLoss(
            main_weight=getattr(self.args, 'main_weight', 1.0),
            behavior_weight=getattr(self.args, 'behavior_weight', 1.0),
            known_weight=getattr(self.args, 'known_weight', 1.0),
            loss_type=self.args.loss,
        )
        return criterion

    def _forward_step(self, batch_x, batch_y, behavior_x, behavior_y, behavior_matrix):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        behavior_x = behavior_x.float().to(self.device)
        behavior_y = behavior_y.float().to(self.device)
        behavior_matrix = behavior_matrix.float().to(self.device)

        y, be_y, be_y_known = self.model(batch_x, behavior_x, behavior_y, behavior_matrix) # Using the forword function in CRAFT.py

        f_dim = -1 if self.args.features == 'MS' else 0 # f_dim = -1 because we are using multiple variables from the start and one variable at the output
        y = y[:, -self.args.predict_window:, f_dim:]
        batch_y = batch_y[:, -self.args.predict_window:, f_dim:]

        return y, be_y, be_y_known, batch_y, behavior_y, behavior_matrix

    def directional_accuracy(self, pred, true, last_value):
        """
        pred, true:
            [B, predict_window, 1]

        last_value:
            [B, 1] or [B]
        """

        pred_direction = pred[:, 0, 0] > last_value
        true_direction = true[:, 0, 0] > last_value

        accuracy = (
            pred_direction == true_direction
        ).float().mean()

        return accuracy.item()

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        total_accuracy = []

        self.model.eval()

        with torch.no_grad():
            for (
                batch_x,
                batch_y,
                behavior_x,
                behavior_y,
                behavior_matrix
            ) in vali_loader:

                (
                    y,
                    be_y,
                    be_y_known,
                    batch_y,
                    behavior_y,
                    behavior_matrix
                ) = self._forward_step(
                    batch_x,
                    batch_y,
                    behavior_x,
                    behavior_y,
                    behavior_matrix
                )

                total, *_ = criterion(
                    y,
                    be_y,
                    be_y_known,
                    batch_y,
                    behavior_y,
                    behavior_matrix
                )

                total_loss.append(
                    total.detach().cpu().item()
                )

                # Last observed target value
                last_value = batch_x[
                    :, -1, -1
                ].float().to(self.device)

                acc = self.directional_accuracy(
                    y,
                    batch_y,
                    last_value
                )

                total_accuracy.append(acc)

        avg_loss = np.mean(total_loss)
        avg_accuracy = np.mean(total_accuracy)

        self.model.train()

        return avg_loss, avg_accuracy

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train') # train_loader already splits into mini batches from data loader
        vali_data, vali_loader = self._get_data(flag='val') # vali_loader ''
        test_data, test_loader = self._get_data(flag='test') # test_loader ''

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        # --convergence_threshold is the minimum validation-loss improvement that
        # counts as progress. An epoch that improves the best loss by less than
        # this (or does not improve it) increments the early-stopping counter;
        # training stops once the counter reaches --patience.
        min_delta = getattr(self.args, "convergence_threshold", 1e-5)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, delta=min_delta)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        self.epochs_trained = 0

        for epoch in range(self.args.train_epochs):
            # Mini Batch Gradient Descent
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, behavior_x, behavior_y, behavior_matrix) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                y, be_y, be_y_known, batch_y_t, behavior_y_t, behavior_matrix_t = self._forward_step(
                    batch_x, batch_y, behavior_x, behavior_y, behavior_matrix
                )
                loss, main_l, behavior_l, known_l = criterion(
                    y, be_y, be_y_known, batch_y_t, behavior_y_t, behavior_matrix_t
                )
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.5f} (main {3:.5f}, be {4:.5f}, known {5:.5f})".format(
                        i + 1, epoch + 1, loss.item(), main_l.item(), behavior_l.item(), known_l.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {:.4f}s".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss) # Average the losses from all the batches 
            vali_loss, vali_acc = self.vali(vali_data, vali_loader,criterion)

            test_loss, test_acc = self.vali(test_data,test_loader,criterion)

            print(
            "Epoch: {0}, Steps: {1} | "
            "Train Loss: {2:.5f} "
            "Vali Loss: {3:.5f} "
            "Vali Direction Acc: {4:.2%} "
            "Test Loss: {5:.5f} "
            "Test Direction Acc: {6:.2%}".format(
                epoch + 1,
                train_steps,
                train_loss,
                vali_loss,
                vali_acc,
                test_loss,
                test_acc
                )
            )
            early_stopping(vali_loss, self.model, path)
            self.epochs_trained = epoch + 1

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds, trues, inputx = [], [], []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, behavior_x, behavior_y, behavior_matrix) in enumerate(test_loader):
                y, be_y, be_y_known, batch_y_t, behavior_y_t, behavior_matrix_t = self._forward_step(
                    batch_x, batch_y, behavior_x, behavior_y, behavior_matrix
                )

                pred = y.detach().cpu().numpy()
                true = batch_y_t.detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())

                if i % 20 == 0:
                    gt = np.concatenate((inputx[-1][0, :, -1], true[0, :, -1]), axis=0)

                    pd_values = np.concatenate((inputx[-1][0, :, -1], pred[0, :, -1]), axis=0)

                    if test_data.dates is not None:
                        graph_dates = test_data.dates[i: i + self.args.train_window + self.args.predict_window]
                    else:
                        graph_dates = None

                    visual(gt, pd_values, dates=graph_dates, name=os.path.join(folder_path,str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('corr:{:.5f}'.format(corr))
        with open("result.txt", 'a') as f:
            f.write(setting + "  \n")
            f.write('mse:{}, mae:{}, rse:{}, corr:{}\n\n'.format(mse, mae, rse, corr))

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, rse, corr]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, behavior_x, behavior_y, behavior_matrix) in enumerate(pred_loader):
                y, *_ = self._forward_step(batch_x, batch_y, behavior_x, behavior_y, behavior_matrix)
                preds.append(y.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        np.save(folder_path + 'real_prediction.npy', preds)
        return