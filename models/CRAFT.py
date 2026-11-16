import torch
from torch import nn
from base_model.mlp import MLP
from utils.data.scaler import StandardScaler, ColumnStandardScaler
from utils.Linear_Algebraic_Solution.LA_Solution import ridge_regression
from models.craft_modules.decomposition import series_decomp
from models.craft_modules.hire_adjust import hire_adjust


class Model(nn.Module):
    """
    CRAFT: koopman-style trend predictor + hierarchical behavior-guided
    seasonal/trend forecaster.

    forward(batch_x, behavior_x, behavior_y, behavior_matrix)
        batch_x          : [B, train_window, enc_in]        target series history
        behavior_x       : [B, train_window, enc_in_be]     behavior series history
        behavior_y       : [B, predict_window, enc_in_be]   behavior series future (ground truth)
        behavior_matrix  : [B, predict_window, enc_in_be]   binary mask, 1 where behavior_y
                                                              is already known at inference time

    returns (y, be_y, be_y_known)
    """
    def __init__(self, configs, dynamic_dim=64, kernel_size=25):
        """
        kernel_size:
            Size of the local window used for moving-average decomposition (and
            generally the number of neighbouring time steps considered in a local
            operation). It is NOT the same as the look-back window.
            E.g. kernel_size = 5 operates on windows such as
            [1,2,3,4,5], [2,3,4,5,6], ...

        predict_window:
            Number of future time steps the model predicts.

        train_window:
            Number of historical time steps provided as input to the model
            (look-back window).

        enc_in:
            Number of historical input features (columns in the main CSV).

        enc_in_be:
            Number of behaviour / known-future features (columns in the behaviour CSV).

        dynamic_dim:
            Dimension of the model's internal (latent) representation used by the
            encoder and decoder.

        loss_ql:
            Boolean flag indicating whether Quantile Loss ("ql") is being used.

        node_number: Number of nodes in the neural networks
        """
        super(Model, self).__init__()
        self.kernel_size = kernel_size
        self.predict_window = configs.predict_window
        self.train_window = configs.train_window
        self.enc_in_be = configs.enc_in_be
        self.enc_in = configs.enc_in
        self.dynamic_dim = dynamic_dim
        self.loss_ql = (configs.loss == 'ql')
        self.node_number = configs.node_number
        self.dropout = getattr(configs, "dropout", 0.05)
        self.mlp_hidden_layers = getattr(configs, "hidden_layers", 2)
        self.ridge_alpha = getattr(configs, "ridge_alpha", 0.1)
        # behavior_norm: "column" = each behavior column normalised on its own scale;
        #                "global" = legacy, ONE mean/std over the whole behavior tensor.
        # behavior_relu: which behavior outputs get the final clamp at 0 --
        #                "both" (legacy) | "known_only" | "none". The price output y is always clamped.
        self.behavior_norm = getattr(configs, "behavior_norm", "column")
        self.behavior_relu = getattr(configs, "behavior_relu", "known_only")

        # Decomposes the input time series into trend and seasonal/residual components
        # using a moving-average window of size kernel_size.
        self.decompsition = series_decomp(self.kernel_size)

        # Encodes the prediction-window representation into a lower-dimensional
        # latent representation of size dynamic_dim.
        # Shape idea: predict_window -> dynamic_dim
        self.encoder_re = MLP(
            f_in=self.predict_window,
            f_out=self.dynamic_dim,
            hidden_dim=128,
            hidden_layers=self.mlp_hidden_layers,
            dropout=self.dropout,
        )

        # Decodes the latent representation back into the prediction-window length.
        # Shape idea: dynamic_dim -> predict_window
        self.decoder_re = MLP(
            f_in=self.dynamic_dim,
            f_out=self.predict_window,
            hidden_dim=128,
            hidden_layers=self.mlp_hidden_layers,
            dropout=self.dropout,
        )

        # Encodes the combined historical + future sequence into a latent
        # representation of size dynamic_dim.
        # Shape idea: (train_window + predict_window) -> dynamic_dim
        self.encoder = MLP(
            f_in=(self.train_window + self.predict_window),
            f_out=self.dynamic_dim,
            hidden_dim=128,
            hidden_layers=self.mlp_hidden_layers,
            dropout=self.dropout,
        )

        # Decodes the latent representation back into the full historical + future
        # sequence length.
        # Shape idea: dynamic_dim -> (train_window + predict_window)
        self.decoder = MLP(
            f_in=self.dynamic_dim,
            f_out=(self.train_window + self.predict_window),
            hidden_dim=128,
            hidden_layers=self.mlp_hidden_layers,
            dropout=self.dropout,
        )

        # Applies a 1D convolution across the time dimension to capture local
        # temporal patterns between nearby observations.
        # kernel_size=3 means the convolution looks at 3 neighbouring time steps
        # at a time. The number of feature channels remains enc_in.
        self.k_cnn = nn.Conv1d(
            in_channels=self.enc_in,
            out_channels=self.enc_in,
            kernel_size=3,
            padding=1,
            padding_mode='zeros'
        )

        # Applies a learnable linear transformation to the latent representation
        # without changing its dimensionality.
        # Shape idea: dynamic_dim -> dynamic_dim
        self.k_linear = nn.Linear(
            self.dynamic_dim,
            self.dynamic_dim
        )

        # Regularizes the backbone (k_linear + k_cnn) the same way dropout
        # already regularizes the MLP encoders/decoders -- reuses the same
        # --dropout value for consistency rather than adding a separate flag.
        self.backbone_dropout = nn.Dropout(self.dropout)

        # Performs hierarchical adjustment/reconciliation between related nodes
        # using their latent representations.
        self.hire_adjust = hire_adjust(
            configs,
            self.dynamic_dim
        )

        # Maps the historical seasonal/residual component directly from the
        # training window to the prediction window.
        # Shape idea: train_window -> predict_window
        self.Linear_Seasonal = nn.Linear(
            self.train_window,
            self.predict_window
        )

        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.Linear_Seasonal.weight)

    def forward(self, batch_x, behavior_x, behavior_y, behavior_matrix):
        """
        1. Scale the data
        2. Decompose the batch and behavior data into trend and seasonals (residuals)
        3. Do the prediction of the trend and seasonals 
        4. Add seasonal and trend together
        5. Inverse scaling 
        6. Do RELU and return the result 
        """
        yscaler = StandardScaler()
        batch_x, x_mean, x_std = yscaler.fit_transform(batch_x) # fit_transform finds the mean/std of the variable, then (y-mean)/std
        be_scaler = ColumnStandardScaler() if self.behavior_norm == "column" else yscaler
        behavior_x, be_x_mean, be_x_std = be_scaler.fit_transform(behavior_x)
        behavior_y = be_scaler.transform(behavior_y, be_x_mean, be_x_std) # Scales new/existing data using the previously computed mean/std
        behavior_y_known = behavior_y * behavior_matrix

        sea_init_x, trend_init_x = self.decompsition(batch_x)
        sea_init_be_x, trend_init_be_x = self.decompsition(behavior_x)

        # seasonal branch
        sea_init_x, sea_init_be_x = sea_init_x.transpose(1, 2), sea_init_be_x.transpose(1, 2)
        sea_output_x = self.Linear_Seasonal(sea_init_x)
        sea_output_be_x = self.Linear_Seasonal(sea_init_be_x)

        # trend branch
        trend_init_x, trend_init_be_x = trend_init_x.transpose(1, 2), trend_init_be_x.transpose(1, 2)
        behavior_y_known = behavior_y_known.transpose(1, 2)
        trend_output_x, trend_output_be_x, be_y_known = self.Linear_trend(
            trend_init_x, trend_init_be_x, behavior_y_known
        )

        be_y = sea_output_be_x + trend_output_be_x
        y = sea_output_x + trend_output_x

        if self.behavior_norm == "column":   # stats are [B, 1, C]; outputs here are [B, C, P]
            be_m, be_s = be_x_mean.transpose(1, 2), be_x_std.transpose(1, 2)
        else:
            be_m, be_s = be_x_mean, be_x_std
        be_y_known = be_scaler.inverse_transform(be_y_known, be_m, be_s)
        be_y = be_scaler.inverse_transform(be_y, be_m, be_s)
        y = yscaler.inverse_transform(y, x_mean, x_std)

        if self.behavior_relu in ("both", "known_only"):
            be_y_known = torch.relu(be_y_known)
        if self.behavior_relu == "both":
            be_y = torch.relu(be_y)
        be_y_known = be_y_known.transpose(1, 2)
        be_y = be_y.transpose(1, 2)
        y = torch.relu(y).transpose(1, 2)

        return y, be_y, be_y_known

    def Linear_trend(self, batch_x, behavior_x, behavior_y_known):

        B, N, _ = batch_x.shape
        _, M, _ = behavior_x.shape

        if self.train_window < self.predict_window: # Add padding if the train window size is smaller than the predict window 
            pad = self.predict_window - self.train_window
            batch_x_re = torch.cat(
                [torch.zeros(B, N, pad, device=batch_x.device), batch_x], dim=2)
            behavior_x_re = torch.cat(
                [torch.zeros(B, M, pad, device=behavior_x.device), behavior_x], dim=2)
        else:
            batch_x_re = batch_x[:, :, -self.predict_window:]
            behavior_x_re = behavior_x[:, :, -self.predict_window:]

        # koopman predictor module
        enc_be_x = self.encoder_re(behavior_x_re) #enc_be_x = ZC_L^T
        enc_be_y_known = self.encoder_re(behavior_y_known) # enc_be_y_known = ZC_p^ 
        enc_x = self.encoder_re(batch_x_re) 
        self.k_konwn = ridge_regression(enc_be_x, enc_be_y_known, alpha=self.ridge_alpha) #k_known = K_c = ZC_p * Z_L^T * (ZC_L^T * ZC_L^T^tr + lambda * I)^-1
        enc_y = torch.bmm(enc_x, self.k_konwn) # Batch matrix multuplication. e.g. (32, 8, 64) * (32, 64, 64) -> (32, 8, 64), i = 32 where i is the index 
        batch_y = self.decoder_re(enc_y)[:, :, -self.predict_window:]
        be_y_known = self.decoder_re(enc_be_y_known)[:, :, -self.predict_window:]

        # internal trend mining module
        input = torch.cat((batch_x, batch_y), dim=2)
        behavior_input = torch.cat((behavior_x, behavior_y_known), dim=2)

        input = self.encoder(input)
        behavior_input = self.encoder(behavior_input)

        output = self.backbone_dropout(self.k_linear(input))
        behavior_output = self.backbone_dropout(self.k_linear(behavior_input))

        output = self.backbone_dropout(self.k_cnn(output))

        # External Trend Guide Module (Only perform hierarchical adjustment when multiple nodes exist)
        if self.node_number > 1:
            output = self.hire_adjust(output)

        behavior_output = self.decoder(behavior_output)[:, :, -self.predict_window:]
        output = self.decoder(output)[:, :, -self.predict_window:]

        return output, behavior_output, be_y_known