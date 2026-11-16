import torch
from torch import nn


class hire_adjust(nn.Module):
    """
    Hierarchical structure / external trend-guide attention module.

    Reshapes (B, N, D) -> (B // node_number, node_number * N, D) to attend
    within a hierarchical group before reshaping back. Requires that every
    `node_number` consecutive rows in a batch belong to the same group —
    see the note in data_provider/data_loader.py about batching/sampling.
    """
    def __init__(self, configs, dynamic_dim):
        super(hire_adjust, self).__init__()
        self.enc_in_be = configs.enc_in_be
        self.enc_in = configs.enc_in
        self.dynamic_dim = dynamic_dim
        self.node_number = configs.node_number

        self.query_projection = nn.Linear(self.dynamic_dim, self.dynamic_dim)
        self.key_projection = nn.Linear(self.dynamic_dim, self.dynamic_dim)
        self.value_projection = nn.Linear(self.dynamic_dim, self.dynamic_dim)
        nn.init.xavier_uniform_(self.query_projection.weight)
        nn.init.xavier_uniform_(self.key_projection.weight)
        nn.init.xavier_uniform_(self.value_projection.weight)

    def forward(self, y):
        B, N, D = y.shape
        B_group = B // self.node_number

        y = y.reshape(B_group, self.node_number * N, D)
        # Attention-based 
        query = self.query_projection(y)
        key = self.key_projection(y)
        Pa = torch.bmm(query, key.transpose(1, 2)) / (D ** 0.5)
        P = nn.Softmax(dim=-1)(Pa)
        value = self.value_projection(y)
        output_re = torch.bmm(P, value)
        y_pre = (y + output_re).reshape(B, N, D)
        return y_pre
