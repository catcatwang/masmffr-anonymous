import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
from torch.utils.data import DataLoader

class TransformerEncode(nn.Module):
    def __init__(self, in_dim, num_heads, num_layers):
        super(TransformerEncode, self).__init__()
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_dim,
            nhead=num_heads,
            dim_feedforward=in_dim * 2,
        )
        self.transformer_encode = nn.TransformerEncoder(
            self.transformer_encoder_layer,
            num_layers=num_layers
        )

    def forward(self, features):
        h = F.leaky_relu(self.transformer_encode(features))
        return h


