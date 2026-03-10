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

class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dropout, norm):
        super(GraphSAGEEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        hidden_dim = hidden_dim if num_layers > 1 else out_dim
        self.input_conv = dgl.nn.GraphConv(in_dim, hidden_dim, norm=norm)
        self.convs = []
        for _ in range(num_layers - 2):
            self.convs.append(dgl.nn.GraphConv(hidden_dim, hidden_dim, norm=norm))
        if num_layers > 1:
            self.convs.append(dgl.nn.GraphConv(hidden_dim, out_dim, norm=norm))

    def forward(self, g, features):
        h = F.leaky_relu(self.input_conv(g, features))
        h = self.dropout(h)
        for conv in self.convs:
            h = F.leaky_relu(conv(g, h))
            h = self.dropout(h)
        return h

    def transform(self, g, features):
        h = F.leaky_relu(self.input_conv(g, features))
        for conv in self.convs:
            h = F.leaky_relu(conv(g, h))
        return h

class GATEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dropout, num_heads, norm=None):
        super(GATEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_heads = num_heads

        hidden_dim = hidden_dim if num_layers > 1 else out_dim
        self.input_conv = dglnn.GATConv(in_dim, hidden_dim, num_heads=num_heads, allow_zero_in_degree=True)

        self.convs = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.convs.append(dglnn.GATConv(hidden_dim * num_heads, hidden_dim, num_heads=num_heads, allow_zero_in_degree=True))

        if num_layers > 1:
            self.convs.append(dglnn.GATConv(hidden_dim * num_heads, out_dim, num_heads=1, allow_zero_in_degree=True))

    def forward(self, g, features):
        h = self.input_conv(g, features)
        h = F.leaky_relu(h)
        h = h.flatten(1)
        h = self.dropout(h)

        for conv in self.convs[:-1]:
            h = conv(g, h)
            h = F.leaky_relu(h)
            h = h.flatten(1)
            h = self.dropout(h)

        if self.convs:
            h = self.convs[-1](g, h).mean(1)
            h = F.leaky_relu(h)
            h = self.dropout(h)

        return h

class GRUDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dropout):
        super(GRUDecoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(in_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        h, _ = self.gru(x)
        h = self.dropout(h)
        out = self.fc(h)
        return out

class AutoRegressor(nn.Module):
    def __init__(self, instance_dim, num_heads, channel_dim, gnn_hidden_dim, gnn_out_dim, gru_hidden_dim, dropout, tf_layers, gnn_layers, gru_layers):
        super(AutoRegressor, self).__init__()
        self.transformer_encoder = TransformerEncode(instance_dim, num_heads, tf_layers)
        self.gnn_encoder = GATEncoder(channel_dim, gnn_hidden_dim, gnn_out_dim, gnn_layers, dropout, num_heads)
        self.gru_decoder = GRUDecoder(gnn_out_dim, gru_hidden_dim, channel_dim, gru_layers, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g_list, feats_list):
        batch_ts = []
        z_list = []
        for g, feats in zip(g_list, feats_list):
            batch_ts.append(g.ndata['ts'][:,0].tolist())
            # Node feature encoding
            h = self.gnn_encoder(g, feats)
            h = h.unsqueeze(0)  # [1, num_nodes, channel_dim]
            # Temporal feature encoding
            z = self.transformer_encoder(h)
            z_list.append(z)
        # Batch processing
        z_batch = torch.cat(z_list, dim=0)  # [batch_size, num_nodes, channel_dim]
        # Sequence decoding
        reconstructed = self.gru_decoder(z_batch)
        return z_batch, reconstructed

def collate_fn_AR(batch):
    ts_list, graphs_list, feats_list, targets_list = zip(*batch)
    return list(ts_list), list(graphs_list), list(feats_list), list(targets_list)

def create_dataloader_AR(samples, batch_size, shuffle=True):
    dataloader = DataLoader(samples, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn_AR, drop_last=True)
    return dataloader
