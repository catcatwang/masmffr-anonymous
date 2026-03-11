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
        # features: [batch, seq_len, in_dim]
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
            h = self.convs[-1](g, h)
            h = h.squeeze(1)

        return h

    def transform(self, g, features):
        h = self.input_conv(g, features)
        h = F.leaky_relu(h)
        h = h.flatten(1)

        for conv in self.convs[:-1]:
            h = conv(g, h)
            h = F.leaky_relu(h)
            h = h.flatten(1)

        if self.convs:
            h = self.convs[-1](g, h)
            h = h.squeeze(1)

        return h

class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):

        if x.dim() == 2:
            x = torch.einsum('nd,nm->md', (x, A))
        elif x.dim() == 3:
            x = torch.einsum('bnd,nm->bmd', (x, A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self, c_in, c_out, bias=True):
        super(linear, self).__init__()
        self.mlp = nn.Linear(c_in, c_out, bias=bias)

    def forward(self, x):
        return self.mlp(x)

class mixprop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep+1)*c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha


    def forward(self,x,adj):
        if isinstance(adj, dgl.DGLHeteroGraph):
            adj = adj.adjacency_matrix().to_dense().to(x.device)  # (N, N)
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
            out.append(h)
        ho = torch.cat(out,dim=1)
        ho = self.mlp(ho)
        return ho

class SimpleTransformer(nn.Module):
    def __init__(self, d_model, nhead):
        super(SimpleTransformer, self).__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):  # x: [B, T, C]
        attn_output, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_output)
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        return x


class ConfigurableMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=None, num_layers=2,
                 activation='leaky_relu', dropout=0.0):
        super(ConfigurableMLP, self).__init__()

        if hidden_dims is None:

            hidden_dims = [output_dim] * (num_layers - 1)
        elif len(hidden_dims) != num_layers - 1:

            if len(hidden_dims) < num_layers - 1:

                hidden_dims = hidden_dims + [hidden_dims[-1]] * (num_layers - 1 - len(hidden_dims))
            else:

                hidden_dims = hidden_dims[:num_layers - 1]


        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            if i < len(dims) - 2:
                if activation == 'leaky_relu':
                    layers.append(nn.LeakyReLU(0.01))
                elif activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'tanh':
                    layers.append(nn.Tanh())

                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)



class Extractor(nn.Module):
    def __init__(self, tf_in_dim, num_heads, gnn_in_dim, gnn_hidden_dim, gnn_out_dim, gru_hidden_dim, dropout=0, tf_layers=1, gnn_layers=2, gru_layers=1):
        super(Extractor, self).__init__()
        #130 46 130 46  #266 18 266 18
        self.TFEncoderinstance = TransformerEncode(266 , num_heads, tf_layers)
        self.TFEncoderchannel = TransformerEncode(18, num_heads, tf_layers)
        self.GRUEncoderchannel = nn.GRU(266 , gru_hidden_dim, gru_layers, bias=False, batch_first=True)
        self.GRUEncoderinstance = nn.GRU(18, gru_hidden_dim, gru_layers, bias=False, batch_first=True)
        self.LSTMEncoderchannel = nn.LSTM(266 , gru_hidden_dim, gru_layers, bias=False, batch_first=True)
        self.LSTMEncoderinstance = nn.LSTM(18, gru_hidden_dim, gru_layers, bias=False, batch_first=True)
        self.TemporalTransformer = SimpleTransformer(266 , num_heads)
        self.InstanceTransformer = SimpleTransformer(18, num_heads)
        self.lin1 = nn.Linear(266 ,gru_hidden_dim)
        self.lin2 = nn.Linear(18, gru_hidden_dim)
        self.lin3 = nn.Linear(gru_hidden_dim,gnn_out_dim)
        # self.GraphEncoder = GraphSAGEEncoder(gru_hidden_dim, gnn_hidden_dim, gnn_out_dim, gnn_layers, dropout, norm='none')
        self.GraphEncoder = GATEncoder(
            in_dim=gru_hidden_dim,
            hidden_dim=gnn_hidden_dim,
            out_dim=gnn_out_dim,
            num_layers=4,
            dropout=dropout,
            num_heads=4,
            norm=None
        )
        self.graphencoder = mixprop(
            c_in=gru_hidden_dim,
            c_out=gnn_out_dim,
            gdep=10,
            dropout=dropout,
            alpha=0.05
        )
        self.MLPchannel = nn.Sequential(
            nn.Linear(1330, gru_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(gru_hidden_dim, gru_hidden_dim)
        )
        self.MLPinstance = nn.Sequential(
            nn.Linear(90, gru_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(gru_hidden_dim, gru_hidden_dim)
        )
        # self.MLPchannel = ConfigurableMLP(1330, gru_hidden_dim, num_layers=2)
        # self.MLPinstance = ConfigurableMLP(90, gru_hidden_dim, num_layers=2)
        self.linchannel = nn.Linear(1330,gru_hidden_dim)
        self.lininstance = nn.Linear(90, gru_hidden_dim)

    def moving_average(self,signal, window_size):
        return torch.nn.functional.avg_pool1d(signal.unsqueeze(0), window_size, stride=1).squeeze(0)

    def seasonal_decomposition(self,signal, window_size=9):
        trend = self.moving_average(signal, window_size)

        pad_size = window_size // 2
        trend = torch.nn.functional.pad(trend, (pad_size, pad_size), 'replicate')

        seasonal = signal - trend
        return trend, seasonal

    def keep_top_80_percent_fft(self,x_fft):

        magnitudes = torch.abs(x_fft)

        sorted_mag, indices = torch.sort(magnitudes, descending=True)

        num_keep = int(0.6 * len(x_fft))

        mask = torch.zeros_like(x_fft)

        mask[indices[:num_keep]] = 1

        filtered_fft = x_fft * mask

        return filtered_fft

    def forward(self, g, features):
        batch_size, series_len, instance_num, channel_dim = features.shape

        h1 = features.reshape(-1, instance_num, channel_dim)
        h1 = self.TFEncoderinstance(h1)
        h1 = h1.permute(0, 2, 1)

        h2 = features.reshape(-1, channel_dim, instance_num)
        h2 = self.TFEncoderchannel(h2)
        h = h1 + h2

        h3 = h.permute(0,2,1).view(batch_size, series_len, instance_num, channel_dim).permute(0,2,1,3).reshape(-1, series_len, channel_dim) # 184,5,130
        h4 = h.permute(0,2,1).view(batch_size, series_len, instance_num, channel_dim).permute(0,3,1,2).reshape(-1, series_len, instance_num)  #520,5,46

        h3_flattened = h3.reshape(h3.size(0), -1)
        h4_flattened = h4.reshape(h4.size(0), -1)

        h_n3 = self.MLPchannel(h3_flattened)
        h_n4 = self.MLPinstance(h4_flattened)

        h_n4_sliced = h_n4[:h_n3.shape[0], :]
        h_n5 = h_n3 + h_n4_sliced

        h = self.GraphEncoder(g, h_n5)
        h = h.view(batch_size, instance_num, -1)
        return h

class Regressor(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Regressor, self).__init__()
        self.mlp = nn.Linear(in_dim, out_dim)

    def forward(self, features):
        h = F.leaky_relu(self.mlp(features))
        return h

class AutoRegressor(nn.Module):
    def __init__(self, tf_in_dim, num_heads, gnn_in_dim, gnn_hidden_dim, gnn_out_dim, gru_hidden_dim, dropout=0, tf_layers=1, gnn_layers=2, gru_layers=1):
        super(AutoRegressor, self).__init__()
        self.extractor = Extractor(tf_in_dim, num_heads, gnn_in_dim, gnn_hidden_dim, gnn_out_dim, gru_hidden_dim, dropout, tf_layers, gnn_layers, gru_layers)
        self.regressor = Regressor(gru_hidden_dim, gnn_in_dim)

    def forward(self, g, features):
        z = self.extractor(g, features)
        h = self.regressor(z)
        return z, h

def collate_AR(samples):
    timestamps, graphs, feats, targets = map(list, zip(*samples))
    batched_ts = torch.stack(timestamps)
    batched_graphs = dgl.batch(graphs)
    batched_feats = torch.stack(feats)
    batched_targets = torch.stack(targets)
    return  batched_ts, batched_graphs, batched_feats, batched_targets

def create_dataloader_AR(samples, window_size=6, max_gap=60, batch_size=2, shuffle=False):
    series_samples = [samples[i:i+window_size] for i in range(len(samples) - window_size + 1)]
    series_samples = [
        series_sample for series_sample in series_samples
            if all(abs(series_sample[i][0] - series_sample[i+1][0]) <= max_gap
                for i in range(len(series_sample) - 1))
    ]
    dataset = [[
            torch.tensor(series_sample[-1][0]),
            series_sample[-1][1],
            torch.stack([step[2] for step in series_sample[:-1]]),
            torch.tensor(series_sample[-1][2])
        ] for _, series_sample in enumerate(series_samples)]
    dataloader = DataLoader(dataset, batch_size, shuffle, collate_fn=collate_AR)
    return dataloader
