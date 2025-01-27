import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch

def initialize_weight(x):
    nn.init.xavier_uniform_(x.weight)
    if x.bias is not None:
        nn.init.constant_(x.bias, 0)

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

class Swish(nn.Module):

    def __init__(self, inplace=False):
        super().__init__()

        self.inplace = True

    def forward(self, x):
        if self.inplace:
            x.mul_(F.sigmoid(x))
            return x
        else:
            return x * F.sigmoid(x)

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.act = Swish(inplace=True)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            self.act,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)

class IFSxMLP(nn.Module):
    def __init__(self, feature_dim, seq_dim, token_dim, channel_dim, seq_inner_num):
        super().__init__()
        # feature mlp
        self.feature_mix = nn.Sequential(
            nn.LayerNorm(feature_dim),
            MLP(feature_dim, token_dim),
        )
        # sequence mlp
        self.seq_mix = nn.Sequential(
            MLP(seq_dim, channel_dim),
        )
        # subsequence feature mlp
        self.subseq_feature_mix = nn.Sequential(
            nn.LayerNorm(feature_dim),
            MLP(feature_dim, token_dim),
        )
        # subsequence mlp
        self.subseq_seq_mix = nn.Sequential(
            MLP((seq_dim - 1) // seq_inner_num, channel_dim // seq_inner_num),
        )
        self.seq_inner_num = seq_inner_num

        ## LFL
        self.proj_norm1 = nn.LayerNorm(feature_dim)
        self.proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.proj_norm2 = nn.LayerNorm(feature_dim)

    def forward(self, x1, inner_x):
        # split sequence, the number of subsequence is 8
        block_inner_x = torch.chunk(inner_x, self.seq_inner_num, dim=1)

        ## inner FSxMLP
        out_chunks = []
        for x_chunk in block_inner_x:
            x_chunk = self.subseq_feature_mix(x_chunk)
            x_chunk = x_chunk.permute(0, 2, 1)
            x_chunk = self.subseq_seq_mix(x_chunk)
            x_chunk = x_chunk.permute(0, 2, 1)
            out_chunks.append(x_chunk)
        x2 = torch.cat(out_chunks, dim=1)
        x2 = inner_x + x2

        B, N, _ = x2.size()
        ## LFL Block
        temp = x1[:, 1:, :] + self.proj_norm2(self.proj(self.proj_norm1(x2.reshape(B, N, -1))))
        x = torch.cat((x1[:, 0:1, :], temp), dim=1)

        shortcut = x
        ## outer FSxMLP
        x = self.feature_mix(x)
        x = x.permute(0, 2, 1)
        x = self.seq_mix(x)
        x = x.permute(0, 2, 1)

        return x + shortcut, x2

class net1(nn.Module):
    def __init__(
            self,
            seq_dim=384,
            seq_inner_num=8,
            input_dim=128,
            token_dim=1024,
            channel_dim=1536,
            feature_dim=256,
            depth=6
    ):
        super().__init__()
        self.seq_dim = seq_dim + 1
        self.feature_dim = feature_dim
        self.input_dim = input_dim
        self.linear_encoding = nn.Linear(self.input_dim, self.feature_dim)
        initialize_weight(self.linear_encoding)
        self.global_feature = nn.Sequential(
            nn.Linear(522, self.feature_dim * 2),
            nn.LayerNorm(self.feature_dim * 2),
            GELU(),
            nn.Linear(self.feature_dim * 2, self.feature_dim)
        )
        self.layer_norm = nn.LayerNorm(self.represent_dim)
        self.net = nn.ModuleList([])
        for _ in range(depth):
            self.net.append(IFSxMLP(feature_dim=self.feature_dim, seq_dim=self.seq_dim, token_dim=token_dim,
                                                  channel_dim=channel_dim, seq_inner_num=seq_inner_num))

    def forward(self, x, g_f):
        n = x.shape[0]
        x = x.view(n, -1, self.feature_dim)
        x = self.linear_encoding(x)
        g_f = g_f.view(n, -1, 522)
        g_f = self.global_feature(g_f)
        x = torch.cat((g_f, x), dim=1)
        # split Cls_token
        inner_x = x[:, 1:, :]
        for IFSxMLP in self.net:
            x, inner_x = IFSxMLP(x, inner_x)
        x = self.layer_norm(x)
        x = x[:, 0]

        return x
