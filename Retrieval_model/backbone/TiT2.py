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
            #             nn.LayerNorm(hidden_dim),
            self.act,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class net(nn.Module):
    def __init__(self, dim, seq_len, token_dim, channel_dim, seq_inner_num):
        super().__init__()

        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            MLP(dim, token_dim),
        )

        self.channel_mix = nn.Sequential(

            MLP(seq_len, channel_dim),
        )

        self.inner_token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            MLP(dim, token_dim),
        )

        self.inner_channel_mix = nn.Sequential(
            MLP((seq_len - 1) // seq_inner_num, channel_dim // seq_inner_num),
        )
        self.seq_inner_num = seq_inner_num

        self.proj_norm1 = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.proj_norm2 = nn.LayerNorm(dim)

    def forward(self, x1, inner_x):
        input = inner_x
        # split Cls_token
        inner_x = inner_x[:, 1:, :]
        # split blocks
        block_inner_x = torch.chunk(inner_x, self.seq_inner_num, dim=1)

        out_chunks = []
        for x_chunk in block_inner_x:
            x_chunk = self.inner_token_mix(x_chunk)
            x_chunk = x_chunk.permute(0, 2, 1)
            x_chunk = self.inner_channel_mix(x_chunk)
            x_chunk = x_chunk.permute(0, 2, 1)
            out_chunks.append(x_chunk)
        x2 = torch.cat(out_chunks, dim=1)
        x2 = inner_x + x2

        B, N, _ = x2.size()
        ## LFL Block
        temp = x1[:, 1:, :] + self.proj_norm2(self.proj(self.proj_norm1(x2.reshape(B, N, -1))))
        x = torch.cat((x1[:, 0:1, :], temp), dim=1)

        shortcut = x
        ## TMLP block
        x = self.token_mix(x)
        x = x.permute(0, 2, 1)
        x = self.channel_mix(x)
        x = x.permute(0, 2, 1)

        x2 = torch.cat((input[:, 0:1, :], x2), dim=1)

        return x + shortcut, x2

class TiT2(nn.Module):
    def __init__(
            self,
            seq_len=384,
            seq_inner_num=8,
            d_model=128,
            token_dim=1024,
            channel_dim=1536,
            represent_dim=256,
            depth=6
    ):
        super().__init__()
        self.seq_len = seq_len + 1
        self.d_model = d_model
        self.represent_dim = represent_dim
        self.linear_encoding = nn.Linear(self.d_model, self.represent_dim)
        initialize_weight(self.linear_encoding)
        self.huffman_code = nn.Sequential(
            nn.Linear(522, self.represent_dim * 2),
            nn.LayerNorm(self.represent_dim * 2),
            GELU(),
            nn.Linear(self.represent_dim * 2, self.represent_dim)
        )
        self.layer_norm = nn.LayerNorm(self.represent_dim)
        self.TiT = nn.ModuleList([])
        for _ in range(depth):
            self.TiT.append(net(dim=self.represent_dim, seq_len=self.seq_len, token_dim=token_dim,
                                                  channel_dim=channel_dim, seq_inner_num=seq_inner_num))

    def forward(self, x, huffman_code):
        n = x.shape[0]
        x = x.view(n, -1, self.d_model)
        x = self.linear_encoding(x)
        huffman_code = huffman_code.view(n, -1, 522)
        huffman_code = self.huffman_code(huffman_code)
        x = torch.cat((huffman_code, x), dim=1)
        inner_x = x
        for TiT in self.TiT:
            x, inner_x = TiT(x, inner_x)
        x = self.layer_norm(x)
        x = x[:, 0]

        return x