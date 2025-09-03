# %%
import torch
import torch.nn as nn
import random
import math


def gaussian_attention(distances, shift, width):
    width = width.clamp(min=1e-1)
    # shift = shift.clamp(min=1e-1)
    return torch.exp(-((distances - shift) ** 2) / width)


def laplacian_attention(distances, shift, width):
    width = width.clamp(min=1e-1)
    # shift = shift.clamp(min=1e-1)
    return torch.exp(-torch.abs(distances - shift) / width)


def cauchy_attention(distances, shift, width):
    width = width.clamp(min=1e-1)
    # shift = shift.clamp(min=1e-1)
    return 1 / (1 + ((distances - shift) / width) ** 2)


def sigmoid_attention(distances, shift, width):
    width = width.clamp(min=1e-1)
    # shift = shift.clamp(min=1e-1)
    return 1 / (1 + torch.exp((-distances + shift) / width))


def triangle_attention(distances, shift, width):
    # TODO: 실험을 위해 수정해야하는 부분
    pass


def get_focus(attention_type):
    if attention_type == "gaussian":
        return gaussian_attention
    elif attention_type == "laplacian":
        return laplacian_attention
    elif attention_type == "cauchy":
        return cauchy_attention
    elif attention_type == "sigmoid":
        return sigmoid_attention
    elif attention_type == "triangle":
        return triangle_attention
    else:
        raise ValueError("Invalid attention type")


class GaussianNoise(nn.Module):
    def __init__(self, std=0.01):
        super(GaussianNoise, self).__init__()
        self.std = std

    def forward(self, x):
        if self.training:
            return x + torch.randn_like(x) * self.std
        return x


class FFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FFN, self).__init__()
        self.ffn = nn.Sequential(
            GaussianNoise(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.ffn(x)


class MoireAttention(nn.Module):

    def __init__(
        self,
        in_dims,
        out_dims,
        heads,
        focus,
        shifts,
        widths,
        self_loop_weight,
    ):
        super(MoireAttention, self).__init__()
        self.num_heads = heads
        self.head_dim = out_dims // heads
        assert (
            self.head_dim * heads == out_dims
        ), "output_dim must be divisible by num_heads"
        self.attn_func = focus
        self.shifts = nn.Parameter(
            torch.tensor(shifts, dtype=torch.float).view(1, heads, 1, 1)
        )
        self.widths = nn.Parameter(
            torch.tensor(widths, dtype=torch.float).view(1, heads, 1, 1)
        )
        self.self_loop_W = nn.Parameter(
            torch.tensor(self_loop_weight, dtype=torch.float).view(1, heads, 1, 1),
            requires_grad=False,
        )
        self.qkv_proj = nn.Linear(in_dims, 3 * out_dims)
        self.scale1 = self.head_dim
        self.scale2 = math.sqrt(self.head_dim)

    def forward(self, x, adj, mask):
        batch_size, num_nodes, _ = x.size()
        qkv = (
            self.qkv_proj(x)
            .view(batch_size, num_nodes, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        Q, K, V = qkv[0], qkv[1], qkv[2]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale2
        moire_adj = self.attn_func(adj.unsqueeze(1), self.shifts, self.widths).clamp(
            min=1e-9
        )
        # moire_adj = moire_adj + self.self_loop_W * I
        adjusted_scores = (
            scores * torch.log(moire_adj)
            + torch.eye(num_nodes, device=x.device).unsqueeze(0).unsqueeze(0) * 0.1
        )
        # I = torch.eye(num_nodes, device=x.device).unsqueeze(0).unsqueeze(0)
        # adjusted_scores.add_(
        #     torch.eye(num_nodes, device=x.device).unsqueeze(0).unsqueeze(0)
        # )
        mask_2d = mask.unsqueeze(1) & mask.unsqueeze(2)
        adjusted_scores.masked_fill_(~mask_2d.unsqueeze(1), -1e3)
        attention_weights = torch.softmax(adjusted_scores, dim=-1)
        return (
            torch.matmul(attention_weights, V)
            .transpose(1, 2)
            .reshape(batch_size, num_nodes, -1)
        )


class MoireLayer(nn.Module):
    def __init__(
        self,
        in_dims,
        out_dims,
        heads,
        focus,
        shift,
        width,
    ):
        super(MoireLayer, self).__init__()
        self.residual_weight = nn.Parameter(
            torch.tensor(0.5, dtype=torch.float32), requires_grad=True
        )
        if shift <= 0.5 or width <= 0.5:
            raise ValueError("initial_values must be greater than 0.5")
        self.attention = MoireAttention(
            in_dims,
            out_dims,
            heads,
            focus,
            [shift + random.uniform(-0.2, 0.2) for _ in range(heads)],
            [width + random.uniform(-0.2, 0.2) for _ in range(heads)],
            [1 + random.uniform(-0.2, 0.2) for _ in range(heads)],
        )
        self.ffn = FFN(out_dims, out_dims, out_dims)
        self.projection_for_residual = nn.Linear(in_dims, out_dims)

    def forward(self, x, adj, mask):
        h = self.attention(x, adj, mask)
        h.mul_(mask.unsqueeze(-1))
        h = self.ffn(h)
        h.mul_(mask.unsqueeze(-1))
        x_proj = self.projection_for_residual(x)
        h = self.residual_weight * h + (1 - self.residual_weight) * x_proj
        return h
