import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


def get_subnet(c=None):

    def subnet_cons(in_channels, out_channels):
        return ResidualSubnet(in_channels, out_channels, c=c)

    return subnet_cons


class ResidualSubnet(nn.Module):

    def __init__(self, in_channels, out_channels, c=None):
        super(ResidualSubnet, self).__init__()

        self.gate1 = nn.Conv2d(in_channels, c.hidden_channels, kernel_size=1, padding=0)

        self.conv = ConvBlock(c.hidden_channels, c.hidden_channels, drop_prob=c.drop_prob)
        self.attn = GatedAttn(c.hidden_channels, num_heads=c.num_heads, drop_prob=c.drop_prob)

        self.gate2 = nn.Conv2d(c.hidden_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.gate1(x)
        x = self.conv(x) + x
        x = self.attn(x) + x
        x = self.gate2(x)
        return x


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, drop_prob=0.):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.drop = nn.Dropout2d(drop_prob)

    def forward(self, x):
        x = self.conv(x)
        x = self.drop(x)
        return x


class GatedAttn(nn.Module):
    """Gated Multi-Head Self-Attention Block

    Based on the paper:
    "Attention Is All You Need"
    by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
        Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
    (https://arxiv.org/abs/1706.03762).

    Args:
        d_model (int): Number of channels in the input.
        num_heads (int): Number of attention heads.
        drop_prob (float): Dropout probability.
    """

    def __init__(self, d_model, num_heads=4, drop_prob=0.):
        super(GatedAttn, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.drop_prob = drop_prob
        self.in_proj = weight_norm(nn.Linear(d_model, 3 * d_model, bias=False))

    def forward(self, x):
        # Flatten and encode position
        b, c, h, w = x.size()
        x = x.permute(0, 2, 3, 1)
        x = x.view(b, h * w, c)
        _, seq_len, num_channels = x.size()
        pos_encoding = self.get_pos_enc(seq_len, num_channels, x.device)
        x = x + pos_encoding

        # Compute q, k, v
        memory, query = torch.split(self.in_proj(x), (2 * c, c), dim=-1)
        q = self.split_last_dim(query, self.num_heads)
        k, v = [self.split_last_dim(tensor, self.num_heads) for tensor in torch.split(memory, self.d_model, dim=2)]

        # Compute attention and reshape
        key_depth_per_head = self.d_model // self.num_heads
        q = q * (key_depth_per_head**-0.5)
        x = self.dot_product_attention(q, k, v)
        x = self.combine_last_two_dim(x.permute(0, 2, 1, 3))
        x = x.transpose(1, 2).view(b, c, h, w)  # (b, c, h, w)

        return x

    def dot_product_attention(self, q, k, v, bias=False):
        """Dot-product attention.

        Args:
            q (torch.Tensor): Queries of shape (batch, heads, length_q, depth_k)
            k (torch.Tensor): Keys of shape (batch, heads, length_kv, depth_k)
            v (torch.Tensor): Values of shape (batch, heads, length_kv, depth_v)
            bias (bool): Use bias for attention.

        Returns:
            attn (torch.Tensor): Output of attention mechanism.
        """
        weights = torch.matmul(q, k.permute(0, 1, 3, 2))
        if bias:
            weights += self.bias
        weights = F.softmax(weights, dim=-1)
        weights = F.dropout(weights, self.drop_prob, self.training)
        attn = torch.matmul(weights, v)

        return attn

    @staticmethod
    def split_last_dim(x, n):
        """Reshape x so that the last dimension becomes two dimensions.
        The first of these two dimensions is n.
        Args:
            x (torch.Tensor): Tensor with shape (..., m)
            n (int): Size of second-to-last dimension.
        Returns:
            ret (torch.Tensor): Resulting tensor with shape (..., n, m/n)
        """
        old_shape = list(x.size())
        last = old_shape[-1]
        new_shape = old_shape[:-1] + [n] + [last // n if last else None]
        ret = x.view(new_shape)

        return ret.permute(0, 2, 1, 3)

    @staticmethod
    def combine_last_two_dim(x):
        """Merge the last two dimensions of `x`.

        Args:
            x (torch.Tensor): Tensor with shape (..., m, n)

        Returns:
            ret (torch.Tensor): Resulting tensor with shape (..., m * n)
        """
        old_shape = list(x.size())
        a, b = old_shape[-2:]
        new_shape = old_shape[:-2] + [a * b if a and b else None]
        ret = x.contiguous().view(new_shape)

        return ret

    @staticmethod
    def get_pos_enc(seq_len, num_channels, device):
        position = torch.arange(seq_len, dtype=torch.float32, device=device)
        num_timescales = num_channels // 2
        log_timescale_increment = math.log(10000.) / (num_timescales - 1)
        inv_timescales = torch.arange(num_timescales, dtype=torch.float32, device=device)
        inv_timescales *= -log_timescale_increment
        inv_timescales = inv_timescales.exp_()
        scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
        encoding = torch.cat([scaled_time.sin(), scaled_time.cos()], dim=1)
        encoding = F.pad(encoding, [0, num_channels % 2, 0, 0])
        encoding = encoding.view(1, seq_len, num_channels)

        return encoding
