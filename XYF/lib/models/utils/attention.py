import torch
from torch import nn, einsum

from timm.models.layers import DropPath
from einops import rearrange


class LinearWithMask(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, mask):
        ''' 要求input的下方padding都是0, i_h >= m_h
        :param x: torch.size([b, i_h, w])
        :param mask: torch.size([b, m_h, w])
        :return: torch.size([b, i_h, w])
        '''
        b, m_h, m_w = mask.shape
        x[:, :m_h] = self.linear(x[:, :m_h])
        return x


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4, dropout=0.):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        if mask is not None:
            b, m_h, m_w = mask.shape
            x[:, :m_h] = self.mlp(x[:, :m_h])
        else:
            x = self.mlp(x)
        return x


class Attention(nn.Module):
    def __init__(self, q_dim, kv_dim=None, heads=8, head_dim=64, dropout=0.):
        super().__init__()
        inner_dim = head_dim * heads

        if kv_dim is None:
            kv_dim = q_dim

        self.heads = heads
        self.scale = head_dim ** -0.5

        self.to_q = nn.Linear(q_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(kv_dim, inner_dim * 2, bias=False)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, q_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, kv=None):
        q = self.to_q(x)

        if kv is None:
            kv = x
        k, v = self.to_kv(kv).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=self.heads), (q, k, v))

        # TODO fix bug 可能token数量不同造成了bug，在dataset中padding或sample特征
        # print(q.shape, k.shape)

        dots = einsum('b i d, b j d -> b i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=self.heads)
        return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(self, q_dim, kv_dim, heads=8, head_dim=64, dropout=0., drop_path=0.):
        super().__init__()
        self.q_norm = nn.LayerNorm(q_dim)
        self.kv_norm = nn.LayerNorm(kv_dim)
        self.attention = Attention(q_dim, kv_dim, heads=heads, head_dim=head_dim, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.LayerNorm(q_dim),
            MLP(q_dim)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, q, kv, mask=None):
        if mask is not None:
            b, m_h, m_w = mask.shape
            kv = kv[:, :m_h]

        q, kv = self.q_norm(q), self.kv_norm(kv)
        x = q + self.drop_path(self.attention(q, kv))
        x = x + self.drop_path(self.mlp(x))
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim, heads=8, head_dim=64, dropout=0., drop_path=0.):
        super().__init__()
        self.attention = nn.Sequential(
            nn.LayerNorm(dim),
            Attention(dim, heads=heads, head_dim=head_dim, dropout=dropout)
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            MLP(dim)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, mask=None):
        if mask is not None:
            b, m_h, m_w = mask.shape
            x_temp = x
            x = x[:, :m_h]

        x = x + self.drop_path(self.attention(x))
        x = x + self.drop_path(self.mlp(x))

        if mask is not None:
            x_temp[:, :m_h] = x
            return x_temp

        return x
