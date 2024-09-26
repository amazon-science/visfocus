# --------------------------------------------------------
# Swin Transformer V2
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu

# Modifications Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[0::2] = np.sin(sinusoid_table[0::2])  # dim 2i
        sinusoid_table[1::2] = np.cos(sinusoid_table[1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(1) # -> [L,B,dim]

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()

class CrossAttention(nn.Module):
    """
        borrowed from https://github.com/openai/CLIP/blob/main/clip/model.py (AttentionPool2d)
    """
    def __init__(self,
                 dim: int,
                 kv_dim: int,
                 output_dim: int = None,
                 num_heads: int = None,
                 context_length: int = None,
                 norm_layer=nn.LayerNorm,
                 learned_ape=True,
                 **kwargs):
        super().__init__()
        embed_dim = dim
        output_dim = output_dim
        self.learned_ape = learned_ape
        if learned_ape:
            self.positional_embedding = nn.Parameter(torch.randn(context_length, embed_dim) / embed_dim ** 0.5)
        else:
            self.positional_embedding = PositionalEncoding(embed_dim, context_length)
        self.context_length = context_length
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(kv_dim, embed_dim)
        self.v_proj = nn.Linear(kv_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
        self.norm = norm_layer(dim)

    def forward(self, x_q, x_kv, print_maps=False):
        x_q = x_q.permute(1, 0, 2)  # NLW -> LNC
        x_kv = x_kv.permute(1, 0, 2)  # NCS -> SNC
        # x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        if self.learned_ape:
            x_q = x_q + self.positional_embedding[:x_q.shape[0], None, :].to(x_q.dtype)  # (HW+1)NC
        else:
            x_q = self.positional_embedding(x_q)
        x, _ = F.multi_head_attention_forward(
            query=x_q, key=x_kv, value=x_kv,
            embed_dim_to_check=x_q.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
            # print_maps=print_maps
        )
        if self.norm:
            x = self.norm(x)
        x = x.permute(1, 0, 2) # LNC -> NLW
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 pretrained_window_size=[0, 0]):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))

        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h,
                            relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None, v_length=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01)).to(self.logit_scale.device)).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn[..., :v_length, :v_length] = attn[..., :v_length, :v_length] + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, ' \
               f'pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        pretrained_window_size (int): Window size in pre-training.
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pretrained_window_size=0, lm_d_model=None, vl_sa=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size))

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.vl_sa = vl_sa
        if vl_sa:
            self.proj_cp = nn.Linear(lm_d_model, dim)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

        #     mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        #     mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        #     attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        #     attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        # else:
        #     attn_mask = None

        # self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, context_prompts=None):
        # H, W = self.input_resolution
        # B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        # shortcut = x
        # x = x.view(B, H, W, C)

        # # cyclic shift
        # if self.shift_size > 0:
        #     shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        # else:
        #     shifted_x = x

        B, L, C = x.shape
        H, W = self.input_resolution
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        # x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            # attn_mask = mask_matrix
        else:
            shifted_x = x
            # attn_mask = None


        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # concat prompt if available
        if self.vl_sa:
            assert context_prompts is not None
            nWtB = x_windows.shape[0]
            context_prompts = self.proj_cp(context_prompts)
            context_prompts = torch.cat([p.unsqueeze(0).expand(nWtB // B, -1, -1) for p in context_prompts]) # -> [B*nw, len(prompt), feature_dim]
            x_windows = torch.cat([x_windows, context_prompts], dim=1)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, v_length=self.window_size * self.window_size) # , mask=self.attn_mask)  # nW*B, window_size*window_size, C

        if self.vl_sa:
            attn_windows = attn_windows[:, :self.window_size * self.window_size]

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(self.norm1(x))

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x, **kwargs):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.reduction(x)
        x = self.norm(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        flops += H * W * self.dim // 2
        return flops


class ContextPatchMerging(nn.Module):
    """ 
        Applies merging together with Group Attention (https://arxiv.org/pdf/2305.13245.pdf)
        Code base from https://github.com/fkodom/grouped-query-attention-pytorch/blob/main/grouped_query_attention_pytorch/attention.py
    """

    def __init__(self, input_resolution, dim, num_heads, kv_input_dim=512, norm_layer=nn.LayerNorm, **kwargs):
        from .grouped_query_attention import MultiheadGQA

        super().__init__()
        self.input_resolution = input_resolution
        self.embed_dim = dim

        context_length = self.input_resolution[0] * self.input_resolution[1]
        self.positional_embedding = nn.Parameter(torch.randn(context_length, self.embed_dim) / self.embed_dim ** 0.5)

        q_heads = num_heads * 4
        kv_heads = num_heads

        self.mgq_attn = MultiheadGQA(self.embed_dim, q_heads, kv_heads, kv_input_dim=kv_input_dim, layer_norm=norm_layer)

        # self.TMP = nn.Linear(1, 512)

    def forward(self, x, context_prompts): # q, k, v):
        x = x + self.positional_embedding[None, ...].to(x.dtype)
        x = self.mgq_attn(x, context_prompts, context_prompts)
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        flops += H * W * self.dim // 2
        return flops


class PatchMergingAttention(nn.Module):
    r""" Patch Merging Layer with cross attention.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self,
                 input_resolution,
                 dim,
                 num_heads,
                 lm_d_model,
                 vl_learned_ape,
                 norm_layer=nn.LayerNorm,
                 **kwargs):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)
        self.cross_attn = CrossAttention(dim=2 * dim,
                                         kv_dim=lm_d_model,
                                         context_length=self.input_resolution[0] // 2 * self.input_resolution[1] // 2,
                                         output_dim=2 * dim,
                                         num_heads=num_heads,
                                         learned_ape=vl_learned_ape
                                         )
        nn.init.eye_(self.cross_attn.q_proj.weight)
        nn.init.constant_(self.cross_attn.q_proj.bias, 0)
        self.cross_attn.q_proj.requires_grad_(False)
        self.vl_alpha = 0.5

    def forward(self, x, context_prompts, **kwargs):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.reduction(x)
        x = self.norm(x)

        x_vl = self.cross_attn(x, context_prompts)
        x = self.vl_alpha * x_vl + (1 - self.vl_alpha) * x

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        flops += H * W * self.dim // 2
        return flops


class PatchMergingAttentionV2(nn.Module):
    r""" Patch Merging Layer with cross attention.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self,
                 input_resolution,
                 dim,
                 num_heads,
                 lm_d_model,
                 vl_learned_ape=True,
                 norm_layer=nn.LayerNorm,
                 **kwargs):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)
        self.cross_attn = CrossAttention(dim=dim,
                                         kv_dim=lm_d_model,
                                         context_length=self.input_resolution[0] * self.input_resolution[1],
                                         output_dim=dim,
                                         num_heads=num_heads,
                                         learned_ape=vl_learned_ape
                                         )
        nn.init.eye_(self.cross_attn.q_proj.weight)
        nn.init.constant_(self.cross_attn.q_proj.bias, 0)
        self.cross_attn.q_proj.requires_grad_(False)
        self.vl_alpha = 0.5

    def forward(self, x, context_prompts, **kwargs):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x_vl = self.cross_attn(x, context_prompts)
        x = self.vl_alpha * x_vl + (1 - self.vl_alpha) * x

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.reduction(x)
        x = self.norm(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        flops += H * W * self.dim // 2
        return flops


class PatchMergingAttentionV3(nn.Module):
    r""" Patch Merging Layer with cross attention.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self,
                 input_resolution,
                 dim,
                 num_heads,
                 lm_d_model,
                 vl_learned_ape=True,
                 norm_layer=nn.LayerNorm,
                 reduce=True,
                 **kwargs):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False) if reduce else nn.Linear(4 * dim, 4 * dim, bias=False)
        self.norm = norm_layer(2 * dim) if reduce else norm_layer(4 * dim)
        self.cross_attn = CrossAttention(dim=dim * 4,
                                         kv_dim=lm_d_model,
                                         context_length=self.input_resolution[0] // 2 * self.input_resolution[1] // 2,
                                         output_dim=dim * 4,
                                         num_heads=num_heads,
                                         learned_ape=vl_learned_ape
                                         )
        nn.init.eye_(self.cross_attn.q_proj.weight)
        nn.init.constant_(self.cross_attn.q_proj.bias, 0)
        self.cross_attn.q_proj.requires_grad_(False)
        self.vl_alpha = 0.5

    def forward(self, x, context_prompts, **kwargs):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x_vl = self.cross_attn(x, context_prompts)
        x = self.vl_alpha * x_vl + (1 - self.vl_alpha) * x

        x = self.reduction(x)
        x = self.norm(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        flops += H * W * self.dim // 2
        return flops


class PatchMergingGatedAttentionV3(PatchMergingAttentionV3):
    r"""
        PatchMergingAttentionV3 with Tanh-gating
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vl_alpha = nn.Parameter(torch.tensor([0.]))

    def forward(self, x, context_prompts, **kwargs):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x_vl = self.cross_attn(x, context_prompts)

        x = self.vl_alpha.tanh() * x_vl + x

        x = self.reduction(x)
        x = self.norm(x)

        return x


class PatchMerger(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.input_resolution = input_resolution
        self.scale = dim ** -0.5
        self.norm_layer = norm_layer(dim)
        num_tokens_out = (input_resolution[0] * input_resolution[1]) // 4
        self.queries = nn.Parameter(torch.randn(num_tokens_out, dim))
        self.linear = nn.Linear(dim, 2 * dim, bias=False)

    def forward(self, x):
        x = self.norm_layer(x)
        sim = torch.matmul(self.queries, x.transpose(-1, -2)) * self.scale
        attn = sim.softmax(dim = -1)
        return self.linear(torch.matmul(attn, x))


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        pretrained_window_size (int): Local window size in pre-training.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 pretrained_window_size=0, do_shift=True, lm_d_model=None, vl_sa=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth if do_shift else 1 # do not add SWA layers
        self.use_checkpoint = use_checkpoint
        self.vl_sa = vl_sa
        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if ((i % 2 == 0) or (not do_shift)) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 pretrained_window_size=pretrained_window_size,
                                 lm_d_model=lm_d_model,
                                 vl_sa=vl_sa)
            for i in range(self.depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution=input_resolution,
                                         dim=dim,
                                         norm_layer=norm_layer,
                                         num_heads=num_heads,
                                         lm_d_model=lm_d_model
                                         )
        else:
            self.downsample = None

    def forward(self, x, context_prompts=None):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, context_prompts=context_prompts)
        if self.downsample is not None:
            x = self.downsample(x, context_prompts=context_prompts)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding
    Args:
        img_size (int or tuple): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class PatchEmbed1D(nn.Module):
    r""" 1D Image to Patch Embedding (if for example patches are prextracted)
    Args:
        img_size (int or tuple): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, in_chans=3, embed_dim=96, norm_layer=None, img_size=-1, patch_size=-1, **kwargs):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=1, stride=1)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, L, C = x.shape # [batch, num_patches, numof_patch_pixels]
        x = x.permute(0, 2, 1)
        x = self.proj(x).flatten(2).permute(0, 2, 1)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x


class Pix2StructVisionEmbeddings(nn.Module):
    r"""
    Construct the embeddings from patch. In `Pix2Struct` the input is different from classic Vision-transformer models.
    Here the input is a sequence of `seq_len` flattened patches that also combines padding patches (tokens). Each patch
    is represented by a vector of `hidden_size` values.
    """

    def __init__(self,
                 in_chans=3,
                 embed_dim=96,
                 norm_layer=None,
                 img_size=-1,
                 patch_size=-1,
                 **kwargs) -> None:
        super().__init__()

        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=1, stride=1)
        self.row_embedder = nn.Embedding(self.num_patches, embed_dim)
        self.column_embedder = nn.Embedding(self.num_patches, embed_dim)
        
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, flattened_patches: torch.Tensor) -> torch.Tensor:
        # the row and column indices are stored in the first and second position of the flattened_patches
        # flattened_patches: `batch_size`, `seq_len`, `hidden_size` + 2
        row_indices = flattened_patches[:, :, 0].long()
        col_indices = flattened_patches[:, :, 1].long()
        flattened_patches = flattened_patches[:, :, 2:]

        flattened_patches = flattened_patches.permute(0, 2, 1)
        embeddings = self.proj(flattened_patches).permute(0, 2, 1)
        if self.norm is not None:
            embeddings = self.norm(embeddings)

        row_embeddings = self.row_embedder(row_indices)
        col_embeddings = self.column_embedder(col_indices)

        # sum all embeddings together
        embeddings = embeddings + row_embeddings + col_embeddings

        return embeddings


class SwinTransformerV2(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        pretrained_window_sizes (tuple(int)): Pretrained window sizes of each layer.
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0],
                 embedd_matcher_dim=512, do_shift=True, downsampling_method=PatchMerging,
                 vl_cross_attn_layers=[], cross_attention_cls_key='cross_attention', vl_alpha=0.5, lm_d_model=512, input_type='rgb', vl_learned_ape=True,
                 vl_self_attn_layers=[], **kwargs):
        super().__init__()
        self.model_name = 'swin_v2'

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.input_type = input_type

        if input_type == 'flattened_patches':
            self.patch_embed = Pix2StructVisionEmbeddings(
                in_chans=(in_chans*patch_size**2),
                embed_dim=embed_dim,
                norm_layer=norm_layer if self.patch_norm else None,
                img_size=img_size,
                patch_size=patch_size)
        else:
            # split image into non-overlapping patches
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
                norm_layer=norm_layer if self.patch_norm else None)

        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        downsampling_method_key = downsampling_method
        downsampling_method = DOWNDALING_MAP[downsampling_method]

        self.vl_cross_attn_layers = nn.ModuleDict({str(i): None for i in vl_cross_attn_layers})
        self.vl_alpha = vl_alpha
        
        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=downsampling_method if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               pretrained_window_size=pretrained_window_sizes[i_layer],
                               do_shift=do_shift,
                               lm_d_model=lm_d_model,
                               vl_sa=i_layer in vl_self_attn_layers)
            self.layers.append(layer)
            if str(i_layer) in self.vl_cross_attn_layers:
                cross_attention_cls = CROSS_ATTENTION_MAP[cross_attention_cls_key]
                layer_factor = i_layer + int(i_layer < self.num_layers - 1)
                self.vl_cross_attn_layers.update({
                    str(i_layer): cross_attention_cls(
                        dim=int(embed_dim * 2 ** layer_factor),
                        kv_dim=lm_d_model,
                        context_length=patches_resolution[0] // (2 ** layer_factor) * patches_resolution[1] // (2 ** layer_factor),
                        num_heads=num_heads[i_layer],
                        vl_learned_ape=vl_learned_ape)
                })

        self.norm = norm_layer(self.num_features)

        self.embedd_matcher_dim = embedd_matcher_dim
        
        self.apply(self._init_weights)
        for bly in self.layers:
            bly._init_respostnorm()

        self.cross_attention_cls_key = cross_attention_cls_key
        if self.cross_attention_cls_key == 'merge_gated_attention_v3':
            self.gated_attn_alpha = nn.Parameter(torch.tensor([0.]))
        if downsampling_method_key == 'merge_attention_v3_at_final':
            self.last_ds = downsampling_method(
                input_resolution=(48,24),
                dim=768,
                num_heads=24,
                lm_d_model=lm_d_model,
                reduce=False
                )
            self.norm_last_ds = norm_layer(self.num_features * 4)
        elif downsampling_method_key == 'patch_merging_at_final':
            self.last_ds = downsampling_method(input_resolution=(48,24), dim=768)
            self.norm_last_ds = norm_layer(self.num_features * 2)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"cpb_mlp", "logit_scale", 'relative_position_bias_table'}

    def forward_features(self, x, context_prompts=None):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for i, layer in enumerate(self.layers):
            if isinstance(layer.downsample, ContextPatchMerging) or isinstance(layer.downsample, PatchMergingAttention) or isinstance(layer.downsample, PatchMergingAttentionV2) or isinstance(layer.downsample, PatchMergingAttentionV3) or layer.vl_sa or isinstance(layer.downsample, PatchMergingGatedAttentionV3):
                assert context_prompts is not None, 'Context prompt is None'
                x = layer(x, context_prompts)
            else:
                x = layer(x)
            if str(i) in self.vl_cross_attn_layers.keys():
                x_vl = self.vl_cross_attn_layers[str(i)](x, context_prompts, print_maps=True)
                if self.cross_attention_cls_key == 'merge_gated_attention_v3':
                    x = self.gated_attn_alpha.tanh() * x_vl + x
                else:
                    x = self.vl_alpha * x_vl + (1 - self.vl_alpha) * x

        if hasattr(self, 'last_ds'):
            x = self.last_ds(x, context_prompts=context_prompts)
            x = self.norm_last_ds(x)
        else:
            x = self.norm(x)  # B L C

        return x

    def forward(self, x, **kwargs):
        x = self.forward_features(x, **kwargs)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        return flops


DOWNDALING_MAP = {
    'patch_merging': PatchMerging,
    'patch_merging_at_final': PatchMerging,
    'patch_merger': PatchMerger,
    'context_merging': ContextPatchMerging,
    'merge_attention': PatchMergingAttention,
    'merge_attention_v2': PatchMergingAttentionV2,
    'merge_attention_v3': PatchMergingAttentionV3,
    'merge_attention_v3_at_final': PatchMergingAttentionV3,
    'merge_gated_attention_v3': PatchMergingGatedAttentionV3,
}

CROSS_ATTENTION_MAP = {
    'cross_attention': CrossAttention,
    # 'cross_attention_v2': CrossAttentionV2,
}