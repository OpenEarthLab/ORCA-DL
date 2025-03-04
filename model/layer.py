import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

import numpy as np

from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_
from transformers.activations import ACT2FN

from .utils import (
    window_partition_2d,
    window_reverse_2d,
    get_window_size,
    compute_mask_2d
)


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class Mlp(nn.Module):
    def __init__(self, config, in_dims, out_dims=None):
        super().__init__()
        hidden_dims = int(in_dims * config.mlp_ratio)
        if config.hidden_act == 'swiglu':
            self.fc1 = nn.Linear(in_dims, hidden_dims*2)
            self.act = SwiGLU()
        else:
            self.fc1 = nn.Linear(in_dims, hidden_dims)
            self.act = ACT2FN[config.hidden_act]
        if out_dims is None:
            out_dims = in_dims
        self.fc2 = nn.Linear(hidden_dims, out_dims)
        self.drop = nn.Dropout(config.drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    

class MlpMoe(nn.Module):
    def __init__(self, config, in_dims, out_dims=None) -> None:
        super().__init__()
        self.mlps = nn.ModuleList([Mlp(config, in_dims, out_dims) for _ in range(config.max_t)])
        self.max_t = config.max_t
        if out_dims is None:
            out_dims = in_dims
        self.out_dims = out_dims

    def forward(self, x, leat_t):
        lead_t = leat_t % self.max_t
        B, H, W, C = x.shape
        out = torch.zeros((B, H, W, self.out_dims), dtype=x.dtype, device=x.device)
        for i in range(self.max_t):
            mlp = self.mlps[i]
            idx = lead_t == i
            selected_x = x[idx]
            if len(selected_x) == 0:
                continue
            out[idx] = mlp(selected_x)
        return out


class PatchEmbed(nn.Module):
    def __init__(self, input_shape, patch_size, in_chans, embed_dim, patch_norm, ape, mask_token=False):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.ape = ape

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if patch_norm:
            self.norm = nn.LayerNorm(embed_dim)
        else:
            self.norm = None

        if self.ape:
            Hp = int(np.ceil(input_shape[0] / patch_size[0]))
            Wp = int(np.ceil(input_shape[1] / patch_size[1]))
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, Hp * Wp, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        if mask_token == True:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            trunc_normal_(self.mask_token, mean=0., std=.02)

    def forward(self, x, mask=None):
        
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x) # B C Wh Ww
        Wh, Ww = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        B, L, _ = x.shape

        if mask is not None:
            mask_tokens = self.mask_token.expand(B, L, -1)
            w = mask.flatten(1).unsqueeze(-1).type_as(mask_tokens)
            x = x * (1.0 - w) + mask_tokens * w

        if self.norm is not None:
            x = self.norm(x)
        
        if self.ape:
            x = x + self.absolute_pos_embed
        
        x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x


class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        Args:
            x: Input feature, tensor size (B, H, W, C).
        """
        B, H, W, C = x.shape

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchExpanding(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = norm_layer(dim // 2)

    def forward(self, x):
        """
        Args:
            x: Input feature, tensor size (B, H, W, C).
        """
        x = self.expand(x)
        B, H, W, C = x.shape
        x = rearrange(x, 'b h w (p2 p1 c) -> b (h p1) (w p2) c', p1=2, p2=2, c=C//4).contiguous()
        x = self.norm(x)

        return x


class RotaryTimeEmbed(torch.nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        theta = 10000 ** (torch.arange(dim).div(2, rounding_mode='floor') * 2 / dim) 
        self.register_buffer('theta', theta)

    def forward(self, x, time):
        theta = torch.einsum('i,j->ij', time, self.theta)
        cos_enc = torch.cos(theta)[:, :, None, None]
        sin_enc = torch.sin(theta)[:, :, None, None]
        x_rot = torch.stack([-x[:, 1::2], x[:, ::2]], dim=1).reshape_as(x)
        x = x * cos_enc + x_rot * sin_enc
        return x


class RotaryPosEmbed2D(nn.Module):
    def __init__(self, shape, dim) -> None:
        super().__init__()
        
        coords_0 = torch.arange(shape[0])
        coords_1 = torch.arange(shape[1])
        coords = torch.stack(torch.meshgrid([coords_0, coords_1], indexing="ij")).reshape(2, -1)

        half_size = dim // 2
        self.dim1_size = half_size // 2
        self.dim2_size = half_size - half_size // 2
        freq_seq1 = torch.arange(0, self.dim1_size) / self.dim1_size
        freq_seq2 = torch.arange(0, self.dim2_size) / self.dim2_size
        inv_freq1 = 10000 ** -freq_seq1
        inv_freq2 = 10000 ** -freq_seq2

        sinusoid1 = coords[0].unsqueeze(-1) * inv_freq1    
        sinusoid2 = coords[1].unsqueeze(-1) * inv_freq2     

        self.register_buffer('sin1', torch.sin(sinusoid1).reshape(*shape, sinusoid1.shape[-1]))
        self.register_buffer('cos1', torch.cos(sinusoid1).reshape(*shape, sinusoid1.shape[-1]))
        self.register_buffer('sin2', torch.sin(sinusoid2).reshape(*shape, sinusoid2.shape[-1]))
        self.register_buffer('cos2', torch.cos(sinusoid2).reshape(*shape, sinusoid2.shape[-1]))

    def forward(self, x):

        x11, x21, x12, x22 = x.split([self.dim1_size, self.dim2_size, \
                                        self.dim1_size, self.dim2_size], dim=-1)
        
        res = torch.cat([x11 * self.cos1 - x12 * self.sin1, x21 * self.cos2 - x22 * self.sin2, \
                        x12 * self.cos1 + x11 * self.sin1, x22 * self.cos2 + x21 * self.sin2], dim=-1)
        return res


class WindowAttention(nn.Module):
    def __init__(self, config, dim, window_size, num_heads):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = config.qk_scale or head_dim ** -0.5
        self.head_dim = head_dim

        self.pos_embed = RotaryPosEmbed2D(window_size, head_dim)

        self.qkv = nn.Linear(dim, dim * 3, bias=config.qkv_bias)
        self.attn_drop = nn.Dropout(config.attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(config.drop_rate)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = self.pos_embed(q.reshape(-1, *self.window_size, C // self.num_heads)).reshape(B_, self.num_heads, -1, C // self.num_heads)
        k = self.pos_embed(k.reshape(-1, *self.window_size, C // self.num_heads)).reshape(B_, self.num_heads, -1, C // self.num_heads)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

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


class SwinLayer(nn.Module):
    def __init__(self, config, window_size, dim, num_heads, shift_size=(0,0), drop_path=0., is_moe=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = config.mlp_ratio

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"

        self.norm1 = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.attn = WindowAttention(config, dim, window_size=self.window_size, num_heads=num_heads)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=config.layer_norm_eps)

        self.is_moe = is_moe
        if is_moe:
            self.mlp = MlpMoe(config, in_dims=dim)
        else:
            self.mlp = Mlp(config, in_dims=dim)

        self.gradient_checkpointing = False

    def forward_part1(self, x, mask_matrix, mask_matrix_shifted):
        B, H, W, C = x.shape
        window_size, shift_size = get_window_size((H, W), self.window_size, self.shift_size)
        x = self.norm1(x)
        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
        pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
            attn_mask = mask_matrix_shifted
        else:
            shifted_x = x
            attn_mask = mask_matrix
        # partition windows
        x_windows = window_partition_2d(shifted_x, window_size)  # B*nW, Wd*Wh*Ww, C
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # B*nW, Wd*Wh*Ww, C
        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size+(C,)))
        
        shifted_x = window_reverse_2d(attn_windows, window_size, B, Hp, Wp)  # B D' H' W' C
        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        return x

    def forward_part2(self, x, lead_t):
        if self.is_moe:
            return self.drop_path(self.mlp(self.norm2(x), lead_t))
        else:
            return self.drop_path(self.mlp(self.norm2(x)))


    def forward(self, x, mask_matrix, mask_matrix_shifted, lead_t=None):
        """
        Args:
            x: Input feature, tensor size (B, H, W, C).
            mask_matrix: Attention mask for cyclic shift.
        """

        shortcut = x
        if self.gradient_checkpointing:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix, mask_matrix_shifted)
        else:
            x = self.forward_part1(x, mask_matrix, mask_matrix_shifted)
        x = shortcut + self.drop_path(x)

        if self.gradient_checkpointing:
            x = x + checkpoint.checkpoint(self.forward_part2, x, lead_t)
        else:
            x = x + self.forward_part2(x, lead_t)

        return x


class SwinEncoderStage(nn.Module):
    def __init__(self, config, window_size, dim, depth, num_heads, drop_path=0., downsample=None, is_atmo=False):

        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth

        self.is_moe = getattr(config, 'is_moe', False)
        if is_atmo:
            self.is_moe_encoder = getattr(config, 'is_moe_atmo', True)
        else:
            self.is_moe_encoder = getattr(config, 'is_moe_encoder', True)

        # build blocks
        self.blocks = nn.ModuleList([
            SwinLayer(
                config,
                window_size,
                dim=dim,
                num_heads=num_heads,
                shift_size=(0,0) if (i % 2 == 0) else self.shift_size,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                is_moe=(self.is_moe and self.is_moe_encoder)
            )
            for i in range(depth)])

        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(dim=dim)

    def forward(self, x, land_mask_pad=None, land_mask_pad_shifted=None, lead_t=None):
        """
        Args:
            x: Input feature, tensor size (B, H, W, C).
        """
        B, H, W, C = x.shape
        window_size, shift_size = get_window_size((H,W), self.window_size, self.shift_size)

        Hp = int(np.ceil(H / window_size[0])) * window_size[0]
        Wp = int(np.ceil(W / window_size[1])) * window_size[1]
        attn_mask_shifted = compute_mask_2d(Hp, Wp, window_size, shift_size, x.device)
        if land_mask_pad_shifted is not None:
            attn_mask_shifted = attn_mask_shifted & land_mask_pad_shifted.to(x.device)
        attn_mask_shifted = torch.zeros(attn_mask_shifted.shape, device=x.device).masked_fill(attn_mask_shifted, float(-100.0))
        if land_mask_pad is not None:
            attn_mask = torch.zeros(land_mask_pad.shape, device=x.device).masked_fill(land_mask_pad.to(x.device), float(-100.0))
        else:
            attn_mask = None

        for blk in self.blocks:
            if self.is_moe:
                x = blk(x, attn_mask, attn_mask_shifted, lead_t)
            else:
                x = blk(x, attn_mask, attn_mask_shifted)
        x = x.view(B, H, W, -1)

        x_before_downsample = x

        if self.downsample is not None:
            x = self.downsample(x)

        return x, x_before_downsample


class SwinDecoderStage(nn.Module):
    def __init__(self, config, window_size, dim, depth, num_heads, drop_path=0., upsample=None):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth

        self.is_moe = getattr(config, 'is_moe', False)
        self.is_moe_decoder = getattr(config, 'is_moe_decoder', True)

        # build blocks
        self.blocks = nn.ModuleList([
            SwinLayer(
                config,
                window_size,
                dim=dim,
                num_heads=num_heads,
                shift_size=(0,0,0) if (i % 2 == 0) else self.shift_size,
                drop_path=drop_path[depth-1-i] if isinstance(drop_path, list) else drop_path,
                is_moe=(self.is_moe and self.is_moe_decoder)
            )
            for i in range(depth)])

        self.upsample = upsample
        if self.upsample is not None:
            self.upsample = upsample(dim=dim)

    def forward(self, x, land_mask_pad=None, land_mask_pad_shifted=None, lead_t=None):
        """ 
        Args:
            x: Input feature, tensor size (B, H, W, C).
        """
        # calculate attention mask for SW-MSA
        B, H, W, C = x.shape
        window_size, shift_size = get_window_size((H,W), self.window_size, self.shift_size)
        # x = rearrange(x, 'b c d h w -> b d h w c')
        Hp = int(np.ceil(H / window_size[0])) * window_size[0]
        Wp = int(np.ceil(W / window_size[1])) * window_size[1]
        attn_mask_shifted = compute_mask_2d(Hp, Wp, window_size, shift_size, x.device)
        if land_mask_pad_shifted is not None:
            attn_mask_shifted = attn_mask_shifted & land_mask_pad_shifted.to(x.device)
        attn_mask_shifted = torch.zeros(attn_mask_shifted.shape, device=x.device).masked_fill(attn_mask_shifted, float(-100.0))
        if land_mask_pad is not None:
            attn_mask = torch.zeros(land_mask_pad.shape, device=x.device).masked_fill(land_mask_pad.to(x.device), float(-100.0))
        else:
            attn_mask = None

        for blk in self.blocks:
            if self.is_moe:
                x = blk(x, attn_mask, attn_mask_shifted, lead_t)
            else:
                x = blk(x, attn_mask, attn_mask_shifted)
        x = x.view(B, H, W, -1)

        if self.upsample is not None:
            x = self.upsample(x)

        return x