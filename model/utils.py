import torch
import torch.nn.functional as F
import numpy as np
from operator import mul
from global_land_mask import globe
from functools import reduce, lru_cache


def window_partition_2d(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, Wh*Ww, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, reduce(mul, window_size), C)
    return windows


def window_reverse_2d(windows, window_size, B, H, W):
    """
    Args:
        windows: (num_windows*B, Wd, Ww, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    # B = int(windows.shape[0] / (H * W / window_size[0] / window_size))
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x



def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


@lru_cache
def compute_mask_2d(H, W, window_size, shift_size, device):
    img_mask = torch.zeros((1, H, W, 1), device=device)  # 1 Hp Wp 1
    cnt = 0
    for h in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0],None):
        for w in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1],None):
            img_mask[:, h, w, :] = cnt
            cnt += 1

    mask_windows = window_partition_2d(img_mask, window_size)  # nW, ws[0]*ws[1], 1
    mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    # attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask != 0


@lru_cache
def compute_land_mask(lat_space, lon_space):
    lat = torch.linspace(*lat_space)
    lon = torch.linspace(*lon_space)
    lat, lon = torch.meshgrid(lat, lon)
    land_mask = torch.roll(torch.from_numpy(globe.is_ocean(lat, lon-180)), 180, dims=1)
    return land_mask


def compute_land_mask_matrix_2d(land_mask, window_size):
    # land_mask = compute_land_mask(lat_space, lon_space)
    land_mask = land_mask.unsqueeze(0).unsqueeze(-1) # 1 H W 1
    # land_mask = land_mask.repeat((Dp, 1, 1, 1)).unsqueeze(0)
    land_mask_windows = window_partition_2d(land_mask, window_size)
    land_mask_windows = land_mask_windows.squeeze(-1)
    land_attn_mask = land_mask_windows.unsqueeze(1) & land_mask_windows.unsqueeze(2)
    return ~land_attn_mask


def prepare_land_mask_2d(land_mask, patch_size, window_size, H, W, num_stages, padding=0, stride=None, pad_before=True):
    
    if pad_before:
        Hp = int(np.ceil(H / patch_size[-2])) * patch_size[-2]
        Wp = int(np.ceil(W / patch_size[-1])) * patch_size[-1]
        land_mask = F.pad(land_mask, (0, Wp-W, 0, Hp-H))

    if stride is None:
        stride = patch_size
    
    weight = torch.zeros(1,1,patch_size[-2], patch_size[-1])
    land_mask = F.conv2d(land_mask.float()[None, None], weight, stride=stride, padding=padding).squeeze()
    land_mask = land_mask > 0

    # land_mask = land_mask.view(Hp//patch_size[-2], patch_size[-2], Wp//patch_size[-1], patch_size[-1])
    # land_mask = land_mask.permute(0,2,1,3).contiguous().sum((-1,-2)) != 0

    H, W = land_mask.shape #int(np.ceil(H / patch_size[-2])), int(np.ceil(W / patch_size[-1]))
    shift_size = tuple(i // 2 for i in window_size)

    all_land_mask_pad = []
    all_land_mask_pad_shifted = []
    for i in range(num_stages):
        window_size, shift_size = get_window_size((H,W), window_size, shift_size)
        Hp = int(np.ceil(H / window_size[-2])) * window_size[-2]
        Wp = int(np.ceil(W / window_size[-1])) * window_size[-1]
        land_mask_pad = F.pad(land_mask, (0, Wp-W, 0, Hp-H), value=False)
        land_mask_pad_shifted = torch.roll(land_mask_pad, shifts=(-shift_size[-2], -shift_size[-1]), dims=(0, 1))
        land_mask_pad_shifted = compute_land_mask_matrix_2d(land_mask_pad_shifted, window_size)
        land_mask_pad = compute_land_mask_matrix_2d(land_mask_pad, window_size)
        all_land_mask_pad.append(land_mask_pad)
        all_land_mask_pad_shifted.append(land_mask_pad_shifted)
        if i < num_stages - 1:
            H, W = int(np.ceil(H / 2)), int(np.ceil(W / 2))
            land_mask = land_mask[0::2, 0::2] | land_mask[1::2, 0::2] | \
                land_mask[0::2, 1::2] | land_mask[1::2, 1::2]

    return all_land_mask_pad, all_land_mask_pad_shifted


class MaskGenerator:
    def __init__(self, config):
        self.input_size = config.input_shape
        self.mask_patch_size = config.mask_patch_size
        self.model_patch_size = config.patch_size
        self.mask_ratio = config.mask_ratio
        
        # assert self.input_size % self.mask_patch_size == 0
        # assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_size = (self.input_size[0] // self.mask_patch_size[0], self.input_size[1] // self.mask_patch_size[1])
        self.scale = (self.mask_patch_size[0] // self.model_patch_size[0], self.mask_patch_size[1] // self.model_patch_size[1])
        
        self.token_count = self.rand_size[0] * self.rand_size[1]
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self):
        mask_idx = torch.randperm(self.token_count)[:self.mask_count]
        mask = torch.zeros(self.token_count)
        mask[mask_idx] = 1.0
        mask = mask.reshape((self.rand_size[0], self.rand_size[1]))
        mask = mask.repeat_interleave(self.scale[0], dim=0).repeat_interleave(self.scale[1], dim=1)

        return mask






