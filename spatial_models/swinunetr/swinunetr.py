# spatial_models/swinunetr/swinunetr.py
# Implementation based on MONAI's SwinUNETR with adaptations for satellite imagery
import math
from typing import Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm

class BandAttention(nn.Module):
    """
    Channel attention module for emphasizing important spectral bands.
    Similar to Squeeze-and-Excitation block but adapted for satellite bands.
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super(BandAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class PatchEmbed(nn.Module):
    """
    Patch embedding block for converting image to tokens
    """
    def __init__(self, patch_size=4, in_channels=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = (patch_size, patch_size, patch_size)
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, t, h, w = x.shape
        # Ensure dimensions are divisible by patch size
        pad_t = (self.patch_size[0] - t % self.patch_size[0]) % self.patch_size[0]
        pad_h = (self.patch_size[1] - h % self.patch_size[1]) % self.patch_size[1]
        pad_w = (self.patch_size[2] - w % self.patch_size[2]) % self.patch_size[2]
        if pad_h > 0 or pad_w > 0 or pad_t > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_t))
        
        x = self.proj(x)
        x = self.norm(x)
        return x

class WindowAttention(nn.Module):
    """
    Window-based multi-head self-attention module
    with relative position encoding.
    """
    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                num_heads
            )
        )
        
        # Get pair-wise relative position index for each token in the window
        coords_t = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_t, coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)
        ].reshape(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block.
    """
    def __init__(
        self,
        dim,
        num_heads,
        window_size=(7, 7, 7),
        shift_size=(0, 0, 0),
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        use_checkpoint=False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward_part1(self, x, mask_matrix):
        b, d, h, w, c = x.shape
        window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)

        x = self.norm1(x)
        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - h % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - w % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, dp, hp, wp, _ = x.shape
        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(
                x,
                shifts=(-shift_size[0], -shift_size[1], -shift_size[2]),
                dims=(1, 2, 3),
            )
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # B*nW, Wd*Wh*Ww, C
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # B*nW, Wd*Wh*Ww, C
        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size + (c,)))
        shifted_x = window_reverse(attn_windows, window_size, b, dp, hp, wp)  # B D' H' W' C
        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(
                shifted_x,
                shifts=(shift_size[0], shift_size[1], shift_size[2]),
                dims=(1, 2, 3),
            )
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :d, :h, :w, :].contiguous()
        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x, mask_matrix):
        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)
        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)
        return x

class Mlp(nn.Module):
    """
    Multilayer perceptron.
    """
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
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

def window_partition(x, window_size):
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size
    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    b, d, h, w, c = x.shape
    x = x.view(
        b,
        d // window_size[0],
        window_size[0],
        h // window_size[1],
        window_size[1],
        w // window_size[2],
        window_size[2],
        c,
    )
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0] * window_size[1] * window_size[2], c)
    return windows

def window_reverse(windows, window_size, b, d, h, w):
    """
    Reverse window partition.
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        b (int): Batch size
        d (int): Depth
        h (int): Height
        w (int): Width
    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(
        b,
        d // window_size[0],
        h // window_size[1],
        w // window_size[2],
        window_size[0],
        window_size[1],
        window_size[2],
        -1,
    )
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(b, d, h, w, -1)
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

class BasicLayer(nn.Module):
    """
    A basic Swin Transformer layer for one stage.
    """
    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size=(7, 7, 7),
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
            )
            for i in range(depth)
        ])

        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)

    def forward(self, x):
        b, c, d, h, w = x.shape
        x = x.permute(0, 2, 3, 4, 1)
        
        # Calculate attention mask for SW-MSA
        dp = int(np.ceil(d / self.window_size[0])) * self.window_size[0]
        hp = int(np.ceil(h / self.window_size[1])) * self.window_size[1]
        wp = int(np.ceil(w / self.window_size[2])) * self.window_size[2]
        
        img_mask = torch.zeros((1, dp, hp, wp, 1), device=x.device)
        for d in slice(-self.window_size[0]), slice(-self.window_size[0], -self.shift_size[0]), slice(-self.shift_size[0], None):
            for h in slice(-self.window_size[1]), slice(-self.window_size[1], -self.shift_size[1]), slice(-self.shift_size[1], None):
                for w in slice(-self.window_size[2]), slice(-self.window_size[2], -self.shift_size[2]), slice(-self.shift_size[2], None):
                    img_mask[:, d, h, w, :] = 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2])
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            x = blk(x, attn_mask)
        
        x = x.permute(0, 4, 1, 2, 3)  # (B, D, H, W, C) -> (B, C, D, H, W)
        
        if self.downsample is not None:
            x = self.downsample(x)
            
        return x

class PatchMerging(nn.Module):
    """
    Patch Merging Layer.
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(8 * dim)

    def forward(self, x):
        b, c, d, h, w = x.shape
        pad_input = (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, d % 2, 0, h % 2, 0, w % 2))
        
        x = x.permute(0, 2, 3, 4, 1)  # B, C, D, H, W -> B, D, H, W, C
        
        x0 = x[:, 0::2, 0::2, 0::2, :]  # B D/2 H/2 W/2 C
        x1 = x[:, 1::2, 0::2, 0::2, :]  # B D/2 H/2 W/2 C
        x2 = x[:, 0::2, 1::2, 0::2, :]  # B D/2 H/2 W/2 C
        x3 = x[:, 0::2, 0::2, 1::2, :]  # B D/2 H/2 W/2 C
        x4 = x[:, 1::2, 1::2, 0::2, :]  # B D/2 H/2 W/2 C
        x5 = x[:, 0::2, 1::2, 1::2, :]  # B D/2 H/2 W/2 C
        x6 = x[:, 1::2, 0::2, 1::2, :]  # B D/2 H/2 W/2 C
        x7 = x[:, 1::2, 1::2, 1::2, :]  # B D/2 H/2 W/2 C
        
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # B D/2 H/2 W/2 8*C
        x = self.norm(x)
        x = self.reduction(x)  # B D/2 H/2 W/2 2*C
        
        x = x.permute(0, 4, 1, 2, 3)  # B 2*C D/2 H/2 W/2
        return x

class SwinUNETR(nn.Module):
    """
    SwinUNETR model for 3D satellite image segmentation with focus on active fire detection.
    
    Args:
        in_channels (int): Number of input channels (spectral bands)
        out_channels (int): Number of output channels (classes)
        img_size (tuple): Input image size
        patch_size (int): Patch size
        feature_size (int): Feature size
        depths (tuple): Depths of Swin layers
        num_heads (tuple): Number of attention heads
        window_size (tuple): Window size
        norm_name (str): Normalization name
        use_checkpoint (bool): Whether to use checkpointing
        spatial_dims (int): Number of spatial dimensions (3 for 3D data)
    """
    def __init__(
        self, 
        in_channels=8,
        out_channels=2,
        img_size=(10, 256, 256),
        patch_size=4,
        feature_size=48,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=(2, 7, 7),
        norm_name="instance",
        use_checkpoint=False,
        spatial_dims=3,
    ):
        super(SwinUNETR, self).__init__()
        
        self.normalize = nn.InstanceNorm3d if norm_name == "instance" else nn.BatchNorm3d
        
        self.embed_dim = feature_size
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size
        self.patch_size = patch_size
        
        # Add band attention module for emphasizing important spectral bands
        self.band_attention = BandAttention(in_channels)
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=feature_size,
            norm_layer=nn.LayerNorm
        )
        
        # Encoder stages
        self.stages = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        
        for i_layer in range(len(depths)):
            stage = BasicLayer(
                dim=int(feature_size * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=4.0,
                qkv_bias=True,
                drop=0.0,
                attn_drop=0.0,
                drop_path=0.0,
                norm_layer=nn.LayerNorm,
                downsample=PatchMerging if i_layer < len(depths) - 1 else None,
                use_checkpoint=use_checkpoint,
            )
            self.stages.append(stage)
            
        # Decoder path with skip connections
        self.decoder1 = nn.Conv3d(feature_size * 8, feature_size * 4, kernel_size=1)
        self.decoder2 = nn.Conv3d(feature_size * 4, feature_size * 2, kernel_size=1)
        self.decoder3 = nn.Conv3d(feature_size * 2, feature_size, kernel_size=1)
        self.decoder4 = nn.Conv3d(feature_size, feature_size // 2, kernel_size=1)
        
        self.decoder_upsampler = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        
        # Final layers
        self.out = nn.Conv3d(feature_size // 2, out_channels, kernel_size=1)
        
        # Multiscale Context Module
        self.context_module = MultiscaleContextModule(feature_size * 8)
    
    def forward(self, x):
        # Add band attention
        x = self.band_attention(x)
        
        # Initial patch embedding
        x = self.patch_embed(x)
        
        skips = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.stages) - 1:
                skips.append(x)
        
        # Apply context module at bottleneck
        x = self.context_module(x)
        
        # Decoder path with skip connections
        x = self.decoder1(x)
        x = self.decoder_upsampler(x)
        x = x + skips[2]
        
        x = self.decoder2(x)
        x = self.decoder_upsampler(x)
        x = x + skips[1]
        
        x = self.decoder3(x)
        x = self.decoder_upsampler(x)
        x = x + skips[0]
        
        x = self.decoder4(x)
        x = self.decoder_upsampler(x)
        
        # Final layer
        x = self.out(x)
        
        return x

class MultiscaleContextModule(nn.Module):
    """
    Multiscale context module for capturing features at different scales.
    Similar to ASPP (Atrous Spatial Pyramid Pooling).
    """
    def __init__(self, in_channels, reduction_ratio=8):
        super(MultiscaleContextModule, self).__init__()
        
        out_channels = in_channels // reduction_ratio
        
        self.branch1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU()
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU()
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=3, dilation=3),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU()
        )
        
        self.branch4 = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU()
        )
        
        self.fusion = nn.Sequential(
            nn.Conv3d(out_channels * 4, in_channels, kernel_size=1),
            nn.InstanceNorm3d(in_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        size = x.size()[2:]
        
        feat1 = self.branch1(x)
        feat2 = self.branch2(x)
        feat3 = self.branch3(x)
        feat4 = F.interpolate(self.branch4(x), size=size, mode='trilinear', align_corners=False)
        
        # Concatenate features from different scales
        out = torch.cat([feat1, feat2, feat3, feat4], dim=1)
        out = self.fusion(out)
        
        # Residual connection
        return x + out

# Make sure to add an empty __init__.py file in the swinunetr directory