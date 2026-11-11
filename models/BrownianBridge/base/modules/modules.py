import torch
import torch.nn as nn
from functools import partial

# 3D Spatial Rescaler, adjusting spatial dimensions for 3D conditioning maps (CT and PET volumes)
class SpatialRescaler(nn.Module):
    def __init__(self, n_stages=1, method='trilinear', multiplier=0.5, in_channels=3, out_channels=None, bias=False):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in ['nearest', 'linear', 'bilinear', 'trilinear', 'bicubic', 'area']
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None
        
        # If output channels are specified, use a convolution to remap the channels after resizing
        if self.remap_output:
            print(f'Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing.')
            self.channel_mapper = nn.Conv3d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        # Iterate through resizing stages
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)

        # If remapping output channels is requested, use the channel mapper
        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)
