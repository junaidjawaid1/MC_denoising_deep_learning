"""Building blocks of the 3D U-Net (Fig. 1 of the paper).

Each block keeps its layers in an ``nn.ModuleDict`` whose keys are prefixed with the
block ``name``. The names are part of the checkpoint keys, so do not change them if you
want to load checkpoints saved by this code.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

NEGATIVE_SLOPE = 0.3
NORM_EPS = 1e-3


def conv_norm_act(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, inplace=False):
    """3DConv + InstanceNorm + LeakyReLU(0.3), the "3DConv+NL" block of Fig. 1."""
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.InstanceNorm3d(out_channels, affine=True, eps=NORM_EPS),
        nn.LeakyReLU(NEGATIVE_SLOPE, inplace=inplace),
    )


class MultiScaleElaboration(nn.Module):
    """Four parallel branches (1x1x1, 3x3x3, 2x 3x3x3, 3x 3x3x3) merged by a 1x1x1 conv.

    The output is the input concatenated with ``growth`` new channels.
    """

    BRANCH_CHANNELS = 12

    def __init__(self, in_channels, growth=24, name="ms_block"):
        super().__init__()
        self.name = name
        self.in_channels = in_channels
        self.out_channels = in_channels + growth
        c = self.BRANCH_CHANNELS
        self.layers = nn.ModuleDict({
            f"{name}_conv1": conv_norm_act(in_channels, c, kernel_size=1),
            f"{name}_conv2": conv_norm_act(in_channels, c, kernel_size=3, padding=1),
            f"{name}_conv3": conv_norm_act(in_channels, c, kernel_size=3, padding=1),
            f"{name}_conv3_1": conv_norm_act(c, c, kernel_size=3, padding=1),
            f"{name}_conv4": conv_norm_act(in_channels, c, kernel_size=3, padding=1),
            f"{name}_conv4_1": conv_norm_act(c, c, kernel_size=3, padding=1),
            f"{name}_conv4_2": conv_norm_act(c, c, kernel_size=3, padding=1),
            f"{name}_conv5": nn.Conv3d(4 * c, growth, kernel_size=1, bias=False),
        })

    def forward(self, x):
        layer = lambda suffix: self.layers[f"{self.name}_{suffix}"]
        out1 = layer("conv1")(x)
        out2 = layer("conv2")(x)
        out3 = layer("conv3_1")(layer("conv3")(x))
        out4 = layer("conv4_2")(layer("conv4_1")(layer("conv4")(x)))
        out5 = layer("conv5")(torch.cat([out1, out2, out3, out4], dim=1))
        return torch.cat([x, out5], dim=1)


class Reduction(nn.Module):
    """Halves the spatial size: max pool, average pool and a strided conv, merged by a 1x1x1 conv."""

    BRANCH_CHANNELS = 8

    def __init__(self, in_channels, name="reduction"):
        super().__init__()
        self.name = name
        self.in_channels = in_channels
        self.out_channels = in_channels
        c = self.BRANCH_CHANNELS
        self.layers = nn.ModuleDict({
            f"{name}_conv_branch": conv_norm_act(in_channels, c, kernel_size=2, stride=2, bias=True, inplace=True),
            f"{name}_conv_final": conv_norm_act(2 * in_channels + c, in_channels, kernel_size=1, bias=True,
                                                inplace=True),
        })

    def forward(self, x):
        max_pool = F.max_pool3d(x, kernel_size=2, stride=2)
        avg_pool = F.avg_pool3d(x, kernel_size=2, stride=2)
        conv = self.layers[f"{self.name}_conv_branch"](x)
        return self.layers[f"{self.name}_conv_final"](torch.cat([max_pool, avg_pool, conv], dim=1))


class Expansion(nn.Module):
    """Doubles the spatial size: trilinear upsampling and a transposed conv, merged by a 1x1x1 conv."""

    def __init__(self, in_channels, name="expansion"):
        super().__init__()
        self.name = name
        self.in_channels = in_channels
        channels = in_channels // 2
        self.out_channels = channels
        self.layers = nn.ModuleDict({
            f"{name}_conv1": conv_norm_act(in_channels, channels, kernel_size=1, bias=True, inplace=True),
            f"{name}_conv_transpose": nn.ConvTranspose3d(channels, channels, kernel_size=3, stride=2, padding=1,
                                                         output_padding=1, bias=True),
            f"{name}_conv2": conv_norm_act(2 * channels, channels, kernel_size=1, bias=True, inplace=True),
        })

    def forward(self, x):
        conv1 = self.layers[f"{self.name}_conv1"](x)
        up_sample = F.interpolate(conv1, scale_factor=2, mode="trilinear", align_corners=False)
        conv_transpose = self.layers[f"{self.name}_conv_transpose"](conv1)
        return self.layers[f"{self.name}_conv2"](torch.cat([up_sample, conv_transpose], dim=1))
