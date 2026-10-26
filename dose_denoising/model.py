"""3D U-Net with the stepwise 2 -> 1 channel output head (Fig. 1 of the paper)."""
import torch
import torch.nn as nn

from .blocks import NEGATIVE_SLOPE, Expansion, MultiScaleElaboration, Reduction


class DoseDenoisingUNet(nn.Module):
    """Refines a noisy 5e3-history MC dose to the quality of a 1e6-history dose.

    Input:  [B, 2, D, H, W] (normalized CT, min-max normalized 5e3-history dose).
    Output: [B, 1, D, H, W] (unnormalized; divide by its max as done in training).
    D, H and W must be multiples of 8.

    Attribute names are part of the checkpoint keys; keep them unchanged to load saved checkpoints.
    """

    def __init__(self, in_channels=2, growth=24):
        super().__init__()

        def elab(channels, name):
            return MultiScaleElaboration(channels, growth=growth, name=name)

        # Encoder
        self.enc_multi_elab_1 = elab(in_channels, "enc_multi_elab_enc_1")
        self.enc_multi_elab_2 = elab(self.enc_multi_elab_1.out_channels, "enc_multi_elab_enc_2")
        skip1 = self.enc_multi_elab_2.out_channels
        self.reduction_1 = Reduction(skip1, name="reduction_1")

        self.enc_multi_elab_3 = elab(skip1, "enc_multi_elab_enc_3")
        self.enc_multi_elab_4 = elab(self.enc_multi_elab_3.out_channels, "enc_multi_elab_enc_4")
        skip2 = self.enc_multi_elab_4.out_channels
        self.reduction_2 = Reduction(skip2, name="reduction_2")

        self.enc_multi_elab_5 = elab(skip2, "enc_multi_elab_enc_5")
        self.enc_multi_elab_6 = elab(self.enc_multi_elab_5.out_channels, "enc_multi_elab_enc_6")
        skip3 = self.enc_multi_elab_6.out_channels
        self.reduction_3 = Reduction(skip3, name="ct_reduction_3")

        self.bottleneck_multi_elab_1 = elab(skip3, "bottleneck_multi_elab_1")
        self.bottleneck_multi_elab_2 = elab(self.bottleneck_multi_elab_1.out_channels, "bottleneck_multi_elab_2")

        # Decoder
        self.expansion_1 = Expansion(self.bottleneck_multi_elab_2.out_channels, name="expansion_1")
        self.dec_multi_elab_1 = elab(self.expansion_1.out_channels + skip3, "dec_multi_elab_1")
        self.dec_multi_elab_2 = elab(self.dec_multi_elab_1.out_channels, "dec_multi_elab_2")

        self.expansion_2 = Expansion(self.dec_multi_elab_2.out_channels, name="expansion_2")
        self.dec_multi_elab_3 = elab(self.expansion_2.out_channels + skip2, "dec_multi_elab_3")
        self.dec_multi_elab_4 = elab(self.dec_multi_elab_3.out_channels, "dec_multi_elab_4")

        self.expansion_3 = Expansion(self.dec_multi_elab_4.out_channels, name="expansion_3")
        self.dec_multi_elab_5 = elab(self.expansion_3.out_channels + skip1, "dec_multi_elab_5")
        self.dec_multi_elab_6 = elab(self.dec_multi_elab_5.out_channels, "dec_multi_elab_6")

        # Stepwise output head: 2-channel conv + LeakyReLU, then the 1-channel conv
        self.conv = nn.Conv3d(self.dec_multi_elab_6.out_channels, 2, kernel_size=1)
        self.head_activation = nn.LeakyReLU(NEGATIVE_SLOPE)
        self.final_conv = nn.Conv3d(2, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.enc_multi_elab_2(self.enc_multi_elab_1(x))
        x2 = self.enc_multi_elab_4(self.enc_multi_elab_3(self.reduction_1(x1)))
        x3 = self.enc_multi_elab_6(self.enc_multi_elab_5(self.reduction_2(x2)))
        bottleneck = self.bottleneck_multi_elab_2(self.bottleneck_multi_elab_1(self.reduction_3(x3)))

        # Decoder
        d1 = torch.cat((self.expansion_1(bottleneck), x3), dim=1)
        d1 = self.dec_multi_elab_2(self.dec_multi_elab_1(d1))
        d2 = torch.cat((self.expansion_2(d1), x2), dim=1)
        d2 = self.dec_multi_elab_4(self.dec_multi_elab_3(d2))
        d3 = torch.cat((self.expansion_3(d2), x1), dim=1)
        d3 = self.dec_multi_elab_6(self.dec_multi_elab_5(d3))

        return self.final_conv(self.head_activation(self.conv(d3)))


def max_normalize(output, eps=1e-10):
    """Divide each sample by its maximum (applied to the network output in training and testing)."""
    return output / (output.amax(dim=(2, 3, 4), keepdim=True) + eps)
