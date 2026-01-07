"""
U-Net and U-Net++ implementations for cilia segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard U-Net architecture.

    Args:
        n_channels: Number of input channels (3 for RGB)
        n_classes: Number of output classes (1 for binary segmentation)
        bilinear: Use bilinear upsampling instead of transposed convolutions
        base_channels: Number of channels in first layer (default 64)
    """

    def __init__(self, n_channels=3, n_classes=1, bilinear=True, base_channels=64):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor)
        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear)
        self.outc = OutConv(base_channels, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class NestedUp(nn.Module):
    """U-Net++ nested upsampling block"""

    def __init__(self, in_channels_list, out_channels, bilinear=True):
        super().__init__()
        total_in_channels = sum(in_channels_list)

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(total_in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels_list[0], in_channels_list[0], kernel_size=2, stride=2)
            self.conv = DoubleConv(total_in_channels, out_channels)

    def forward(self, upsampled, *skip_connections):
        upsampled = self.up(upsampled)

        # Pad if needed
        for skip in skip_connections:
            diffY = skip.size()[2] - upsampled.size()[2]
            diffX = skip.size()[3] - upsampled.size()[3]
            upsampled = F.pad(upsampled, [diffX // 2, diffX - diffX // 2,
                                          diffY // 2, diffY - diffY // 2])
            break

        x = torch.cat([upsampled] + list(skip_connections), dim=1)
        return self.conv(x)


class UNetPlusPlus(nn.Module):
    """
    U-Net++ (Nested U-Net) architecture for improved segmentation.

    Reference: Zhou et al. "UNet++: A Nested U-Net Architecture for Medical Image Segmentation"

    Args:
        n_channels: Number of input channels
        n_classes: Number of output classes
        deep_supervision: Use deep supervision with multiple outputs
        base_channels: Number of channels in first layer
    """

    def __init__(self, n_channels=3, n_classes=1, deep_supervision=False, base_channels=64):
        super(UNetPlusPlus, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.deep_supervision = deep_supervision

        nb_filter = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8, base_channels * 16]

        # Encoder
        self.conv0_0 = DoubleConv(n_channels, nb_filter[0])
        self.conv1_0 = Down(nb_filter[0], nb_filter[1])
        self.conv2_0 = Down(nb_filter[1], nb_filter[2])
        self.conv3_0 = Down(nb_filter[2], nb_filter[3])
        self.conv4_0 = Down(nb_filter[3], nb_filter[4])

        # Nested decoder - level 1
        self.conv0_1 = NestedUp([nb_filter[1], nb_filter[0]], nb_filter[0])
        self.conv1_1 = NestedUp([nb_filter[2], nb_filter[1]], nb_filter[1])
        self.conv2_1 = NestedUp([nb_filter[3], nb_filter[2]], nb_filter[2])
        self.conv3_1 = NestedUp([nb_filter[4], nb_filter[3]], nb_filter[3])

        # Nested decoder - level 2
        self.conv0_2 = NestedUp([nb_filter[1], nb_filter[0], nb_filter[0]], nb_filter[0])
        self.conv1_2 = NestedUp([nb_filter[2], nb_filter[1], nb_filter[1]], nb_filter[1])
        self.conv2_2 = NestedUp([nb_filter[3], nb_filter[2], nb_filter[2]], nb_filter[2])

        # Nested decoder - level 3
        self.conv0_3 = NestedUp([nb_filter[1], nb_filter[0], nb_filter[0], nb_filter[0]], nb_filter[0])
        self.conv1_3 = NestedUp([nb_filter[2], nb_filter[1], nb_filter[1], nb_filter[1]], nb_filter[1])

        # Nested decoder - level 4
        self.conv0_4 = NestedUp([nb_filter[1], nb_filter[0], nb_filter[0], nb_filter[0], nb_filter[0]], nb_filter[0])

        if self.deep_supervision:
            self.final1 = OutConv(nb_filter[0], n_classes)
            self.final2 = OutConv(nb_filter[0], n_classes)
            self.final3 = OutConv(nb_filter[0], n_classes)
            self.final4 = OutConv(nb_filter[0], n_classes)
        else:
            self.final = OutConv(nb_filter[0], n_classes)

    def forward(self, x):
        # Encoder
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(x0_0)
        x2_0 = self.conv2_0(x1_0)
        x3_0 = self.conv3_0(x2_0)
        x4_0 = self.conv4_0(x3_0)

        # Nested decoder
        x0_1 = self.conv0_1(x1_0, x0_0)
        x1_1 = self.conv1_1(x2_0, x1_0)
        x2_1 = self.conv2_1(x3_0, x2_0)
        x3_1 = self.conv3_1(x4_0, x3_0)

        x0_2 = self.conv0_2(x1_1, x0_0, x0_1)
        x1_2 = self.conv1_2(x2_1, x1_0, x1_1)
        x2_2 = self.conv2_2(x3_1, x2_0, x2_1)

        x0_3 = self.conv0_3(x1_2, x0_0, x0_1, x0_2)
        x1_3 = self.conv1_3(x2_2, x1_0, x1_1, x1_2)

        x0_4 = self.conv0_4(x1_3, x0_0, x0_1, x0_2, x0_3)

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]
        else:
            output = self.final(x0_4)
            return output


# Small variants for faster training
class UNetSmall(UNet):
    """Smaller U-Net with 32 base channels"""

    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        super().__init__(n_channels, n_classes, bilinear, base_channels=32)


class UNetTiny(UNet):
    """Tiny U-Net with 16 base channels"""

    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        super().__init__(n_channels, n_classes, bilinear, base_channels=16)
