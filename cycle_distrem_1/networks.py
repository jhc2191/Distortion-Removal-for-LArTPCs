# networks.py

import torch
import torch.nn.functional as F
from torch import nn

##################################################################################
# Base Network Class
##################################################################################
# networks.py

class BaseNet(nn.Module):
    def __init__(self, input_dim=3, output_dim=3, n_downsample=3, skip=True, dim=32, n_res=8,
                 norm='in', activ='relu', pad_type='reflect', final_activ='tanh'):
        super(BaseNet, self).__init__()

        self.skip = skip
        self.norm_type = norm
        self.activ_type = activ
        self.pad_type = pad_type

        # Initial convolution
        self.conv_in = Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)

        # Downsampling layers
        self.down_blocks = nn.ModuleList()
        for i in range(n_downsample):
            self.down_blocks.append(
                Conv2dBlock(dim, dim * 2, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)
            )
            dim *= 2  # Double the dimension at each downsampling step

        # Residual blocks
        self.res_blocks = nn.ModuleList()
        for i in range(n_res):
            self.res_blocks.append(ResBlock(dim, norm=norm, activation=activ, pad_type=pad_type))

        # Upsampling layers
        self.up_blocks = nn.ModuleList()
        for i in range(n_downsample):
            self.up_blocks.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    Conv2dBlock(dim, dim // 2, 5, 1, 2, norm=norm, activation=activ, pad_type=pad_type)
                )
            )
            dim = dim // 2  # Halve the dimension at each upsampling step

        # Output convolution
        self.conv_out = Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation=final_activ, pad_type=pad_type)

    def forward(self, x):
        # Initial convolution
        x = self.conv_in(x)

        # Downsampling with skip connections stored before downsampling
        skips = []
        for down_block in self.down_blocks:
            if self.skip:
                skips.append(x)
            x = down_block(x)

        # Residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)

        # Upsampling with concatenated skip connections
        for up_block in self.up_blocks:
            x = up_block(x)
            if self.skip and len(skips) > 0:
                skip = skips.pop()
                x = torch.cat([x, skip], dim=1)  # Concatenate along the channel dimension

                # Reduce channels after concatenation
                x = Conv2dBlock(
                    input_dim=x.shape[1],
                    output_dim=x.shape[1] // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm=self.norm_type,
                    activation=self.activ_type,
                    pad_type=self.pad_type
                ).to(x.device)(x)

        # Output convolution
        x = self.conv_out(x)

        return x


##################################################################################
# Correction Network
##################################################################################
class CorrectionNet(BaseNet):
    def __init__(self, **kwargs):
        super(CorrectionNet, self).__init__(**kwargs)
        # No additional modifications needed for now

##################################################################################
# Distortion Network
##################################################################################
class DistortionNet(BaseNet):
    def __init__(self, **kwargs):
        super(DistortionNet, self).__init__(**kwargs)
        # No additional modifications needed for now

##################################################################################
# Basic Blocks
##################################################################################
class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        model = []
        model += [
            Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)
        ]
        model += [
            Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True

        # Initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, f"Unsupported padding type: {pad_type}"

        # Initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim, affine=True)
        elif norm == 'ln':
            self.norm = nn.LayerNorm([norm_dim, 1, 1])
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, f"Unsupported normalization: {norm}"

        # Initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, f"Unsupported activation: {activation}"

        # Initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x
