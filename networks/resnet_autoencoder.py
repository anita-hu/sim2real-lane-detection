# Modified from: https://github.com/julianstastny/VAE-ResNet18-PyTorch/blob/master/model.py

import torch
from torch import nn
import torch.nn.functional as F
from networks.basic_blocks import Conv2dBlock


class ResizeConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest', norm='bn',
                 pad_type='zeros'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = Conv2dBlock(in_channels, out_channels, kernel_size, stride=1, activation='none', padding=1,
                                pad_type=pad_type, norm=norm)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x


class BasicBlockEnc(nn.Module):
    def __init__(self, in_planes, stride=1, activ='relu', norm='bn', pad_type='zeros'):
        super().__init__()
        planes = in_planes * stride
        self.conv1 = Conv2dBlock(in_planes, planes, kernel_size=3, stride=stride, activation=activ, padding=1,
                                 pad_type=pad_type, norm=norm, use_bias=False)
        self.conv2 = Conv2dBlock(planes, planes, kernel_size=3, stride=1, activation='none', padding=1,
                                 pad_type=pad_type, norm=norm, use_bias=False)
        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = Conv2dBlock(in_planes, planes, kernel_size=1, stride=stride, activation='none', norm=norm,
                                        use_bias=False)
        if activ == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activ == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activ == 'prelu':
            self.activation = nn.PReLU()
        elif activ == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activ == 'tanh':
            self.activation = nn.Tanh()
        else:
            assert 0, "Unsupported activation: {}".format(activ)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.shortcut(x)
        out = self.activation(out)
        return out


class BasicBlockDec(nn.Module):
    def __init__(self, in_planes, stride=1, activ='relu', norm='bn', pad_type='zeros'):
        super().__init__()
        planes = int(in_planes / stride)
        self.conv2 = Conv2dBlock(in_planes, in_planes, kernel_size=3, stride=1, activation=activ, padding=1,
                                 pad_type=pad_type, norm=norm, use_bias=False)
        if stride == 1:
            self.conv1 = Conv2dBlock(in_planes, planes, kernel_size=3, stride=1, activation='none', padding=1,
                                     pad_type=pad_type, norm=norm, use_bias=False)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride, norm=norm,
                                      pad_type=pad_type)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride, norm=norm, pad_type=pad_type),
            )
        if activ == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activ == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activ == 'prelu':
            self.activation = nn.PReLU()
        elif activ == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activ == 'tanh':
            self.activation = nn.Tanh()
        else:
            assert 0, "Unsupported activation: {}".format(activ)

    def forward(self, x):
        out = self.conv2(x)
        out = self.conv1(out)
        out += self.shortcut(x)
        out = self.activation(out)
        return out


class ResNetEnc(nn.Module):
    def __init__(self, num_blocks=(2, 2, 2, 2), nc=3, activ='relu', norm='bn', pad_type='zeros'):
        super().__init__()
        self.in_planes = 64
        self.activ, self.norm, self.pad_type = activ, norm, pad_type
        self.conv1 = Conv2dBlock(nc, 64, kernel_size=3, stride=2, activation=activ, padding=1,
                                 pad_type=pad_type, norm=norm, use_bias=False)
        self.layer1 = self._make_layer(BasicBlockEnc, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlockEnc, 128, num_blocks[1], stride=2)
        # self.layer3 = self._make_layer(BasicBlockEnc, 256, num_blocks[2], stride=2)
        # self.layer4 = self._make_layer(BasicBlockEnc, 512, num_blocks[3], stride=2)

    def _make_layer(self, BasicBlockEnc, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, stride, self.activ, self.norm, self.pad_type)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x2 = self.layer2(x)
        # x3 = self.layer3(x2)
        # x4 = self.layer4(x3)
        return x2


class ResNetDec(nn.Module):
    def __init__(self, num_blocks=(2, 2, 2, 2), nc=3, activ='relu', norm='bn', pad_type='zeros'):
        super().__init__()
        self.in_planes = 128  # 512
        self.activ, self.norm, self.pad_type = activ, norm, pad_type
        # self.layer4 = self._make_layer(BasicBlockDec, 256, num_blocks[3], stride=2)
        # self.layer3 = self._make_layer(BasicBlockDec, 128, num_blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_blocks[0], stride=1)
        self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=2)

    def _make_layer(self, BasicBlockDec, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride, self.activ, self.norm, self.pad_type)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        # x = self.layer4(x)
        # x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = torch.tanh(self.conv1(x))
        return x
