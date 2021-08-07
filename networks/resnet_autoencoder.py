# Modified from: https://github.com/julianstastny/VAE-ResNet18-PyTorch/blob/master/model.py
import torch
from torch import nn
import torch.nn.functional as F
from networks.norm import LayerNorm, AdaptiveInstanceNorm2d, NoneNorm


class ResizeConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x


class BasicBlockEnc(nn.Module):
    def __init__(self, in_planes, stride=1, norm='bn'):
        super().__init__()

        planes = in_planes * stride

        if norm == 'bn':
            norm_layer = nn.BatchNorm2d
        elif norm == 'in':
            norm_layer = nn.InstanceNorm2d
        elif norm == 'ln':
            norm_layer = LayerNorm
        elif norm == 'adain':
            norm_layer = AdaptiveInstanceNorm2d
        elif norm == 'none':
            norm_layer = NoneNorm
        else:
            raise NotImplementedError("Unsupported normalization: {}".format(norm))
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = norm_layer(planes)

        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes)
            )

    def forward(self, x):
        out = torch.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class BasicBlockDec(nn.Module):
    def __init__(self, in_planes, stride=1, norm='bn'):
        super().__init__()

        planes = int(in_planes / stride)

        if norm == 'bn':
            norm_layer = nn.BatchNorm2d
        elif norm == 'in':
            norm_layer = nn.InstanceNorm2d
        elif norm == 'ln':
            norm_layer = LayerNorm
        elif norm == 'adain':
            norm_layer = AdaptiveInstanceNorm2d
        elif norm == 'none':
            norm_layer = NoneNorm
        else:
            raise NotImplementedError("Unsupported normalization: {}".format(norm))
        
        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = norm_layer(in_planes)

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.norm1 = norm_layer(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.norm1 = norm_layer(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                norm_layer(planes)
            )

    def forward(self, x):
        out = torch.relu(self.norm2(self.conv2(x)))
        out = self.norm1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNetEnc(nn.Module):
    def __init__(self, num_blocks=(2, 2, 2, 2), nc=3, norm='bn'):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(nc, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(64)  # nn.InstanceNorm2d(64)
        self.layer1 = self._make_layer(BasicBlockEnc, 64, num_blocks[0], stride=1, norm=norm)
        self.layer2 = self._make_layer(BasicBlockEnc, 128, num_blocks[1], stride=2, norm=norm)
        # self.layer3 = self._make_layer(BasicBlockEnc, 256, num_blocks[2], stride=2, norm=norm)
        # self.layer4 = self._make_layer(BasicBlockEnc, 512, num_blocks[3], stride=2, norm=norm)

    def _make_layer(self, BasicBlockEnc, planes, num_blocks, stride, norm):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, stride, norm)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.norm1(self.conv1(x)))
        x = self.layer1(x)
        x2 = self.layer2(x)
        # x3 = self.layer3(x2)
        # x4 = self.layer4(x3)
        return x2


class ResNetDec(nn.Module):
    def __init__(self, num_blocks=(2, 2, 2, 2), nc=3, norm='bn'):
        super().__init__()
        self.in_planes = 128  # 512
        # self.layer4 = self._make_layer(BasicBlockDec, 256, num_blocks[3], stride=2, norm=norm)
        # self.layer3 = self._make_layer(BasicBlockDec, 128, num_blocks[2], stride=2, norm=norm)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_blocks[1], stride=2, norm=norm)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_blocks[0], stride=1, norm=norm)
        self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=2)

    def _make_layer(self, BasicBlockDec, planes, num_blocks, stride, norm):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride, norm)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        # x = self.layer4(x)
        # x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = torch.tanh(self.conv1(x))
        return x
