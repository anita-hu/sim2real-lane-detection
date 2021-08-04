# MIT License
#
# Copyright (c) 2020 cfzd (from https://github.com/cfzd/Ultra-Fast-Lane-Detection)
# Copyright (c) 2021 Anita Hu (modifications)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np

import torch
import torchvision
from torch import nn
from networks.norm import LayerNorm
from networks.resnet_autoencoder import BasicBlockEnc


class Resnet(nn.Module):
    def __init__(self, layers, pretrained=False):
        super(Resnet, self).__init__()
        if layers == '18':
            model = torchvision.models.resnet18(pretrained=pretrained)
        elif layers == '34':
            model = torchvision.models.resnet34(pretrained=pretrained)
        elif layers == '50':
            model = torchvision.models.resnet50(pretrained=pretrained)
        elif layers == '101':
            model = torchvision.models.resnet101(pretrained=pretrained)
        elif layers == '152':
            model = torchvision.models.resnet152(pretrained=pretrained)
        elif layers == '50next':
            model = torchvision.models.resnext50_32x4d(pretrained=pretrained)
        elif layers == '101next':
            model = torchvision.models.resnext101_32x8d(pretrained=pretrained)
        elif layers == '50wide':
            model = torchvision.models.wide_resnet50_2(pretrained=pretrained)
        elif layers == '101wide':
            model = torchvision.models.wide_resnet101_2(pretrained=pretrained)
        else:
            raise NotImplementedError

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x2, x3, x4


class ConvNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False, norm='bn'):
        super(ConvNormRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=bias)
        if norm == 'bn':
            norm_layer = nn.BatchNorm2d
        elif norm == 'in':
            norm_layer = nn.InstanceNorm2d
        elif norm == 'ln':
            norm_layer = LayerNorm
        else:
            raise NotImplementedError("Unsupported normalization: {}".format(norm))
        self.norm = norm_layer(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class UltraFastLaneDetector(nn.Module):
    def __init__(self, hyperparams, feature_dims=None, size=(288, 800), baseline=False):
        super(UltraFastLaneDetector, self).__init__()

        norm = hyperparams["norm"]
        num_gridding = hyperparams["griding_num"]+1
        row_anchors = hyperparams["cls_num_per_lane"]
        num_lanes = hyperparams["num_lanes"]
        self.cls_dim = (num_gridding, row_anchors, num_lanes)
        self.use_aux = hyperparams["use_aux"]
        self.total_dim = int(np.prod(self.cls_dim))
        self.baseline = baseline

        # input : nchw,
        # output: (w+1) * sample_rows * 4

        if not baseline:
            self.layer2 = nn.Sequential(BasicBlockEnc(128, 1, norm=norm))
            self.layer3 = nn.Sequential(BasicBlockEnc(128, 2, norm=norm), BasicBlockEnc(256, 1, norm=norm),
                                        BasicBlockEnc(256, 1, norm=norm))
            self.layer4 = nn.Sequential(BasicBlockEnc(256, 2, norm=norm), BasicBlockEnc(512, 1, norm=norm))

        if self.use_aux:
            self.aux_header2 = nn.Sequential(
                ConvNormRelu(feature_dims[0], 128, kernel_size=3, stride=1, padding=1, norm=norm),
                ConvNormRelu(128, 128, 3, padding=1, norm=norm),
                ConvNormRelu(128, 128, 3, padding=1, norm=norm),
                ConvNormRelu(128, 128, 3, padding=1, norm=norm),
            )
            self.aux_header3 = nn.Sequential(
                ConvNormRelu(feature_dims[1], 128, kernel_size=3, stride=1, padding=1, norm=norm),
                ConvNormRelu(128, 128, 3, padding=1, norm=norm),
                ConvNormRelu(128, 128, 3, padding=1, norm=norm),
            )
            self.aux_header4 = nn.Sequential(
                ConvNormRelu(feature_dims[2], 128, kernel_size=3, stride=1, padding=1, norm=norm),
                ConvNormRelu(128, 128, 3, padding=1, norm=norm),
            )
            self.aux_combine = nn.Sequential(
                ConvNormRelu(384, 256, 3, padding=2, dilation=2, norm=norm),
                ConvNormRelu(256, 128, 3, padding=2, dilation=2, norm=norm),
                ConvNormRelu(128, 128, 3, padding=2, dilation=2, norm=norm),
                ConvNormRelu(128, 128, 3, padding=4, dilation=4, norm=norm),
                nn.Conv2d(128, self.cls_dim[-1] + 1, 1)
                # output : n, num_of_lanes+1, h, w
            )
            initialize_weights(self.aux_header2, self.aux_header3, self.aux_header4, self.aux_combine)

        self.pool = nn.Conv2d(feature_dims[2], 8, 1)

        self.cls_in = int(size[0]/16*size[1]/16*8) if not baseline else int(size[0]/32*size[1]/32*8)
        self.cls = nn.Sequential(
            nn.Linear(self.cls_in, 2048),
            nn.ReLU(),
            nn.Linear(2048, self.total_dim),
        )
        initialize_weights(self.cls)

    def forward(self, x):
        if self.baseline:
            x2, x3, fea = x
        else:
            x2 = self.layer2(x)
            x3 = self.layer3(x2)
            fea = self.layer4(x3)

        if self.use_aux:
            x2 = self.aux_header2(x2)
            x3 = self.aux_header3(x3)
            x3 = nn.functional.interpolate(x3, scale_factor=2, mode='bilinear')
            x4 = self.aux_header4(fea)
            x4 = nn.functional.interpolate(x4, scale_factor=4, mode='bilinear')
            aux_seg = torch.cat([x2, x3, x4], dim=1)
            aux_seg = self.aux_combine(aux_seg)
        else:
            aux_seg = None

        fea = self.pool(fea).view(-1, self.cls_in)
        group_cls = self.cls(fea).view(-1, *self.cls_dim)

        if self.use_aux:
            return group_cls, aux_seg

        return group_cls


def initialize_weights(*models):
    for model in models:
        real_init_weights(model)


def real_init_weights(m):
    if isinstance(m, list):
        for mini_m in m:
            real_init_weights(mini_m)
    else:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0.0, std=0.01)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Module):
            for mini_m in m.children():
                real_init_weights(mini_m)
        else:
            print('Unknown module', m)
