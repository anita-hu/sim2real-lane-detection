import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel
from networks.unit import Conv2dBlock


class FeatureDis(nn.Module):
    def __init__(self, input_dim, params, multi_gpu=False):
        super(FeatureDis, self).__init__()
        n_layer = params['n_layer']
        pad_type = params['pad_type']
        self.dim = params['dim']
        self.input_dim = input_dim

        layers = [Conv2dBlock(input_dim, self.dim, kernel_size=3, stride=2, padding=1, pad_type=pad_type)]
        for i in range(n_layer - 1):
            layers.append(Conv2dBlock(self.dim, self.dim * 2, kernel_size=3, stride=2, padding=1, pad_type=pad_type))
            self.dim *= 2
        layers.append(nn.AdaptiveAvgPool2d(1))
        self.conv_net = nn.Sequential(*layers)
        self.fc = nn.Linear(self.dim, 1)
        if multi_gpu:
            self.conv_net = DataParallel(self.conv_net)
            self.fc = DataParallel(self.fc)

    def forward(self, x):
        x = self.conv_net(x).view(-1, self.dim)
        x = self.fc(x)
        return x

    def calc_dis_loss(self, input_fake, input_real):
        # calculate the loss to train D
        out0 = self.forward(input_fake)
        out1 = self.forward(input_real)
        all0 = torch.zeros_like(out0).cuda().detach()
        all1 = torch.ones_like(out1).cuda().detach()
        loss = torch.mean(F.binary_cross_entropy_with_logits(out0, all0) +
                          F.binary_cross_entropy_with_logits(out1, all1))

        return loss

    def calc_gen_loss(self, input_fake, input_real):
        # calculate the loss to train G
        out0 = self.forward(input_fake)
        out1 = self.forward(input_real)
        all0 = torch.zeros_like(out1).cuda().detach()
        all1 = torch.ones_like(out0).cuda().detach()
        loss = torch.mean(F.binary_cross_entropy_with_logits(out0, all1) +
                          F.binary_cross_entropy_with_logits(out1, all0))

        return loss
