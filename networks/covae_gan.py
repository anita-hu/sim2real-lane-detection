from torch import nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
from torch.nn.parallel import DataParallel
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass

from networks.resnet_autoencoder import BasicBlockEnc, BasicBlockDec, ResizeConv2d
from networks.unit import Conv2dBlock


class CoVAEGen(nn.Module):
    # VAE architecture
    def __init__(self, input_dim, params, multi_gpu=False):
        super(CoVAEGen, self).__init__()
        norm = params['norm']
        num_blocks = (2, 2)

        # encoder
        self.in_planes = 64
        self.enc_conv1 = nn.Conv2d(input_dim, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.enc_norm1 = nn.BatchNorm2d(64)
        self.enc_layer1 = self._enc_make_layer(BasicBlockEnc, 64, num_blocks[0], stride=1, norm=norm)
        self.enc_layer2_a = self._enc_make_layer(BasicBlockEnc, 128, num_blocks[1], stride=2, norm=norm)
        self.in_planes = 64
        self.enc_layer2_b = self._enc_make_layer(BasicBlockEnc, 128, num_blocks[1], stride=2, norm=norm)

        # decoder
        self.dec_layer2_a = self._dec_make_layer(BasicBlockDec, 64, num_blocks[1], stride=2, norm=norm)
        self.in_planes = 128
        self.dec_layer2_b = self._dec_make_layer(BasicBlockDec, 64, num_blocks[1], stride=2, norm=norm)
        self.dec_layer1 = self._dec_make_layer(BasicBlockDec, 64, num_blocks[0], stride=1, norm=norm)
        self.dec_conv1 = ResizeConv2d(64, input_dim, kernel_size=3, scale_factor=2)

        if multi_gpu:
            for attr in dir(self):
                if "enc_" in attr or "dec_" in attr:
                    setattr(self, attr, DataParallel(getattr(self, attr)))

    def _enc_make_layer(self, BasicBlockEnc, planes, num_blocks, stride, norm):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, stride, norm)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def _dec_make_layer(self, BasicBlockDec, planes, num_blocks, stride, norm):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride, norm)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, images):
        # This is a reduced VAE implementation where we assume the outputs are multivariate Gaussian distribution with
        # mean = hiddens and std_dev = all ones.
        hiddens = self.encode(images)
        if self.training == True:
            noise = Variable(torch.randn(hiddens.size()).cuda(hiddens.data.get_device()))
            images_recon = self.decode(hiddens + noise)
        else:
            images_recon = self.decode(hiddens)
        return images_recon, hiddens

    def encode(self, images, domain="a"):
        assert domain in ["a", "b"]
        x = torch.relu(self.enc_norm1(self.enc_conv1(images)))
        x = self.enc_layer1(x)
        if domain == "a":
            hiddens = self.enc_layer2_a(x)
        else:
            hiddens = self.enc_layer2_b(x)
        noise = Variable(torch.randn(hiddens.size()).cuda(hiddens.data.get_device()))
        return hiddens, noise

    def decode(self, hiddens, domain="a"):
        assert domain in ["a", "b"]
        if domain == "a":
            x = self.dec_layer2_a(hiddens)
        else:
            x = self.dec_layer2_b(hiddens)
        x = self.dec_layer1(x)
        images = torch.tanh(self.dec_conv1(x))
        return images


class CoDis(nn.Module):
    def __init__(self, input_dim, params):
        super(CoDis, self).__init__()
        dim = params['dim']
        norm = params['norm']
        activ = params['activ']
        pad_type = params['pad_type']
        self.dis_conv1_a = Conv2dBlock(input_dim, dim, 4, 2, 1, norm='none', activation=activ, pad_type=pad_type)
        self.dis_conv1_b = Conv2dBlock(input_dim, dim, 4, 2, 1, norm='none', activation=activ, pad_type=pad_type)
        self.dis_conv2_a = Conv2dBlock(dim, dim * 2, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)
        self.dis_conv2_b = Conv2dBlock(dim, dim * 2, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)
        self.dis_conv3 = Conv2dBlock(dim * 2, dim * 4, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)
        self.dis_conv4 = Conv2dBlock(dim * 4, dim * 8, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)
        self.dis_out = nn.Conv2d(dim * 8, 1, 1, 1, 0)

    def forward(self, x, domain):
        assert domain in ["a", "b"]
        if domain == "a":
            x = self.dis_conv1_a(x)
            x = self.dis_conv2_a(x)
        else:
            x = self.dis_conv1_b(x)
            x = self.dis_conv2_b(x)
        x = self.dis_conv3(x)
        x = self.dis_conv4(x)
        return x


class CoMsImageDis(nn.Module):
    # Multi-scale discriminator architecture
    def __init__(self, input_dim, params, multi_gpu=False):
        super(CoMsImageDis, self).__init__()
        self.gan_type = params['gan_type']
        self.num_scales = params['num_scales']
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.cnns = nn.ModuleList()
        for _ in range(self.num_scales):
            cnn = DataParallel(CoDis(input_dim, params)) if multi_gpu else CoDis(input_dim, params)
            self.cnns.append(cnn)

    def forward(self, x, domain):
        assert domain in ["a", "b"]
        outputs = []
        for model in self.cnns:
            outputs.append(model(x, domain))
            x = self.downsample(x)
        return outputs

    def calc_dis_loss(self, input_fake, input_real, domain):
        # calculate the loss to train D
        outs0 = self.forward(input_fake, domain)
        outs1 = self.forward(input_real, domain)
        loss = 0

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 0)**2) + torch.mean((out1 - 1)**2)
            elif self.gan_type == 'nsgan':
                all0 = Variable(torch.zeros_like(out0.data).cuda(), requires_grad=False)
                all1 = Variable(torch.ones_like(out1.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
                                   F.binary_cross_entropy(F.sigmoid(out1), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

    def calc_gen_loss(self, input_fake, domain):
        # calculate the loss to train G
        outs0 = self.forward(input_fake, domain)
        loss = 0
        for it, (out0) in enumerate(outs0):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 1)**2) # LSGAN
            elif self.gan_type == 'nsgan':
                all1 = Variable(torch.ones_like(out0.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss
