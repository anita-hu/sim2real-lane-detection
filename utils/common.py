"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
import math
import time
import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler
import torch.nn.init as init
import torchvision.utils as vutils
import yaml
import wandb

from networks.unit import Vgg16
from utils.lr_schedulers import MultiStepLR, CosineAnnealingLR


# Methods
# get_config                : load yaml file
# write_2images             : save output image
# prepare_sub_folder        : create checkpoints for saving outputs
# write_loss
# load_vgg16
# vgg_preprocess
# get_scheduler
# weights_init


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.safe_load(stream)


def make_grid(image_outputs, display_image_num):
    image_outputs = [images.expand(-1, 3, -1, -1) for images in image_outputs]  # expand gray-scale images to 3 channels
    image_tensor = torch.cat([images[:display_image_num] for images in image_outputs], 0)
    image_grid = vutils.make_grid(image_tensor.data, nrow=display_image_num, padding=0, normalize=True)
    return image_grid


def write_2images(image_outputs, display_image_num, epoch, postfix, step=None):
    n = len(image_outputs)
    a2b = wandb.Image(make_grid(image_outputs[0:n // 2], display_image_num))
    b2a = wandb.Image(make_grid(image_outputs[n // 2:n], display_image_num))
    wandb.log({f"a2b_{postfix}": a2b, f"b2a_{postfix}": b2a, "epoch": epoch}, step=step)


def prepare_sub_folder(output_directory):
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory


def write_loss(iterations, trainer):
    members = [attr for attr in dir(trainer) \
               if not callable(getattr(trainer, attr)) and not attr.startswith("_") and (
                           'loss' in attr or 'grad' in attr or 'nwd' in attr)]
    log_dict = {m: getattr(trainer, m) for m in members}
    log_dict.update(trainer.log_dict)  # additional losses
    wandb.log(log_dict, step=iterations)


def load_vgg16(model_path):
    """
    Downloaded model from https://github.com/abhiskk/fast-neural-style/blob/master/neural_style/utils.py
    Converted from .t7 to .pth using https://github.com/clcarwin/convert_torch_to_pytorch
    """
    vgg = Vgg16()
    weights = torch.load(model_path)
    for src, dst in zip(weights.values(), vgg.parameters()):
        dst.data[:] = src
    return vgg


def vgg_preprocess(batch):
    tensortype = type(batch.data)
    (r, g, b) = torch.chunk(batch, 3, dim=1)
    batch = torch.cat((b, g, r), dim=1)  # convert RGB to BGR
    batch = (batch + 1) * 255 * 0.5  # [-1, 1] -> [0, 255]
    mean = tensortype(batch.data.size()).cuda()
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    batch = batch.sub(Variable(mean))  # subtract mean
    return batch


def get_scheduler(optimizer, hyperparameters, epochs=-1):
    if 'lr_policy' not in hyperparameters or hyperparameters['lr_policy'] == 'constant':
        scheduler = None  # constant scheduler
    elif hyperparameters['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=hyperparameters['step_size'],
                                        gamma=hyperparameters['gamma'], last_epoch=epochs)
    elif hyperparameters['lr_policy'] == 'cos':
        iterations = epochs * hyperparameters['iter_per_epoch'] if epochs > 0 else -1
        total_iters = hyperparameters['max_epoch'] * hyperparameters['iter_per_epoch']
        scheduler = CosineAnnealingLR(optimizer, total_iters, eta_min=0, warmup=hyperparameters['warmup'],
                                      warmup_iters=hyperparameters['warmup_iters'], last_iter=iterations)
    elif hyperparameters['lr_policy'] == 'multi':
        iterations = epochs * hyperparameters['iter_per_epoch'] if epochs > 0 else -1
        scheduler = MultiStepLR(optimizer, hyperparameters['lr_steps'], gamma=hyperparameters['gamma'],
                                iters_per_epoch=hyperparameters['iter_per_epoch'], warmup=hyperparameters['warmup'],
                                warmup_iters=hyperparameters['warmup_iters'], last_iter=iterations)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', hyperparameters['lr_policy'])
    return scheduler


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun


class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))
