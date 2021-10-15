"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
import math
import time
import numpy as np
import scipy
import cv2
import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.nn import functional as F
import torch.nn.init as init
import torchvision.utils as vutils
import torchvision.transforms as tf
import yaml
import wandb
from matplotlib import cm

from networks.unit import Vgg16
from utils.lr_schedulers import MultiStepLR, CosineAnnealingLR

# Methods
# get_config                : load yaml file
# write_translation_images  : log example visuals of i2i translation
# write_lane_images         : log example visuals of lane predictions
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


def write_lane_images(images, display_num, epoch, postfix, step=None):
    image_grid = vutils.make_grid(images.data, nrow=display_num, padding=0)
    imgs = wandb.Image(image_grid)
    wandb.log({f"lanes_{postfix}": imgs, "epoch": epoch}, step=step)


def write_translation_images(image_outputs, display_image_num, epoch, postfix, step=None):
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
               if not callable(getattr(trainer, attr)) and not attr.startswith("_") and ('loss' in attr)]
    loss_folder = []
    for m in members:
        if 'lane' in m:
            loss_folder.append('lane_loss')
        elif 'dis' in m:
            loss_folder.append('dis_loss')
        else:
            loss_folder.append('gen_loss')
    log_dict = {f'{f}/{m}': getattr(trainer, m) for f, m in zip(loss_folder, members)}
    log_dict.update(trainer.log_dict)  # additional logs
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


def viz_lanes(det_out, seg_out, dataset_root, names, model_in_dims, griding_num, row_anchor, num_lanes):
    # Send to CPU
    det_out, seg_out = det_out.cpu(), seg_out.cpu()

    # Generate colors for individual lanes
    c_idx = np.linspace(0.0, 1.0, num_lanes) # 1 color for each lane
    colors = [tuple(int(255 * c) for c in cm.gist_rainbow(idx)[:-1]) for idx in c_idx] # remove alpha from rgba and convert to 0-255

    det_images = []
    seg_images = []
    n_images = det_out.shape[0]

    if seg_out is not None:
        # Get predictions as one hot; seg_out shape: (batch_size, num_lanes+1, h, w)
        aux_preds = torch.argmax(seg_out, dim=1)
        masks = F.one_hot(aux_preds.view(aux_preds.shape[0], -1), num_classes=seg_out.shape[1])
        masks = masks.bool().transpose(1, 2) # (n_images, num_lanes+1, h*w)
        masks = torch.reshape(masks, seg_out.shape)

        # Remove no lane mask
        masks = masks[:, 1:]

        # Resize to model input size for display
        resize_mask = tf.Resize((model_in_dims[1], model_in_dims[0]), interpolation=tf.InterpolationMode.NEAREST)
        masks = resize_mask(masks)

    for x in range(n_images):
        col_sample = np.linspace(0, model_in_dims[0] - 1, griding_num)
        col_sample_w = col_sample[1] - col_sample[0]

        out_j = det_out[x].data.cpu().numpy()
        out_j = out_j[:, ::-1, :]
        prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
        idx = np.arange(griding_num) + 1
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0)
        out_j = np.argmax(out_j, axis=0)
        loc[out_j == griding_num] = 0
        out_j = loc

        filename = os.path.join(dataset_root, names[x])
        img = cv2.imread(filename)  # read the original file from the dataset
        if img is None:
            raise ValueError(f"OpenCV failed to read a frame. Does the file {filename} exist?")
        # Resize image to model input size for display
        img = cv2.resize(img, model_in_dims, interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if seg_out is not None:
            seg_img = img.copy()
            seg_img = torch.from_numpy(seg_img).permute(2, 0, 1)
            seg_img = vutils.draw_segmentation_masks(seg_img, masks=masks[x], alpha=0.7, colors=colors)
            seg_images.append(seg_img)

        for i in range(out_j.shape[1]):
            if np.sum(out_j[:, i] != 0) > 2:
                for k in range(out_j.shape[0]):
                    if out_j[k, i] > 0:
                        ppp = (int(out_j[k, i] * col_sample_w) - 1, 
                               int(row_anchor[num_lanes-1-k]) - 1)
                        img = cv2.circle(img, ppp, 5, colors[i], -1)
        det_images.append(img)

    # Need to convert to tensor for torchvision grid function
    disp_images = np.stack(det_images).transpose((0, 3, 1, 2)) # (num_images, 3, H, W)
    disp_images = torch.from_numpy(disp_images).float() / 255.0
    if seg_out is not None:
        seg_images = torch.stack(seg_images).float() / 255.0
        disp_images = torch.cat((disp_images, seg_images)) # (2*num_images, 3, H, W)

    return disp_images


class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))
