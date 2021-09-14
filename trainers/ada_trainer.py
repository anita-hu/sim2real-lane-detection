"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

TODO: Figure out license
"""
from networks.ada_discriminator import FeatureDis
from networks.resnet_autoencoder import ResNetEnc
from networks.lane_detector import UltraFastLaneDetector
from utils import weights_init, get_scheduler
from lane_losses import UltraFastLaneDetectionLoss
from lane_metrics import get_metric_dict, update_metrics, reset_metrics
import torch
import torch.nn as nn
from torch.cuda import amp
from torch.nn.parallel import DataParallel
import os


class ADA_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(ADA_Trainer, self).__init__()
        # Initiate the networks
        nc = hyperparameters['input_dim_b']
        n_res = hyperparameters['gen']['n_res']
        activ = hyperparameters['gen']['activ']
        norm = hyperparameters['gen']['norm']
        pad_type = hyperparameters['gen']['pad_type']
        self.gen_b = ResNetEnc(num_blocks=n_res, nc=nc, activ=activ, norm=norm,
                               pad_type=pad_type)  # encoder for both domains
        if hyperparameters['multi_gpu']:
            self.gen_b = DataParallel(self.gen_b)
        self.dis = FeatureDis(128, hyperparameters['dis_fea'],
                              multi_gpu=hyperparameters['multi_gpu'])  # feature discriminator
        input_size = (hyperparameters['input_height'], hyperparameters['input_width'])
        self.lane_model = UltraFastLaneDetector(hyperparameters['lane'], feature_dims=(128, 256, 512),
                                                size=input_size)
        self.lane_loss = UltraFastLaneDetectionLoss(hyperparameters['lane'])

        if hyperparameters['multi_gpu']:
            self.lane_model = DataParallel(self.lane_model)

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        lr = hyperparameters['dis_fea']['lr']
        self.dis_opt = torch.optim.Adam([p for p in self.dis.parameters() if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        lr = hyperparameters['gen']['lr']
        self.gen_opt = torch.optim.Adam([p for p in self.gen_b.parameters() if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        lr = hyperparameters['lane']['lr']
        self.lane_opt = torch.optim.Adam([p for p in self.lane_model.parameters() if p.requires_grad],
                                         lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)
        self.lane_scheduler = get_scheduler(self.lane_opt, hyperparameters)
        self.warmup_iteration = hyperparameters['dis_fea']['warmup_iters']  # iterations to only train supervised task

        # Mixed precision training
        self.scaler = amp.GradScaler() if hyperparameters["mixed_precision"] else None

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))
        self.dis.apply(weights_init('gaussian'))

        # Logging additional losses and metrics
        self.log_dict = {}  # updated per log iter
        self.metric_log_dict = {}  # updated per epoch
        self.metric_dict = get_metric_dict(hyperparameters['lane'])

    def forward(self, x):
        fea = self.gen_b(x)
        preds = self.lane_model(fea)
        return preds

    def eval_lanes(self, x):
        self.eval()
        preds = self.forward(x)
        self.train()
        return preds

    def reset_metrics(self):
        reset_metrics(self.metric_dict)

    def _log_lane_metrics(self, metric_dict, preds, labels, postfix):
        det_label, cls_label, seg_label = labels
        det_out, cls_out, seg_out = preds
        if cls_out is not None and seg_out is not None:
            results = {'det_out': torch.argmax(det_out, dim=1), 'det_label': det_label,
                       'seg_out': torch.argmax(seg_out, dim=1), 'seg_label': seg_label,
                       'cls_out': torch.argmax(cls_out, dim=1), 'cls_label': cls_label}
        elif seg_out is not None:
            results = {'det_out': torch.argmax(det_out, dim=1), 'det_label': det_label,
                       'seg_out': torch.argmax(seg_out, dim=1), 'seg_label': seg_label}
        elif cls_out is not None:
            results = {'det_out': torch.argmax(det_out, dim=1), 'det_label': det_label,
                       'cls_out': torch.argmax(cls_out, dim=1), 'cls_label': cls_label}
        else:
            results = {'det_out': torch.argmax(det_out, dim=1), 'det_label': det_label}

        update_metrics(metric_dict, results)
        for me_name, me_op in zip(metric_dict['name'], metric_dict['op']):
            self.metric_log_dict[f"eval_metrics/lane_metric_{me_name}_{postfix}"] = me_op.get()

    def _log_lane_losses(self, postfix):
        for k, v in self.lane_loss.current_losses.items():
            self.log_dict[f"lane_loss/lane_{k}_{postfix}"] = v

    def gen_update(self, x_a, x_b, y_a, hyperparameters):
        # assume x_a from simulation data with labels y_a and x_b from real data
        self.gen_opt.zero_grad()
        self.lane_opt.zero_grad()
        with amp.autocast(enabled=hyperparameters["mixed_precision"]):
            # encode
            fea_a = self.gen_b(x_a)
            fea_b = self.gen_b(x_b)
            # lane detection
            pred_a = self.lane_model(fea_a)
            # lane detection loss
            self.total_lane_loss = self.lane_loss(pred_a, y_a)
            self._log_lane_losses("x_a")
            self._log_lane_metrics(self.metric_dict, pred_a, y_a, "x_a")
            # GAN loss
            self.loss_gen_adv = self.dis.calc_gen_loss(fea_a, fea_b)
            # total loss
            self.warmup_iteration -= 1
            self.warmup_iteration = min(0, self.warmup_iteration)
            if self.warmup_iteration > 0:
                hyperparameters['gan_w'] = 0
            self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv + self.total_lane_loss

        if hyperparameters["mixed_precision"]:
            self.scaler.scale(self.loss_gen_total).backward()
            self.scaler.step(self.gen_opt)
            self.scaler.step(self.lane_opt)
        else:
            self.loss_gen_total.backward()
            self.gen_opt.step()
            self.lane_opt.step()

    def dis_update(self, x_a, x_b, hyperparameters):
        self.dis_opt.zero_grad()
        with amp.autocast(enabled=hyperparameters["mixed_precision"]):
            # encode
            fea_a = self.gen_b(x_a)
            fea_b = self.gen_b(x_b)
            # D loss
            self.loss_dis_total = self.dis.calc_dis_loss(fea_a.detach(), fea_b.detach())

        if hyperparameters["mixed_precision"]:
            self.scaler.scale(self.loss_dis_total).backward()
            self.scaler.step(self.dis_opt)
        else:
            self.loss_dis_total.backward()
            self.dis_opt.step()

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()
        if self.lane_scheduler is not None:
            self.lane_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        state_dict = torch.load(os.path.join(checkpoint_dir, "gen.pt"))
        self.gen_b.load_state_dict(state_dict['b'])
        epoch = state_dict["epoch"]
        # Load lane model
        self.lane_model.load_state_dict(state_dict['lane'])
        # Load discriminators
        state_dict = torch.load(os.path.join(checkpoint_dir, "dis.pt"))
        self.dis.load_state_dict(state_dict)
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        self.lane_opt.load_state_dict(state_dict['lane'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, epoch)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, epoch)
        self.lane_scheduler = get_scheduler(self.lane_opt, hyperparameters, epoch)
        print('Resume from epoch %d' % epoch)
        return epoch

    def save(self, snapshot_dir, epoch):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen.pt')
        dis_name = os.path.join(snapshot_dir, 'dis.pt')
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'b': self.gen_b.state_dict(), 'lane': self.lane_model.state_dict(),
                    'epoch': epoch + 1}, gen_name)
        torch.save({'dis': self.dis.state_dict(), 'epoch': epoch + 1}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict(),
                    'lane': self.lane_opt.state_dict(), 'epoch': epoch + 1}, opt_name)
