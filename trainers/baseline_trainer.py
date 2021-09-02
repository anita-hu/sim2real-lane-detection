"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

TODO: Figure out license
"""
from networks.lane_detector import Resnet, UltraFastLaneDetector
from utils import get_scheduler
from lane_losses import UltraFastLaneDetectionLoss
from lane_metrics import get_metric_dict, update_metrics, reset_metrics
import torch
import torch.nn as nn
from torch.cuda import amp
from torch.nn.parallel import DataParallel
import os


class Baseline_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(Baseline_Trainer, self).__init__()
        lr = hyperparameters['lr']
        # Initiate the networks
        backbone = str(hyperparameters["lane"]["backbone"])
        assert backbone in ['18', '34', '50', '101', '152', '50next', '101next', '50wide', '101wide']
        self.backbone = Resnet(backbone, pretrained=hyperparameters["lane"]["pretrained"])
        input_size = (hyperparameters['input_height'], hyperparameters['input_width'])
        feature_dims = (128, 256, 512) if backbone in ['34', '18'] else (512, 1024, 2048)
        self.lane_model = UltraFastLaneDetector(hyperparameters['lane'], feature_dims=feature_dims,
                                                size=input_size, baseline=True)
        self.lane_loss = UltraFastLaneDetectionLoss(hyperparameters['lane'])

        if hyperparameters['multi_gpu']:
            self.backbone = DataParallel(self.backbone)
            self.lane_model = DataParallel(self.lane_model)

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        model_params = list(self.backbone.parameters()) + list(self.lane_model.parameters())
        self.opt = torch.optim.Adam(model_params, lr=lr, betas=(beta1, beta2),
                                    weight_decay=hyperparameters['weight_decay'])
        self.gen_scheduler = get_scheduler(self.opt, hyperparameters)
        self.dis_scheduler = None
        self.lane_scheduler = None

        # Mixed precision training
        self.scaler = amp.GradScaler() if hyperparameters["mixed_precision"] else None

        # Logging additional losses and metrics
        self.log_dict = {}  # updated per log iter
        self.metric_log_dict = {}  # updated per epoch
        self.metric_dict = get_metric_dict(hyperparameters['lane'])

    def forward(self, x):
        fea = self.backbone(x)
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
        if isinstance(labels, tuple):
            cls_label, seg_label = labels
            cls_out, seg_out = preds
            results = {'cls_out': torch.argmax(cls_out, dim=1), 'cls_label': cls_label,
                       'seg_out': torch.argmax(seg_out, dim=1), 'seg_label': seg_label}
        else:
            cls_label = labels
            cls_out = preds
            results = {'cls_out': torch.argmax(cls_out, dim=1), 'cls_label': cls_label}

        update_metrics(metric_dict, results)
        for me_name, me_op in zip(metric_dict['name'], metric_dict['op']):
            self.metric_log_dict[f"lane_metric_{me_name}_{postfix}"] = me_op.get()

    def _log_lane_losses(self, postfix):
        for k, v in self.lane_loss.current_losses.items():
            self.log_dict[f"lane_{k}_{postfix}"] = v

    def gen_update(self, x_a, x_b, y_a, hyperparameters):
        # assume x_a from simulation data with labels y_a and x_b from real data
        self.opt.zero_grad()
        with amp.autocast(enabled=hyperparameters["mixed_precision"]):
            pred_a = self.forward(x_a)
            self.total_lane_loss = self.lane_loss(pred_a, y_a)
            self._log_lane_losses("x_a")
            self._log_lane_metrics(self.metric_dict, pred_a, y_a, "x_a")

        if hyperparameters["mixed_precision"]:
            self.scaler.scale(self.total_lane_loss).backward()
            self.scaler.step(self.opt)
        else:
            self.total_lane_loss.backward()
            self.opt.step()

    def dis_update(self, x_a, x_b, hyperparameters):
        return

    def update_learning_rate(self):
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        state_dict = torch.load(os.path.join(checkpoint_dir, "gen.pt"))
        self.backbone.load_state_dict(state_dict['b'])
        epoch = state_dict["epoch"]
        # Load lane model
        self.lane_model.load_state_dict(state_dict['lane'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.opt.load_state_dict(state_dict['gen'])
        # Reinitilize schedulers
        self.gen_scheduler = get_scheduler(self.opt, hyperparameters, epoch)
        print('Resume from epoch %d' % epoch)
        return epoch

    def save(self, snapshot_dir, epoch):
        # Save backbone and lane detector
        gen_name = os.path.join(snapshot_dir, 'gen.pt')
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'b': self.backbone.state_dict(), 'lane': self.lane_model.state_dict(), 'epoch': epoch + 1},
                   gen_name)
        torch.save({'gen': self.opt.state_dict(), 'epoch': epoch + 1}, opt_name)
