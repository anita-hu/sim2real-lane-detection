"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

TODO: Figure out license
"""
from networks.unit import MsImageDis, VAEGen
from networks.lane_detector import UltraFastLaneDetector
from utils import weights_init, vgg_preprocess, load_vgg16, get_scheduler
from lane_losses import UltraFastLaneDetectionLoss
from lane_metrics import get_metric_dict, update_metrics, reset_metrics
import torch
import torch.nn as nn
from torch.cuda import amp
from torch.nn.parallel import DataParallel
import os


class UNIT_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(UNIT_Trainer, self).__init__()
        # Initiate the networks
        self.gen_a = VAEGen(hyperparameters['input_dim_a'], hyperparameters['gen'],
                            multi_gpu=hyperparameters['multi_gpu'])  # auto-encoder for domain a
        self.gen_b = VAEGen(hyperparameters['input_dim_b'], hyperparameters['gen'],
                            multi_gpu=hyperparameters['multi_gpu'])  # auto-encoder for domain b
        self.dis_a = MsImageDis(hyperparameters['input_dim_a'], hyperparameters['dis'],
                                multi_gpu=hyperparameters['multi_gpu'])  # discriminator for domain a
        self.dis_b = MsImageDis(hyperparameters['input_dim_b'], hyperparameters['dis'],
                                multi_gpu=hyperparameters['multi_gpu'])  # discriminator for domain b
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        input_size = (hyperparameters['input_height'], hyperparameters['input_width'])
        self.lane_model = UltraFastLaneDetector(hyperparameters['lane'], feature_dims=(128, 256, 512),
                                                size=input_size)
        self.lane_loss = UltraFastLaneDetectionLoss(hyperparameters['lane'])

        if hyperparameters['multi_gpu']:
            self.lane_model = DataParallel(self.lane_model)

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
        lr = hyperparameters['dis']['lr']
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        lr = hyperparameters['gen']['lr']
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        lr = hyperparameters['lane']['lr']
        self.lane_opt = torch.optim.Adam([p for p in self.lane_model.parameters() if p.requires_grad],
                                         lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        if 'warmup_iters' in hyperparameters['dis']:
            hyperparameters['warmup_iters'] = hyperparameters['dis']['warmup_iters']
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        if 'warmup_iters' in hyperparameters['gen']:
            hyperparameters['warmup_iters'] = hyperparameters['gen']['warmup_iters']
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)
        if 'warmup_iters' in hyperparameters['lane']:
            hyperparameters['warmup_iters'] = hyperparameters['lane']['warmup_iters']
        self.lane_scheduler = get_scheduler(self.lane_opt, hyperparameters)

        # Mixed precision training
        self.scaler = amp.GradScaler() if hyperparameters["mixed_precision"] else None

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))
        self.dis_a.apply(weights_init('gaussian'))
        self.dis_b.apply(weights_init('gaussian'))

        # Load VGG model if needed
        if 'vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0:
            self.vgg = load_vgg16(hyperparameters['vgg_model_path'] + '/models/vgg16.pth')
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False
            if hyperparameters['multi_gpu']:
                self.vgg = DataParallel(self.vgg)

        # Logging additional losses and metrics
        self.log_dict = {}  # updated per log iter
        self.metric_log_dict = {}  # updated per epoch
        self.metric_dict = get_metric_dict(hyperparameters['lane'])
        self.metric_dict_cyc = get_metric_dict(hyperparameters['lane'])

    def _recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def forward(self, x_a, x_b):
        self.eval()
        h_a, _ = self.gen_a.encode(x_a)
        h_b, _ = self.gen_b.encode(x_b)
        x_ba = self.gen_a.decode(h_b)
        x_ab = self.gen_b.decode(h_a)
        self.train()
        return x_ab, x_ba

    def eval_lanes(self, x):
        self.eval()
        h_b, _ = self.gen_b.encode(x)
        preds = self.lane_model(h_b)
        self.train()
        return preds

    def _compute_kl(self, mu):
        # def _compute_kl(self, mu, sd):
        # mu_2 = torch.pow(mu, 2)
        # sd_2 = torch.pow(sd, 2)
        # encoding_loss = (mu_2 + sd_2 - torch.log(sd_2)).sum() / mu_2.size(0)
        # return encoding_loss
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

    def reset_metrics(self):
        reset_metrics(self.metric_dict)
        reset_metrics(self.metric_dict_cyc)

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
        self.gen_opt.zero_grad()
        self.lane_opt.zero_grad()
        with amp.autocast(enabled=hyperparameters["mixed_precision"]):
            # encode
            h_a, n_a = self.gen_a.encode(x_a)
            h_b, n_b = self.gen_b.encode(x_b)
            # lane detection (within domain)
            pred_a = self.lane_model(h_a)
            # decode (within domain)
            x_a_recon = self.gen_a.decode(h_a + n_a)
            x_b_recon = self.gen_b.decode(h_b + n_b)
            # decode (cross domain)
            x_ba = self.gen_a.decode(h_b + n_b)
            x_ab = self.gen_b.decode(h_a + n_a)
            # encode again
            h_b_recon, n_b_recon = self.gen_a.encode(x_ba)
            h_a_recon, n_a_recon = self.gen_b.encode(x_ab)
            # lane detection (cyclic)
            pred_a_cyc = self.lane_model(h_a_recon)
            # decode again (if needed)
            x_aba = self.gen_a.decode(h_a_recon + n_a_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None
            x_bab = self.gen_b.decode(h_b_recon + n_b_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None

            # lane detection loss
            self.total_lane_loss_x_a = self.lane_loss(pred_a, y_a)
            self._log_lane_losses("x_a")
            self._log_lane_metrics(self.metric_dict, pred_a, y_a, "x_a")
            self.total_lane_loss_cyc_x_a = self.lane_loss(pred_a_cyc, y_a)
            self._log_lane_losses("cyc_x_a")
            self._log_lane_metrics(self.metric_dict_cyc, pred_a_cyc, y_a, "cyc_x_a")
            # reconstruction loss
            self.loss_gen_recon_x_a = self._recon_criterion(x_a_recon, x_a)
            self.loss_gen_recon_x_b = self._recon_criterion(x_b_recon, x_b)
            self.loss_gen_recon_kl_a = self._compute_kl(h_a)
            self.loss_gen_recon_kl_b = self._compute_kl(h_b)
            self.loss_gen_cyc_x_a = self._recon_criterion(x_aba, x_a)
            self.loss_gen_cyc_x_b = self._recon_criterion(x_bab, x_b)
            self.loss_gen_recon_kl_cyc_aba = self._compute_kl(h_a_recon)
            self.loss_gen_recon_kl_cyc_bab = self._compute_kl(h_b_recon)
            # GAN loss
            self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
            self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)
            # domain-invariant perceptual loss
            self.loss_gen_vgg_a = self._compute_vgg_loss(self.vgg, x_ba, x_b) if hyperparameters['vgg_w'] > 0 else 0
            self.loss_gen_vgg_b = self._compute_vgg_loss(self.vgg, x_ab, x_a) if hyperparameters['vgg_w'] > 0 else 0
            # total loss
            self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
                                  hyperparameters['gan_w'] * self.loss_gen_adv_b + \
                                  hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                                  hyperparameters['recon_kl_w'] * self.loss_gen_recon_kl_a + \
                                  hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                                  hyperparameters['recon_kl_w'] * self.loss_gen_recon_kl_b + \
                                  hyperparameters['recon_x_cyc_w'] * self.loss_gen_cyc_x_a + \
                                  hyperparameters['recon_kl_cyc_w'] * self.loss_gen_recon_kl_cyc_aba + \
                                  hyperparameters['recon_x_cyc_w'] * self.loss_gen_cyc_x_b + \
                                  hyperparameters['recon_kl_cyc_w'] * self.loss_gen_recon_kl_cyc_bab + \
                                  hyperparameters['vgg_w'] * self.loss_gen_vgg_a + \
                                  hyperparameters['vgg_w'] * self.loss_gen_vgg_b + \
                                  hyperparameters['lane_w'] * self.total_lane_loss_x_a + \
                                  hyperparameters['lane_cyc_w'] * self.total_lane_loss_cyc_x_a

        if hyperparameters["mixed_precision"]:
            self.scaler.scale(self.loss_gen_total).backward()
            self.scaler.step(self.gen_opt)
            self.scaler.step(self.lane_opt)
        else:
            self.loss_gen_total.backward()
            self.gen_opt.step()
            self.lane_opt.step()

    def _compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def sample(self, x_a, x_b):
        self.eval()
        x_a_recon, x_b_recon, x_ba, x_ab = [], [], [], []
        for i in range(x_a.size(0)):
            h_a, _ = self.gen_a.encode(x_a[i].unsqueeze(0))
            h_b, _ = self.gen_b.encode(x_b[i].unsqueeze(0))
            x_a_recon.append(self.gen_a.decode(h_a))
            x_b_recon.append(self.gen_b.decode(h_b))
            x_ba.append(self.gen_a.decode(h_b))
            x_ab.append(self.gen_b.decode(h_a))
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba = torch.cat(x_ba)
        x_ab = torch.cat(x_ab)
        self.train()
        return x_a, x_a_recon, x_ab, x_b, x_b_recon, x_ba

    def dis_update(self, x_a, x_b, hyperparameters):
        self.dis_opt.zero_grad()
        with amp.autocast(enabled=hyperparameters["mixed_precision"]):
            # encode
            h_a, n_a = self.gen_a.encode(x_a)
            h_b, n_b = self.gen_b.encode(x_b)
            # decode (cross domain)
            x_ba = self.gen_a.decode(h_b + n_b)
            x_ab = self.gen_b.decode(h_a + n_a)
            # D loss
            self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
            self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)
            self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_a + hyperparameters['gan_w'] * self.loss_dis_b

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
        self.gen_a.load_state_dict(state_dict['a'])
        self.gen_b.load_state_dict(state_dict['b'])
        epoch = state_dict["epoch"]
        # Load lane model
        self.lane_model.load_state_dict(state_dict['lane'])
        # Load discriminators
        state_dict = torch.load(os.path.join(checkpoint_dir, "dis.pt"))
        self.dis_a.load_state_dict(state_dict['a'])
        self.dis_b.load_state_dict(state_dict['b'])
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
        torch.save({'a': self.gen_a.state_dict(), 'b': self.gen_b.state_dict(),
                    'lane': self.lane_model.state_dict(), 'epoch': epoch + 1}, gen_name)
        torch.save({'a': self.dis_a.state_dict(), 'b': self.dis_b.state_dict(), 'epoch': epoch + 1}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict(),
                    'lane': self.lane_opt.state_dict(), 'epoch': epoch + 1}, opt_name)
