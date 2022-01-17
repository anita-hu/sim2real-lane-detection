# MIT License
#
# Copyright (c) 2020 cfzd (from https://github.com/cfzd/Ultra-Fast-Lane-Detection)
# Copyright (c) 2021 Anita Hu (UltraFastLaneDetectionLoss class)
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

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class UltraFastLaneDetectionLoss(nn.Module):
    def __init__(self, hyperparameters):
        """ Consists of 4 losses as described in "Ultra Fast Structure-aware Deep Lane Detection"
            https://arxiv.org/abs/2004.11757
            with issues of structural losses further explained in
            https://github.com/cfzd/Ultra-Fast-Lane-Detection/issues/8#issuecomment-647004142

            segmentation loss (optional):
                auxiliary segmentation branch, uses pixel-wise cross entropy loss where
                the model output is (batch, num_of_lanes+1, input height / 4, input width / 4)
            classification loss:
                w+1 classes, where w is the number of gridding cells and
                an extra dimension is used to indicate the absence of lane. The model
                predicts the probability of selecting (w+1) gridding cells for the i-th lane
                and j-th row anchor, where the output dimension of the model is
                    (w+1) * num row anchors * num lanes
            similarity loss:
                lane points in adjacent row anchors should be close to each other. This is
                done by constraining the distribution of classification vectors over adjacent
                row anchors.
                NOTE: This works well with higher number of row anchors since the difference
                between rows is less (i.e. TuSimple->56 row anchors). However if the difference
                is large (i.e. CULane->18 row anchors) optimizing this loss may lead to
                inferior performance
            shape loss:
                the lane shape is constrained with the second-order difference which is zero
                when straight (generally most lanes are straight, even with curved lanes the
                majority is straight due to the perspective effect)
                NOTE: since the model output is w+1 where the first w dimensions is for
                localization and the last is for "no lane" case, this could harm the learning
                of the shape loss
        """
        super(UltraFastLaneDetectionLoss, self).__init__()
        self.use_aux = hyperparameters["use_aux"]
        self.use_cls = hyperparameters["use_cls"]

        cls_loss_w = 1.0
        if "cls_loss_w" in hyperparameters:
            cls_loss_w = hyperparameters["cls_loss_w"]

        self.loss_weights = {
            "det_loss": 1.0,
            "seg_loss": 1.0,
            "sim_loss": hyperparameters["sim_loss_w"],
            "shp_loss": hyperparameters["shp_loss_w"],
            "cls_loss": cls_loss_w
        }
        self.lane_losses = {
            "det_loss": SoftmaxFocalLoss(2),
            "seg_loss": torch.nn.CrossEntropyLoss(),
            "sim_loss": ParsingRelationLoss(),
            "shp_loss": ParsingRelationDis(),
            "cls_loss": torch.nn.CrossEntropyLoss()
        }
        self.current_losses = None  # for wandb logging

    def forward(self, preds, labels):
        det_label, cls_label, seg_label = labels
        det_out, cls_out, seg_out = preds

        self.current_losses = {
            "det_loss": self.lane_losses["det_loss"](det_out, det_label),
            "seg_loss": self.lane_losses["seg_loss"](seg_out, seg_label) if self.use_aux else 0,
            "sim_loss": self.lane_losses["sim_loss"](det_out),
            "shp_loss": self.lane_losses["shp_loss"](det_out),
            "cls_loss": self.lane_losses["cls_loss"](cls_out, cls_label) if self.use_cls else 0
        }

        lane_loss = 0
        for loss_name, loss_value in self.current_losses.items():
            lane_loss += loss_value * self.loss_weights[loss_name]

        return lane_loss


class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction="none")

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss > self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)


class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma, ignore_lb=255, *args, **kwargs):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss(ignore_index=ignore_lb)

    def forward(self, logits, labels):
        scores = F.softmax(logits, dim=1)
        factor = torch.pow(1. - scores, self.gamma)
        log_score = F.log_softmax(logits, dim=1)
        log_score = factor * log_score
        loss = self.nll(log_score, labels)
        return loss


class ParsingRelationLoss(nn.Module):
    def __init__(self):
        super(ParsingRelationLoss, self).__init__()

    def forward(self, logits):
        n, c, h, w = logits.shape
        loss_all = []
        for i in range(0, h - 1):
            loss_all.append(logits[:, :, i, :] - logits[:, :, i + 1, :])
        # loss0 : n,c,w
        loss = torch.cat(loss_all)
        return torch.nn.functional.smooth_l1_loss(loss, torch.zeros_like(loss))


class ParsingRelationDis(nn.Module):
    def __init__(self):
        super(ParsingRelationDis, self).__init__()
        self.l1 = torch.nn.L1Loss()
        # self.l1 = torch.nn.MSELoss()

    def forward(self, x):
        n, dim, num_rows, num_cols = x.shape
        x = torch.nn.functional.softmax(x[:, :dim - 1, :, :], dim=1)
        embedding = torch.Tensor(np.arange(dim - 1)).float().to(x.device).view(1, -1, 1, 1)
        pos = torch.sum(x * embedding, dim=1)

        diff_list1 = []
        for i in range(0, num_rows // 2):
            diff_list1.append(pos[:, i, :] - pos[:, i + 1, :])

        loss = 0
        for i in range(len(diff_list1) - 1):
            loss += self.l1(diff_list1[i], diff_list1[i + 1])
        loss /= len(diff_list1) - 1
        return loss
