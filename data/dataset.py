"""
This file taken from https://github.com/cfzd/Ultra-Fast-Lane-Detection/blob/master/data/dataset.py
"""
import torch
from PIL import Image
import os
import pdb
import numpy as np
from data.mytransforms import find_start_pos


def loader_func(path):
    return Image.open(path)


class LaneTestDataset(torch.utils.data.Dataset):
    def __init__(self, path, list_path, img_transform=None, use_cls=False, cls_map=None):
        super(LaneTestDataset, self).__init__()
        self.path = path
        self.img_transform = img_transform
        self.use_cls = use_cls
        self.cls_map = cls_map
        with open(list_path, 'r') as f:
            self.list = f.readlines()
        self.list = [l[1:] if l[0] == '/' else l for l in self.list]  # exclude the incorrect path prefix '/' of CULane

    def __getitem__(self, index):
        l_info = self.list[index].split()
        name = l_info[0]

        img_path = os.path.join(self.path, name)
        img = loader_func(img_path)

        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.use_cls:
            # Get classification labels from list
            cls_label = np.array(l_info[-4:]).astype(int)
            if self.cls_map:
                cls_label = np.vectorize(self.cls_map.get)(cls_label)
        else:
            # can't return None from dataloader so use an empty list
            cls_label = []

        return img, name, cls_label

    def __len__(self):
        return len(self.list)


class LaneDataset(torch.utils.data.Dataset):
    def __init__(self, path, list_path, img_transform=None, target_transform=None, simu_transform=None, griding_num=50,
                 image_dim=(288, 800), row_anchor=None, use_aux=False, segment_transform=None, num_lanes=4,
                 use_cls=False, return_label=False, cls_map=None):
        super(LaneDataset, self).__init__()
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.segment_transform = segment_transform
        self.simu_transform = simu_transform
        self.path = path
        self.griding_num = griding_num
        self.use_aux = use_aux
        self.num_lanes = num_lanes
        self.use_cls = use_cls
        self.return_label = return_label
        self.resize_dim = image_dim
        self.cls_map = cls_map

        # read the list of files
        with open(list_path, 'r') as f:
            self.list = f.readlines()

        self.row_anchor = row_anchor
        self.row_anchor.sort()

    def __getitem__(self, index):
        l = self.list[index]
        l_info = l.split()
        img_name, label_name = l_info[0], l_info[1]
        if img_name[0] == '/':
            img_name = img_name[1:]
            label_name = label_name[1:]

        label_path = os.path.join(self.path, label_name)
        label = loader_func(label_path)

        img_path = os.path.join(self.path, img_name)
        img = loader_func(img_path)

        if self.use_cls:
            # get classification labels from list
            cls_label_list = list(map(int, l_info[-4:]))
            cls_label = np.array(cls_label_list)
            if self.cls_map:
                cls_label = np.vectorize(self.cls_map.get)(cls_label)
        else:
            # can't return None from dataloader so use an empty list
            cls_label = []

        if self.simu_transform is not None:
            img, label = self.simu_transform(img, label)

        # get the coordinates of lanes at row anchors
        lane_pts = self._get_index(label)

        # make the coordinates to detection label
        w, h = img.size
        det_label = self._grid_pts(lane_pts, self.griding_num, w)

        if self.use_aux:
            assert self.segment_transform is not None
            seg_label = self.segment_transform(label)
        else:
            # can't return None from dataloader so use an empty list
            seg_label = []

        if self.img_transform is not None:
            img = self.img_transform(img)

        # for training unsupervised tasks, we don't need the label
        if not self.return_label:
            return img

        return img, det_label, cls_label, seg_label

    def __len__(self):
        return len(self.list)

    def _grid_pts(self, pts, num_cols, w):
        # pts : numlane,n,2
        num_lane, n, n2 = pts.shape
        col_sample = np.linspace(0, w - 1, num_cols)

        assert n2 == 2
        to_pts = np.zeros((n, num_lane))
        for i in range(num_lane):
            pti = pts[i, :, 1]
            to_pts[:, i] = np.asarray(
                [int(pt // (col_sample[1] - col_sample[0])) if pt != -1 else num_cols for pt in pti])
        return to_pts.astype(int)

    def _get_index(self, label):
        w, h = label.size
        scale_f = lambda x: int((x * 1.0 / self.resize_dim[0]) * h)
        sample_tmp = list(map(scale_f, self.row_anchor))

        all_idx = np.zeros((self.num_lanes, len(sample_tmp), 2))
        for i, r in enumerate(sample_tmp):
            label_r = np.asarray(label)[int(round(r))]
            for lane_idx in range(1, self.num_lanes + 1):
                pos = np.where(label_r == lane_idx)[0]
                if len(pos) == 0:
                    all_idx[lane_idx - 1, i, 0] = r
                    all_idx[lane_idx - 1, i, 1] = -1
                    continue
                pos = np.mean(pos)
                all_idx[lane_idx - 1, i, 0] = r
                all_idx[lane_idx - 1, i, 1] = pos

        # data augmentation: extend the lane to the boundary of image

        all_idx_cp = all_idx.copy()
        for i in range(self.num_lanes):
            if np.all(all_idx_cp[i, :, 1] == -1):
                continue
            # if there is no lane

            valid = all_idx_cp[i, :, 1] != -1
            # get all valid lane points' index
            valid_idx = all_idx_cp[i, valid, :]
            # get all valid lane points
            if valid_idx[-1, 0] == all_idx_cp[0, -1, 0]:
                # if the last valid lane point's y-coordinate is already the last y-coordinate of all rows
                # this means this lane has reached the bottom boundary of the image
                # so we skip
                continue
            if len(valid_idx) < 6:
                continue
            # if the lane is too short to extend

            valid_idx_half = valid_idx[len(valid_idx) // 2:, :]
            p = np.polyfit(valid_idx_half[:, 0], valid_idx_half[:, 1], deg=1)
            start_line = valid_idx_half[-1, 0]
            pos = find_start_pos(all_idx_cp[i, :, 0], start_line) + 1

            fitted = np.polyval(p, all_idx_cp[i, pos:, 0])
            fitted = np.array([-1 if y < 0 or y > w - 1 else y for y in fitted])

            assert np.all(all_idx_cp[i, pos:, 1] == -1)
            all_idx_cp[i, pos:, 1] = fitted
        if -1 in all_idx[:, :, 0]:
            pdb.set_trace()
        return all_idx_cp

class DatasetConverter(LaneClsDataset):
    """
    Subclass that does the same as the LaneClsDataset, but also
    returns the image path such that the file can be overwritten.
    __getitem__ is the only method changed.
    """
    def __getitem__(self, index):
        l = self.list[index]
        l_info = l.split()
        img_name, label_name = l_info[0], l_info[1]
        if img_name[0] == '/':
            img_name = img_name[1:]
            label_name = label_name[1:]

        label_path = os.path.join(self.path, label_name)
        label = loader_func(label_path)

        img_path = os.path.join(self.path, img_name)
        img = loader_func(img_path)

        if self.simu_transform is not None:
            img, label = self.simu_transform(img, label)
        lane_pts = self._get_index(label)

        # get the coordinates of lanes at row anchors

        w, h = img.size
        cls_label = self._grid_pts(lane_pts, self.griding_num, w)
        # make the coordinates to classification label
        if self.use_aux:
            assert self.segment_transform is not None
            seg_label = self.segment_transform(label)

        if self.img_transform is not None:
            img = self.img_transform(img)

        return img, cls_label, img_path

