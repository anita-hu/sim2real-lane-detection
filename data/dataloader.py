"""
https://github.com/cfzd/Ultra-Fast-Lane-Detection/blob/master/data/dataloader.py
"""

import torch, os
import numpy as np

import torchvision.transforms as transforms
import data.mytransforms as mytransforms
from data.dataset import LaneClsDataset, LaneTestDataset


def get_tusimple_row_anchor(image_height):
    return [int((160+i*10)/720*image_height) for i in range(56)]


def get_culane_row_anchor(image_height):
    return [int(image_height-i*20/590*image_height)-1 for i in range(18)]


def get_train_loader(batch_size, data_root, griding_num, dataset, use_aux, distributed, num_lanes, image_dim=(288, 800),
                     return_label=False):
    target_transform = transforms.Compose([
        mytransforms.FreeScaleMask(image_dim),
        mytransforms.MaskToTensor(),
    ])
    segment_transform = transforms.Compose([
        mytransforms.FreeScaleMask((image_dim[0]//4, image_dim[1]//4)),
        mytransforms.MaskToTensor(),
    ])
    img_transform = transforms.Compose([
        transforms.Resize(image_dim),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    simu_transform = mytransforms.Compose2([
        mytransforms.RandomRotate(6),
        mytransforms.RandomUDoffsetLABEL(100),
        mytransforms.RandomLROffsetLABEL(200)
    ])
    if dataset == 'CULane':
        culane_row_anchor = get_culane_row_anchor(image_dim[0])
        train_dataset = LaneClsDataset(data_root,
                                       os.path.join(data_root, 'list/train_gt.txt'),
                                       img_transform=img_transform,
                                       target_transform=target_transform,
                                       simu_transform=simu_transform,
                                       segment_transform=segment_transform,
                                       row_anchor=culane_row_anchor,
                                       griding_num=griding_num,
                                       image_dim=image_dim,
                                       use_aux=use_aux,
                                       num_lanes=num_lanes,
                                       return_label=return_label)

    elif dataset == 'TuSimple':
        tusimple_row_anchor = get_tusimple_row_anchor(image_dim[0])
        train_dataset = LaneClsDataset(data_root,
                                       os.path.join(data_root, 'train_gt.txt'),
                                       img_transform=img_transform,
                                       target_transform=target_transform,
                                       simu_transform=simu_transform,
                                       griding_num=griding_num,
                                       row_anchor=tusimple_row_anchor,
                                       image_dim=image_dim,
                                       segment_transform=segment_transform,
                                       use_aux=use_aux,
                                       num_lanes=num_lanes,
                                       return_label=return_label)

    elif dataset == 'WATO':
        tusimple_row_anchor = get_tusimple_row_anchor(image_dim[0])
        train_dataset = LaneClsDataset(data_root,
                                       os.path.join(data_root, 'train_gt.txt'),
                                       img_transform=img_transform,
                                       target_transform=target_transform,
                                       simu_transform=simu_transform,
                                       griding_num=griding_num,
                                       row_anchor=tusimple_row_anchor,
                                       image_dim=image_dim,
                                       segment_transform=segment_transform,
                                       use_aux=use_aux,
                                       num_lanes=num_lanes,
                                       return_label=return_label)
    else:
        raise NotImplementedError

    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        sampler = torch.utils.data.RandomSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4)

    return train_loader


def get_test_loader(batch_size, data_root, dataset, distributed, image_dim=(288, 800)):
    img_transforms = transforms.Compose([
        transforms.Resize(image_dim),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    if dataset == 'CULane':
        test_dataset = LaneTestDataset(data_root, os.path.join(data_root, 'list/test.txt'),
                                       img_transform=img_transforms)
    elif dataset == 'TuSimple':
        test_dataset = LaneTestDataset(data_root, os.path.join(data_root, 'test.txt'), img_transform=img_transforms)

    elif dataset == 'WATO':
        test_dataset = LaneTestDataset(data_root, os.path.join(data_root, 'test.txt'), img_transform=img_transforms)

    if distributed:
        sampler = SeqDistributedSampler(test_dataset, shuffle=False)
    else:
        sampler = torch.utils.data.SequentialSampler(test_dataset)
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler=sampler, num_workers=4)
    return loader


class SeqDistributedSampler(torch.utils.data.distributed.DistributedSampler):
    '''
    Change the behavior of DistributedSampler to sequential distributed sampling.
    The sequential sampling helps the stability of multi-thread testing, which needs multi-thread file io.
    Without sequentially sampling, the file io on thread may interfere other threads.
    '''

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False):
        super().__init__(dataset, num_replicas, rank, shuffle)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        num_per_rank = int(self.total_size // self.num_replicas)

        # sequential sampling
        indices = indices[num_per_rank * self.rank: num_per_rank * (self.rank + 1)]

        assert len(indices) == self.num_samples

        return iter(indices)
