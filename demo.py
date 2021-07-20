"""
Based on https://github.com/cfzd/Ultra-Fast-Lane-Detection/blob/master/demo.py
"""
import argparse
import os

import cv2
import numpy as np
import scipy.special
import torch
import torchvision.transforms as transforms
import tqdm

from data.dataloader import get_culane_row_anchor, get_tusimple_row_anchor
from data.dataset import LaneTestDataset
from trainers.baseline_trainer import Baseline_Trainer
from trainers.munit_trainer import MUNIT_Trainer
from trainers.unit_trainer import UNIT_Trainer
from utils import get_config


def generate_video_demo(model, dataloader: torch.utils.data.DataLoader,
                        dataset_root: str, image_dims, griding_num,
                        outfile="demo.avi", framerate=30.0):
    """
    generates a video of sequential frames from the dataset, using the model
    to predict the lanes.
    :image_dims: (w, h)
    """

    outfile = os.path.join("outputs/", outfile)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    vout = cv2.VideoWriter(outfile, fourcc , framerate, image_dims)

    print(f"writing output to {outfile}.")
    for i, data in enumerate(tqdm.tqdm(dataloader)):
        imgs, names = data
        imgs = imgs.cuda()
        with torch.no_grad():
            out = model.eval_lanes(imgs)  # finally, run the images through the model.

        col_sample = np.linspace(0, 800 - 1, griding_num)
        col_sample_w = col_sample[1] - col_sample[0]


        out_j = out[0].data.cpu().numpy()
        out_j = out_j[:, ::-1, :]
        prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
        idx = np.arange(griding_num) + 1
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0)
        out_j = np.argmax(out_j, axis=0)
        loc[out_j == griding_num] = 0
        out_j = loc

        # import pdb; pdb.set_trace()
        filename = os.path.join(dataset_root, names[0])
        vis = cv2.imread(filename)  # read the original file from the dataset
        if vis is None:
            raise ValueError(f"openCV failed to read a frame. Does the file {filename} exist?")
        for i in range(out_j.shape[1]):
            if np.sum(out_j[:, i] != 0) > 2:
                for k in range(out_j.shape[0]):
                    if out_j[k, i] > 0:
                        # TODO debug this Sinclair, check math
                        ppp = (int(out_j[k, i] * col_sample_w * image_dims[0] / 800) - 1, \
                               int(image_dims[1] * (row_anchor[cls_num_per_lane-1-k]/288)) - 1 )
                        vis = cv2.circle(vis,ppp,5,(0,255,0),-1)
        vout.write(vis)  # write a frame
    vout.release()  # release lock for writing video


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help="model configuration, should match the training config that was used to run the checkpoint")
    parser.add_argument('--checkpoint', type=str, help="state dict for the model, to be loaded in and used for the demo.")
    parser.add_argument('--output_path', type=str, default='demo.avi', help="output video file")
    opts = parser.parse_args()

    print("starting to test")


    ############################################################################
    # Dataset
    ############################################################################

    cfg = get_config(opts.config)

    # image transforms for the testset
    img_transforms = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    if cfg["datasetB"] == 'CULane':
        cls_num_per_lane = 18
        splits = ['test0_normal.txt', 'test1_crowd.txt', 'test2_hlight.txt', 'test3_shadow.txt', 'test4_noline.txt', 'test5_arrow.txt', 'test6_curve.txt', 'test7_cross.txt', 'test8_night.txt']
        datasets = [LaneTestDataset(cfg["dataB_root"], os.path.join(cfg["dataB_root"], 'list/test_split/'+split),img_transform = img_transforms) for split in splits]
        img_w, img_h = 1640, 590
        row_anchor = get_culane_row_anchor(img_h)

    elif cfg["datasetB"] == 'TuSimple':
        cls_num_per_lane = 56
        splits = ['test.txt']
        datasets = [LaneTestDataset(cfg["dataB_root"], os.path.join(cfg["dataB_root"], split),
                                    img_transform = img_transforms) for split in splits]
        img_w, img_h = 1280, 720
        row_anchor = get_tusimple_row_anchor(img_h)
    else:
        raise NotImplementedError


    ############################################################################
    # Model
    ############################################################################

    cfg['vgg_w'] = 0  # do not load vgg model
    cfg['lane']['use_aux'] = False  # no aux segmentation branch

    if cfg['trainer'] == 'MUNIT':
        style_dim = cfg['gen']['style_dim']
        trainer = MUNIT_Trainer(cfg)
    elif cfg['trainer'] == 'UNIT':
        trainer = UNIT_Trainer(cfg)
    elif cfg['trainer'] == 'Baseline':
        trainer = Baseline_Trainer(cfg)
    else:
        sys.exit("Only support MUNIT|UNIT|Baseline")
        trainer = UNIT_Trainer(cfg)  # never reached, but stops linter from complaining

    state_dict = torch.load(opts.checkpoint)
    # assume gen_a is for simulation data and gen_b is for real data
    if cfg['trainer'] == 'Baseline':
        trainer.backbone.load_state_dict(state_dict['b'])
    else:
        trainer.gen_b.load_state_dict(state_dict['b'])
    trainer.lane_model.load_state_dict(state_dict['lane'], strict=False)  # don't load aux

    trainer.cuda()
    trainer.eval()

    for split, dataset in zip(splits, datasets):
        # TODO technically, we would need a different outfile name for each dataset
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle = False, num_workers=1)
        generate_video_demo(trainer, loader, cfg["dataB_root"], (img_w, img_h),
                            griding_num=cfg["lane"]["griding_num"], outfile=opts.output_path)

