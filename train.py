"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from utils import prepare_sub_folder, write_loss, get_config, write_2images, Timer
import argparse
from tqdm import tqdm
from trainers import MUNIT_Trainer, UNIT_Trainer, Baseline_Trainer
import torch
from data.dataloader import get_train_loader, get_test_loader
from evaluation.eval_wrapper import eval_lane

try:
    from itertools import izip as zip
except ImportError:  # will be 3.x series
    pass
import os
import sys
import shutil
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/unit.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument("--resume", action="store_true")
parser.add_argument('--entity', type=str, default='watonomous-perception-research',
                    help="wandb team name, set to None for default entity (username)")
parser.add_argument('--project', type=str, default='sim2real-lane-detection', help="wandb project name")
opts = parser.parse_args()

# Load experiment setting
config = get_config(opts.config)
baseline = config['trainer'] == 'Baseline'
config['vgg_model_path'] = opts.output_path
config["resume"] = opts.resume

# initialize wandb
wandb.init(entity=opts.entity, project=opts.project, config=config)

# set random seed for reproducibility
torch.manual_seed(config["random_seed"])
torch.backends.cudnn.deterministic = True

# Setup data loaders
# NOTE: By convention, dataset A will be simulation, labelled data, while dataset B will be real-world without labels
print(f"Loading {config['datasetA']} as dataset A. (labelled, simulated)")
train_loader_a = get_train_loader(config["batch_size"], config["dataA_root"],
                                  griding_num=config["lane"]["griding_num"], dataset=config["datasetA"],
                                  use_aux=config["lane"]["use_aux"], distributed=False,
                                  num_lanes=config["lane"]["num_lanes"], baseline=baseline,
                                  image_dim=(config["input_height"], config["input_width"]),
                                  return_label=True)

print(f"Loading {config['datasetB']} as dataset B.")
train_loader_b = get_train_loader(config["batch_size"], config["dataB_root"],
                                  griding_num=config["lane"]["griding_num"], dataset=config["datasetB"],
                                  use_aux=config["lane"]["use_aux"], distributed=False,
                                  num_lanes=config["lane"]["num_lanes"], baseline=baseline,
                                  image_dim=(config["input_height"], config["input_width"]))

val_loader_b = get_test_loader(batch_size=config["batch_size"], data_root=config["dataB_root"],
                               dataset=config["datasetB"], distributed=False,
                               image_dim=(config["input_height"], config["input_width"]),
                               partition="val")

# Setup model
print(f"Loading {config['trainer']} trainer")
if config['trainer'] == 'MUNIT':
    trainer = MUNIT_Trainer(config)
elif config['trainer'] == 'UNIT':
    trainer = UNIT_Trainer(config)
elif config['trainer'] == 'Baseline':
    trainer = Baseline_Trainer(config)
else:
    sys.exit("Only support MUNIT|UNIT|Baseline")
trainer.cuda()

if not baseline:
    display_size = config['display_size']
    train_display_images_a = torch.stack([train_loader_a.dataset[i][0] for i in range(display_size)]).cuda()
    train_display_images_b = torch.stack([train_loader_b.dataset[i] for i in range(display_size)]).cuda()
    test_display_images_b = torch.stack([val_loader_b.dataset[i][0] for i in range(display_size)]).cuda()

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))  # copy config file to output folder

# Start training
start_epoch = trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0
print("Beginning training..")
best_val_metric = 0
iter_per_epoch = min(len(train_loader_a), len(train_loader_b))
iterations = start_epoch * iter_per_epoch if opts.resume else 0
for epoch in range(start_epoch, config['max_epoch']):
    print("Training epoch", epoch + 1)
    for image_label_a, image_b in tqdm(zip(train_loader_a, train_loader_b), total=iter_per_epoch):
        if config["lane"]["use_aux"]:
            images_a, cls_label, seg_label = image_label_a
            images_a = images_a.cuda().detach()
            label_a = (cls_label.long().cuda().detach(), seg_label.long().cuda().detach())
        else:
            images_a, cls_label = image_label_a
            images_a, label_a = images_a.cuda(), cls_label.long().cuda()

        images_b = image_b.cuda().detach()

        # Main training code
        trainer.dis_update(images_a, images_b, config)
        trainer.gen_update(images_a, images_b, label_a, config)

        if config["mixed_precision"]:
            trainer.scaler.update()

        torch.cuda.synchronize()

        if (iterations + 1) % config['log_iter'] == 0:
            write_loss(iterations + 1, trainer)

            if trainer.dis_scheduler is not None:
                wandb.log({"dis_lr": trainer.dis_scheduler.get_last_lr()[0]}, step=(iterations + 1))
            if trainer.gen_scheduler is not None:
                wandb.log({"gen_lr": trainer.gen_scheduler.get_last_lr()[0]}, step=(iterations + 1))
            if trainer.lane_scheduler is not None:
                wandb.log({"lane_lr": trainer.lane_scheduler.get_last_lr()[0]}, step=(iterations + 1))

        trainer.update_learning_rate()
        iterations += 1

    # Write train metrics
    log_dict = trainer.metric_log_dict
    log_dict["epoch"] = epoch + 1
    wandb.log(log_dict, step=iterations)
    trainer.reset_metrics()

    # Write images
    if not baseline and (epoch + 1) % config['image_save_epoch'] == 0:
        with torch.no_grad():
            test_image_outputs = trainer.sample(train_display_images_a, test_display_images_b)
            train_image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
        write_2images(test_image_outputs, display_size, epoch + 1, 'test', step=iterations)
        write_2images(train_image_outputs, display_size, epoch + 1, 'train', step=iterations)

    print("Validating epoch", epoch + 1)
    with Timer("Elapsed time in validation: %f"):
        log_dict, val_metric = eval_lane(trainer, config['datasetB'], config['dataB_root'], val_loader_b,
                                         output_directory, config['lane']['griding_num'], config['lane']['use_aux'],
                                         "val")

    log_dict["epoch"] = epoch + 1
    wandb.log(log_dict, step=iterations)

    # Save network weights
    if val_metric > best_val_metric:
        trainer.save(checkpoint_directory, epoch)
        best_val_metric = val_metric
        print("Saved best model at epoch", epoch + 1)
