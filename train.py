"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from utils import prepare_sub_folder, write_loss, get_config, write_2images, Timer
import argparse
from tqdm import tqdm
from trainers import MUNIT_Trainer, UNIT_Trainer, Baseline_Trainer, ADA_Trainer
import torch
from data.dataloader import get_train_loader, get_test_loader
from data.constants import wato_2class_mapping, wato_3class_mapping, tusimple_2class_mapping, tusimple_3class_mapping
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
no_adv_gen = config['trainer'] in ['Baseline', 'ADA']
config['vgg_model_path'] = opts.output_path
config["resume"] = opts.resume

# Initialize wandb
os.environ['WANDB_CONSOLE'] = 'off'
run = wandb.init(entity=opts.entity, project=opts.project, config=config)

# Set random seed for reproducibility
torch.manual_seed(config["random_seed"])
torch.backends.cudnn.deterministic = True

# TuSimple class mapping
train_cls_map, val_cls_map = None, None
if config["lane"]["use_cls"]:
    if config["lane"]["num_classes"] == 3:
        train_cls_map, val_cls_map = wato_2class_mapping, tusimple_2class_mapping
    elif config["lane"]["num_classes"] == 4:
        train_cls_map, val_cls_map = wato_3class_mapping, tusimple_3class_mapping
    else:
        raise ValueError("Only support 3|4 lane classes, see data/constants.py for mapping")

# Setup data loaders
# NOTE: By convention, dataset A will be simulation, labelled data, while dataset B will be real-world without labels
print(f"Loading dataset A (labelled, simulated) from {config['dataA_root']}")
train_loader_a = get_train_loader(
    config["batch_size"],
    config["dataA_root"],
    griding_num=config["lane"]["griding_num"],
    dataset=config["dataset"],
    use_aux=config["lane"]["use_aux"],
    distributed=False,
    num_lanes=config["lane"]["num_lanes"],
    use_cls=config["lane"]["use_cls"],
    baseline=baseline,
    image_dim=(config["input_height"], config["input_width"]),
    return_label=True,
    cls_map=train_cls_map
)

print(f"Loading dataset B (unlabelled, real-world) from {config['dataB_root']}")
train_loader_b = get_train_loader(
    config["batch_size"],
    config["dataB_root"],
    griding_num=config["lane"]["griding_num"],
    dataset=config["dataset"],
    use_aux=config["lane"]["use_aux"],
    distributed=False,
    num_lanes=config["lane"]["num_lanes"],
    use_cls=config["lane"]["use_cls"],
    baseline=baseline,
    image_dim=(config["input_height"], config["input_width"])
)

val_loader_b = get_test_loader(
    batch_size=config["batch_size"],
    data_root=config["dataB_root"],
    distributed=False,
    use_cls=config["lane"]["use_cls"],
    image_dim=(config["input_height"], config["input_width"]),
    partition="val",
    cls_map=val_cls_map,
)

# Setup model
print(f"Loading {config['trainer']} trainer")
if config['trainer'] == 'MUNIT':
    trainer = MUNIT_Trainer(config)
elif config['trainer'] == 'UNIT':
    trainer = UNIT_Trainer(config)
elif config['trainer'] == 'Baseline':
    trainer = Baseline_Trainer(config)
elif config['trainer'] == 'ADA':
    trainer = ADA_Trainer(config)
else:
    raise ValueError("Only support MUNIT|UNIT|Baseline|ADA")

trainer.cuda()

# Sample images for GAN image logging
if not no_adv_gen:
    display_size = config['display_size']
    train_display_images_a = torch.stack([train_loader_a.dataset[i][0] for i in range(display_size)]).cuda()
    train_display_images_b = torch.stack([train_loader_b.dataset[i] for i in range(display_size)]).cuda()
    test_display_images_b = torch.stack([val_loader_b.dataset[i][0] for i in range(display_size)]).cuda()

# Setup logger and output folders
output_directory = os.path.join(opts.output_path + "/outputs", run.name)
if not os.path.exists(output_directory):
    print("Creating directory: {}".format(output_directory))
    os.makedirs(output_directory)
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
        images_a, det_label, cls_label, seg_label = image_label_a

        images_a = images_a.cuda()
        images_b = image_b.cuda()

        det_label = det_label.long().cuda()
        cls_label = cls_label.long().cuda() if config["lane"]["use_cls"] else cls_label
        seg_label = seg_label.long().cuda() if config["lane"]["use_aux"] else seg_label

        label_a = (det_label, cls_label, seg_label)

        # Main training code
        trainer.dis_update(images_a, images_b, config)
        trainer.gen_update(images_a, images_b, label_a, config)

        if config["mixed_precision"]:
            trainer.scaler.update()

        torch.cuda.synchronize()

        # Log train loss and lr
        if (iterations + 1) % config['log_iter'] == 0:
            write_loss(iterations + 1, trainer)

            if trainer.dis_scheduler is not None:
                wandb.log({"lr/dis_lr": trainer.dis_scheduler.get_last_lr()[0]}, step=(iterations + 1))
            if trainer.gen_scheduler is not None:
                wandb.log({"lr/gen_lr": trainer.gen_scheduler.get_last_lr()[0]}, step=(iterations + 1))
            if trainer.lane_scheduler is not None:
                wandb.log({"lr/lane_lr": trainer.lane_scheduler.get_last_lr()[0]}, step=(iterations + 1))

        trainer.update_learning_rate()
        iterations += 1

    # Log train metrics
    log_dict = trainer.metric_log_dict
    log_dict["epoch"] = epoch + 1
    wandb.log(log_dict, step=iterations)
    trainer.reset_metrics()

    # Log GAN images
    if not no_adv_gen and (epoch + 1) % config['image_save_epoch'] == 0:
        with torch.no_grad():
            test_image_outputs = trainer.sample(train_display_images_a, test_display_images_b)
            train_image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
        write_2images(test_image_outputs, display_size, epoch + 1, 'test', step=iterations)
        write_2images(train_image_outputs, display_size, epoch + 1, 'train', step=iterations)

    # Run validation
    print("Validating epoch", epoch + 1)
    with Timer("Elapsed time in validation: %f"):
        log_dict, val_metric = eval_lane(trainer, config['dataset'], config['dataB_root'], val_loader_b,
                                         output_directory, config['lane']['griding_num'],
                                         config['lane']['use_cls'], "val")

    log_dict["epoch"] = epoch + 1
    wandb.log(log_dict, step=iterations)

    # Save network weights
    if val_metric > best_val_metric:
        trainer.save(checkpoint_directory, epoch)
        best_val_metric = val_metric
        print("Saved best model at epoch", epoch + 1)
