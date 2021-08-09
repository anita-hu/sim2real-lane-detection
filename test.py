"""
Based on https://github.com/cfzd/Ultra-Fast-Lane-Detection/blob/master/test.py
"""
import argparse
import os
import sys
import torch
from utils import get_config
from trainers import MUNIT_Trainer, UNIT_Trainer, Baseline_Trainer, ADA_Trainer
from data.dataloader import get_test_loader
from evaluation.eval_wrapper import eval_lane

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help="net configuration")
parser.add_argument('--output_folder', type=str, help="output folder for results")
parser.add_argument('--checkpoint', type=str, help="checkpoint of autoencoders and lane model")
parser.add_argument('--distributed', action='store_true', help="whether use distributed testing")
parser.add_argument('--local_rank', type=int, default=0)
opts = parser.parse_args()

torch.backends.cudnn.benchmark = True

# Load experiment setting
config = get_config(opts.config)
torch.manual_seed(config['random_seed'])
torch.cuda.manual_seed(config['random_seed'])

# Setup model and data loader
config['vgg_w'] = 0  # do not load vgg model
config['lane']['use_aux'] = False  # no aux segmentation branch
if config['trainer'] == 'MUNIT':
    style_dim = config['gen']['style_dim']
    trainer = MUNIT_Trainer(config)
elif config['trainer'] == 'UNIT':
    trainer = UNIT_Trainer(config)
elif config['trainer'] == 'Baseline':
    trainer = Baseline_Trainer(config)
elif config['trainer'] == 'ADA':
    trainer = ADA_Trainer(config)
else:
    sys.exit("Only support MUNIT|UNIT|Baseline")

state_dict = torch.load(opts.checkpoint)
# assume gen_a is for simulation data and gen_b is for real data
if config['trainer'] == 'Baseline':
    trainer.backbone.load_state_dict(state_dict['b'])
else:
    trainer.gen_b.load_state_dict(state_dict['b'])
trainer.lane_model.load_state_dict(state_dict['lane'], strict=False)  # don't load aux

trainer.cuda()
trainer.eval()

distributed = False
if 'WORLD_SIZE' in os.environ:
    distributed = int(os.environ['WORLD_SIZE']) > 1

if distributed:
    torch.cuda.set_device(opts.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

print('Start testing...')

if config['datasetB'] == 'CULane':
    num_anchors = 18
elif config['datasetB'] == 'TuSimple':
    num_anchors = 56
else:
    raise NotImplementedError("Only support CULane|TuSimple")

loader = get_test_loader(
    batch_size=config["batch_size"],
    data_root=config["dataB_root"],
    distributed=False, 
    use_cls=config["lane"]["use_cls"],
    image_dim=(config["input_height"], config["input_width"])
)

if distributed:
    net = torch.nn.parallel.DistributedDataParallel(trainer, device_ids=[opts.local_rank])

if not os.path.exists(opts.output_folder):
    os.mkdir(opts.output_folder)

eval_lane(
    net=trainer,
    dataset=config['dataset'],
    data_root=config['dataB_root'],
    loader=loader,
    work_dir=opts.output_folder,
    griding_num=config['lane']['griding_num'],
    use_cls=config["lane"]["use_cls"]
)
