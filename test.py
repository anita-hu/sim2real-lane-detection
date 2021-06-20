"""
Based on https://github.com/cfzd/Ultra-Fast-Lane-Detection/blob/master/test.py
"""
import argparse
import os
import sys
import torch
from utils import get_config
from trainers import MUNIT_Trainer, UNIT_Trainer
from evaluation.eval_wrapper import eval_lane


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help="net configuration")
parser.add_argument('--input', type=str, help="input image path")
parser.add_argument('--output_folder', type=str, help="output folder for results")
parser.add_argument('--checkpoint', type=str, help="checkpoint of autoencoders and lane model")
parser.add_argument('--seed', type=int, default=10, help="random seed")
parser.add_argument('--distributed', action='store_true', help="whether use distributed testing")
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--output_path', type=str, default='.', help="path for logs, checkpoints, and VGG model weight")
parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")
opts = parser.parse_args()


torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)
torch.backends.cudnn.benchmark = True

# Load experiment setting
config = get_config(opts.config)
opts.num_style = 1 if opts.style != '' else opts.num_style

# Setup model and data loader
config['vgg_model_path'] = opts.output_path
config['lane']['use_aux'] = False
if opts.trainer == 'MUNIT':
    style_dim = config['gen']['style_dim']
    trainer = MUNIT_Trainer(config)
elif opts.trainer == 'UNIT':
    trainer = UNIT_Trainer(config)
else:
    sys.exit("Only support MUNIT|UNIT")

state_dict = torch.load(opts.checkpoint)
# assume gen_a is for simulation data and gen_b is for real data
trainer.gen_b.load_state_dict(state_dict['b'])
trainer.lane_model.load_state_dict(state_dict['lane'])

trainer.cuda()
trainer.eval()

if 'WORLD_SIZE' in os.environ:
    distributed = int(os.environ['WORLD_SIZE']) > 1

if distributed:
    torch.cuda.set_device(opts.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

print('start testing...')

if config['datasetB'] == 'CULane':
    cls_num_per_lane = 18
elif config['datasetB'] == 'Tusimple':
    cls_num_per_lane = 56
else:
    raise NotImplementedError

if distributed:
    net = torch.nn.parallel.DistributedDataParallel(trainer, device_ids=[opts.local_rank])

if not os.path.exists(opts.output_folder):
    os.mkdir(opts.output_folder)

eval_lane(trainer, config['datasetB'], config['dataB_root'], opts.output_folder, config['lane']['griding_num'], False,
          distributed)
