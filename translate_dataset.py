"""
this file is for creating a dataset of _translated images_ using an image translator,
namely munit, unit, or drit. The dataset is replicated, and then for every image in the
simulated dataset it's translated to the "real" domain and then saved, overwritten.
The original dataset is specified in the config file, datasetA. That dataset
is cloned and then modified.
This script uses `cp` to copy the whole dataset before modifying it in place.
If the script is hanging, it's likely that `cp` cannot complete, likely due
to a permission error.
"""
import os
import sys
from data.dataset import DatasetConverter
from trainers import MUNIT_Trainer, UNIT_Trainer
from utils import get_config
import torchvision.transforms as transforms
from data.mytransforms import UnNormalize
from torchvision.utils import save_image
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/tusimple/munit_s2r.yaml', help='Path to the config file.')
parser.add_argument('--new_data_folder', type=str, help='root folder to generate the new dataset.')
parser.add_argument('--checkpoint_dir', type=str, help='directory containing model weights, in .pt files')
parser.add_argument('--vgg_model_path', type=str, default='.', help="parent folder of vgg model.pth")
opts = parser.parse_args()

config = get_config(opts.config)
config['vgg_model_path'] = opts.vgg_model_path

# clone the dataset folder

dataset_root = config["dataA_root"]
new_dataset = opts.new_data_folder

print("Making a copy of the dataset")
print(f"cp -r {dataset_root} {new_dataset}")
os.system(f"cp -r {dataset_root} {new_dataset}")
print("Done copying the dataset")

# Setup model

print(f"Loading {config['trainer']} trainer")
if config['trainer'] == 'MUNIT':
    config['display_size'] = config['batch_size']
    trainer = MUNIT_Trainer(config)
elif config['trainer'] == 'UNIT':
    trainer = UNIT_Trainer(config)
else:
    sys.exit("Only support MUNIT|UNIT")

trainer.resume(opts.checkpoint_dir, config)
trainer.cuda()

# Initialize dataset

image_dim = (config["input_height"], config["input_width"])

img_transform = transforms.Compose([
    transforms.Resize(image_dim),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def get_tusimple_row_anchor(image_height):
    return [int((160 + i * 10) / 720 * image_height) for i in range(56)]


dataset_to_translate = DatasetConverter(new_dataset + "WATO_TuSimple",
                                        os.path.join(new_dataset, 'WATO_TuSimple/list/train_gt.txt'),
                                        img_transform=img_transform,
                                        image_dim=image_dim,
                                        row_anchor=get_tusimple_row_anchor(image_dim[0]),
                                        return_label=False)

dataset_root = config["dataA_root"]
iterator = torch.utils.data.DataLoader(dataset_to_translate, batch_size=config["batch_size"],
                                       drop_last=True)

unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

print("Starting conversion")
upsample = nn.Upsample(size=(720, 1280), mode='nearest')
with torch.no_grad():
    for el in tqdm(iterator):
        images, labels, image_paths = el
        sim2real, real2sim = trainer.forward(images.cuda(), images.cuda())
        for i, image_path in enumerate(image_paths):
            image_tensor = unorm(sim2real[i])
            image_tensor = upsample(image_tensor.unsqueeze(0)).squeeze()
            save_image(image_tensor.cpu(), image_path)  # overwrite
